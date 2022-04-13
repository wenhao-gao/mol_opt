""" Contains for for Gaussian process Bayesian optimization """

import logging
import random
import pprint
import heapq
from typing import Union

import numpy as np
import torch
from tqdm import tqdm 

from function_utils import CachedFunction, CachedBatchFunction
from gp import TanimotoGP, batch_predict_mu_var_numpy
from graph_ga.graph_ga import run_ga_maximization


# Logger with standard handler
logger = logging.getLogger("gp_bo")
if len(logger.handlers) == 0:
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)


# Optimize acquisition function with genetic algorithm
def maximize_acquisition_func_ga(
    gp_model: TanimotoGP,
    acq_func_np: callable,
    starting_smiles: list,
    smiles_to_np_fingerprint: callable,
    **ga_kwargs,
):

    # Construct acquisition function for GA
    def _acq_func_smiles(smiles_list):
        fp_array = np.stack(list(map(smiles_to_np_fingerprint, smiles_list)))
        if gp_model.train_inputs[0].dtype == torch.float32:
            fp_array = fp_array.astype(np.float32)
        elif gp_model.train_inputs[0].dtype == torch.float64:
            fp_array = fp_array.astype(np.float64)
        else:
            raise ValueError(gp_model.train_inputs[0].dtype)
        mu_pred, var_pred = batch_predict_mu_var_numpy(
            gp_model, torch.as_tensor(fp_array), batch_size=2 ** 15
        )
        acq_vals = acq_func_np(mu_pred, var_pred)
        return list(map(float, acq_vals))

    cached_acq_function = CachedBatchFunction(_acq_func_smiles)

    # Run GA
    _, smiles_2_acq_dict, _ = run_ga_maximization(
        starting_population_smiles=list(starting_smiles),
        scoring_function=cached_acq_function,
        **ga_kwargs,
    )

    # Sort and return results (highest acq func first)
    sm_ac_list = list(smiles_2_acq_dict.items())
    sm_ac_list.sort(reverse=True, key=lambda t: t[1])
    smiles_out = [s for s, v in sm_ac_list]
    acq_out = [v for s, v in sm_ac_list]
    return smiles_out, acq_out


# Whole GP BO loop
def gp_bo_loop(
    gp_model,
    scoring_function,
    smiles_to_np_fingerprint: callable,
    acq_func_of_time: callable,
    max_bo_iter: int,
    bo_batch_size: int = 1,
    y_transform: callable = None,
    gp_train_smiles: list = None,
    smiles_pool: list = None,
    max_func_calls: int = None,
    ga_pool_num_best: int = 250,
    ga_pool_num_carryover: int = 250,  # number of SMILES with high acq funcs to carry over from last time
    max_ga_start_population_size: int = 1000,
    ga_population_size: int = 500,
    ga_max_generations: int = 25,
    ga_offspring_size: int = 1000,
    ga_mutation_rate: float = 1e-2,
    ga_num_cpu: int = 1,
    refit_gp_func: callable = None,
    n_top_log: int = 10,  # When I log "topN" of something, what should N be?
    log_ga_smiles: bool = False,  # whether to log all SMILES evaluated with GA.
    numpy_dtype=np.float32,  # numpy dtype to be using
):
    logger.info("Starting GP BO")

    # Set up which SMILES the GP should be trained on
    # If not given, it is assumed that the GP is trained on all known smiles
    start_cache = scoring_function.mol_buffer 
    start_cache_size = len(scoring_function) 
    if gp_train_smiles is None:
        logger.debug(
            "No GP training SMILES given. "
            f"Will default to training on the {start_cache_size} SMILES with known scores."
        )
        gp_train_smiles_set = set(start_cache.keys())
    else:
        gp_train_smiles_set = set(gp_train_smiles)
    del gp_train_smiles  # should refer to new variables later on; don't want to use by mistake

    # Keep a pool of all SMILES encountered (used for seeding GA)
    if smiles_pool is None:
        smiles_pool = set()
    else:
        smiles_pool = set(smiles_pool)
    smiles_pool.update(start_cache.keys())
    smiles_pool.update(gp_train_smiles_set)
    logger.debug(f"SMILES pool created, size={len(smiles_pool)}")
    assert (
        len(smiles_pool) > 0
    ), "No SMILES were provided to the algorithm as training data, known scores, or a SMILES pool."

    # Handle edge case of no training data
    if len(gp_train_smiles_set) == 0:
        logger.warning(
            f"No SMILES were provided to train GP. A random one will be chosen from the pool to start training."
        )
        random_smiles = random.choice(list(smiles_pool))
        logger.debug(f"The following SMILES was chosen:\n\t{random_smiles}")
        gp_train_smiles_set.add(random_smiles)
        del random_smiles
    if len(gp_train_smiles_set) > 0:
        logger.debug(
            f"Plan to condition GP on {len(gp_train_smiles_set)} training points."
        )

    # Evaluate scores of training data (ideally should all be known)
    num_train_data_not_known = len(gp_train_smiles_set - set(start_cache.keys()))
    if num_train_data_not_known > 0:
        logger.warning(
            f"Need to evaluate {num_train_data_not_known} unknown GP training points."
            " Probably the training points should have known scores which should be provided."
        )
    logger.debug("Scoring training points.")
    gp_train_smiles_list = list(gp_train_smiles_set)
    gp_train_smiles_scores = scoring_function(gp_train_smiles_list)
    logger.debug("Scoring of training points done.")

    # Store GP training data
    x_train_np = np.stack(
        list(map(smiles_to_np_fingerprint, gp_train_smiles_list))
    ).astype(numpy_dtype)
    y_train_np = np.array(gp_train_smiles_scores).astype(numpy_dtype)
    gp_model.set_train_data(
        inputs=torch.as_tensor(x_train_np),
        targets=torch.as_tensor(y_train_np),
        strict=False,
    )
    logger.debug("Created initial GP training data")

    # State variables for BO loop
    carryover_smiles_pool = set()
    bo_query_res = list()
    bo_state_dict = dict(
        gp_model=gp_model,
        gp_train_smiles_list=gp_train_smiles_list,
        bo_query_res=bo_query_res,
        scoring_function=scoring_function,
    )

    # Possibly re-fit GP hyperparameters
    if refit_gp_func is not None:
        logger.info("Initial fitting of GP hyperparameters")
        refit_gp_func(bo_iter=0, gp_model=gp_model, bo_state_dict=bo_state_dict)

    # Actual BO loop
    for bo_iter in tqdm(range(1, max_bo_iter + 1)):

        print(">>>> # of used oracal call ", len(scoring_function))
        if scoring_function.finish: 
            break 

        logger.info(f"Start iter {bo_iter}")

        # Make starting population for GA from a combination of
        #     1) best `ga_pool_num_best` known scores
        #     2) Up to `ga_pool_num_carryover` promising SMILES from last iteration
        #     3) Random smiles from `smiles_pool` to pad the pool
        top_smiles_at_bo_iter_start = [
            s
            for _, s in heapq.nlargest(
                ga_pool_num_best,
                [
                    (scoring_function(smiles), smiles)
                    for smiles in scoring_function.mol_buffer.keys()
                ],
            )
        ]
        ga_start_smiles = set(top_smiles_at_bo_iter_start)  # start with best
        ga_start_smiles.update(carryover_smiles_pool)  # add carryover
        if len(ga_start_smiles) < max_ga_start_population_size:
            samples_from_pool = random.sample(
                smiles_pool, min(len(smiles_pool), max_ga_start_population_size)
            )

            # Pad with random SMILES until full
            for s in samples_from_pool:
                ga_start_smiles.add(s)
                if len(ga_start_smiles) >= max_ga_start_population_size:
                    break
            del samples_from_pool

        # Current acquisition function
        curr_acq_func = acq_func_of_time(bo_iter, bo_state_dict)

        # Optimize acquisition function
        logger.debug(
            f"Maximizing acqusition function with {len(ga_start_smiles)} starting SMILES."
        )
        acq_smiles, acq_vals = maximize_acquisition_func_ga(
            gp_model=gp_model,
            acq_func_np=curr_acq_func,
            starting_smiles=list(ga_start_smiles),
            smiles_to_np_fingerprint=smiles_to_np_fingerprint,
            max_generations=ga_max_generations,
            population_size=ga_population_size,
            offspring_size=ga_offspring_size,
            mutation_rate=ga_mutation_rate,
            num_cpu=ga_num_cpu,
        )
        logger.debug(f"Acquisition function optimized, {len(acq_smiles)} evaluated.")
        _n_top = max(n_top_log, bo_batch_size + 3)
        logger.debug(
            f"Top {_n_top} acquisition function values: "
            + ", ".join([f"{v:.2f}" for v in acq_vals[:_n_top]])
        )
        del _n_top

        # Now that new SMILES were generated, add them to the pool
        _start_size = len(smiles_pool)
        smiles_pool.update(acq_smiles)
        _end_size = len(smiles_pool)
        logger.debug(
            f"{_end_size - _start_size} smiles added to pool "
            f"(size went from {_start_size} to {_end_size})"
        )
        del _start_size, _end_size

        # Greedily choose SMILES to be in the BO batch
        smiles_batch = []
        smiles_batch_acq = []
        for candidate_smiles, acq in zip(acq_smiles, acq_vals):
            if (
                candidate_smiles not in gp_train_smiles_set
                and candidate_smiles not in smiles_batch
            ):
                smiles_batch.append(candidate_smiles)
                smiles_batch_acq.append(acq)
            if len(smiles_batch) >= bo_batch_size:
                break
        del candidate_smiles, acq
        logger.debug(f"Batch created, size {len(smiles_batch)}/{bo_batch_size}")
        assert (
            len(smiles_batch) > 0
        ), "Empty batch, shouldn't happen. Must be problem with GA."
        smiles_batch_np = np.stack(
            list(map(smiles_to_np_fingerprint, smiles_batch))
        ).astype(x_train_np.dtype)

        # Get predictions about SMILES batch before training on it
        smiles_batch_mu_pre, smiles_batch_var_pre = batch_predict_mu_var_numpy(
            gp_model, torch.as_tensor(smiles_batch_np)
        )
        logger.debug("Made mean/var predictions for new SMILES batch")

        # Score these SMILES
        logger.debug(
            f"Evaluating scoring function on SMILES batch of size {len(smiles_batch)}."
        )
        smiles_batch_scores = scoring_function(smiles_batch)
        logger.debug(f"Scoring complete.")

        # Add new points to GP training data
        gp_train_smiles_list += smiles_batch
        gp_train_smiles_set.update(gp_train_smiles_list)
        x_train_np = np.concatenate([x_train_np, smiles_batch_np], axis=0)
        y_train_np = np.concatenate(
            [y_train_np, np.asarray(smiles_batch_scores, dtype=y_train_np.dtype)],
            axis=0,
        )
        gp_model.set_train_data(
            inputs=torch.as_tensor(x_train_np),
            targets=torch.as_tensor(y_train_np),
            strict=False,
        )
        logger.debug(f"GP training data reset, now of size {len(x_train_np)}")

        # Potentially refit GP hyperparameters
        if refit_gp_func is not None:
            logger.info("Re-fitting GP hyperparameters")
            refit_gp_func(
                bo_iter=bo_iter, gp_model=gp_model, bo_state_dict=bo_state_dict
            )

        # Add SMILES with high acquisition function values to the priority pool,
        # Since maybe they will have high acquisition function values next time
        carryover_smiles_pool = set()
        for s in acq_smiles:
            if (
                len(carryover_smiles_pool) < ga_pool_num_carryover
                and s not in gp_train_smiles_set
            ):
                carryover_smiles_pool.add(s)
            else:
                break

        # Get predictions about SMILES batch AFTER training on it
        smiles_batch_mu_post1, smiles_batch_var_post1 = batch_predict_mu_var_numpy(
            gp_model, torch.as_tensor(smiles_batch_np)
        )

        # Assemble full batch results
        batch_results = []
        for i, s in enumerate(smiles_batch):
            transformed_score = scoring_function(s)
            pred_dict = dict(
                mu=float(smiles_batch_mu_pre[i]),
                std=float(np.sqrt(smiles_batch_var_pre[i])),
                acq=smiles_batch_acq[i],
            )
            pred_dict["pred_error_in_stds"] = (
                pred_dict["mu"] - transformed_score
            ) / pred_dict["std"]
            pred_dict_post1 = dict(
                mu=float(smiles_batch_mu_post1[i]),
                std=float(np.sqrt(smiles_batch_var_post1[i])),
            )
            res = dict(
                bo_iter=bo_iter,
                smiles=s,
                raw_score=scoring_function(s),
                transformed_score=transformed_score,
                predictions=pred_dict,
                predictions_after_fit=pred_dict_post1,
            )
            batch_results.append(res)

            del pred_dict, pred_dict_post1, res, transformed_score
        bo_query_res.extend(batch_results)
        logger.debug("Full batch results:\n" + pprint.pformat(batch_results))

        # Potentially add GA info to batch
        if log_ga_smiles:
            batch_results[0]["ga_info"] = dict(
                ga_start_smiles=ga_start_smiles,
                ga_eval_smiles=acq_smiles,
            )

        # Log batch information
        bo_iter_status_update = f"End of iter {bo_iter}. Status update:"
        _batch_argsort = np.argsort(
            -np.asarray([float(r["transformed_score"]) for r in batch_results])
        )
        # bo_iter_status_update += "\n\tBatch scores (raw): "
        # bo_iter_status_update += ", ".join([str(r["raw_score"]) for r in batch_results])
        bo_iter_status_update += "\n\tBatch scores (transformed): "
        bo_iter_status_update += ", ".join(
            [str(batch_results[pos]["transformed_score"]) for pos in _batch_argsort]
        )
        bo_iter_status_update += "\n\tBatch acquisition function values: "
        bo_iter_status_update += ", ".join(
            f"{smiles_batch_acq[pos]:.2e}" for pos in _batch_argsort
        )
        bo_iter_status_update += (
            "\n\tAcquisition function values of top known smiles : "
        )
        _acq_val_dict = dict(zip(acq_smiles, acq_vals))
        bo_iter_status_update += ", ".join(
            f"{_acq_val_dict[s]:.2e}" for s in top_smiles_at_bo_iter_start[:n_top_log]
        )
        del _acq_val_dict, _batch_argsort

        # Overall progress towards optimizing function
        new_bo_smiles = [
            r["smiles"] for r in bo_query_res if r["smiles"] not in start_cache
        ]
        new_bo_smiles = list(set(new_bo_smiles))
        bo_iter_status_update += "\n\tTop new scores so far: "
        bo_iter_status_update += ", ".join(
            f"#{i+1}={v:.3f}"
            for i, v in enumerate(
                heapq.nlargest(n_top_log, [scoring_function(s) for s in new_bo_smiles])
            )
        )
        func_evals_so_far = len(scoring_function.mol_buffer) - start_cache_size
        bo_iter_status_update += f"\n\tFunction calls so far: {func_evals_so_far}"
        logger.info(bo_iter_status_update)

        # Delete variables that shouldn't persist for multiple runs of loop
        # (not strictly necessary but reduces probability of bug)
        del ga_start_smiles
        del smiles_batch, acq_smiles, acq_vals, smiles_batch_scores, smiles_batch_np
        del (
            smiles_batch_mu_pre,
            smiles_batch_mu_post1,
            smiles_batch_var_pre,
            smiles_batch_var_post1,
        )
        del batch_results, bo_iter_status_update

        # Potentially do early stopping if too many function calls were made
        if max_func_calls is not None and func_evals_so_far >= max_func_calls:
            logger.info("Maximum number of function evaluations reached. STOPPING.")
            break
        del func_evals_so_far

    logger.info("End of BO loop.")
    return (bo_query_res, scoring_function.mol_buffer)
