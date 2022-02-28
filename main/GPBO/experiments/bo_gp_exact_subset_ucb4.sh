#!/usr/bin/env bash
# Graph GA on guacamol dataset (of varying sizes)
objective_arr=( \
    "logP" \
    "QED" \
    "celecoxib-rediscovery" \
    "troglitazone-rediscovery" \
    "median-molecules-1" \
    "median-molecules-2" \
    "osimertinib-mpo" \
    "zaleplon-mpo" \
    \
    "plogP" \
)
dataset_size_arr=( 100 1000 10000 100000 )
method_name="bo_gp_exact_subset_ucb4"

# These must match...
max_func_calls_arr=( 10 100 1000 10000 100000 )
batch_size_arr=(      1   1   10   100   1000 )
ga_offspring_arr=(  100 100  250  1000   5000 )
ga_best_arr=(        25  25   50   200   1000 )
ga_prom_arr=(        50  50  100   500   3000 )

# logging
log_dir="./results/ai4sci/log/"
mkdir -p "$log_dir"

curr_expt_idx=0
for target in "${objective_arr[@]}" ; do

    # Result dir for this target
    res_dir="./results/ai4sci/res/${method_name}/${target}"
    mkdir -p "${res_dir}"

    # max func calls
    for budget_i in "${!max_func_calls_arr[@]}" ; do 
        max_func_calls="${max_func_calls_arr[budget_i]}"
        batch_size="${batch_size_arr[budget_i]}"

        # Dataset size
        for dataset_size in "${dataset_size_arr[@]}" ; do

            # Multiple trials
            for trial in {0..2}; do
                expt_id_str="budget-${max_func_calls}_Ndata-${dataset_size}_trial-${trial}"
                output_path="${res_dir}/${expt_id_str}.json" 
                dataset_path="./data/guacamol-splits/seed_${trial}/rand_${dataset_size}.tsv"

                if [[ -z "$expt_idx" || "$expt_idx" = "$curr_expt_idx" ]] ; then

                    if [[ -f "$output_path" ]]; then
                        echo "Results for expt_idx ${curr_expt_idx} ${method_name} ${target} ${expt_id_str} exists! Skipping."
                    else


                        echo "Running expt_idx ${curr_expt_idx} ${method_name} ${target} ${expt_id_str}..."

                        PYTHONPATH="$(pwd)/src:$PYTHONPATH" python src/mol_opt/run_${method_name}.py \
                            --num_cpu=6 \
                            \
                            --dataset="$dataset_path" \
                            --objective="${target}" \
                            --maximize \
                            --max_func_calls="${max_func_calls}" \
                            \
                            --n_train_gp_best=2000 \
                            --n_train_gp_rand=3000 \
                            --bo_batch_size="${batch_size}" \
                            \
                            --fp_radius=2 \
                            --fp_nbits=4096 \
                            \
                            --ga_max_generations=5 \
                            --ga_offspring_size="${ga_offspring_arr[budget_i]}" \
                            --ga_pop_params "${ga_best_arr[budget_i]}" "${ga_prom_arr[budget_i]}" "${ga_offspring_arr[budget_i]}" \
                            \
                            --output_path="${output_path}" \
                            &> "${log_dir}/${method_name}_${target}_${expt_id_str}.log"
                    
                    fi
                fi

                # Increment experiment index after every potential experiment
                curr_expt_idx=$(( curr_expt_idx + 1 ))

            done
        done
    done
done
