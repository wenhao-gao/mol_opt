import random
import numpy as np
import torch
# import neptune

from joblib import delayed
from guacamol.utils.chemistry import canonicalize


class GeneticExpertGuidedLearningTrainer:
    def __init__(
        self,
        apprentice_storage,
        expert_storage,
        apprentice_handler,
        expert_handler,
        char_dict,
        num_keep,
        apprentice_sampling_batch_size,
        expert_sampling_batch_size,
        apprentice_training_batch_size,
        num_apprentice_training_steps,
        init_smis,
    ):
        self.apprentice_storage = apprentice_storage
        self.expert_storage = expert_storage

        self.apprentice_handler = apprentice_handler
        self.expert_handler = expert_handler

        self.char_dict = char_dict

        self.num_keep = num_keep
        self.apprentice_sampling_batch_size = apprentice_sampling_batch_size
        self.expert_sampling_batch_size = expert_sampling_batch_size
        self.apprentice_training_batch_size = apprentice_training_batch_size
        self.num_apprentice_training_steps = num_apprentice_training_steps

        self.init_smis = init_smis

    def init(self, scoring_function, device, pool):
        if len(self.init_smis) > 0:
            smis, scores = self.canonicalize_and_score_smiles(
                smis=self.init_smis, scoring_function=scoring_function, pool=pool
            )

            self.apprentice_storage.add_list(smis=smis, scores=scores)
            self.expert_storage.add_list(smis=smis, scores=scores)

    def step(self, scoring_function, device, pool):
        apprentice_smis, apprentice_scores = self.update_storage_by_apprentice(
            scoring_function, device, pool
        )
        expert_smis, expert_scores = self.update_storage_by_expert(scoring_function, pool)
        loss, fit_size = self.train_apprentice_step(device)

        # neptune.log_metric("apprentice_loss", loss)
        # neptune.log_metric("fit_size", fit_size)

        return apprentice_smis + expert_smis, apprentice_scores + expert_scores

    def update_storage_by_apprentice(self, scoring_function, device, pool):
        with torch.no_grad():
            self.apprentice_handler.model.eval()
            smis, _, _, _ = self.apprentice_handler.sample(
                num_samples=self.apprentice_sampling_batch_size, device=device
            )

        smis, scores = self.canonicalize_and_score_smiles(
            smis=smis, scoring_function=scoring_function, pool=pool
        )

        self.apprentice_storage.add_list(smis=smis, scores=scores)
        self.apprentice_storage.squeeze_by_kth(k=self.num_keep)

        return smis, scores

    def update_storage_by_expert(self, scoring_function, pool):
        expert_smis, expert_scores = self.apprentice_storage.sample_batch(
            self.expert_sampling_batch_size
        )
        smis = self.expert_handler.query(
            query_size=self.expert_sampling_batch_size, mating_pool=expert_smis, pool=pool
        )
        smis, scores = self.canonicalize_and_score_smiles(
            smis=smis, scoring_function=scoring_function, pool=pool
        )

        self.expert_storage.add_list(smis=smis, scores=scores)
        self.expert_storage.squeeze_by_kth(k=self.num_keep)

        return smis, scores

    def train_apprentice_step(self, device):
        avg_loss = 0.0
        apprentice_smis, _ = self.apprentice_storage.get_elems()
        expert_smis, _ = self.expert_storage.get_elems()
        total_smis = list(set(apprentice_smis + expert_smis))

        self.apprentice_handler.model.train()
        for _ in range(self.num_apprentice_training_steps):
            smis = random.choices(population=total_smis, k=self.apprentice_training_batch_size)
            loss = self.apprentice_handler.train_on_batch(smis=smis, device=device)

            avg_loss += loss / self.num_apprentice_training_steps

        fit_size = len(total_smis)

        return avg_loss, fit_size

    def canonicalize_and_score_smiles(self, smis, scoring_function, pool):
        smis = pool(
            delayed(lambda smi: canonicalize(smi, include_stereocenters=False))(smi) for smi in smis
        )
        smis = list(filter(lambda smi: (smi is not None) and self.char_dict.allowed(smi), smis))
        # scores = pool(delayed(scoring_function.score)(smi) for smi in smis)
        scores = [scoring_function(smiles) for smiles in smis]
        # scores = [0.0 for smi in smis]

        median_score = np.mean(scores)

        filtered_smis_and_scores = list(
            filter(
                lambda smi_and_score: smi_and_score[1]
                > median_score,
                zip(smis, scores),
            )
        )

        smis, scores = (
            map(list, zip(*filtered_smis_and_scores))
            if len(filtered_smis_and_scores) > 0
            else ([], [])
        )
        return smis, scores
