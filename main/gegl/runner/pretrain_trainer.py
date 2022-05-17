import os
from tqdm import tqdm
import torch
import neptune
from util.smiles.function import smis_to_actions


class PreTrainer:
    def __init__(
        self, char_dict, dataset, generator_handler, num_epochs, batch_size, save_dir, device,
    ):
        self.generator_handler = generator_handler
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.device = device

        action_dataset, _ = smis_to_actions(char_dict=char_dict, smis=dataset)
        action_dataset = torch.LongTensor(action_dataset)
        self.dataset_loader = torch.utils.data.DataLoader(
            dataset=action_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        os.makedirs(save_dir, exist_ok=True)

    def train(self):
        for epoch in tqdm(range(self.num_epochs)):
            for actions in self.dataset_loader:
                loss = self.generator_handler.train_on_action_batch(
                    actions=actions, device=self.device
                )
                neptune.log_metric("loss", loss)

            self.generator_handler.save(self.save_dir)
