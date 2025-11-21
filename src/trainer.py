from typing import Dict, Tuple, Callable, List, Any
import sys
sys.path.append("..")
import os
import random
import logging
import time
import numpy as np
import torch
from tqdm import tqdm
from torch import nn, optim
from nnfabrik.utility.nn_helpers import move_to_device
from SUPPORT.src.utils.dataset import random_transform
class SUPPORTTrainer:
    def __init__(
            self,
            model: nn.Module,
            dataloader: Dict,
            seed: int,
            config
    ) -> None:

        self.model, _ = move_to_device(model,gpu=True)
        self.trainloader = dataloader["train"]
        self.seed = seed
        self.epochs = config.get("epochs",50)
        if "lr" in config:
            lr = config["lr"]
        else:
            lr = 5e-4

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        if "use_amp" in config:
            self.scaler = torch.cuda.amp.GradScaler(enabled=config["use_amp"])
        self.use_amp = True if self.scaler is not None else False
        self.l1_pixelwise = nn.L1Loss()
        self.l2_pixelwise = nn.MSELoss()

        self.loss_coef = config.get("loss_coef",[0.5,0.5])
        self.rng = np.random.default_rng(seed=self.seed)
        self.is_rotate = True if self.model.bs_size[0]==self.model.bs_size[1] else False

    def train_batch(self, noisy_image: torch.Tensor, noisy_target: torch.Tensor) -> Tuple[float, float, float]:
        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            denoised_image = self.model(noisy_image)
            l1_loss = self.l1_pixelwise(denoised_image, noisy_target)
            l2_loss = self.l2_pixelwise(denoised_image, noisy_target)
            loss_sum = self.loss_coef[0] * l1_loss + self.loss_coef[1] * l2_loss
        self.scaler.scale(loss_sum).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss_sum.item(), l1_loss.item(), l2_loss.item()

    def train(self) -> Tuple[float,Tuple[List[float], List[float], List[float]], Dict]:
        epoch_loss_list = []
        epoch_l1_loss_list = []
        epoch_l2_loss_list = []
        torch.manual_seed(self.seed)
        for epoch in range(self.epochs):
            self.model.train()
            loss_list_l1 = []
            loss_list_l2 = []
            loss_list = []
            self.trainloader.dataset.precompute_indices()

            for i, data in enumerate(tqdm(self.trainloader)):
                if not self.trainloader.dataset.load_to_memory:
                    (noisy_image, _, ds_idx, noisy_image_avg, noisy_image_std) = data
                    noisy_image_avg = torch.reshape(noisy_image_avg, (-1, 1, 1, 1)).cuda()
                    noisy_image_std = torch.reshape(noisy_image_std, (-1, 1, 1, 1)).cuda()
                    noisy_image = (noisy_image - noisy_image_avg) / noisy_image_std
                else:
                    (noisy_image, _, ds_idx) = data
                B, T, X, Y = noisy_image.shape
                noisy_image = noisy_image.cuda()

                noisy_image, _ = random_transform(noisy_image,None,self.rng,self.is_rotate)


                noisy_target = torch.unsqueeze(noisy_image[:, int(T / 2), :, :], dim=1)

                loss, l1_loss, l2_loss = self.train_batch(noisy_image, noisy_target)
                loss_list.append(loss)
                loss_list_l1.append(l1_loss)
                loss_list_l2.append(l2_loss)
            epoch_loss = np.mean(np.array(loss_list))
            epoch_l1_loss = np.mean(np.array(loss_list_l1))
            epoch_l2_loss = np.mean(np.array(loss_list_l2))
            epoch_loss_list.append(epoch_loss)
            epoch_l1_loss_list.append(epoch_l1_loss)
            epoch_l2_loss_list.append(epoch_l2_loss)

        return epoch_loss_list[-1], (epoch_loss_list, epoch_l1_loss_list, epoch_l2_loss_list), self.model.state_dict()


def SUPPORT_trainer_fn(model: nn.Module,dataloaders: Dict, seed: int, uid: Dict, cb: Callable, **config) \
        -> Tuple[float,Any,Dict]:
    trainer = SUPPORTTrainer(model,dataloaders,seed,config)
    out = trainer.train()
    return out
