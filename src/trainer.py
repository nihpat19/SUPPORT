from typing import Dict, Tuple, Callable, List, Any

import os
import random
import logging
import time
import numpy as np
import torch
from tqdm import tqdm
from torch import nn, optim
from nnfabrik.utility.nn_helpers import move_to_device

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

