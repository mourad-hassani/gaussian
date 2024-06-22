import torch
from transformers.optimization import get_linear_schedule_with_warmup
from parameters import LR, EPOCHS, WEIGHT_DECAY, NUM_WARMUP_RATIO
import torch.nn as nn

def create_optimizer(model: nn.Module, train_steps_per_epoch: int) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        no_decay = {"bias", "LayerNorm.weight"}
        optimizer_grouped_parameters = [
            {
                "params": [param for name, param in model.named_parameters() if name not in no_decay
                ],
                "weight_decay": WEIGHT_DECAY,
            },
            {
                "params": [param for name, param in model.named_parameters() if name in no_decay
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LR)

        num_training_steps = train_steps_per_epoch * EPOCHS
        num_warmup_steps = int(num_training_steps * NUM_WARMUP_RATIO)

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        return optimizer, lr_scheduler