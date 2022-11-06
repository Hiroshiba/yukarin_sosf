from typing import List, TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from yukarin_sosf.config import ModelConfig
from yukarin_sosf.dataset import OutputData
from yukarin_sosf.network.predictor import Predictor


class ModelOutput(TypedDict):
    loss: Tensor
    loss_f0: Tensor
    loss_voiced: Tensor
    precision_voiced: Tensor
    recall_voiced: Tensor
    data_num: int


def calc(output: Tensor, target: Tensor):
    loss = F.binary_cross_entropy_with_logits(output, target.float())
    tp = ((output >= 0) & (target == 1)).float().sum()
    fp = ((output >= 0) & (target == 0)).float().sum()
    fn = ((output < 0) & (target == 1)).float().sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return loss, precision, recall


class Model(nn.Module):
    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def forward(self, data: OutputData) -> ModelOutput:
        output_list: List[Tensor]
        _, output_list = self.predictor(
            discrete_f0_list=data["discrete_f0"],
            phoneme_list=data["phoneme"],
        )

        output = torch.cat(output_list)
        output_f0 = output[:, 0]
        output_voiced = output[:, 1]

        target_f0 = torch.cat(data["continuous_f0"]).squeeze(1)
        target_voiced = torch.cat(data["voiced"]).squeeze(1)

        mask = target_voiced
        loss_f0 = F.l1_loss(output_f0[mask], target_f0[mask])

        loss_voiced, precision_voiced, recall_voiced = calc(
            output_voiced, target_voiced
        )

        loss = loss_f0 + loss_voiced

        return ModelOutput(
            loss=loss,
            loss_f0=loss_f0,
            loss_voiced=loss_voiced,
            precision_voiced=precision_voiced,
            recall_voiced=recall_voiced,
            data_num=len(data),
        )
