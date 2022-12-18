from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import Literal, TypedDict

from yukarin_sosf.dataset import OutputData
from yukarin_sosf.generator import Generator, GeneratorOutput
from yukarin_sosf.model import calc


class EvaluatorOutput(TypedDict):
    diff_f0: Tensor
    precision_voiced: Tensor
    recall_voiced: Tensor
    precision_unvoiced: Tensor
    recall_unvoiced: Tensor
    value: Tensor
    data_num: int


class Evaluator(nn.Module):
    judge: Literal["min", "max"] = "min"

    def __init__(self, generator: Generator):
        super().__init__()
        self.generator = generator

    def forward(self, data: OutputData) -> EvaluatorOutput:
        device = data["continuous_f0"][0].device

        output_list: List[GeneratorOutput] = self.generator(
            discrete_f0_list=data["discrete_f0"],
            phoneme_list=data["phoneme"],
            speaker_id=torch.stack(data["speaker_id"]),
        )

        target_f0 = torch.cat(data["continuous_f0"]).squeeze(1)
        target_voiced = torch.cat(data["voiced"]).squeeze(1)

        output_f0 = torch.cat([output["f0"] for output in output_list]).to(device)
        output_voiced = torch.cat([output["voiced"] for output in output_list]).to(
            device
        )

        mask = target_voiced
        diff_f0 = F.l1_loss(output_f0[mask], target_f0[mask])

        _, precision_voiced, recall_voiced = calc(output_voiced, target_voiced)
        _, precision_unvoiced, recall_unvoiced = calc(
            output_voiced * -1, target_voiced != True
        )

        value = diff_f0

        return EvaluatorOutput(
            diff_f0=diff_f0,
            precision_voiced=precision_voiced,
            recall_voiced=recall_voiced,
            precision_unvoiced=precision_unvoiced,
            recall_unvoiced=recall_unvoiced,
            value=value,
            data_num=len(data),
        )
