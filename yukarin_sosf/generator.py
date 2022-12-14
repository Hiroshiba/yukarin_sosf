from pathlib import Path
from typing import Any, List, Optional, Union

import numpy
import torch
from torch import Tensor, nn
from typing_extensions import TypedDict

from yukarin_sosf.config import Config
from yukarin_sosf.network.predictor import Predictor, create_predictor


class GeneratorOutput(TypedDict):
    f0: Tensor
    voiced: Tensor


def to_tensor(array: Union[Tensor, numpy.ndarray, Any]):
    if not isinstance(array, (Tensor, numpy.ndarray)):
        array = numpy.asarray(array)
    if isinstance(array, numpy.ndarray):
        return torch.from_numpy(array)
    else:
        return array


class Generator(nn.Module):
    def __init__(
        self,
        config: Config,
        predictor: Union[Predictor, Path],
        use_gpu: bool,
    ):
        super().__init__()

        self.config = config
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")

        if isinstance(predictor, Path):
            state_dict = torch.load(predictor, map_location=self.device)
            predictor = create_predictor(config.network)
            predictor.load_state_dict(state_dict)
        self.predictor = predictor.eval().to(self.device)

    def forward(
        self,
        discrete_f0_list: List[Union[numpy.ndarray, Tensor]],
        phoneme_list: List[Union[numpy.ndarray, Tensor]],
        speaker_id: Union[numpy.ndarray, Tensor],
    ):
        discrete_f0_list = [to_tensor(f0).to(self.device) for f0 in discrete_f0_list]
        phoneme_list = [to_tensor(phoneme).to(self.device) for phoneme in phoneme_list]
        speaker_id = to_tensor(speaker_id).to(self.device)

        with torch.inference_mode():
            output_list = self.predictor.inference(
                discrete_f0_list=discrete_f0_list,
                phoneme_list=phoneme_list,
                speaker_id=speaker_id,
            )

        return [
            GeneratorOutput(f0=output[:, 0], voiced=output[:, 1])
            for output in output_list
        ]
