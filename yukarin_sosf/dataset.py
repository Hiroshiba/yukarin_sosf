import json
from dataclasses import dataclass
from enum import Enum
from functools import partial
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy
import torch
from acoustic_feature_extractor.data.phoneme import OjtPhoneme
from acoustic_feature_extractor.data.sampling_data import SamplingData
from torch import Tensor
from torch.utils.data import Dataset
from typing_extensions import TypedDict

from yukarin_sosf.config import DatasetConfig, DatasetFileConfig

mora_phoneme_list = ["a", "i", "u", "e", "o", "I", "U", "E", "N", "cl", "pau"]
voiced_phoneme_list = (
    ["a", "i", "u", "e", "o", "A", "I", "U", "E", "O", "N"]
    + ["n", "m", "y", "r", "w", "g", "z", "j", "d", "b"]
    + ["ny", "my", "ry", "gy", "by"]
)
unvoiced_mora_phoneme_list = ["A", "I", "U", "E", "O", "cl", "pau"]


class F0ProcessMode(str, Enum):
    normal = "normal"
    phoneme_mean = "phoneme_mean"
    mora_mean = "mora_mean"
    voiced_mora_mean = "voiced_mora_mean"


def f0_mean(
    f0: numpy.ndarray,
    rate: float,
    split_second_list: List[float],
    weight: numpy.ndarray,
):
    f0 = f0.copy()
    indexes = numpy.floor(numpy.array(split_second_list) * rate).astype(int)
    for a, b in zip(numpy.split(f0, indexes), numpy.split(weight, indexes)):
        a[:] = numpy.sum(a[a > 0] * b[a > 0]) / numpy.sum(b[a > 0])
    f0[numpy.isnan(f0)] = 0
    return f0


def make_phoneme_array(phoneme_list: List[OjtPhoneme], frame_rate: float, length: int):
    to_index = lambda x: int(x * frame_rate)
    phoneme = numpy.zeros(length, dtype=numpy.int32)
    for p in phoneme_list:
        phoneme[to_index(p.start) : to_index(p.end)] = p.phoneme_id
    return phoneme[:length]


def get_notsilence_range(silence: numpy.ndarray, prepost_silence_length: int):
    """
    最初と最後の無音を除去したrangeを返す。
    一番最初や最後が無音でない場合はノイズとみなしてその区間も除去する。
    最小でもprepost_silence_lengthだけは確保する。
    """
    length = len(silence)

    ps = numpy.argwhere(numpy.logical_and(silence[:-1], ~silence[1:]))
    pre_length = ps[0][0] + 1 if len(ps) > 0 else 0
    pre_index = max(0, pre_length - prepost_silence_length)

    ps = numpy.argwhere(numpy.logical_and(~silence[:-1], silence[1:]))
    post_length = length - (ps[-1][0] + 1) if len(ps) > 0 else 0
    post_index = length - max(0, post_length - prepost_silence_length)
    return range(pre_index, post_index)


@dataclass
class InputData:
    f0: SamplingData
    phoneme_list: List[OjtPhoneme]
    silence: SamplingData
    volume: SamplingData
    speaker_id: int


@dataclass
class LazyInputData:
    f0_path: Path
    phoneme_list_path: Path
    silence_path: Path
    volume_path: Path
    speaker_id: int

    def generate(self):
        return InputData(
            f0=SamplingData.load(self.f0_path),
            phoneme_list=OjtPhoneme.load_julius_list(self.phoneme_list_path),
            silence=SamplingData.load(self.silence_path),
            volume=SamplingData.load(self.volume_path),
            speaker_id=self.speaker_id,
        )


class OutputData(TypedDict):
    continuous_f0: Tensor
    discrete_f0: Tensor
    silence: Tensor
    phoneme: Tensor
    voiced: Tensor
    speaker_id: Tensor


def preprocess(
    d: InputData,
    frame_rate: float,
    prepost_silence_length: int,
    f0_process_mode: F0ProcessMode,
    max_sampling_length: Optional[int],
):
    f0: numpy.ndarray = d.f0.resample(frame_rate).astype(numpy.float32)
    silence: numpy.ndarray = d.silence.resample(frame_rate)
    volume: numpy.ndarray = d.volume.resample(frame_rate)
    phoneme = make_phoneme_array(
        phoneme_list=d.phoneme_list, frame_rate=frame_rate, length=len(f0)
    )

    assert numpy.abs(len(f0) - len(silence)) < 5, f"{len(f0)} != {len(silence)}"
    assert numpy.abs(len(f0) - len(volume)) < 5, f"{len(f0)} != {len(volume)}"
    assert numpy.abs(len(f0) - len(phoneme)) < 5, f"{len(f0)} != {len(phoneme)}"

    length = min(len(f0), len(silence), len(volume), len(phoneme))
    f0 = f0[:length]
    silence = silence[:length]
    volume = volume[:length]
    phoneme = phoneme[:length]

    # discrete f0
    weight = volume
    if f0_process_mode == F0ProcessMode.phoneme_mean:
        split_second_list = [p.end for p in d.phoneme_list[:-1]]
    else:
        split_second_list = [
            p.end for p in d.phoneme_list[:-1] if p.phoneme in mora_phoneme_list
        ]

    if f0_process_mode == F0ProcessMode.voiced_mora_mean:
        if weight is None:
            weight = numpy.ones_like(volume)

        for p in d.phoneme_list:
            if p.phoneme not in voiced_phoneme_list:
                weight[int(p.start * frame_rate) : int(p.end * frame_rate)] = 0

    discrete_f0 = f0_mean(
        f0=f0,
        rate=frame_rate,
        split_second_list=split_second_list,
        weight=weight[:length],
    )

    # voiced
    f0[silence] = 0
    voiced = f0 != 0

    # 最初と最後の無音を除去する
    notsilence_range = get_notsilence_range(
        silence=silence[:length],
        prepost_silence_length=prepost_silence_length,
    )
    f0 = f0[notsilence_range]
    discrete_f0 = discrete_f0[notsilence_range]
    voiced = voiced[notsilence_range]
    silence = silence[notsilence_range]
    volume = volume[notsilence_range]
    phoneme = phoneme[notsilence_range]
    length = len(f0)

    # サンプリング長調整
    if max_sampling_length is not None and length > max_sampling_length:
        offset = numpy.random.randint(length - max_sampling_length + 1)
        offset_slice = slice(offset, offset + max_sampling_length)
        f0 = f0[offset_slice]
        discrete_f0 = discrete_f0[offset_slice]
        voiced = voiced[offset_slice]
        silence = silence[offset_slice]
        volume = volume[offset_slice]
        phoneme = phoneme[offset_slice]
        length = max_sampling_length

    output_data = OutputData(
        continuous_f0=torch.from_numpy(f0),
        discrete_f0=torch.from_numpy(discrete_f0),
        silence=torch.from_numpy(silence),
        phoneme=torch.from_numpy(phoneme),
        voiced=torch.from_numpy(voiced),
        speaker_id=torch.tensor(d.speaker_id),
    )
    return output_data


class FeatureTargetDataset(Dataset):
    def __init__(
        self,
        datas: Sequence[Union[InputData, LazyInputData]],
        frame_rate: float,
        prepost_silence_length: int,
        f0_process_mode: F0ProcessMode,
        max_sampling_length: Optional[int],
    ):
        self.datas = datas
        self.preprocessor = partial(
            preprocess,
            frame_rate=frame_rate,
            prepost_silence_length=prepost_silence_length,
            f0_process_mode=f0_process_mode,
            max_sampling_length=max_sampling_length,
        )

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, i):
        data = self.datas[i]
        if isinstance(data, LazyInputData):
            data = data.generate()
        return self.preprocessor(data)


def get_datas(config: DatasetFileConfig):
    f0_paths = {Path(p).stem: Path(p) for p in glob(config.f0_glob)}
    fn_list = sorted(f0_paths.keys())
    assert len(fn_list) > 0

    phoneme_list_paths = {
        Path(p).stem: Path(p) for p in sorted(glob(config.phoneme_list_glob))
    }
    assert set(fn_list) == set(phoneme_list_paths.keys())

    silence_paths = {Path(p).stem: Path(p) for p in sorted(glob(config.silence_glob))}
    assert set(fn_list) == set(silence_paths.keys())

    volume_paths = {Path(p).stem: Path(p) for p in sorted(glob(config.volume_glob))}
    assert set(fn_list) == set(volume_paths.keys())

    fn_each_speaker: Dict[str, List[str]] = json.loads(
        config.speaker_dict_path.read_text()
    )
    speaker_ids = {
        fn: speaker_id
        for speaker_id, (_, fns) in enumerate(fn_each_speaker.items())
        for fn in fns
    }
    assert set(fn_list).issubset(set(speaker_ids.keys()))

    datas = [
        LazyInputData(
            f0_path=f0_paths[fn],
            phoneme_list_path=phoneme_list_paths[fn],
            silence_path=silence_paths[fn],
            volume_path=volume_paths[fn],
            speaker_id=speaker_ids[fn],
        )
        for fn in fn_list
    ]
    return datas


def create_dataset(config: DatasetConfig):
    datas = get_datas(config.train_file)
    if config.seed is not None:
        numpy.random.RandomState(config.seed).shuffle(datas)

    tests, trains = datas[: config.test_num], datas[config.test_num :]

    valids = get_datas(config.valid_file)

    def dataset_wrapper(datas, is_eval: bool):
        dataset = FeatureTargetDataset(
            datas=datas,
            frame_rate=config.frame_rate,
            prepost_silence_length=config.prepost_silence_length,
            f0_process_mode=F0ProcessMode(config.f0_process_mode),
            max_sampling_length=(config.max_sampling_length if not is_eval else None),
        )
        return dataset

    return {
        "train": dataset_wrapper(trains, is_eval=False),
        "test": dataset_wrapper(tests, is_eval=False),
        "eval": dataset_wrapper(tests, is_eval=True),
        "valid": dataset_wrapper(valids, is_eval=True),
    }
