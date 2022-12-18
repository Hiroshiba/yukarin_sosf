from typing import List, Optional

import torch
from espnet_pytorch_library.conformer.encoder import Encoder
from espnet_pytorch_library.nets_utils import make_non_pad_mask
from espnet_pytorch_library.tacotron2.decoder import Postnet
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from yukarin_sosf.config import NetworkConfig


class Predictor(nn.Module):
    def __init__(
        self,
        speaker_size: int,
        speaker_embedding_size: int,
        phoneme_size: int,
        phoneme_embedding_size: int,
        hidden_size: int,
        block_num: int,
        post_layer_num: int,
    ):
        super().__init__()

        self.speaker_embedder = nn.Embedding(
            num_embeddings=speaker_size,
            embedding_dim=speaker_embedding_size,
        )

        self.phoneme_embedder = nn.Embedding(
            num_embeddings=phoneme_size,
            embedding_dim=phoneme_embedding_size,
        )

        input_size = 1 + phoneme_embedding_size + speaker_embedding_size
        self.pre = torch.nn.Linear(input_size, hidden_size)

        self.encoder = Encoder(
            idim=None,
            attention_dim=hidden_size,
            attention_heads=2,
            linear_units=hidden_size * 4,
            num_blocks=block_num,
            input_layer=None,
            dropout_rate=0.2,
            positional_dropout_rate=0.2,
            attention_dropout_rate=0.2,
            normalize_before=True,
            positionwise_layer_type="conv1d",
            positionwise_conv_kernel_size=3,
            macaron_style=True,
            pos_enc_layer_type="rel_pos",
            selfattention_layer_type="rel_selfattn",
            activation_type="swish",
            use_cnn_module=True,
            cnn_module_kernel=31,
        )

        self.post = torch.nn.Linear(hidden_size, 2)

        if post_layer_num > 0:
            self.postnet = Postnet(
                idim=2,
                odim=2,
                n_layers=post_layer_num,
                n_chans=hidden_size,
                n_filts=5,
                use_batch_norm=True,
                dropout_rate=0.5,
            )
        else:
            self.postnet = None

    def _mask(self, length: Tensor):
        x_masks = make_non_pad_mask(length).to(length.device)
        return x_masks.unsqueeze(-2)

    def forward(
        self,
        discrete_f0_list: List[Tensor],  # [(L, )]
        phoneme_list: List[Tensor],  # [(L, )]
        speaker_id: Tensor,  # (B, )
    ):
        """
        B: batch size
        L: length
        """
        length_list = [t.shape[0] for t in discrete_f0_list]

        length = torch.tensor(length_list, device=discrete_f0_list[0].device)
        h = pad_sequence(discrete_f0_list, batch_first=True)  # (B, L, ?)

        phoneme = pad_sequence(phoneme_list, batch_first=True)  # (B, L)
        phoneme = self.phoneme_embedder(phoneme)  # (B, L, ?)

        speaker_id = self.speaker_embedder(speaker_id)
        speaker_id = speaker_id.unsqueeze(dim=1)  # (B, 1, ?)
        speaker_feature = speaker_id.expand(
            speaker_id.shape[0], h.shape[1], speaker_id.shape[2]
        )  # (B, L, ?)

        h = torch.cat((h, phoneme, speaker_feature), dim=2)  # (B, L, ?)
        h = self.pre(h)

        mask = self._mask(length)
        h, _ = self.encoder(h, mask)

        output1 = self.post(h)
        if self.postnet is not None:
            output2 = output1 + self.postnet(output1.transpose(1, 2)).transpose(1, 2)
        else:
            output2 = output1

        return (
            [output1[i, :l] for i, l in enumerate(length_list)],
            [output2[i, :l] for i, l in enumerate(length_list)],
        )

    def inference(
        self,
        discrete_f0_list: List[Tensor],  # [(L, )]
        phoneme_list: List[Tensor],  # [(L, )]
        speaker_id: Tensor,  # (B, )
    ):
        _, h = self(
            discrete_f0_list=discrete_f0_list,
            phoneme_list=phoneme_list,
            speaker_id=speaker_id,
        )
        return h


def create_predictor(config: NetworkConfig):
    return Predictor(
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
        phoneme_size=config.phoneme_size,
        phoneme_embedding_size=config.phoneme_embedding_size,
        hidden_size=config.hidden_size,
        block_num=config.block_num,
        post_layer_num=config.post_layer_num,
    )
