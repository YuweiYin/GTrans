# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
from torch import Tensor
from fairseq import checkpoint_utils
from fairseq.data.legacy.masked_lm_dictionary import MaskedLMDictionary
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    TransformerModel,
    TransformerDecoderLayer,
    base_architecture as transformer_base_architecture,
)
from fairseq.models.transformer_from_pretrained_infoxlm import (
    TransformerFromPretrainedInfoXLMModel,
    TransformerEncoderFromPretrainedInfoXLM,
    TransformerDecoderFromPretrainedInfoXLM,
    upgrade_state_dict_with_infoxlm_weights
)
from fairseq.modules.transformer_layer import (
    TransformerDecoderLayer
)
from fairseq.modules.multihead_attention import MultiheadAttention
from fairseq.modules import LayerNorm
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from fairseq import utils
from fairseq.file_io import PathManager
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def CopyDecoderBottom2Top(state_dict, decoder_group_layers, decoder_group_number):
    def get_layer(i, j):
        return i + j * decoder_group_layers

    for i in range(decoder_group_layers):
        for j in range(1, decoder_group_number):
            state_dict[f'layers.{get_layer(i, j)}.self_attn.k_proj.weight'] = state_dict[
                f'layers.{i}.self_attn.k_proj.weight']
            state_dict[f'layers.{get_layer(i, j)}.self_attn.k_proj.bias'] = state_dict[
                f'layers.{i}.self_attn.k_proj.bias']
            state_dict[f'layers.{get_layer(i, j)}.self_attn.v_proj.weight'] = state_dict[
                f'layers.{i}.self_attn.v_proj.weight']
            state_dict[f'layers.{get_layer(i, j)}.self_attn.v_proj.bias'] = state_dict[
                f'layers.{i}.self_attn.v_proj.bias']
            state_dict[f'layers.{get_layer(i, j)}.self_attn.q_proj.weight'] = state_dict[
                f'layers.{i}.self_attn.q_proj.weight']
            state_dict[f'layers.{get_layer(i, j)}.self_attn.q_proj.bias'] = state_dict[
                f'layers.{i}.self_attn.q_proj.bias']
            state_dict[f'layers.{get_layer(i, j)}.self_attn.out_proj.weight'] = state_dict[
                f'layers.{i}.self_attn.out_proj.weight']
            state_dict[f'layers.{get_layer(i, j)}.self_attn.out_proj.bias'] = state_dict[
                f'layers.{i}.self_attn.out_proj.bias']
            state_dict[f'layers.{get_layer(i, j)}.self_attn_layer_norm.weight'] = state_dict[
                f'layers.{i}.self_attn_layer_norm.weight']
            state_dict[f'layers.{get_layer(i, j)}.self_attn_layer_norm.bias'] = state_dict[
                f'layers.{i}.self_attn_layer_norm.bias']

            state_dict[f'layers.{get_layer(i, j)}.encoder_attn.k_proj.weight'] = state_dict[
                f'layers.{i}.encoder_attn.k_proj.weight']
            state_dict[f'layers.{get_layer(i, j)}.encoder_attn.k_proj.bias'] = state_dict[
                f'layers.{i}.encoder_attn.k_proj.bias']
            state_dict[f'layers.{get_layer(i, j)}.encoder_attn.v_proj.weight'] = state_dict[
                f'layers.{i}.encoder_attn.v_proj.weight']
            state_dict[f'layers.{get_layer(i, j)}.encoder_attn.v_proj.bias'] = state_dict[
                f'layers.{i}.encoder_attn.v_proj.bias']
            state_dict[f'layers.{get_layer(i, j)}.encoder_attn.q_proj.weight'] = state_dict[
                f'layers.{i}.encoder_attn.q_proj.weight']
            state_dict[f'layers.{get_layer(i, j)}.encoder_attn.q_proj.bias'] = state_dict[
                f'layers.{i}.encoder_attn.q_proj.bias']
            state_dict[f'layers.{get_layer(i, j)}.encoder_attn.out_proj.weight'] = state_dict[
                f'layers.{i}.encoder_attn.out_proj.weight']
            state_dict[f'layers.{get_layer(i, j)}.encoder_attn.out_proj.bias'] = state_dict[
                f'layers.{i}.encoder_attn.out_proj.bias']
            state_dict[f'layers.{get_layer(i, j)}.encoder_attn_layer_norm.weight'] = state_dict[
                f'layers.{i}.encoder_attn_layer_norm.weight']
            state_dict[f'layers.{get_layer(i, j)}.encoder_attn_layer_norm.bias'] = state_dict[
                f'layers.{i}.encoder_attn_layer_norm.bias']

            state_dict[f'layers.{get_layer(i, j)}.fc1.weight'] = state_dict[f'layers.{i}.fc1.weight']
            state_dict[f'layers.{get_layer(i, j)}.fc1.bias'] = state_dict[f'layers.{i}.fc1.bias']
            state_dict[f'layers.{get_layer(i, j)}.fc2.weight'] = state_dict[f'layers.{i}.fc2.weight']
            state_dict[f'layers.{get_layer(i, j)}.fc2.bias'] = state_dict[f'layers.{i}.fc2.bias']
            state_dict[f'layers.{get_layer(i, j)}.fc3.weight'] = state_dict[f'layers.{i}.fc3.weight']
            state_dict[f'layers.{get_layer(i, j)}.fc3.bias'] = state_dict[f'layers.{i}.fc3.bias']
            state_dict[f'layers.{get_layer(i, j)}.fc4.weight'] = state_dict[f'layers.{i}.fc4.weight']
            state_dict[f'layers.{get_layer(i, j)}.fc4.bias'] = state_dict[f'layers.{i}.fc4.bias']
            state_dict[f'layers.{get_layer(i, j)}.ffn_layer_norm.weight'] = state_dict[
                f'layers.{i}.ffn_layer_norm.weight']
            state_dict[f'layers.{get_layer(i, j)}.ffn_layer_norm.bias'] = state_dict[f'layers.{i}.ffn_layer_norm.bias']
            state_dict[f'layers.{get_layer(i, j)}.final_layer_norm.weight'] = state_dict[
                f'layers.{i}.final_layer_norm.weight']
            state_dict[f'layers.{get_layer(i, j)}.final_layer_norm.bias'] = state_dict[
                f'layers.{i}.final_layer_norm.bias']
    return state_dict


def CopyEncoderBottom2Top(state_dict, encoder_group_layers, encoder_group_number):
    def get_layer(i, j):
        return i + j * encoder_group_layers

    for i in range(encoder_group_layers):
        for j in range(1, encoder_group_number):
            state_dict[f'layers.{get_layer(i, j)}.self_attn.k_proj.weight'] = state_dict[
                f'layers.{i}.self_attn.k_proj.weight']
            state_dict[f'layers.{get_layer(i, j)}.self_attn.k_proj.bias'] = state_dict[
                f'layers.{i}.self_attn.k_proj.bias']
            state_dict[f'layers.{get_layer(i, j)}.self_attn.v_proj.weight'] = state_dict[
                f'layers.{i}.self_attn.v_proj.weight']
            state_dict[f'layers.{get_layer(i, j)}.self_attn.v_proj.bias'] = state_dict[
                f'layers.{i}.self_attn.v_proj.bias']
            state_dict[f'layers.{get_layer(i, j)}.self_attn.q_proj.weight'] = state_dict[
                f'layers.{i}.self_attn.q_proj.weight']
            state_dict[f'layers.{get_layer(i, j)}.self_attn.q_proj.bias'] = state_dict[
                f'layers.{i}.self_attn.q_proj.bias']
            state_dict[f'layers.{get_layer(i, j)}.self_attn.out_proj.weight'] = state_dict[
                f'layers.{i}.self_attn.out_proj.weight']
            state_dict[f'layers.{get_layer(i, j)}.self_attn.out_proj.bias'] = state_dict[
                f'layers.{i}.self_attn.out_proj.bias']
            state_dict[f'layers.{get_layer(i, j)}.self_attn_layer_norm.weight'] = state_dict[
                f'layers.{i}.self_attn_layer_norm.weight']
            state_dict[f'layers.{get_layer(i, j)}.self_attn_layer_norm.bias'] = state_dict[
                f'layers.{i}.self_attn_layer_norm.bias']

            state_dict[f'layers.{get_layer(i, j)}.fc1.weight'] = state_dict[f'layers.{i}.fc1.weight']
            state_dict[f'layers.{get_layer(i, j)}.fc1.bias'] = state_dict[f'layers.{i}.fc1.bias']
            state_dict[f'layers.{get_layer(i, j)}.fc2.weight'] = state_dict[f'layers.{i}.fc2.weight']
            state_dict[f'layers.{get_layer(i, j)}.fc2.bias'] = state_dict[f'layers.{i}.fc2.bias']
            state_dict[f'layers.{get_layer(i, j)}.final_layer_norm.weight'] = state_dict[
                f'layers.{i}.final_layer_norm.weight']
            state_dict[f'layers.{get_layer(i, j)}.final_layer_norm.bias'] = state_dict[
                f'layers.{i}.final_layer_norm.bias']
    return state_dict


def upgrade_state_dict_for_two_ffn(
        state_dict: Dict[str, Any], pretrained_infoxlm_checkpoint: str, num_layers: int
) -> Dict[str, Any]:
    if not os.path.exists(pretrained_infoxlm_checkpoint):
        raise IOError("Model file not found: {}".format(pretrained_infoxlm_checkpoint))

    # state = checkpoint_utils.load_checkpoint_to_cpu(pretrained_infoxlm_checkpoint)
    with open(PathManager.get_local_path(pretrained_infoxlm_checkpoint), "rb") as f:
        state = torch.load(f, map_location=torch.device("cpu"))
    infoxlm_state_dict = state["model"]

    for key in infoxlm_state_dict.keys():
        if 'layers' in key and int(key.split('.')[3]) > 2 * num_layers - 1:
            continue
        if not key.startswith('decoder.'):
            continue
        if 'lm_head' not in key:
            if 'in_proj_weight' in key:
                q, k, v = infoxlm_state_dict[key].chunk(3, dim=0)
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 0:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'q_proj.weight').replace(
                            f'.{i_layer}', f'.{i_layer // 2}')] = q
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'k_proj.weight').replace(
                            f'.{i_layer}', f'.{i_layer // 2}')] = k
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'v_proj.weight').replace(
                            f'.{i_layer}', f'.{i_layer // 2}')] = v
                else:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'q_proj.weight').replace(
                            f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = q
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'k_proj.weight').replace(
                            f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = k
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'v_proj.weight').replace(
                            f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = v
            elif 'in_proj_bias' in key:
                q, k, v = infoxlm_state_dict[key].chunk(3, dim=0)
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 0:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'q_proj.bias').replace(
                            f'.{i_layer}', f'.{i_layer // 2}')] = q
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'k_proj.bias').replace(
                            f'.{i_layer}', f'.{i_layer // 2}')] = k
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'v_proj.bias').replace(
                            f'.{i_layer}', f'.{i_layer // 2}')] = v
                else:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'q_proj.bias').replace(
                            f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = q
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'k_proj.bias').replace(
                            f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = k
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'v_proj.bias').replace(
                            f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = v
            elif 'fc1' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 0:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('fc1', 'fc3').replace(f'.{i_layer}',
                                                                                                          f'.{i_layer // 2}')] = \
                    infoxlm_state_dict[key]
                else:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = \
                    infoxlm_state_dict[key]
            elif 'fc2' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 0:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('fc2', 'fc4').replace(f'.{i_layer}',
                                                                                                          f'.{i_layer // 2}')] = \
                    infoxlm_state_dict[key]
                else:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = \
                    infoxlm_state_dict[key]
            elif 'final_layer_norm' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 0:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('final_layer_norm',
                                                                                    'ffn_layer_norm').replace(
                        f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
                else:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = \
                    infoxlm_state_dict[key]
            elif 'self_attn.out_proj' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 1:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('self_attn.out_proj',
                                                                                    'encoder_attn.out_proj').replace(
                        f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
                else:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = \
                    infoxlm_state_dict[key]
            elif 'self_attn.k_proj' in key or 'self_attn.v_proj' in key or 'self_attn.q_proj' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 1:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('self_attn', 'encoder_attn').replace(
                            f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
                else:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = \
                    infoxlm_state_dict[key]
            elif 'self_attn_layer_norm' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 1:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('self_attn_layer_norm',
                                                                                    'encoder_attn_layer_norm').replace(
                        f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
                else:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = \
                    infoxlm_state_dict[key]
            elif 'emb_layer_norm' in key:
                state_dict[key.replace('decoder.sentence_encoder.emb_layer_norm', 'layernorm_embedding')] = \
                infoxlm_state_dict[key]
            elif 'embed_positions' in key:
                state_dict[key.replace('decoder.sentence_encoder.', '')] = infoxlm_state_dict[key][:state_dict[
                    key.replace('decoder.sentence_encoder.', '')].size(0)]
            elif 'embed_tokens' in key:
                state_dict[key.replace('decoder.sentence_encoder.', '')][:infoxlm_state_dict[key].size(0)] = \
                infoxlm_state_dict[key]
            else:
                state_dict[key.replace('decoder.sentence_encoder.', '')] = infoxlm_state_dict[key]

    return state_dict


def upgrade_state_dict_for_ca_first_two_ffn(
        state_dict: Dict[str, Any], pretrained_infoxlm_checkpoint: str, num_layers: int
) -> Dict[str, Any]:
    if not os.path.exists(pretrained_infoxlm_checkpoint):
        raise IOError("Model file not found: {}".format(pretrained_infoxlm_checkpoint))

    # state = checkpoint_utils.load_checkpoint_to_cpu(pretrained_infoxlm_checkpoint)
    with open(PathManager.get_local_path(pretrained_infoxlm_checkpoint), "rb") as f:
        state = torch.load(f, map_location=torch.device("cpu"))
    infoxlm_state_dict = state["model"]

    for key in infoxlm_state_dict.keys():
        if 'layers' in key and int(key.split('.')[3]) > 2 * num_layers - 1:
            continue
        if not key.startswith('decoder.'):
            continue
        if 'lm_head' not in key:
            if 'in_proj_weight' in key:
                q, k, v = infoxlm_state_dict[key].chunk(3, dim=0)
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 1:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'q_proj.weight').replace(
                            f'.{i_layer}', f'.{i_layer // 2}')] = q
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'k_proj.weight').replace(
                            f'.{i_layer}', f'.{i_layer // 2}')] = k
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'v_proj.weight').replace(
                            f'.{i_layer}', f'.{i_layer // 2}')] = v
                else:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'q_proj.weight').replace(
                            f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = q
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'k_proj.weight').replace(
                            f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = k
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'v_proj.weight').replace(
                            f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = v
            elif 'in_proj_bias' in key:
                q, k, v = infoxlm_state_dict[key].chunk(3, dim=0)
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 1:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'q_proj.bias').replace(
                            f'.{i_layer}', f'.{i_layer // 2}')] = q
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'k_proj.bias').replace(
                            f'.{i_layer}', f'.{i_layer // 2}')] = k
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'v_proj.bias').replace(
                            f'.{i_layer}', f'.{i_layer // 2}')] = v
                else:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'q_proj.bias').replace(
                            f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = q
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'k_proj.bias').replace(
                            f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = k
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'v_proj.bias').replace(
                            f'.{i_layer}', f'.{i_layer // 2}').replace('self_attn', 'encoder_attn')] = v
            elif 'fc1' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 0:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('fc1', 'fc3').replace(f'.{i_layer}',
                                                                                                          f'.{i_layer // 2}')] = \
                    infoxlm_state_dict[key]
                else:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = \
                    infoxlm_state_dict[key]
            elif 'fc2' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 0:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('fc2', 'fc4').replace(f'.{i_layer}',
                                                                                                          f'.{i_layer // 2}')] = \
                    infoxlm_state_dict[key]
                else:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = \
                    infoxlm_state_dict[key]
            elif 'final_layer_norm' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 0:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('final_layer_norm',
                                                                                    'ffn_layer_norm').replace(
                        f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
                else:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = \
                    infoxlm_state_dict[key]
            elif 'self_attn.out_proj' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 0:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('self_attn.out_proj',
                                                                                    'encoder_attn.out_proj').replace(
                        f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
                else:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = \
                    infoxlm_state_dict[key]
            elif 'self_attn.k_proj' in key or 'self_attn.v_proj' in key or 'self_attn.q_proj' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 0:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace('self_attn', 'encoder_attn').replace(
                            f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
                else:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = \
                    infoxlm_state_dict[key]
            elif 'self_attn_layer_norm' in key:
                i_layer = int(key.split('.')[3])
                if i_layer % 2 == 0:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('self_attn_layer_norm',
                                                                                    'encoder_attn_layer_norm').replace(
                        f'.{i_layer}', f'.{i_layer // 2}')] = infoxlm_state_dict[key]
                else:
                    state_dict[
                        key.replace('decoder.sentence_encoder.', '').replace(f'.{i_layer}', f'.{i_layer // 2}')] = \
                    infoxlm_state_dict[key]
            elif 'emb_layer_norm' in key:
                state_dict[key.replace('decoder.sentence_encoder.emb_layer_norm', 'layernorm_embedding')] = \
                infoxlm_state_dict[key]
            elif 'embed_positions' in key:
                state_dict[key.replace('decoder.sentence_encoder.', '')] = infoxlm_state_dict[key][:state_dict[
                    key.replace('decoder.sentence_encoder.', '')].size(0)]
            elif 'embed_tokens' in key:
                state_dict[key.replace('decoder.sentence_encoder.', '')][:infoxlm_state_dict[key].size(0)] = \
                infoxlm_state_dict[key]
            else:
                state_dict[key.replace('decoder.sentence_encoder.', '')] = infoxlm_state_dict[key]

    return state_dict


def upgrade_mt_state_for_deep_model(
        state_dict: Dict[str, Any], pretrained_mt_checkpoint: str
) -> Dict[str, Any]:
    if not os.path.exists(pretrained_mt_checkpoint):
        raise IOError("Model file not found: {}".format(pretrained_mt_checkpoint))

    # state = checkpoint_utils.load_checkpoint_to_cpu(pretrained_infoxlm_checkpoint)
    with open(PathManager.get_local_path(pretrained_mt_checkpoint), "rb") as f:
        state = torch.load(f, map_location=torch.device("cpu"))
    mt_state_dict = state["model"]
    for key in mt_state_dict.keys():
        state_dict[key] = mt_state_dict[key]
    return state_dict


@register_model("GTrans")
class GTransModel(TransformerFromPretrainedInfoXLMModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerFromPretrainedInfoXLMModel.add_args(parser)
        parser.add_argument(
            "--variant",
            type=str,
            metavar="STR",
        )
        parser.add_argument(
            "--encoder-group-layers",
            type=int,
            metavar="STR",
        )
        parser.add_argument(
            "--encoder-group-type",
            type=str,
            metavar="STR",
            default="sparse"
        )
        parser.add_argument(
            "--decoder-group-layers",
            type=int,
            metavar="STR",
        )
        parser.add_argument(
            "--decoder-group-type",
            type=str,
            metavar="STR",
            default="sparse"
        )
        parser.add_argument(
            "--softmax-temperature",
            type=int,
            metavar="STR",
        )
        parser.add_argument(
            "--pretrained-mt-checkpoint",
            type=str,
            metavar="STR",
        )
        parser.add_argument(
            "--use-adapter",
            action="store_true",
        )
        parser.add_argument(
            "--freeze-encoder",
            action="store_true",
        )
        parser.add_argument(
            "--freeze-decoder",
            action="store_true",
        )
        parser.add_argument(
            "--language-specific-lr",
            type=str,
            metavar="STR",
        )
        parser.add_argument(
            "--random-freeze-subnet",
            action="store_true",
        )
        parser.add_argument(
            "--random-freeze-subnet-parameter",
            type=str,
            metavar="STR",
        )
        parser.add_argument(
            "--same-weight",
            type=float,
        )
        parser.add_argument(
            "--adapter-dim",
            type=int,
        )
        parser.add_argument(
            "--recurrent-init",
            action="store_true",
        )
        parser.add_argument(
            "--high-langs",
            type=str,
            metavar="STR",
        )
        parser.add_argument(
            "--low-langs",
            type=str,
            metavar="STR",
        )

    def get_freeze_prob(self, epoch):
        if epoch <= self.start_freeze_epoch:
            return 0
        elif epoch < self.start_freeze_epoch + self.freeze_warmup_epoch:
            return self.freeze_max_prob / self.freeze_warmup_epoch * (epoch - self.start_freeze_epoch)
        else:
            return self.freeze_max_prob

    def random_freeze(self, update_num):
        if torch.rand((1,)) < self.get_freeze_prob(update_num):
            freeze_index = torch.randint(1, self.decoder.decoder_group_number + 1, (1,))
            for param in self.encoder.parameters():
                param.requires_grad = False
            for i in range(freeze_index * self.decoder.decoder_group_layers):
                for param in self.decoder.layers[i].parameters():
                    param.requires_grad = False
        else:
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.decoder.parameters():
                param.requires_grad = True

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        if hasattr(args, "pretrained_mt_checkpoint") and os.path.exists(args.pretrained_mt_checkpoint):
            mt_loaded_state_dict = upgrade_mt_state_for_deep_model(
                state_dict=self.state_dict(),
                pretrained_mt_checkpoint=args.pretrained_mt_checkpoint
            )
            self.load_state_dict(mt_loaded_state_dict, strict=True)
            logger.info("Loading encoder and decoder from {0}".format(args.pretrained_mt_checkpoint))
        self.random_freeze_subnet = getattr(args, "random_freeze_subnet", False)
        self.random_freeze_subnet_parameter = getattr(args, "random_freeze_subnet_parameter", "4,2,0.5")
        if self.random_freeze_subnet_parameter is not None:
            self.start_freeze_epoch, self.freeze_warmup_epoch, self.freeze_max_prob = self.random_freeze_subnet_parameter.split(
                ",")
            self.start_freeze_epoch = int(self.start_freeze_epoch)
            self.freeze_warmup_epoch = int(self.freeze_warmup_epoch)
            self.freeze_max_prob = float(self.freeze_max_prob)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return GTransDecoder(args, tgt_dict, embed_tokens)

    @classmethod
    def build_encoder(cls, args, tgt_dict, embed_tokens):
        return GTransEncoder(args, tgt_dict, embed_tokens)

    @torch.jit.unused
    def forward_non_torchscript(self, net_input: Dict[str, Tensor]):
        encoder_input = {
            k: v for k, v in net_input.items() if k != "prev_output_tokens"
        }
        return self.forward(**encoder_input)

    def forward_torchscript(self, net_input: Dict[str, Tensor]):
        """A TorchScript-compatible version of forward.

        Encoders which use additional arguments may want to override
        this method for TorchScript compatibility.
        """
        if torch.jit.is_scripting():
            return self.forward(
                src_tokens=net_input["src_tokens"],
                src_lengths=net_input["src_lengths"],
                src_lang_id=net_input["src_lang_id"],
                tgt_lang_id=net_input["tgt_lang_id"]
            )
        else:
            return self.forward_non_torchscript(net_input)

    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            return_all_hiddens: bool = True,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lang_id=None,
            tgt_lang_id=None,
            epoch=1,
            prepend_target=None
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        if self.random_freeze_subnet and self.training:
            self.random_freeze(epoch)
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=True, src_lang_id=src_lang_id,
            tgt_lang_id=tgt_lang_id
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            src_lang_id=src_lang_id,
            tgt_lang_id=tgt_lang_id
        )
        return decoder_out


class GTransEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        if hasattr(args, "pretrained_infoxlm_checkpoint") and os.path.exists(args.pretrained_infoxlm_checkpoint):
            infoxlm_loaded_state_dict = upgrade_state_dict_with_infoxlm_weights(
                state_dict=self.state_dict(),
                pretrained_infoxlm_checkpoint=args.pretrained_infoxlm_checkpoint,
                num_layers=args.encoder_layers,
            )
            if args.encoder_layers > args.encoder_group_layers and args.encoder_layers % args.encoder_group_layers == 0 and args.recurrent_init:
                infoxlm_loaded_state_dict = CopyEncoderBottom2Top(infoxlm_loaded_state_dict, args.encoder_group_layers,
                                                                  args.encoder_layers // args.encoder_group_layers)
                logger.info("Recurrent initialization !")
            self.load_state_dict(infoxlm_loaded_state_dict, strict=False)
            logger.info("Loading encoder from {0}".format(args.pretrained_infoxlm_checkpoint))
        else:
            logger.info("Random Initialization for Encoder")

        assert args.encoder_group_layers <= args.encoder_layers
        self.encoder_group_type = getattr(args, "encoder_group_type", "sprase")
        self.encoder_group_layers = args.encoder_group_layers
        self.encoder_group_number = args.encoder_layers // args.encoder_group_layers
        self.langs = getattr(args, "langs", [])
        if hasattr(args, "high_langs") and hasattr(args, "low_langs"):
            self.high_langs = args.high_langs.split(',')
            self.low_langs = args.low_langs.split(',')
        else:
            self.high_langs = self.langs[: len(self.langs) // 2]
            self.low_langs = self.langs[len(self.langs) // 2: -1]
        if self.encoder_group_type == "sparse":
            self.encoder_fusion_weight = Weight(self.encoder_group_number)
        elif self.encoder_group_type == "ls:sparse":
            self.encoder_fusion_weight = nn.ParameterList(
                [Weight(self.encoder_group_number) for _ in range(len(self.high_langs + self.low_langs))])
        self.fusion_layer_norm = LayerNorm(embed_tokens.embedding_dim)
        if getattr(args, 'freeze_encoder', False):
            for param in self.parameters():
                param.requires_grad = False

    def forward(
            self,
            src_tokens,
            src_lengths,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
            src_lang_id=None,
            tgt_lang_id=None
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = []
        # encoder layers
        for i, layer in enumerate(self.layers):
            x = layer(x, encoder_padding_mask)
            assert encoder_states is not None
            encoder_states.append(x)

        if self.encoder_group_type == "sparse":
            x = torch.stack([encoder_states[i] for i in
                             range(self.encoder_group_layers - 1, len(encoder_states), self.encoder_group_layers)],
                            dim=0)
            x = x * torch.sigmoid(self.encoder_fusion_weight).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            x = x.mean(0)
        elif self.encoder_group_type == "ls:sparse":
            x = torch.stack([encoder_states[i] for i in
                             range(self.encoder_group_layers - 1, len(encoder_states), self.encoder_group_layers)],
                            dim=0)
            x = x * torch.sigmoid(self.encoder_fusion_weight).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            x = x.mean(0)
        elif self.encoder_group_type == "sparse-avg":
            x = torch.stack([encoder_states[i] for i in
                             range(self.encoder_group_layers - 1, len(encoder_states), self.encoder_group_layers)],
                            dim=0)
            x = x * torch.sigmoid(self.encoder_fusion_weight)
            x = x.mean(0)

        x = self.fusion_layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }


def Weight(in_features, value=0.0):
    m = nn.Parameter(torch.Tensor(in_features))
    nn.init.constant_(m, value)
    # nn.init.constant_(m, 1.0 / in_features)
    # nn.init.constant_(m, 1.0)
    return m


def ShallowGatedLinear(in_features, out_features, dropout=0.0, bias=True):
    return nn.Sequential(
        nn.Linear(in_features, out_features * 2, dropout, bias),
        nn.GLU(),
    )


class AdapterLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8) or 8
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        export = getattr(args, "char_inputs", False)
        self.cross_self_attention = getattr(args, "cross_self_attention", False)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu') or "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.adapter_dim = getattr(args, "adapter_dim", args.encoder_embed_dim)
        self.fc1 = self.build_fc1(
            self.embed_dim,
            self.adapter_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            self.adapter_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.gate = nn.Linear(self.embed_dim, self.embed_dim)
        self.normalize_before = args.decoder_normalize_before
        self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(
            self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def adapter_forward_with_residual(
            self,
            x,
            encoder_out: Optional[torch.Tensor] = None,
            encoder_padding_mask: Optional[torch.Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            prev_self_attn_state: Optional[List[torch.Tensor]] = None,
            prev_attn_state: Optional[List[torch.Tensor]] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            need_attn: bool = False,
            need_head_weights: bool = False,
    ):
        residual = x
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
                incremental_state is not None
                and _self_attn_input_buffer is not None
                and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = x + residual
        x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = x + residual
            x = self.encoder_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = x + residual
        x = self.final_layer_norm(x)
        return x

    def forward(
            self,
            x,
            encoder_out: Optional[torch.Tensor] = None,
            encoder_padding_mask: Optional[torch.Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            prev_self_attn_state: Optional[List[torch.Tensor]] = None,
            prev_attn_state: Optional[List[torch.Tensor]] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            need_attn: bool = False,
            need_head_weights: bool = False,
    ):
        residual = x
        x = self.adapter_forward_with_residual(x, encoder_out, encoder_padding_mask, incremental_state,
                                               prev_self_attn_state, prev_attn_state, self_attn_mask,
                                               self_attn_padding_mask, need_attn, need_head_weights)
        x = torch.sigmoid(self.gate(x)) * x + residual
        return x


class GTransDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        if hasattr(args, "pretrained_infoxlm_checkpoint") and os.path.exists(args.pretrained_infoxlm_checkpoint):
            if args.variant == 'addffn':
                infoxlm_loaded_state_dict = upgrade_state_dict_for_two_ffn(
                    state_dict=self.state_dict(),
                    pretrained_infoxlm_checkpoint=args.pretrained_infoxlm_checkpoint,
                    num_layers=args.decoder_layers,
                )
                if args.decoder_layers > args.decoder_group_layers and args.decoder_layers % args.decoder_group_layers == 0 and args.recurrent_init:
                    infoxlm_loaded_state_dict = CopyDecoderBottom2Top(infoxlm_loaded_state_dict,
                                                                      args.decoder_group_layers,
                                                                      args.decoder_layers // args.decoder_group_layers)
                    logger.info("Recurrent Initialization !")
            elif args.variant == 'addffn':
                infoxlm_loaded_state_dict = upgrade_state_dict_for_two_ffn(
                    state_dict=self.state_dict(),
                    pretrained_infoxlm_checkpoint=args.pretrained_infoxlm_checkpoint,
                    num_layers=args.decoder_layers,
                )
            elif args.variant == 'cafirst_addffn':
                infoxlm_loaded_state_dict = upgrade_state_dict_for_ca_first_two_ffn(
                    state_dict=self.state_dict(),
                    pretrained_infoxlm_checkpoint=args.pretrained_infoxlm_checkpoint,
                    num_layers=args.decoder_layers,
                )
            else:
                infoxlm_loaded_state_dict = upgrade_state_dict_with_infoxlm_weights(
                    state_dict=self.state_dict(),
                    pretrained_infoxlm_checkpoint=args.pretrained_infoxlm_checkpoint,
                    num_layers=args.decoder_layers,
                )
            self.load_state_dict(infoxlm_loaded_state_dict, strict=True)
            logger.info("Loading decoder from {0}".format(args.pretrained_infoxlm_checkpoint))
        else:
            logger.info("Random Initialization for Decoder")
        #
        self.langs = getattr(args, "langs", [])
        if hasattr(args, "high_langs") and hasattr(args, "low_langs"):
            self.high_langs = args.high_langs.split(',')
            self.low_langs = args.low_langs.split(',')
        else:
            self.high_langs = self.langs[: len(self.langs) // 2]
            self.low_langs = self.langs[len(self.langs) // 2: -1]
        self.softmax_temperature = getattr(args, "softmax_temperature", math.sqrt(args.decoder_embed_dim))
        self.decoder_group_layers = args.decoder_group_layers
        self.decoder_group_number = args.decoder_layers // args.decoder_group_layers
        self.decoder_group_type = getattr(args, "decoder_group_type", "sparse")
        assert args.decoder_group_layers <= args.decoder_layers, "Please ensure args.decoder_group_layers <= args.decoder_layers !"
        if self.decoder_group_type == "dense":
            self.decoder_fusion_weight = Weight(args.decoder_layers)
            self.loss_weight = Weight(self.decoder_group_number)
        elif self.decoder_group_type == "sparse-without-ensemble":
            self.decoder_fusion_weight = Weight(self.decoder_group_number)
            self.loss_weight = Weight(1)
        elif self.decoder_group_type == "ls:sparse-without-ensemble":
            self.decoder_fusion_weight = nn.ParameterList(
                [Weight(self.decoder_group_number) for _ in range(len(self.langs))])
            self.loss_weight = Weight(1)
        elif self.decoder_group_type == "ls:sparse":
            self.decoder_fusion_weight = None
            self.loss_weight = nn.ParameterList([Weight(self.decoder_group_number) for _ in range(len(self.langs))])
        elif self.decoder_group_type == "sparse":
            self.decoder_fusion_weight = None
            self.loss_weight = Weight(self.decoder_group_number)
        elif self.decoder_group_type == "last":
            self.loss_weight = Weight(1)

        self.use_adapter = getattr(args, "use_adapter", False)
        if self.use_adapter:
            self.adapter = nn.ModuleList([])
            self.adapter.extend(
                nn.ModuleList(
                    [
                        AdapterLayer(args)
                        for _ in range(len(self.high_langs + self.low_langs))
                    ])
            )
        else:
            self.adapter = None

        self.language_sepcific_lr = [float(w) for w in args.language_specific_lr.split(",")] if hasattr(args,
                                                                                                        "language_specific_lr") else None

        if getattr(args, 'freeze_decoder', False):
            for param in self.parameters():
                param.requires_grad = False
            if self.adapter is not None:
                for param in self.adapter.parameters():
                    param.requires_grad = True
            if self.decoder_fusion_weight is not None:
                for param in self.decoder_fusion_weight.parameters():
                    param.requires_grad = True
            if self.layer_norm is not None:
                for param in self.layer_norm.parameters():
                    param.requires_grad = True
            for param in self.loss_weight.parameters():
                param.requires_grad = True
        self.same_weight = getattr(args, "same_weight", -1)

    def build_decoder_layer(self, args, no_encoder_attn=False):
        if args.variant == 'first':
            layer = XLMTCrossAttnFirstLayer(args, no_encoder_attn)
        elif args.variant == 'large':
            layer = XLMTCrossAttnLargeLayer(args, no_encoder_attn)
        elif args.variant == 'halfffn':
            layer = XLMTTwoHalfFFN(args, no_encoder_attn)
        elif args.variant == 'addffn':
            layer = XLMTAddFFN(args, no_encoder_attn)
        elif args.variant == 'first_large_halfffn':
            layer = XLMTCaFirstQKLargeTwoHalfFFN(args, no_encoder_attn)
        elif args.variant == 'ca_sa_large':
            layer = XLMTCrossAttnSelfAttnLargeLayer(args, no_encoder_attn)
        elif args.variant == 'cafirst_addffn':
            layer = XLMTCaFirstAddFFN(args, no_encoder_attn)
        else:
            layer = TransformerDecoderLayer(args, no_encoder_attn)
        if getattr(args, "checkpoint_activations", False):
            layer = checkpoint_wrapper(layer)
        return layer

    def forward(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
            src_lang_id=None,
            tgt_lang_id=None
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lang_id=src_lang_id,
            tgt_lang_id=tgt_lang_id
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def get_lang_id(self, lang_id):
        # assert lang_id is not None, "lang_id in Batch can not be None Type !"
        if isinstance(lang_id, int):
            return lang_id
        else:
            return int(lang_id[0].cpu())

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]],
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lang_id=None,
            tgt_lang_id=None
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1
        if tgt_lang_id is not None:
            tgt_lang_id = self.get_lang_id(tgt_lang_id) - 1
        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = []
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                        encoder_out is not None
                        and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

            # if (idx + 1) % self.decoder_group_layers == 0 and (idx + 1) // self.decoder_group_layers == self.decoder_group_number - 1:
            #     if self.langs[tgt_lang_id] in self.low_langs:
            #         break

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        # T x B X C -> layers x T x B x C -> groups x group_layers x T x B x C
        if self.decoder_group_type == "dense":
            decoder_fusion_weight = torch.sigmoid(self.decoder_fusion_weight.unsqueeze(1).unsqueeze(2).unsqueeze(3))
            x = torch.stack(inner_states, dim=0) * decoder_fusion_weight
            x = x.view(self.decoder_group_number, self.decoder_group_layers, x.size(1), x.size(2), x.size(3))
            x = x.mean(dim=1)  # G x T x B x C
            loss_weight = self.loss_weight
            loss_weight = utils.softmax(loss_weight / self.softmax_temperature, dim=-1)
        elif self.decoder_group_type == "sparse":
            x = torch.stack([inner_states[i] for i in
                             range(self.decoder_group_layers - 1, len(inner_states), self.decoder_group_layers)])
            loss_weight = self.loss_weight
            loss_weight = utils.softmax(loss_weight / self.softmax_temperature, dim=-1)
        elif self.decoder_group_type == "ls:sparse":
            selected_states = [inner_states[i] for i in
                               range(self.decoder_group_layers - 1, len(inner_states), self.decoder_group_layers)]
            if self.use_adapter:
                if torch.rand((1,)) < 0.5 or not self.training:
                    for i in range(len(selected_states)):
                        if incremental_state is None and not full_context_alignment:
                            self_attn_mask = self.buffered_future_mask(x)
                        else:
                            self_attn_mask = None
                        selected_states[i] = self.adapter[tgt_lang_id](
                            selected_states[i],
                            encoder_out["encoder_out"][0],
                            encoder_out["encoder_padding_mask"][0],
                            incremental_state,
                            self_attn_mask=self_attn_mask,
                            self_attn_padding_mask=self_attn_padding_mask,
                        )
            if self.training and torch.rand((1,)) < self.same_weight:
                loss_weight = x.new(len(selected_states)).fill_(1.0)
            else:
                loss_weight = self.loss_weight[tgt_lang_id]
            loss_weight = utils.softmax(loss_weight / self.softmax_temperature, dim=-1)
            x = torch.stack(selected_states)
        elif self.decoder_group_type == "ls:sparse-without-ensemble":
            decoder_fusion_weight = torch.sigmoid(
                self.decoder_fusion_weight[tgt_lang_id].unsqueeze(1).unsqueeze(2).unsqueeze(3))
            x = torch.stack([inner_states[i] for i in
                             range(self.decoder_group_layers - 1, len(inner_states), self.decoder_group_layers)])
            x = x * decoder_fusion_weight
            x = x.mean(0).unsqueeze(0)
            loss_weight = self.loss_weight
            loss_weight = utils.softmax(loss_weight / self.softmax_temperature, dim=-1)
        elif self.decoder_group_type == "sparse-without-ensemble":
            decoder_fusion_weight = torch.sigmoid(self.decoder_fusion_weight.unsqueeze(1).unsqueeze(2).unsqueeze(3))
            x = torch.stack([inner_states[i] for i in
                             range(self.decoder_group_layers - 1, len(inner_states), self.decoder_group_layers)])
            x = x * decoder_fusion_weight
            x = x.mean(0).unsqueeze(0)
            loss_weight = self.loss_weight
            loss_weight = utils.softmax(loss_weight / self.softmax_temperature, dim=-1)
        elif self.decoder_group_type == "last":
            x = torch.stack([x], dim=0)
            loss_weight = self.loss_weight
            loss_weight = utils.softmax(loss_weight / self.softmax_temperature, dim=-1)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # groups x T x B x C -> groups x B x T x C
        x = x.transpose(1, 2)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if self.language_sepcific_lr is not None and self.training:
            loss_weight = self.language_sepcific_lr[tgt_lang_id] * loss_weight

        return x, {"attn": [attn], "inner_states": inner_states, "loss_weight": loss_weight}

    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Scriptable helper function for get_normalized_probs in ~BaseFairseqModel"""
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0]
        if log_probs:
            probs = utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
            if probs.size(0) == 1:
                return probs.squeeze(0)
            else:
                probs = probs * net_output[1]["loss_weight"].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                return probs.sum(0)
        else:
            probs = utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
            probs = probs * net_output[1]["loss_weight"].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            return probs.sum(0)


class XLMTCaFirstQKLargeTwoHalfFFN(TransformerDecoderLayer):

    def __init__(
            self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim // 2,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim // 2,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.fc3 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim // 2,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc4 = self.build_fc2(
            args.decoder_ffn_embed_dim // 2,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.ffn_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=embed_dim,
            vdim=embed_dim,
            qdim=embed_dim,
            outdim=embed_dim,
            qkprojdim=1152,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def forward(
            self,
            x,
            encoder_out: Optional[torch.Tensor] = None,
            encoder_padding_mask: Optional[torch.Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            prev_self_attn_state: Optional[List[torch.Tensor]] = None,
            prev_attn_state: Optional[List[torch.Tensor]] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            need_attn: bool = False,
            need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        ###############################################

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        ###############################################

        residual = x
        if self.normalize_before:
            x = self.ffn_layer_norm(x)

        x = self.activation_fn(self.fc3(x))
        x = self.activation_dropout_module(x)
        x = self.fc4(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.ffn_layer_norm(x)

        ###############################################

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
                incremental_state is not None
                and _self_attn_input_buffer is not None
                and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        ###############################################

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        ###############################################

        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None


class XLMTCaFirstAddFFN(TransformerDecoderLayer):

    def __init__(
            self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.fc3 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc4 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.ffn_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def forward(
            self,
            x,
            encoder_out: Optional[torch.Tensor] = None,
            encoder_padding_mask: Optional[torch.Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            prev_self_attn_state: Optional[List[torch.Tensor]] = None,
            prev_attn_state: Optional[List[torch.Tensor]] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            need_attn: bool = False,
            need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        ###############################################

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        ###############################################

        residual = x
        if self.normalize_before:
            x = self.ffn_layer_norm(x)

        x = self.activation_fn(self.fc3(x))
        x = self.activation_dropout_module(x)
        x = self.fc4(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.ffn_layer_norm(x)

        ###############################################

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
                incremental_state is not None
                and _self_attn_input_buffer is not None
                and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        ###############################################

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None


class XLMTAddFFN(TransformerDecoderLayer):

    def __init__(
            self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.fc3 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc4 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.ffn_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def forward(
            self,
            x,
            encoder_out: Optional[torch.Tensor] = None,
            encoder_padding_mask: Optional[torch.Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            prev_self_attn_state: Optional[List[torch.Tensor]] = None,
            prev_attn_state: Optional[List[torch.Tensor]] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            need_attn: bool = False,
            need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        ###############################################

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
                incremental_state is not None
                and _self_attn_input_buffer is not None
                and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        ###############################################

        residual = x
        if self.normalize_before:
            x = self.ffn_layer_norm(x)

        x = self.activation_fn(self.fc3(x))
        x = self.activation_dropout_module(x)
        x = self.fc4(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.ffn_layer_norm(x)

        ###############################################

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        ###############################################

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None


class XLMTTwoHalfFFN(TransformerDecoderLayer):

    def __init__(
            self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim // 2,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim // 2,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.fc3 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim // 2,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc4 = self.build_fc2(
            args.decoder_ffn_embed_dim // 2,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.ffn_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def forward(
            self,
            x,
            encoder_out: Optional[torch.Tensor] = None,
            encoder_padding_mask: Optional[torch.Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            prev_self_attn_state: Optional[List[torch.Tensor]] = None,
            prev_attn_state: Optional[List[torch.Tensor]] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            need_attn: bool = False,
            need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
                incremental_state is not None
                and _self_attn_input_buffer is not None
                and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.ffn_layer_norm(x)

        x = self.activation_fn(self.fc3(x))
        x = self.activation_dropout_module(x)
        x = self.fc4(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.ffn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None


class XLMTCrossAttnSelfAttnLargeLayer(TransformerDecoderLayer):
    def build_self_attention(
            self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=embed_dim,
            vdim=embed_dim,
            qdim=embed_dim,
            outdim=embed_dim,
            qkprojdim=1152,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=embed_dim,
            vdim=embed_dim,
            qdim=embed_dim,
            outdim=embed_dim,
            qkprojdim=1152,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )


class XLMTCrossAttnLargeLayer(TransformerDecoderLayer):
    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=embed_dim,
            vdim=embed_dim,
            qdim=embed_dim,
            outdim=embed_dim,
            qkprojdim=1152,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )


class XLMTCrossAttnFirstLayer(TransformerDecoderLayer):

    def forward(
            self,
            x,
            encoder_out: Optional[torch.Tensor] = None,
            encoder_padding_mask: Optional[torch.Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            prev_self_attn_state: Optional[List[torch.Tensor]] = None,
            prev_attn_state: Optional[List[torch.Tensor]] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            need_attn: bool = False,
            need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
                incremental_state is not None
                and _self_attn_input_buffer is not None
                and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None


@register_model_architecture(
    "GTrans", "GTrans_base"
)
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.layer_wise_attention = getattr(args, "layer_wise_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)

    args.init_encoder_only = getattr(args, "init_encoder_only", False)
    args.init_decoder_only = getattr(args, "init_decoder_only", False)
    args.max_positions = getattr(args, "max_positions", 512)

    # GTrans
    args.encoder_group_layers = getattr(args, "encoder_group_layers", args.encoder_layers)
    args.decoder_group_layers = getattr(args, "decoder_group_layers", args.decoder_layers)
    args.encoder_group_type = getattr(args, "encoder_group_type", "sparse")
    args.decoder_group_type = getattr(args, "decoder_group_type", "sparse")
    args.adapter_dim = getattr(args, "adapter_dim", args.decoder_embed_dim)
    args.recurrent_init = getattr(args, "recurrent_init", False)


@register_model_architecture(
    "GTrans", "GTrans_large"
)
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.layer_wise_attention = getattr(args, "layer_wise_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)

    args.init_encoder_only = getattr(args, "init_encoder_only", False)
    args.init_decoder_only = getattr(args, "init_decoder_only", False)
    args.max_positions = getattr(args, "max_positions", 512)
