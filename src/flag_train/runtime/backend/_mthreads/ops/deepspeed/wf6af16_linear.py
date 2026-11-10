# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MUSA implementation of the FP6-weight FP16-activation linear operator."""

import triton

from flag_train.deepspeed.cuda_wf6af16_linear import (
    _cuda_wf6af16_linear_gemm,
    _gemm_configs,
    _launch_cuda_wf6af16_linear,
)
from flag_train.utils import libentry

# Multi-stage uint32 loads are lowered to async global-to-shared copies on MUSA
# and can fault with a misaligned address during the split-K decode case.
cuda_wf6af16_linear_kernel = libentry()(
    triton.autotune(
        configs=_gemm_configs(num_stages=1),
        key=["M", "N", "K", "split_k"],
    )(_cuda_wf6af16_linear_gemm)
)


def wf6af16_linear(
    output,
    hidden_states,
    weights_2bit,
    weights_4bit,
    scale,
    out_channels=None,
    tokens=None,
    in_channels=None,
    split_k=None,
    wide_load=None,
):
    """Run FP6 linear with the MUSA autotune configurations."""
    return _launch_cuda_wf6af16_linear(
        cuda_wf6af16_linear_kernel,
        output,
        hidden_states,
        weights_2bit,
        weights_4bit,
        scale,
        out_channels,
        tokens,
        in_channels,
        split_k,
        wide_load,
    )
