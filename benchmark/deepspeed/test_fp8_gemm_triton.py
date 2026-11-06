# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Benchmark FP8 GEMM against eager Torch dequantization and matmul."""

import math

import pytest

import flag_train
from baseline.deepspeed.fp8_gemm_triton import DTYPES, OP_NAME, build_inputs, grid, native
from flag_train.deepspeed import fp8_gemm_triton

from .. import base


class Fp8GemmBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPE_DESC = "M, K, N"

    def set_shapes(self, shape_file=None):
        self.shapes = [binding["shape"] for binding in grid()]

    def set_more_shapes(self):
        return []

    def _measure_input(self, inputs, case_id=None):
        metric = super()._measure_input(inputs, case_id=case_id)
        assert metric.error_msg is None, f"{case_id}: {metric.error_msg}"
        if metric.speedup is not None:
            assert math.isfinite(metric.speedup) and metric.speedup > 0.8, (
                f"{case_id}: speedup {metric.speedup}x must be > 0.8x"
            )
        return metric


def fp8_input_fn(shape, dtype, device):
    args, _ = build_inputs({"shape": shape}, dtype, device)
    yield args


@pytest.mark.fp8_gemm_triton
@pytest.mark.skipif(not flag_train.runtime.device.support_bf16, reason="BF16 required")
def test_fp8_gemm_triton_perf():
    bench = Fp8GemmBenchmark(
        input_fn=fp8_input_fn, op_name=OP_NAME, torch_op=native(), dtypes=DTYPES
    )
    bench.set_train(fp8_gemm_triton)
    bench.run()
