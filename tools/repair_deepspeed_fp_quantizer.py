# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Repair the audited DeepSpeed 0.19.7 quantizer before its first JIT build.

The upstream rename conflates input FP8 and output FP16/BF16 exponent widths
in two dequantizers. Restore distinct names and the output bias/sign width.
Quantization, Python APIs and the timed FP8 GEMM are not modified.
Run explicitly from NVIDIA CI setup, never from operator/test execution.
"""

import hashlib
from importlib import metadata
from pathlib import Path

ORIGINAL_SHA256 = "cf52494decbde30cd5d8d3413951123a70a568af35c99fefde5ecc1c8b52e923"
REPAIRED_SHA256 = "8f61258c9b479a615d1130322b9e8dad6c1954a991e3c460c411aa6a08484d78"
GEMM_SHA256 = "5c376c95cca24995283a13ca2a6ec7773cbad20aaa90e74e0ea7a18fe8826096"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def repair_source(source):
    replacements = (
        (
            "constexpr int q_exponent_bits = total_q_bits - mantisa_bits - 1;",
            "constexpr int output_exponent_bits = total_q_bits - mantisa_bits - 1;",
        ),
        (
            "if (dst_exponent != (1 << q_exponent_bits) - 1)",
            "if (dst_exponent != (1 << output_exponent_bits) - 1)",
        ),
        (
            "                               (1 << (q_exponent_bits - 1)) - 1;",
            "                               (1 << (output_exponent_bits - 1)) - 1;",
        ),
        (
            "((sign << (q_exponent_bits + mantisa_bits)) |",
            "((sign << (output_exponent_bits + mantisa_bits)) |",
        ),
    )
    # Restrict every replacement to the two dequantization kernel bodies.
    for name in ("apply_dequantization", "apply_selective_dequantization"):
        start = source.index("__global__ void " + name + "(")
        end = source.index("\n}\n", start) + 2
        block = source[start:end]
        for before, after in replacements:
            if block.count(before) != 1:
                raise ValueError("unexpected DeepSpeed source in " + name)
            block = block.replace(before, after)
        source = source[:start] + block + source[end:]
    return source


def repair(package_root, version):
    if version != "0.19.7":
        raise RuntimeError("Re-audit the quantizer repair for DeepSpeed " + version)
    source = package_root / "ops/csrc/fp_quantizer/fp_quantize_impl.cu"
    gemm = package_root / "ops/fp_quantizer/fp8_gemm_triton.py"
    if digest(gemm.read_bytes()) != GEMM_SHA256:
        raise RuntimeError("Unexpected DeepSpeed GEMM source; refusing repair")
    data = source.read_bytes()
    original_hash = digest(data)
    if original_hash == REPAIRED_SHA256:
        print("DeepSpeed 0.19.7 quantizer repair already applied:", original_hash)
        return False
    if original_hash != ORIGINAL_SHA256:
        raise RuntimeError("Unexpected quantizer SHA256: " + original_hash)
    repaired = repair_source(data.decode("utf-8")).encode("utf-8")
    if digest(repaired) != REPAIRED_SHA256:
        raise RuntimeError("Unexpected repaired quantizer SHA256")
    source.write_bytes(repaired)
    print("DeepSpeed 0.19.7 quantizer repair:", original_hash, "->", digest(repaired))
    print("Unmodified FP8 GEMM SHA256:", GEMM_SHA256)
    return True


if __name__ == "__main__":
    distribution = metadata.distribution("deepspeed")
    repair(Path(distribution.locate_file("deepspeed")), distribution.version)
