# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only setup regressions: python -m unittest discover -s tools/tests."""

import importlib.util
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "repair_quantizer", TOOLS / "repair_deepspeed_fp_quantizer.py"
)
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


def kernel(name):
    return (
        "__global__ void " + name + "()\n{\n"
        "    constexpr int q_exponent_bits = total_q_bits - mantisa_bits - 1;\n"
        "    constexpr int input_bits = q_mantisa_bits + q_exponent_bits + 1;\n"
        "    if (dst_exponent != (1 << q_exponent_bits) - 1) {}\n"
        "    auto bias = (1 << (q_exponent_bits - 1)) -\n"
        "                               (1 << (q_exponent_bits - 1)) - 1;\n"
        "    auto output = ((sign << (q_exponent_bits + mantisa_bits)) | 0);\n"
        "}\n"
    )


class QuantizerRepairTests(unittest.TestCase):
    def test_nvidia_image_runs_repair_after_install(self):
        dockerfile = (TOOLS.parent / ".github/docker/gpu-runner.Dockerfile").read_text(
            encoding="utf-8"
        )
        copy = "COPY tools/repair_deepspeed_fp_quantizer.py "
        run = "RUN python3 /opt/flagtrain-ci/repair_deepspeed_fp_quantizer.py"
        self.assertLess(dockerfile.index('"deepspeed==0.19.7"'), dockerfile.index(copy))
        self.assertLess(dockerfile.index(copy), dockerfile.index(run))

    def setUp(self):
        self.source = (
            "// untouched quantization\n"
            + kernel("apply_dequantization")
            + kernel("apply_selective_dequantization")
            + "// untouched launch\n"
        )

    def test_only_output_width_changes(self):
        fixed = repair.repair_source(self.source)
        self.assertEqual(fixed.count("output_exponent_bits"), 8)
        self.assertEqual(fixed.count("q_mantisa_bits + q_exponent_bits + 1"), 2)
        self.assertEqual(fixed.count("auto bias = (1 << (q_exponent_bits - 1))"), 2)
        self.assertTrue(fixed.startswith("// untouched quantization\n"))
        self.assertTrue(fixed.endswith("// untouched launch\n"))

    def test_unexpected_source_rejected(self):
        with self.assertRaises(ValueError):
            repair.repair_source(self.source.replace("total_q_bits", "changed"))

    def test_output_exponent_width(self):
        # Exact bit-field equivalence to the pre-rename upstream decoder,
        # including BF16/FP16, all signs and representative input widths.
        for output_mantissa in (7, 10):
            output_exponent = 16 - output_mantissa - 1
            for input_mantissa, input_exponent in ((1, 2), (2, 3), (3, 4), (7, 4)):
                for code in range(1 << (input_mantissa + input_exponent + 1)):
                    sign = code >> (input_mantissa + input_exponent)
                    exponent = (code >> input_mantissa) & ((1 << input_exponent) - 1)
                    mantissa = code & ((1 << input_mantissa) - 1)
                    adjusted = exponent
                    if exponent != (1 << output_exponent) - 1:
                        adjusted += (1 << (output_exponent - 1)) - (
                            1 << (input_exponent - 1)
                        )
                    expected = (
                        (sign << 15)
                        | (adjusted << output_mantissa)
                        | (mantissa << (output_mantissa - input_mantissa))
                    )
                    actual = (
                        (sign << (output_exponent + output_mantissa))
                        | (adjusted << output_mantissa)
                        | (mantissa << (output_mantissa - input_mantissa))
                    )
                    self.assertEqual(actual, expected)

    def test_hash_guards_and_idempotence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "ops/csrc/fp_quantizer/fp_quantize_impl.cu"
            gemm = root / "ops/fp_quantizer/fp8_gemm_triton.py"
            source.parent.mkdir(parents=True)
            gemm.parent.mkdir(parents=True)
            original = self.source.encode()
            fixed = repair.repair_source(self.source).encode()
            source.write_bytes(original)
            gemm.write_bytes(b"original GEMM")
            with patch.multiple(
                repair,
                ORIGINAL_SHA256=repair.digest(original),
                REPAIRED_SHA256=repair.digest(fixed),
                GEMM_SHA256=repair.digest(gemm.read_bytes()),
            ):
                with self.assertRaises(RuntimeError):
                    repair.repair(root, "0.19.8")
                self.assertEqual(source.read_bytes(), original)
                self.assertTrue(repair.repair(root, "0.19.7"))
                self.assertFalse(repair.repair(root, "0.19.7"))
                self.assertEqual(source.read_bytes(), fixed)
                self.assertEqual(gemm.read_bytes(), b"original GEMM")
                source.write_bytes(b"unknown source")
                with self.assertRaises(RuntimeError):
                    repair.repair(root, "0.19.7")
                self.assertEqual(source.read_bytes(), b"unknown source")
                source.write_bytes(original)
                gemm.write_bytes(b"changed GEMM")
                with self.assertRaises(RuntimeError):
                    repair.repair(root, "0.19.7")
                self.assertEqual(source.read_bytes(), original)


class CompilerRetryTests(unittest.TestCase):
    def test_thead_pins_required_baseline(self):
        config = (TOOLS.parent / "backends.yaml").read_text(encoding="utf-8")
        thead = config.split("\n  thead:\n", 1)[1]
        self.assertIn("- deepspeed==0.19.7", thead)

    def test_thead_baseline_setup_is_scoped_and_fails_closed(self):
        bash = shutil.which("bash")
        if not bash:
            self.skipTest("bash required for setup shell tests")
        script = (TOOLS / "setup_env.sh").read_text(encoding="utf-8")
        marker = "# T-Head uses the original DeepSpeed FP8 API"
        start = script.index(marker)
        self.assertLess(script.index('-e "${REPO_ROOT}[test]"'), start)
        block = script[start : script.index('\necho "checking that FlagTrain', start)]
        stub = 'python3() { printf "CALL %s\\n" "$*"; return "$FAIL_STATUS"; }\n'
        for backend, failure, expected_calls in (
            ("thead", "0", 2),
            ("thead", "23", 1),
            ("ascend-cann900", "0", 0),
            ("nvidia-cuda133", "0", 0),
        ):
            with self.subTest(backend=backend, failure=failure):
                result = subprocess.run(
                    [bash, "--noprofile", "--norc", "-e", "-c", stub + block],
                    env=dict(
                        os.environ,
                        BACKEND=backend,
                        FAIL_STATUS=failure,
                        REPO_ROOT="/repo",
                    ),
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, int(failure), result.stderr)
                self.assertEqual(result.stdout.count("CALL "), expected_calls)
                if expected_calls:
                    self.assertIn(
                        "/repo/tools/repair_deepspeed_fp_quantizer.py", result.stdout
                    )
                if expected_calls == 2:
                    self.assertIn("FP_Quantize, matmul_fp8", result.stdout)

    def run_install(self, backend, failures):
        bash = shutil.which("bash")
        if not bash:
            self.skipTest("bash required for setup shell tests")
        script = (TOOLS / "setup_env.sh").read_text()
        function = script.split("install_compiler() {", 1)[1].split("\n}\n", 1)[0]
        command = "install_compiler() {" + function + "\n}\n"
        command += """
count=0
python3() {
    count=$((count + 1))
    printf 'CALL %s\n' "$*"
    [ "$count" -gt "$FAILURES" ] || return 19
}
sleep() { :; }
set +e
install_compiler --no-cache-dir --index-url https://example.invalid/simple flagtree==0.7.0+ppu3.6
status=$?
printf 'RESULT %s %s\n' "$status" "$count"
exit "$status"
"""
        env = dict(os.environ, BACKEND=backend, FAILURES=str(failures))
        return subprocess.run(
            [bash, "--noprofile", "--norc", "-c", command],
            env=env,
            capture_output=True,
            text=True,
        )

    def test_retry_succeeds_without_weakening_pip(self):
        result = self.run_install("thead", 2)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("RESULT 0 3", result.stdout)
        calls = [
            line for line in result.stdout.splitlines() if line.startswith("CALL ")
        ]
        self.assertEqual(len(set(calls)), 1)
        self.assertIn("--no-cache-dir", calls[0])
        self.assertNotIn("--trusted-host", calls[0])

    def test_persistent_failure_is_not_hidden(self):
        result = self.run_install("thead", 3)
        self.assertEqual(result.returncode, 19)
        self.assertIn("RESULT 19 3", result.stdout)

    def test_other_backends_not_retried(self):
        result = self.run_install("nvidia-cuda133", 1)
        self.assertEqual(result.returncode, 19)
        self.assertIn("RESULT 19 1", result.stdout)


if __name__ == "__main__":
    unittest.main()
