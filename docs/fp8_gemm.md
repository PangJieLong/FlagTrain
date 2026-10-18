# DeepSpeed FP8 GEMM — portable BF16 validation

Implementation: `src/flag_train/deepspeed/fp8_gemm_triton.py`, exported as
`flag_train.deepspeed.matmul_fp8_triton(x, weight, scale, quantization_group_size)`.
The kernel fuses DeepSpeed custom uint8 E4M3 decoding, scaling and GEMM with FP32
accumulation. Output has the activation dtype. It is forward-only. Weights are
not `torch.float8_e4m3fn`: byte zero represents `2**-7` before scaling; groups
index flattened `[K, N]` elements. Production code imports no DeepSpeed helpers.

## Current acceptance scope

The generic implementation and shared BF16 tests use FlagTrain's active device
and public component API, following the LAMB and `GenericBenchmark` structure.
There is no NVIDIA-only entry guard or vendor-wide test skip. NVIDIA is the
only hardware on which this contribution has been measured; Ascend (`ascend`),
Hygon (`hygon`), T-Head (`thead`), MetaX (`metax`) and other backends are **pending
hardware validation**, not claimed supported or fast merely by collecting tests.
Existing FP16 entry points, gradients and end-to-end training are not covered.

The baseline is the **unmodified** public API
`deepspeed.ops.fp_quantizer.matmul_fp8`. No replacement kernels, launch overrides,
scale padding or fallback baseline are installed. Missing dependencies are errors
in original-baseline tests, not a reason to silently substitute torch. Portable
candidate correctness tests do not import or require DeepSpeed.

## Cross-platform workflow

1. Install the target vendor's PyTorch/Triton stack and FlagTrain, then verify
   `flag_train.vendor_name` and `flag_train.device`. Use the same public import
   `from flag_train.deepspeed import matmul_fp8_triton` on every backend.
2. Run portable correctness (all 15 core shapes plus edge/contract tests):
   `python -m pytest tests/deepspeed/test_fp8_gemm.py -q -ra -k "not against_original_deepspeed"`.
   Add `--ref cpu` to execute decoding and reference matmul on CPU, not simply
   copy a reference result computed on the accelerator.
3. Install/build original DeepSpeed's FP quantizer for that platform, then run
   the full correctness and default performance commands below. Baseline build,
   execution and numerical failures remain visible failures. Only the narrowly
   identified NVIDIA H20 limitation below is skipped.
4. If the baseline is not yet available, obtain diagnostic candidate latency:
   `python -m pytest benchmark/deepspeed/test_fp8_gemm.py -q -s --mode=kernel --metrics latency --record=json --output=/tmp/fp8_latency.json`.
   This explicitly bypasses DeepSpeed and uses portable encoded inputs. Reports
   contain **no baseline latency or speedup** and mark acceptance as not evaluated.
   A passing latency-only run is not a pass against the requested 0.8x floor.
5. Reuse the generic kernel if correctness and original-baseline performance both
   pass. If compilation, correctness or performance fails, investigate it and
   add a vendor implementation through the existing backend registration, not a
   copy of the test suite or a broad skip.

Vendor implementations can follow the existing Hygon component layout under
`src/flag_train/runtime/backend/_<vendor>/ops/deepspeed/`. Re-export the same
`matmul_fp8_triton` name in that package's `__init__.py` and include it in
`__all__`; the runtime discovers the backend ops package and its subpackages.
`SpecOpRegistrar` in `flag_train.deepspeed` replaces the generic export when an
override exists; both test suites call that public export. No untested vendor
implementation or per-vendor tuning claim is added by this refactor.

Device guards and synchronization use `torch_device_fn`. Kernel-mode timing is
selected by the shared benchmark framework: `do_bench_npu` on Ascend and
`triton.testing.do_bench` on the other configured paths. NVIDIA precision and
shared-memory workarounds are not applied to NPU or other vendors indiscriminately.
Devices declared unable to run BF16 are explicitly skipped, not reported as passed.

All core shapes use `K=4096`, `N=8192`, `quantization_group_size=128`:

| Group | M | Default behavior on every platform |
| --- | --- | --- |
| Common base (7 upstream + 4 supplemental) | 1, 2, 4, 8, 16, 17, 32, 48, 64, 96, 128 | Attempt original baseline comparison and timing |
| Large-M extension (upstream) | 256, 512, 1024, 2048 | Also attempt original baseline comparison and timing; only confirmed limitations may skip |

Both groups run by default. A `shape_group` label distinguishes them in benchmark
case plans, timing-row provenance and pytest/JUnit properties (the shared accuracy
JSON does not serialize these properties); it does not disable large shapes. All 15
candidate shapes run against the torch reference. H20/Triton 3.6 baseline
resource skips below are **not** copied to Ascend, Hygon, T-Head or MetaX.
Passing the base group does not establish large-M performance acceptance.

Supplemental shapes were chosen before timing: M=16/17 exercises a tile-size
boundary, while M=48/96 exercises partial tiles. Shapes are not removed for
candidate correctness failures or speedups below the threshold.

## Narrow skip policy

Four original BF16 launches require 263168 bytes of shared memory per block,
exceeding the measured H20 limit of 232448 bytes. The skip requires all of:

- Exactly those four M values and the stated K/N/group size.
- GEMM source SHA256 `5c376c95cca24995283a13ca2a6ec7773cbad20aaa90e74e0ea7a18fe8826096`
  (observed at DeepSpeed commit `79caae1c04fca210345bdeb03bcacf87b1ac2f23`).
- Triton 3.6.x, H20 device name, and per-block shared-memory limit below 263168.

Other versions, shapes and errors do not inherit this skip. Numerical failures
are not skipped. These are 11 usable comparisons and four visible upstream skips,
not 15 baseline passes.

## Correctness

Every candidate core shape uses deterministic CPU-generated custom E4M3 bytes,
scales and activations, transferred to the active backend. The independent
arithmetic decoder uses FP32 (no accelerator FP64 requirement) and checks the
candidate against torch matmul. These bytes are test inputs, not an alternative
DeepSpeed quantizer or performance acceptance baseline.

Separate original-baseline tests use actual DeepSpeed quantization and upstream's
reference `torch.matmul(x, fpq.dequantize(weight, scale=fpq.scale))`. The independent
decoder must also exactly match DeepSpeed's decoder; both implementations are
checked against the reference and compared directly with each other. Under
`--ref cpu`, reference operands are moved before the decoder and matmul execute.

Repository `train_assert_close` ultimately uses `torch.testing.assert_close`:
BF16 rtol=0.016 and fixed atol=0.025. Core checks retain upstream's native-dtype
mean-relative-error formula and threshold `<0.004`, plus an FP32 relative mean
as an additional stable assertion. Neither check replaces a failed assertion.
There are also 25 candidate-only edge/oracle/invalid-input checks; these do not
claim that the original wrapper supports the extended input contracts.

The functional suite additionally checks that four non-NVIDIA vendor identities
cannot inherit the H20 skip and that a portable candidate case never loads
DeepSpeed. Vendor-identity policy tests are not hardware execution tests.
In total there are 60 nodes: 45 portable checks and 15 original-baseline checks.

## Performance

Each core shape is a separate pytest node. NVIDIA kernel mode uses
`triton.testing.do_bench()` via `GenericBenchmark` with its configured warmup,
repetitions and median statistic. Both timed calls allocate their output and
consume identical inputs. Quantization, reference construction and correctness
checks occur outside timing. Every measured case must satisfy
`DeepSpeed_latency / FlagTrain_latency > 0.8`; a lower result fails, not skips.

The JSON retains all 15 pytest outcomes and all actually measured rows (11 on
the pinned H20 stack; potentially 15 elsewhere); skipped cases have no speedup.
`--list-cases` lists all shapes without loading DeepSpeed.
`--case-id` supports an explicit known skip and still rejects unknown IDs.

## Commands

Install DeepSpeed with `DS_BUILD_FP_QUANTIZER=1 pip install -e .`, then FlagTrain
with `pip install -e .`, in the same environment. On the development server:

```bash
cd /workspace/fp8_gemm_dev/FlagTrain-nv-pr
source /workspace/fp8_gemm_dev/venv/bin/activate
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$PWD/src:$PWD"
python -m pytest tests/deepspeed/test_fp8_gemm.py -q -ra \
  --record=json --output=/tmp/fp8_bf16_accuracy.json
python -m pytest benchmark/deepspeed/test_fp8_gemm.py -q -s -ra --mode=kernel \
  --record=json --output=/tmp/fp8_bf16_benchmark.json
```

Use `python -m pytest` to avoid a system executable selecting another Python.
No baseline-selection environment variable is needed.

The development environment uses PyTorch 2.11.0a0+eb65b36914.nv26.02, CUDA 13.1,
Triton 3.6.0 and DeepSpeed 0.17.7+79caae1 on H20-3e. Two pre-existing external
DeepSpeed build adaptations permit Triton 3.6 and rename a CUDA translation unit
to avoid duplicate object names. They do not modify the GEMM kernel or launch
and are not part of this FlagTrain contribution.

## Original NVIDIA acceptance before portability refactor (2026-10-14)

The contribution was integrated onto FlagTrain main commit
`e4abdf01b4490f5dc2522577922bc8ec50e0be2f` and revalidated through the component
public API. The top-level exports, shared test framework, CI and other operators
are unchanged. The development commands above explicitly select the PR worktree
through `PYTHONPATH`, without changing the original checkout's editable install.

On the environment above, the BF16 functional suite completed with **51 passed,
4 skipped**. The four skips affect only original-baseline comparisons: all 15
candidate core shapes passed against the torch reference. Performance completed
with **11 passed, 4 skipped**; every measured ratio exceeded 0.8x.

| M | Origin | DeepSpeed (ms) | FlagTrain (ms) | Ratio |
| --- | --- | ---: | ---: | ---: |
| 1 | Upstream | 0.051744 | 0.056608 | 0.914x |
| 2 | Upstream | 0.059712 | 0.056352 | 1.060x |
| 4 | Upstream | 0.059584 | 0.056608 | 1.053x |
| 8 | Upstream | 0.059520 | 0.056384 | 1.056x |
| 16 | Supplemental | 0.059808 | 0.056864 | 1.052x |
| 17 | Supplemental | 0.075968 | 0.060384 | 1.258x |
| 32 | Upstream | 0.075872 | 0.056672 | 1.339x |
| 48 | Supplemental | 0.072064 | 0.074688 | 0.965x |
| 64 | Upstream | 0.070592 | 0.074720 | 0.945x |
| 96 | Supplemental | 0.146784 | 0.089056 | 1.648x |
| 128 | Upstream | 0.147328 | 0.096704 | 1.523x |

Ratios below 1 mean slower than DeepSpeed, despite satisfying the requested
0.8x floor. These are measurements on one pinned environment, not a guarantee
for other devices or software versions.

A second complete run also produced 11 passed and four skipped, with ratios
from 0.917x to 1.641x (`benchmark_repeat.json`). Single-case selection was checked:
a supported ID records exactly one timing row; a known unsupported ID records
no timing rows and an explicit skip; an unknown ID fails with exit code 1.
Unselected pytest parameter nodes can report passed but do not count as timed
cases; use recorded timing rows, not that CLI run's total passed count.

The 25 edge/oracle/invalid-input checks also passed under Compute Sanitizer
memcheck with **0 errors**. LAMB and packbits functional regression:
**106 passed**. Black, isort, repository-configured flake8, component export
checks and `git diff --check` passed.

Raw logs and JSON are retained outside the repository at
`/workspace/fp8_gemm_dev/results/nv_submission_20261014/`, including
`accuracy.json`, `benchmark.json`, `memcheck.log` and `other_ops.log`.
Previous mixed-baseline results are not evidence for this BF16-only acceptance.

## Portability refactor regression (2026-10-14)

On the same NVIDIA H20 stack, after removing the vendor-only gate and separating
portable tests from baseline tests:

- Device-reference correctness: **56 passed, 4 skipped**.
- `--ref cpu` correctness: **56 passed, 4 skipped**; reference computation runs on CPU.
- Original-DeepSpeed performance, two complete runs: **11 passed, 4 skipped** each;
  ratios **0.920329x–1.651016x** and **0.910532x–1.646170x**. Every measured case
  remains above 0.8x, not necessarily above 1x.
- With all DeepSpeed imports deliberately blocked, portable correctness:
  **45 passed, 15 deselected**; explicit latency-only benchmarking: **15 passed**.
  All 15 recorded latency-only rows have no baseline latency and no speedup;
  the JSON explicitly says DeepSpeed speedup acceptance was not evaluated.
- The same missing-DeepSpeed harness fails an original-baseline test as expected:
  there is no implicit replacement baseline or false acceptance pass.
- LAMB and packbits regression: **106 passed**.
- Repository-configured Black, isort and flake8 checks passed.

Evidence is outside the source tree at
`/workspace/fp8_gemm_dev/results/nv_portability_20261014/`. The initial CPU-reference
oracle-contract test revealed a device-placement assertion, which was fixed and
the entire CPU-reference suite rerun; the final results above supersede that
initial attempt. No numerical threshold was relaxed. The five additional policy
and dependency-isolation tests are not a substitute for real Ascend/Hygon/T-Head/
MetaX validation. Run the same commands on those platforms before reporting support.

## Base + large-M regression (2026-10-18)

Both groups remain enabled by default. On the pinned NVIDIA H20 environment:

- Full device-reference and full CPU-reference correctness each completed with
  **56 passed, 4 skipped**. All 15 candidate shapes passed; the four skips remain
  the narrowly identified original-baseline resource limitation.
- Two final original-DeepSpeed performance runs each completed with **11 passed,
  4 skipped**, at **0.912847x–1.640266x** and **0.910271x–1.643422x**. All measured
  ratios exceeded 0.8x. Large-M speedup is still unmeasured on this H20 baseline.
- With DeepSpeed imports blocked, **45 portable correctness checks passed** and
  all **15 candidate-only timing cases passed**. Baseline acceptance still fails
  when its required dependency is missing; diagnostic rows have no baseline or
  speedup and explicitly label acceptance as not evaluated.
- Sanitizer: **30 checks passed, 0 errors**. LAMB + packbits: **106 passed**.
- Report/case-ID checks and repository-configured formatting/lint checks passed.

Final evidence is retained at
`/workspace/fp8_gemm_dev/results/nv_portability_20261018/`. Neither policy tests
using vendor names nor NVIDIA regressions count as execution on other hardware.
The proposed uniform exclusion of large shapes was not adopted: other vendors
must attempt the large-M original baseline and report the actual outcome.
