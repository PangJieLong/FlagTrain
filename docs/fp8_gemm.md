# DeepSpeed FP8 GEMM — portable BF16 validation

Public API:
`flag_train.deepspeed.matmul_fp8_triton(x, weight, scale, quantization_group_size)`.
The generic implementation in `src/flag_train/deepspeed/fp8_gemm_triton.py` fuses
DeepSpeed custom uint8 E4M3 decoding, scaling and GEMM with FP32 accumulation.
An Ascend override in
`src/flag_train/runtime/backend/_ascend/ops/deepspeed/fp8_gemm_triton.py` uses
separate Triton decode and GEMM kernels for selected large-M BF16 inputs and
delegates other supported inputs to the unchanged generic implementation.
Output has the activation dtype. The API is forward-only. Weights are not
`torch.float8_e4m3fn`: byte zero represents `2**-7` before scaling; groups index
flattened `[K, N]` elements. Production code imports no DeepSpeed helpers.

## Current acceptance scope

The generic implementation, Ascend override and shared BF16 tests use FlagTrain's
active device and public component API, following the LAMB and
`GenericBenchmark` structure.
There is no NVIDIA-only entry guard or vendor-wide test skip. Historical NVIDIA
performance and Ascend (`ascend`) portable correctness have been measured.
The generic kernel missed the 0.8x floor on three large Ascend cases against the
fixed Torch lookup-table composite baseline. The Ascend two-stage path passed
full BF16 correctness and two complete performance runs against that same
baseline on 910B4 with CANN 9.0.0; see the dated acceptance results below.
The latest shared-test revision and Ascend registration changes have not been
rerun on NVIDIA hardware. The generic production kernel is unchanged, but the
dated NVIDIA results below are historical, not physical regression results for
this revision. Hygon (`hygon`), T-Head (`thead`), MetaX (`metax`) and other backends
are **pending hardware validation**, not claimed supported or fast merely by
collecting tests.
Existing FP16 entry points, gradients and end-to-end training are not covered.

On NVIDIA and other non-Ascend vendors the baseline remains the **unmodified**
public API `deepspeed.ops.fp_quantizer.matmul_fp8`, with real DeepSpeed quantization.
On Ascend the explicitly selected functional and performance baseline is
**Torch eager dequantization followed by BF16 `torch.matmul`**. The decoder uses
a fixed 256-entry FP32 custom E4M3 codebook (1 KiB per device) and
`torch.index_select`. Every timed call converts all input bytes to int32 indices,
looks up every weight, multiplies FP32 scales, converts the decoded weights to
BF16 and performs matmul. Only the format-constant codebook is initialized
outside timing; it is independent of input values, scales and tensor shapes.
Input-derived indices, decoded/scaled weights and outputs are never cached.
Neither `torch.compile` nor the independent correctness oracle is used as this
baseline.
Both calls receive identical fixed-seed, pre-generated custom E4M3 weights and
scales. This tests the decode-and-GEMM workload, not DeepSpeed's unsupported NPU
quantizer, original Triton entry or high-level fallback.

The vendor policy is fixed, not selected by catching an execution failure.
Reports identify the baseline mode, input pipeline, timed scope and source hash
(the Torch helper on Ascend, the upstream GEMM source elsewhere). No DeepSpeed
kernel or launch settings are modified. Ascend tests and benchmarks do not
require DeepSpeed. Missing DeepSpeed dependencies on non-Ascend original-baseline
paths remain errors, not a reason to silently substitute Torch. Portable
candidate correctness tests also do not import or require DeepSpeed.

## Cross-platform workflow

1. Install the target vendor's PyTorch/Triton stack and FlagTrain, then verify
   `flag_train.vendor_name` and `flag_train.device`. Use the same public import
   `from flag_train.deepspeed import matmul_fp8_triton` on every backend.
2. Run portable correctness (all 15 core shapes plus edge/contract tests):
   `python -m pytest tests/deepspeed/test_fp8_gemm.py -q -ra -k "not against_baseline"`.
   Add `--ref cpu` to execute decoding and reference matmul on CPU, not simply
   copy a reference result computed on the accelerator.
3. On non-Ascend paths, install original DeepSpeed and build its FP quantizer.
   Ascend's selected Torch baseline requires no DeepSpeed installation. Run the
   full correctness and default performance commands below. Baseline build,
   execution and numerical failures remain visible failures. Only the narrowly
   identified NVIDIA H20 limitation below is skipped.
4. If the baseline is not yet available, obtain diagnostic candidate latency:
   `python -m pytest benchmark/deepspeed/test_fp8_gemm.py -q -s --mode=kernel --metrics latency --record=json --output=/tmp/fp8_latency.json`.
   This explicitly bypasses DeepSpeed and uses portable encoded inputs. Reports
   contain **no baseline latency or speedup** and mark acceptance as not evaluated.
   A passing latency-only run is not a pass against the requested 0.8x floor.
5. Reuse the generic kernel if correctness and selected-baseline performance both
   pass. If compilation, correctness or performance fails, investigate it and
   add a vendor implementation through the existing backend registration, not a
   copy of the test suite or a broad skip. Ascend now has such an override for
   eligible large-M inputs. Its full device-reference, CPU-reference and repeated
   performance runs use the same shared suites and unchanged 15 core shapes.

Vendor implementations can follow the existing Hygon component layout under
`src/flag_train/runtime/backend/_<vendor>/ops/deepspeed/`. Re-export the same
`matmul_fp8_triton` name in that package's `__init__.py` and include it in
`__all__`; the runtime discovers the backend ops package and its subpackages.
`SpecOpRegistrar` in `flag_train.deepspeed` replaces the generic export when an
override exists; both test suites call that public export. The Ascend package's
`__init__.py` registers its FP8 GEMM override through this mechanism. No overrides
or hardware-validation claims are added for other vendors.

## Ascend implementation and memory trade-off

The Ascend entry first reuses the generic `_validate` contract. It selects the
two-stage path only when the activation is BF16, `M >= 512`, group size is 128,
and positive K and N are both divisible by 128. Because the two kernels use
int32 offsets, the path additionally requires `K*N <= 2**31 - 1`,
`M*N <= 2**31 - 1` and `(M-1)*AM + (K-1)*AK <= 2**31 - 1`, where AM and AK are
the activation's row and column strides in elements. A range exceeding any of
these bounds routes to the generic `_launch` before allocating the workspace
or launching either stage, as do other valid shapes or dtypes outside the
specialization. This is a fixed dispatch rule, not an exception-triggered retry;
compilation or execution errors are not hidden by changing the path. Routing
oversized tensors to the generic path is not a claim that every such tensor
fits device memory or has been validated on hardware.
Among the 15 core shapes, M=512, 1024 and 2048 select the new path; M=256 and
all smaller core shapes retain the generic kernel.

The first Triton kernel decodes and scales the complete uint8 weight matrix into
a newly allocated BF16 workspace. Each weight is decoded once per public call,
rather than once for each M tile. The second Triton kernel multiplies the BF16
activation and workspace, using FP32 accumulation and
`BLOCK_M=128, BLOCK_N=256, BLOCK_K=128`, four warps and one pipeline stage.
It honors activation strides and masks partial output tiles. The implementation
uses Torch for allocation and device selection, not for decode or matmul
computation; neither production stage calls the Torch baseline or DeepSpeed.

The trade-off is an additional `2*K*N`-byte workspace: **64 MiB** for the core
4096-by-8192 weights, in addition to the output and input tensors. Each call
allocates and repopulates this workspace, then writes and rereads it through
device memory. Neither decoded weights nor outputs are retained for reuse.
This spends extra memory and bandwidth to remove repeated decoding across M
tiles; it is not a zero-workspace fused kernel or a cross-call weight cache.
This trade-off passed the full core suite on the measured Ascend stack below.

## Timing and shapes

Device guards and synchronization use `torch_device_fn`. Kernel-mode timing is
selected by the shared benchmark framework: `do_bench_npu` on Ascend and
`triton.testing.do_bench` on the other configured paths. On Ascend, the intended
kernel-mode metric sums the device-kernel durations for all operations in each
call, not only the final matmul. On the Ascend candidate's two-stage path this
includes both the decode kernel and the BF16 GEMM kernel; neither stage is
precomputed outside the measured call. It is not CPU end-to-end latency. The
installed timer's handling of this multi-kernel workload was checked against
retained profiler task durations, including both candidate kernels.
NVIDIA precision and shared-memory workarounds are not
applied to NPU or other vendors indiscriminately.
Devices declared unable to run BF16 are explicitly skipped, not reported as passed.

All core shapes use `K=4096`, `N=8192`, `quantization_group_size=128`:

| Group | M | Default behavior on every platform |
| --- | --- | --- |
| Common base (7 upstream + 4 supplemental) | 1, 2, 4, 8, 16, 17, 32, 48, 64, 96, 128 | Attempt selected-baseline comparison and timing |
| Large-M extension (upstream) | 256, 512, 1024, 2048 | Also attempt selected-baseline comparison and timing; only the scoped H20 limitation below is skipped |

Both groups run by default. A `shape_group` label distinguishes them in benchmark
case plans, timing-row provenance and pytest/JUnit properties (the shared accuracy
JSON does not serialize these properties); it does not disable large shapes. All 15
candidate shapes run against the torch reference. Ascend attempts all 15 Torch
baseline comparisons, including all four large-M cases. H20/Triton 3.6 baseline
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
DeepSpeed quantizer. Ascend's Torch-composite acceptance also uses these encoded
inputs; both implementations consume the same activation, bytes and scales.

On non-Ascend vendors, separate original-baseline tests use actual DeepSpeed
quantization and upstream's reference
`torch.matmul(x, fpq.dequantize(weight, scale=fpq.scale))`. The independent
decoder must also exactly match DeepSpeed's decoder; both implementations are
checked against the reference and compared directly with each other. Under
`--ref cpu`, reference operands are moved before the decoder and matmul execute.
On Ascend, the Torch composite baseline and candidate GEMM instead compare
against the independent arithmetic decoder and FP32 torch matmul, rounded back
to BF16, using their shared pre-generated operands. The baseline performs
vectorized `torch.index_select` from the fixed 256-entry FP32 format codebook,
applies scales and executes BF16 matmul. It does not reconstruct FP32 bits with
per-weight left shifts. The independent oracle instead uses chunked arithmetic
decoding and can run on CPU with `--ref cpu`; it is not called by the timed
baseline. The arithmetic decoder uses exact integer shifts for powers of two,
avoiding an NPU `exp2` rounding error, and checks all 256 encodings against
scalar `math.ldexp`.

Repository `train_assert_close` ultimately uses `torch.testing.assert_close`:
BF16 rtol=0.016 and fixed atol=0.025. Core checks retain upstream's native-dtype
mean-relative-error formula and threshold `<0.004`, plus an FP32 relative mean
as an additional stable assertion. Neither check replaces a failed assertion.
There are also 30 candidate-only edge/oracle/invalid-input checks; these do not
claim that the original wrapper supports the extended input contracts.
Five of these cover the new large-M dispatch boundary at M=511/512/513, a partial
N tile, noncontiguous large-M activations and a large-M layout delegated to the
generic kernel. They still call the public component API on each active vendor;
collecting them on one backend does not validate another backend.

The functional suite additionally checks that four non-NVIDIA vendor identities
cannot inherit the H20 skip and that a portable candidate case never loads
DeepSpeed. Vendor-identity policy tests are not hardware execution tests.
Entry-routing policy cases and failure-propagation checks additionally verify
that only Ascend uses the Torch composite path and that neither selected
baseline is replaced after an error. Decoder checks cover all 256 codes, zero
and non-power-of-two scales, partial groups and BF16/FP32 outputs. Further checks
verify that every baseline invocation decodes again, that real mutations of
weights and scales affect the next result rather than reusing cached data, and
that Ascend input construction, metadata and baseline execution do not load
DeepSpeed. The full suite contains 15 core baseline-comparison nodes in addition
to portable core, edge, contract and policy tests; report the collected count
from the actual run.

## Performance

Each core shape is a separate pytest node. NVIDIA kernel mode uses
`triton.testing.do_bench()` via `GenericBenchmark` with its configured warmup,
repetitions and median statistic. Both timed calls allocate their output and
consume identical inputs. Encoded-input preparation, quantization where used,
independent reference construction and correctness checks occur outside timing.
On Ascend, input-byte conversion to int32 indices, all weight lookups, FP32
scaling, BF16 conversion, BF16 matmul and their allocations remain inside each
measured baseline invocation. Initialization of the input-independent 1 KiB
format codebook is the only decoder setup outside timing; no tensor derived
from the current weights, scales or shape is retained between calls. The
benchmark must not time only matmul on predecoded weights. Every measured case
must satisfy `selected_baseline_latency / FlagTrain_latency > 0.8`; a lower
result fails, not skips. Ascend ratios are against this Torch composite, not
DeepSpeed, and must not be presented as original-DeepSpeed speedups.
The candidate is also timed through its complete allocating public API. For
eligible Ascend large-M inputs this includes allocation and population of the
BF16 workspace, the decode launch, the GEMM launch and output allocation inside
the call. Kernel-mode results count the device execution of both stages; they
do not separately quantify Python dispatch or host allocation overhead.

A diagnostic on the current Ascend stack at `M=16, K=4096, N=8192` measured
0.81187 ms for the lookup-table Torch composite versus 4.19136 ms for a
multiplication-based bitcast variant that avoided left shifts. The earlier
per-weight `LeftShift` variant exhibited approximately one-second latency on
this stack and is not an acceptance baseline. These diagnostics motivated the
lookup-table baseline choice; they are not the final full-shape performance
results, and neither the discarded variants nor their speedup ratios count
toward acceptance.

With the fixed lookup-table baseline, the generic candidate's full performance
run missed the floor at M=512 (0.791160x), M=1024 (0.459678x) and M=2048
(0.290038x). These failures were not skipped or removed. A two-stage Triton
prototype subsequently measured 1.20705 ms and 1.138x at M=2048 with the tile
configuration above. This single-shape diagnostic motivated the Ascend override;
it is not final acceptance evidence for the integrated implementation. The
separate full-suite results below validate the integrated public override,
including its final int32 range guards.

The JSON retains all 15 pytest outcomes and all actually measured rows (11 on
the pinned H20 stack; potentially 15 elsewhere); skipped cases have no speedup.
`--list-cases` lists all shapes without loading DeepSpeed.
`--case-id` supports an explicit known skip and still rejects unknown IDs.

## Commands

On NVIDIA, install DeepSpeed with `DS_BUILD_FP_QUANTIZER=1 pip install -e .`, then FlagTrain
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

For Ascend, use the same pytest files and kernel-mode benchmark command; the
`ascend` vendor policy selects Torch eager dequantization plus BF16 matmul.
DeepSpeed and its unsupported NPU FP quantizer are not required. In the current
development container `flagtrtee-dev-900-pangjielong`:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/secure/pangjielong/fp8_gemm_ascend_dev/venv-cann900/bin/activate
cd /home/secure/pangjielong/fp8_gemm_ascend_dev/FlagTrain-cann900
export ASCEND_RT_VISIBLE_DEVICES=1
export PYTHONPATH="$PWD/src:$PWD"
python -m pytest tests/deepspeed/test_fp8_gemm.py -q -ra \
  --record=json --output=/tmp/fp8_bf16_accuracy_torch.json
python -m pytest benchmark/deepspeed/test_fp8_gemm.py -q -s -ra --mode=kernel \
  --record=json --output=/tmp/fp8_bf16_benchmark_torch.json
```

These commands use the baseline validated in the Ascend results below.
The earlier Ascend experiment using the original DeepSpeed
Triton entry encountered compiler failures and is not the current acceptance
baseline. Its outcomes cannot establish acceptance against the Torch composite.

The NVIDIA development environment uses PyTorch 2.11.0a0+eb65b36914.nv26.02, CUDA 13.1,
Triton 3.6.0 and DeepSpeed 0.17.7+79caae1 on H20-3e. Two pre-existing external
DeepSpeed build adaptations permit Triton 3.6 and rename a CUDA translation unit
to avoid duplicate object names. They do not modify the GEMM kernel or launch
and are not part of this FlagTrain contribution.

## Ascend Torch-composite acceptance (2026-10-23)

Environment: Ascend 910B4, CANN 9.0.0, Python 3.11.12, PyTorch 2.8.0,
torch_npu 2.8.0.post4.dev20260409 and FlagTree 0.7.0+ascend3.5 (Triton 3.5.1).
Physical device 1 was selected with `ASCEND_RT_VISIBLE_DEVICES=1`.
The public function resolved to the Ascend-specific module before validation.

- Device-reference correctness: **85 passed, 0 skipped**.
- CPU-reference correctness (`--ref cpu`): **85 passed, 0 skipped**.
- Two complete kernel-mode benchmarks: **15 passed, 0 skipped** each;
  ratios **1.155567x–1.707169x** and **1.137021x–1.721949x**.
- With every DeepSpeed import blocked: **85 correctness checks passed**;
  the additional M=2048 benchmark passed at **1.148361x** (14 nodes deselected,
  exactly one timed row). This is supplementary, not a third full-shape run.
- Nine external metadata-only dispatch checks passed, covering int32 bounds,
  threshold and alignment routing without allocating oversized tensors. They
  do not establish hardware support for huge tensors.
- Ascend packbits regression in the same registration package, with CPU
  reference: **84 passed**. This does not imply LAMB or full-training validation.
- Black, isort, repository-configured flake8 and `git diff --check` passed.

All 15 core shapes, including M=256/512/1024/2048, were measured without copying
the NVIDIA baseline skips. No accuracy threshold was relaxed. Ratios are
`Torch eager dequantization + BF16 matmul / FlagTrain`, not original-DeepSpeed
speedups or end-to-end training speedups.

A separate retained-profiler audit at M=2048 found five device kernels per
Torch baseline call (cast, gather, scale, cast, matmul) and two per candidate
call (decode, GEMM). Summing all active task durations and dividing by active
function calls reproduced the installed timer's result to floating-point
roundoff: 1.360104 ms baseline and 1.213012 ms candidate in that audit. Thus
neither decoder was omitted from the kernel-mode metric. Host dispatch and
allocation overhead are not included in the device-duration sum, even though
allocation remains inside each measured function invocation.

The final source hash of the Ascend implementation is
`56d71976c9c767b4d818914c3fb3a766040550cb31965838b2cee428be2139c3`;
the Torch-baseline helper hash is
`156281495f7295e739982b970bea80b08e9638cb36db64b698ce4c0351b8f141`.
Raw JSON, logs, dispatch checks and retained profiler data are outside the
repository at
`/home/secure/pangjielong/fp8_gemm_ascend_dev/results/torch_baseline_20261023/ascend_final_checked/`.
Earlier failed generic-kernel runs and tuning diagnostics are retained separately;
they are not replaced by these final results. NVIDIA has not been physically
rerun with this shared-test revision.

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
The proposed uniform exclusion of large shapes was not adopted. At the time of
this historical regression, the intended comparison on other vendors was also
the original DeepSpeed baseline. The current Ascend policy above supersedes
that baseline selection with the Torch composite, while retaining all large-M
shapes and reporting their actual outcomes. These dated NVIDIA results do not
validate the current Ascend implementation or the latest shared tests on NVIDIA.
