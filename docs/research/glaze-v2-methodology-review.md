# Glaze v2: research comparison and methodology review

Assessment date: 2026-10-05. Scope: primary literature, local implementation, and saved experiment artifacts; no new GPU experiments.

**Verdict:** Glaze v2 is an effective deployment recipe with a substantial measured MATH-500 gain over the tested uniform AutoRound configuration. The current study does not establish a new calibration principle, new Fisher theory, or superiority over contemporary mixed-precision PTQ. Its plausible contribution is a specific, economical integration of sensitivity estimation, fused-kernel constraints, bit/group selection, and vision-to-language storage allocation. That contribution needs matched comparisons and isolated ablations.

## Comparison with prior methods

| Method / precedent | Relevant established mechanism | Glaze v2's distinction and research implication |
| --- | --- | --- |
| [GPTQ](https://arxiv.org/abs/2210.17323) | Second-order, layerwise weight quantization with compensation for rounding error. | Glaze learns rounding/clipping and uses sensitivity for a heterogeneous storage allocation. Beating the tested GPTQ recipe supports practical value, but does not address the closest novelty precedents. |
| [AWQ](https://arxiv.org/abs/2306.00978) | Activation-informed weight scaling protects salient channels. | Glaze uses backward-derived importance and block reconstruction. A methodological difference exists; neither activation awareness nor selective protection is new. |
| [AutoRound / SignRound](https://arxiv.org/abs/2309.05516) | Joint rounding and clipping optimization with signed gradient descent. | Glaze inherits this optimization pattern and changes the objective, data, and allocation. Learned rounding itself is not its contribution. |
| [SignRoundV2](https://arxiv.org/html/2512.04746v2) / [AutoRound AutoScheme](https://github.com/intel/auto-round) | Gradient-weighted quantization perturbations guide mixed precision; dynamic programming assigns bits under a budget, followed by learned rounding. | Glaze instead uses sampled-label Fisher approximations, includes group-size choices and fused-module constraints, and fixes vision to INT8. **Modern AutoRound already supports mixed precision:** the existing comparison tests one uniform configuration, not its strongest relevant capability. |
| [BRECQ](https://arxiv.org/abs/2102.05426) and [GuidedQuant](https://arxiv.org/abs/2505.07004) | Fisher-informed reconstruction and end-loss gradient guidance have prior art. GuidedQuant retains interactions between weights within output channels. | Glaze uses separable token/channel weights for block-output error. This is a specific approximation whose incremental benefit and approximation quality remain untested. BRECQ is an ancestral CNN precedent, not a directly interchangeable VLM baseline. |
| [Optimal Formats for Weight Quantisation](https://arxiv.org/html/2505.12988v1) | Relates output KL to Fisher-weighted squared weight error; estimates predictive Fisher with sampled model labels and allocates tensor bit widths. | Glaze's input/output marginal factorization and discrete bit/group menu differ. The KL–Fisher rationale and model-sampled labels are established mechanisms. |
| [PrismaQuant / AURA](https://github.com/RobTand/prismaquant) | A primary implementation combines KL–Fisher sensitivity, byte-budget optimization, native exports, and held-out validation. | Its estimator retains gradient/error direction through squared inner products, unlike Glaze's marginal squared-error proxy. Its native formats and hardware differ. Treat it as a close implementation precedent, not independently validated peer-reviewed evidence or a comparable headline score. |
| [MABA](https://openaccess.thecvf.com/content/CVPR2026F/html/Zhang_Modality-Aware_Bit_Allocation_for_Mixed-Precision_Quantization_of_Vision-Language_Models_CVPRF_2026_paper.html) | Gradient-guided mixed precision across VLM groups under memory/latency constraints. | Glaze fixes vision precision and allocates language precision using text calibration. It does not jointly estimate both modalities' sensitivities. Cross-modal precision budgeting is established; this particular deployment policy may still be useful. |
| [Self-calibration](https://arxiv.org/abs/2410.17170), [COLA](https://arxiv.org/abs/2510.10618), [AYOT](https://arxiv.org/abs/2608.01078) | Model-generated calibration, capability-focused data curation, and target-model reasoning traces already appear in PTQ research. | Glaze combines public training prompts with its BF16 model's answers and fixed domain proportions. The particular recipe is empirical; using the teacher's own reasoning data is not a new general calibration technique. |
| [GPTQv2](https://arxiv.org/abs/2504.02692) | Asymmetric reconstruction accounts for accumulated upstream quantization errors. | Feeding quantized-chain inputs while targeting the clean BF16 chain is also an established idea. |

This comparison identifies substantial overlap, not proof that every detail of Glaze has previously appeared together. A combination can be publishable if it solves a distinct constraint and demonstrates a reproducible advantage over close alternatives.

## What the saved evidence establishes

The independent audit matched complete question sets and document/prompt hashes across Glaze, AutoRound, and BF16 for these three tasks. Accuracy differences are percentage points; intervals are paired question bootstrap intervals conditional on these fixed runs.

| Task | Glaze | AutoRound | BF16 | Glaze − AutoRound, 95% interval | Glaze − BF16, 95% interval |
| --- | ---: | ---: | ---: | --- | --- |
| MATH-500, 500 questions | 83.2 | 73.2 | 83.4 | +10.0 [+6.6, +14.0] | −0.2 [−3.4, +2.8] |
| IFEval, 541 prompts | 82.8 | 80.6 | 82.3 | +2.2 [−0.6, +5.0] | +0.6 [−2.0, +3.3] |
| MMLU-Pro, fixed 1,001-question panel | 71.1 | 71.4 | 73.6 | −0.3 [−2.4, +2.0] | −2.5 [−4.7, −0.4] |

The MATH-500 gain is convincing for the tested recipe: Glaze alone solves 75 questions and AutoRound alone solves 25. Glaze and BF16 have similar aggregate MATH accuracy but disagree on correctness for 65 questions. Similar totals do not establish equivalent behavior or a predefined non-inferiority margin. The MMLU panel uses a different aggregation from the full-suite headline; its BF16 deficit must not be replaced by the full-suite −0.9 figure.

Saved 4B KL decreases from 0.03049 to 0.01231 in-domain (59.6%) and from 0.04351 to 0.03105 on chat (28.6%). These are validation-set results, for reasons below. Export size is 3,795,765,945 bytes versus AutoRound's 3,800,831,215: 5.07 MB smaller. Native serving and benchmark outputs support deployability; file size alone establishes neither resident GPU memory nor throughput.

Source outputs and recomputed values are preserved in [the evidence audit](/Users/johnzakkam/Projects/qwen35-compression/docs/research/glaze-v2-evidence-audit.json). Producer revisions recorded locally are `e2cb1cb` for the proxy and `d3a2e2b` for 4B.

## Methodology defects that affect the conclusions

1. **Calibration content is confounded with volume.** Proxy A uses 128 generic blocks; B uses 512 Glaze blocks. Its 48.4% in-domain KL reduction measures content plus four times the available calibration blocks. It cannot be attributed to teacher-generated domain data alone.

2. **The weighting ablation changes the optimization package.** B uses AutoRound; C switches to Glaze's pipeline and allows 400 rather than 200 steps, with different reconstruction and selection behavior. The additional 6.0% KL reduction does not isolate Fisher weighting. There is no otherwise identical Glaze run with token/channel weights set to one.

3. **Allocation and vision precision change together.** C leaves vision in BF16; D uses INT8 vision and redistributes storage. The 43.2% reduction is the combined intervention. It does not establish that Fisher ranks options better than random, activation-only, or gradient-based allocation. The proxy D export is also about 5% smaller than A, so this is a no-larger budget comparison rather than an exactly equal-size comparison.

4. **“Held-out KL” reuses selection data.** `rounding.py` repeatedly evaluates the 64 held-out blocks to select layer states; `glaze2_evaluate.py` later reads the same blocks for the KL gate. They are validation data, not an untouched test set. All 32 layers selected iteration 400 in the 4B manifest, so this run also supplies no evidence that early stopping improved quality. External benchmark outcomes remain useful evidence, subject to benchmark-informed design choices.

5. **The Fisher objective is approximate.** Allocation multiplies marginal input second moments by marginal output-gradient second moments, ignores correlations and cross-module interactions, and scores RTN errors before learned rounding. Reconstruction separately factorizes token and channel importance. Neither expression is an exact end-to-end KL measurement. Test whether predicted savings rank measured, post-rounding option gains before making theoretical accuracy claims.

6. **Documentation overstates two properties.** Answer masking appears in KL evaluation, but the Fisher and reconstruction calls receive token IDs without answer masks. “Losses count answer tokens only” therefore does not describe the complete training pipeline. Allocation uses conservative 64 KiB cost bins: dynamic programming is exact for that discretized problem, not necessarily the original byte-level optimum.

7. **Numerical and statistical scope is limited.** KL reloads stored weights into a dequantized BF16 student; it is not served-kernel KL. Benchmarks were run across A30 and A100 hardware. Question bootstrap intervals do not include calibration-seed, optimization-seed, kernel, or hardware variability. IFEval's interval includes zero; broad claims of improvement or “nothing measurably worse” need task-specific uncertainty and predefined margins.

8. **Generalization and contamination checks need extension.** Both sizes belong to one model family; the thinking track is untested. Text calibration does not directly calibrate vision. Exact 13-gram filtering against MATH-500 and MMLU-Pro is useful but does not rule out semantic overlap or audit every evaluated dataset. Dropping truncated teacher answers may bias calibration toward shorter/easier examples.

Implementation evidence: [proxy settings](/Users/johnzakkam/Projects/qwen35-compression/configs/variants/glaze2_proxy.yaml), [Fisher estimation](/Users/johnzakkam/Projects/qwen35-compression/src/qwen35_compression/glaze2/fisher.py), [allocation](/Users/johnzakkam/Projects/qwen35-compression/src/qwen35_compression/glaze2/allocate.py), [reconstruction and selection](/Users/johnzakkam/Projects/qwen35-compression/src/qwen35_compression/glaze2/rounding.py), [KL evaluator](/Users/johnzakkam/Projects/qwen35-compression/src/qwen35_compression/glaze2/evaluate.py).

## A defensible next research design

**Hypothesis:** under the same whole-export byte budget, calibration corpus, and optimization effort, Glaze's sensitivity approximation and constrained bit/group allocation improve untouched-test quality beyond modern mixed-precision AutoRound and simpler allocation policies.

| Question | Required comparison |
| --- | --- |
| Does calibration content help? | Generic versus teacher-generated domain data at both 128 and 512 blocks, holding optimizer and steps fixed. Separately compare prompt-only and prompt-plus-generated-answer inputs. |
| Does Fisher reconstruction help? | Glaze weights on/off under the same fixed allocation, data, steps, and selection schedule. Compare with a GuidedQuant-style weighting control where feasible. |
| Does the allocator help? | Fixed INT8 vision throughout; random, simple heuristic, SignRoundV2-style, and Glaze allocation under the same fused-unit options and actual byte ceiling. Use several random assignments. |
| Is the gain from the rounding engine? | AutoRound and Glaze optimize the identical allocation on identical blocks; compare both equal-step and equal-time settings. |
| Is the vision storage exchange worthwhile? | BF16 versus INT8 vision without language upgrades first; then measure the complete exchange with vision quality and serving memory/latency. |
| Does the approach generalize? | Confirm on 4B, an additional architecture, and at least three calibration/optimization seeds. Test unseen domains, long reasoning, and multimodal inputs. |

Use prompt/source-cluster splits into calibration, validation, and untouched test before packing. Freeze all choices before final benchmark scoring. Decontaminate against every test set used. For KL uncertainty, resample prompts or source clusters rather than treating correlated tokens as independent observations.

Retain GPTQ and AWQ as practical reference baselines, but prioritize current AutoRound AutoScheme/SignRoundV2 for the central claim. Compare Fisher estimators inspired by Optimal Formats and AURA inside a common supported option menu; do not compare incompatible native formats or their unrelated published headline scores. Check adapter and model support before promising a direct baseline run.

Report quality–size–cost curves at multiple feasible budgets, paired task intervals, seed variability, and predefined vision/chat non-inferiority margins. Measure actual GPU memory, prefill/decode speed, and end-to-end compression time. The saved 4B quantization stage took about 44 minutes and 19.4 GiB peak allocation; teacher answer generation adds about 14 minutes. Those two stages alone total approximately 58 minutes, excluding setup and evaluation. Baseline timings and provider rates are required for a cost advantage claim.

**Decision rule:** if the gain survives identical data, stronger mixed-precision baselines, isolated weighting/allocation controls, and untouched evaluation, a narrow algorithmic or systems contribution becomes defensible. If it disappears under those controls, the present result remains a useful deployment recipe and an empirical calibration study; the claim should reflect that outcome.
