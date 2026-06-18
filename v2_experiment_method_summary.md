# v2 Experiment Method Summary

## Overview

The v2 experiments shift the research focus from "always correct by steering" to:

```text
Detect context-ignorance reliably.
Steer only when correction is plausible.
Abstain when the case appears unsafe or unrecoverable.
```

The main reason for this shift is that conflict detection is relatively strong, while steering recovery and cross-dataset transfer are limited.

## Core Experimental Design

All PubMedQA experiments use 10-fold cross validation.

For each fold:

```text
Validation split: 900 examples
  - feature selection
  - context-specific feature filtering
  - alpha tuning
  - conflict-score threshold selection

Test split: 100 examples
  - final recovery/corruption measurement
  - conflict-score AUC
  - CAI policy evaluation
```

The test split is used only once per fold for measurement. Feature IDs, layers, alphas, and thresholds are chosen only on the validation split.

Final reported numbers should be:

```text
mean +- std over 10 folds
```

## Experiment 1: Context-Specific Steering CV

Output:

```text
results/steering_context_specific/qwen3.5-9b_context_specific_cv.json
```

This experiment tests whether SAE features can causally reduce PubMedQA context-ignorance.

For each fold:

1. Split PubMedQA into validation/test.
2. Use validation correct vs wrong cases to find dominant SAE features.
3. Filter those features for context-specificity:
   - PubMedQA with context should activate the feature strongly.
   - PubMedQA without context should activate it less.
   - MedQA should not activate it much.
4. Tune steering alpha on validation.
5. Evaluate on test.

Steering sets:

```text
all_wrong_dominant
context_specific_wrong
single_28696_suppress
multi_layer_context_specific_wrong
```

Metrics:

```text
recovery   = originally wrong cases corrected by steering
corruption = originally correct cases broken by steering
```

## Experiment 2: Conflict Score CV

Output:

```text
results/conflict_score/qwen3.5-9b_conflict_score_cv_L20.json
```

Conflict score:

```text
wrong_signal / (correct_signal + wrong_signal)
```

The goal is to test whether SAE feature signals can detect context-ignorance.

For each fold:

1. Use validation scores to select a threshold.
2. Measure AUC and confusion statistics on test.

This is the main detection experiment.

## Experiment 3: Recovered vs Unrecovered Analysis

Output:

```text
results/ceiling_analysis/qwen3.5-9b_context_specific_wrong_L20.json
```

This experiment investigates why some cases are corrected by steering and others are not.

Groups:

```text
recovered   = wrong before steering, correct after steering
unrecovered = wrong before steering, still wrong after steering
```

Compared variables:

```text
conflict_score
prior_confidence
context_feature_activation
wrong_feature_activation
SAE_reconstruction_error
```

Interpretation:

If unrecovered cases have stronger prior confidence, higher conflict score, or higher reconstruction error, then they are not merely failed steering cases. They are candidates for abstention.

## Experiment 4: MedAbstain Transfer

Output:

```text
results/medabstain_transfer/qwen3.5-9b_medabstain_transfer_L20_fold0.json
```

This tests whether PubMedQA-discovered context-ignorance features transfer causally to MedAbstain.

Procedure:

1. Use PubMedQA validation-selected context-specific wrong features.
2. Suppress those features on MedAbstain prompts.
3. Check whether wrong-answer behavior changes into abstention.

Metric:

```text
wrong_to_abstain
```

If this is low, transfer is weak.

## Experiment 5: MedAbstain Failure Diagnostics

Output:

```text
results/medabstain_diagnostics/qwen3.5-9b_medabstain_feature_diagnostics_fold0.json
```

This diagnoses why MedAbstain transfer is weak.

Two possible explanations are tested.

### A. Mechanism Difference

MedAbstain features are discovered directly using behavior labels:

```text
correct_abstain vs wrong_answered
```

Then the discovered MedAbstain features are compared with PubMedQA context-specific features.

Interpretation:

```text
same layer + overlapping features -> shared mechanism, transfer failed for technical/formal reasons
different layer or no overlap      -> likely different mechanism
```

### B. Format / Token Effect

MedAbstain uses multiple-choice answers with an abstain option, usually `E`.

The diagnostic checks:

```text
abstain letter probability before steering
abstain letter probability after steering
delta in abstain probability
whether steering affects the output distribution at all
```

This separates mechanistic failure from output-format failure.

## Experiment 6: CAI Policy Evaluation

Output:

```text
results/cai_policy/qwen3.5-9b_cai_policy_L20.json
```

CAI policy:

```text
s < t_low       -> pass
t_low <= s <= t_high -> steer
s > t_high      -> abstain
```

where `s` is conflict score.

Thresholds `t_low` and `t_high` are selected on validation and evaluated on test.

Baselines:

```text
No intervention
Always steer
CAI
```

Metrics:

```text
selective_accuracy = accuracy among answered cases
coverage           = fraction of cases answered
safety             = fraction of risky/wrong cases abstained
```

The goal is to show that detection can be used for a safer pass/steer/abstain pipeline.

## Time Reduction Method

The original implementation was too slow because context-specific filtering repeatedly forwarded the same prompts for every candidate feature.

### Old Method

For each candidate feature:

```text
PubMedQA with context:    50 forward passes
PubMedQA without context: 50 forward passes
MedQA:                   50 forward passes
```

If there are 300 candidate features:

```text
300 * 150 = 45,000 forward passes per layer per fold
```

This was repeated across layers and folds, causing extremely long runtimes.

### New Method

For each layer and fold:

```text
PubMedQA with context:    50 forward passes
PubMedQA without context: 50 forward passes
MedQA:                   50 forward passes
```

Total:

```text
150 forward passes per layer per fold
```

Then all candidate feature means are computed from the SAE activation matrix.

This is not an approximation. It reuses the same activations that the old method recomputed repeatedly.

### Alpha Tuning Reduction

Alpha tuning was also limited to a validation subset:

```text
--max-alpha-tune-cases 300
```

This affects only validation-time alpha selection. The final test recovery/corruption evaluation still uses the full test split unless `--max-steering-cases` is explicitly set.

## Recommended Command

```bash
cd /workspace/medical_mi

nohup python3 scripts/run_master_experiments.py \
  --model qwen3.5-9b \
  --folds 10 \
  --max-alpha-tune-cases 300 \
  --run-ceiling \
  --run-conflict \
  --run-medabstain \
  --run-medabstain-diagnostics \
  --run-cai \
  > results/next_experiments_v2_qwen3.5-9b_fast.log 2>&1 &
```

Use `--backup-existing` only when previous result files need to be preserved before overwriting.

## Main Result Files to Inspect

```text
results/steering_context_specific/qwen3.5-9b_context_specific_cv.json
results/conflict_score/qwen3.5-9b_conflict_score_cv_L20.json
results/ceiling_analysis/qwen3.5-9b_context_specific_wrong_L20.json
results/medabstain_transfer/qwen3.5-9b_medabstain_transfer_L20_fold0.json
results/medabstain_diagnostics/qwen3.5-9b_medabstain_feature_diagnostics_fold0.json
results/cai_policy/qwen3.5-9b_cai_policy_L20.json
```
