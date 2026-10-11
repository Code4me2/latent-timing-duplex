# PRELIMINARY build question card

## Objective

Does a causal-cutoff frozen representation improve held-out prediction of discrete cause-specific hazards—user resumption or recorded-partner entry, with event-free continuation in each fully observed interval—over elapsed+acoustic and hidden-state baselines within the locked 8 s follow-up?

## Evidence level

**PRELIMINARY / synthetic build validation only.** The component fits on locked
train lineages and scores development lineages with fixed 1 s discrete-time
bins and paired lineage bootstrap intervals. It is not a pilot result.

## Synthetic build observations

In the deliberately separable 90-item synthetic check (60 train, 30 development), the hidden representation reduced mean per-item discrete-hazard NLL from **1.1238** to **0.0281** versus the uninformative elapsed+acoustic baseline: improvement **1.0957**, paired lineage-bootstrap 95% interval **[1.0957, 1.0957]** with 10,000 draws. The degenerate interval is expected because every constructed development lineage has the same pattern; it is a deterministic software check, not empirical uncertainty. The runnable test also confirms that user resumption wins a nearest-source-sample-frame tie and that a partially observed terminal censor bin contributes no false event-free label.

## Assumptions

- The authenticated feature and future-outcome contracts preserve causal
  cutoff, item identity, and lineage-disjoint train/development splits.
- Censoring is administrative; only fully observed no-event intervals enter the
  likelihood, while an observed event enters its containing interval.
- Recorded-partner entry is an observational cause, not normative assistant
  timing.
- Fixed 1 s bins are a coarse screen through 8 s, not a tuned time scale.

## Failure modes

Sparse cause counts, informative session-end censoring, lineage/corpus shift, coarse-bin misspecification, high-dimensional probe instability, per-prefix state reset/history truncation, or selection of only ratable pilot items can make held-out likelihood noisy or misleading. The balanced deterministic pilot cannot identify population event incidence or population calibration. The frozen extractor is external and has not yet produced authenticated real features.

## Non-conclusions

Synthetic separation does not show that real hidden states encode floor state,
that recorded partners acted appropriately, or that a generated assistant
would enter acceptably. This screen does not compute acceptable-onset hit rate,
premature interruption, late/missed response, or any efficacy claim.

## Discriminating checks when real pilot data arrive

1. Confirm both event causes and at least two informative development lineages;
   report censoring and cause counts before fitting.
2. Require lower development item NLL with a paired lineage-bootstrap interval
   against elapsed+acoustic and hidden baselines; inspect results by corpus and
   predeclared checkpoint-elapsed strata if counts permit.
3. Check observed-versus-predicted event mass by bin as a sample-only diagnostic,
   explicitly not population calibration.
4. Repeat with a finer predeclared binning only if 1 s bins visibly hide timing
   structure and event counts support the added parameters.

## Explicit deferral

**Confirmatory test access remains blocked.** The shared-HMAC prototype checks integrity but not role separation or chronology, and this repository does not yet implement the independently produced sealed test dataset/feature package. External sealed custody and a coordinator-controlled commitment are still required.

**R3 replay and Mimi actuation remain deferred.** No silence-code override,
text padding, oracle/predicted native gate, LoRA, or generation intervention is
implemented before real R1/R2 evidence passes its gates.
