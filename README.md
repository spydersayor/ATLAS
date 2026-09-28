# ATLAS: Adaptive Trust-based Linkage & Entity Resolution System

ATLAS is an entity-resolution pipeline built for the **Amazon ML Challenge 2026**. It links business records across three sources (S1, S2, S3), where each record has a name, an address and a country, and it outputs the S2/S3 records that refer to the same real-world business as each S1 record.

The competition metric is **macro-averaged F0.5**, so precision is weighted more heavily than recall. The whole pipeline is designed around that: retrieval is wide, the final decision is conservative.

> No external business lookups, APIs or data augmentation are used, in line with the competition rules.

---

## Pipeline

```
Raw TSVs
   -> Normalization        (noise-aware cleanup, raw values preserved)
   -> Fingerprinting       (name / address / country keys)
   -> Blocking             (multi-strategy candidate generation)
   -> Candidate fusion     (dedupe + blocker provenance)
   -> Feature engineering  (similarity / evidence features)
   -> ML matcher           (gradient-boosted classifier)
   -> F0.5 threshold optimization
   -> Entity-level decisions (conflict handling, one row per S1 entity)
   -> matching_results.tsv + candidate_pairs.tsv
```

Ground truth is used **only for training and evaluation**. It is never used during candidate generation, and candidates are always generated before any ML scoring.

---

## Data

| File | Columns |
|---|---|
| `train_source1/2/3.tsv`, `test_source1/2/3.tsv` | `entity_id`, `business_name`, `business_address`, `country` |
| `train_ground_truth.tsv` | `source1_entity_id`, `matched_entity_ids` |

- `matched_entity_ids` is a comma-separated list that mixes S2 and S3 ids. An empty value means the S1 entity is a true singleton with no match anywhere.
- Row counts, train: S1 2,206,821, S2 5,034,616, S3 5,285,603.
- Row counts, test: S1 1,732,544, S2 4,887,273, S3 5,082,316.
- Text is multilingual (English, Hindi, French, ...) and the `country` field mixes codes and full names.

Place data under `dataset/train/` and `dataset/test/`.

---

## Blocking (`src/blocking.py`)

Blocking produces a bounded set of candidate pairs so that nothing does an O(N²) comparison. It returns exactly these columns:

`source1_entity_id, candidate_entity_id, candidate_source, blocker_sources, num_blockers`

Blockers, each independently switchable through `BlockingConfig`:

| Blocker | Key |
|---|---|
| `name_exact`, `address_exact` | exact fingerprint |
| `name_country`, `address_country` | fingerprint + country |
| `name_token_country`, `address_token_country` | shared informative token + country |
| `name_prefix_country`, `address_prefix_country` | prefix of longest token + country |
| `name_token_pair_country`, `address_token_pair_country` | two longest non-stopword tokens + country |
| `name_token` | shared token in normalized name, no country key |

Design points:

- **Bounded candidate generation.** Token and prefix signatures are frequency-capped (`max_signature_frequency`) so very common tokens cannot cause a candidate explosion.
- **No large many-to-many merges** for signature blockers. Target-side lookups use compact dictionaries.
- **Memory-safe chunking.** Signature blockers process S1 in chunks (`source1_chunk_size`) and spill completed chunks to temporary files instead of holding every chunk in RAM.
- **Provenance-preserving fusion.** Blocker sources are stored as a bitmask and expanded into a comma-separated `blocker_sources` string with a matching `num_blockers`.
- **Invariant:** each blocker frame may contribute a given (S1, candidate) pair at most once. The fusion step relies on this, and the signature blocker deduplicates per S1 record to enforce it.

Some blockers are much looser than others. On a 2,000 / 10,000 / 10,000 sample the exact and token-pair blockers produced only hundreds of candidates, while the token and prefix blockers each produced tens of thousands. Enable or disable blockers based on your time budget and recall needs.

---

## Repository layout

```
src/
  normalization.py      # add_normalized_columns
  fingerprinting.py     # add_fingerprint_columns
  blocking.py           # candidate generation + fusion
  features.py           # evidence features
  matcher.py            # EntityMatcher (save / load)
  training.py           # MatcherTrainer
  inference.py          # InferencePipeline
  evaluator.py          # ground-truth parsing + metrics
scripts/
  run_submission.py     # end-to-end train -> infer -> submission files
  test_f05.py           # validation F0.5 / threshold report
matching.py             # inference-only run from a saved matcher
dataset/{train,test}/
output/
```

Adjust this to match your actual tree.

---

## Running

Run from the project root with the project on `PYTHONPATH`:

```bash
# Full pipeline: train, save matcher + threshold, run test inference, write submission files
PYTHONPATH=. python scripts/run_submission.py

# Validation report (best threshold, F0.5, precision, recall)
PYTHONPATH=. python scripts/test_f05.py

# Inference only, reusing a previously saved matcher
PYTHONPATH=. python matching.py
```

`run_submission.py` saves `output/matcher.joblib` and `output/threshold.txt`. `matching.py` reads both, so it only works after a training run has produced them.

### Outputs

- `output/matching_results.tsv` has columns `source1_entity_id, matched_entity_ids`, with exactly one row per test S1 entity.
- `output/candidate_pairs.tsv` holds the fused candidate pairs.

Submission constraints checked by the pipeline:

- Every test S1 entity appears exactly once in `matching_results.tsv`.
- Final matches are a subset of `candidate_pairs.tsv`.
- No duplicate candidate pairs.

### Speed vs. recall

`run_submission.py` has a fast-mode configuration (`FAST_BLOCKING_CONFIG` and sampled training data) that keeps only the four vectorized exact and composite blockers. It runs much faster but retrieves far fewer candidates, so recall suffers. Switch back to the default `BlockingConfig()` and full training data when you are not time-limited. Test data is never sampled.

---

## Results

Validation result from `scripts/test_f05.py`:

| Metric | Value |
|---|---|
| Best threshold | 0.7933 |
| F0.5 | 0.996432 |
| Precision | 0.997700 |
| Recall | 0.991392 |
| Predicted matches | 1,517,975 |

**Caveats.**

- This is a validation score on a split of the training data, not a hidden-test leaderboard score, so the two are not directly comparable.
- The threshold was tuned on the same validation set, which biases the score upward slightly.
- If recall is measured only over candidate pairs, matches that blocking never retrieved are not counted. Measure end-to-end recall against ground truth before trusting the number.
- This run was not submitted to the competition.

---

## Known limitations and next steps

- Measure blocking recall against ground truth using the **full** S2/S3 tables. Truncating them (e.g. `head(10000)`) makes almost every true target unreachable and gives meaningless recall.
- Loose blockers can produce very large candidate sets at full scale (tens of millions of pairs), which makes feature engineering and scoring the bottleneck. Tune `max_signature_frequency`, `name_token_min_length` and the enabled blockers against measured recall.
- Keep train-time and test-time blocking configurations consistent to avoid train/serve skew in features such as `num_blockers`.

---

## Notes

- Python 3.10+ (uses `int.bit_count`), pandas.
- Add a `requirements.txt` with pinned versions for reproducibility.
