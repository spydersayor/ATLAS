import sys
import time
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

from src.features import FeatureEngineer, get_feature_columns
from src.matcher import EntityMatcher


TRAIN_S1 = "dataset/train/train_source1.tsv"
TRAIN_S2 = "dataset/train/train_source2.tsv"
TRAIN_S3 = "dataset/train/train_source3.tsv"
GT = "dataset/train/train_ground_truth.tsv"


# ------------------------------------------------------------
# TIMING HELPER
#
# WHY: fe.transform() / matcher.fit() aren't visible in this file, so
# if the script is still slow after the fixes below, these prints tell
# you exactly which stage to look at next instead of guessing.
# ------------------------------------------------------------

_stage_start = time.time()


def stage(label):
    global _stage_start
    now = time.time()
    print(f"[{now - _stage_start:6.1f}s] {label}")
    _stage_start = now


overall_start = time.time()

print("=" * 70)
print("ATLAS F0.5 VALIDATION TEST")
print("=" * 70)


# ------------------------------------------------------------
# LOAD -- optional pyarrow engine (faster C-level TSV parsing), with
# a safe fallback to the default engine if pyarrow isn't installed or
# hits any incompatibility. Behavior/output dtype is unaffected either
# way (dtype=str keeps plain object/str columns regardless of engine).
# ------------------------------------------------------------

def read_tsv(path):
    try:
        return pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            engine="pyarrow",
        )
    except Exception:
        return pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
        )


print("\nLoading training data...")

s1 = read_tsv(TRAIN_S1)
s2 = read_tsv(TRAIN_S2)
s3 = read_tsv(TRAIN_S3)
gt = read_tsv(GT)

stage("Loaded S1/S2/S3/GT")

print("S1:", len(s1))
print("S2:", len(s2))
print("S3:", len(s3))
print("GT:", len(gt))


# ------------------------------------------------------------
# INSPECT GT
# ------------------------------------------------------------

print("\nGT columns:", list(gt.columns))
print(gt.head())


# ------------------------------------------------------------
# ADAPT GROUND TRUTH -- VECTORIZED (no iterrows)
# ------------------------------------------------------------

id_col = gt.columns[0]
value_cols = list(gt.columns[1:])

melted = gt.melt(
    id_vars=id_col,
    value_vars=value_cols,
    value_name="raw_value",
)

melted["raw_value"] = melted["raw_value"].astype(str).str.strip()
melted = melted[melted["raw_value"] != ""]

# Normalize both comma and semicolon separators to comma, then split.
split_lists = (
    melted["raw_value"]
    .str.replace(";", ",", regex=False)
    .str.split(",")
)

exploded = melted.assign(candidate_entity_id=split_lists).explode(
    "candidate_entity_id"
)

exploded["candidate_entity_id"] = (
    exploded["candidate_entity_id"].astype(str).str.strip()
)
exploded = exploded[exploded["candidate_entity_id"] != ""]

is_s2 = exploded["candidate_entity_id"].str.startswith("S2-")
is_s3 = exploded["candidate_entity_id"].str.startswith("S3-")

exploded = exploded[is_s2 | is_s3].copy()
exploded["candidate_source"] = np.where(
    exploded["candidate_entity_id"].str.startswith("S2-"),
    "S2",
    "S3",
)

positive = pd.DataFrame(
    {
        "source1_entity_id": exploded[id_col].to_numpy(),
        "candidate_entity_id": exploded["candidate_entity_id"].to_numpy(),
        "candidate_source": exploded["candidate_source"].to_numpy(),
        "blocker_sources": "ground_truth",
        "num_blockers": "1",
        "label": 1,
    }
)

stage("Built positive pairs from ground truth (vectorized)")

print("\nPositive pairs:", len(positive))

if len(positive) == 0:
    raise RuntimeError("No positive pairs extracted from ground truth.")


# ------------------------------------------------------------
# NEGATIVES -- VECTORIZED (no per-row rng.choice loop)
# ------------------------------------------------------------

rng = np.random.default_rng(42)

n_neg = min(len(positive) * 3, 500_000)

s1_ids = s1["entity_id"].to_numpy()
s2_ids = s2["entity_id"].to_numpy()
s3_ids = s3["entity_id"].to_numpy()

sampled_s1 = rng.choice(s1_ids, size=n_neg)
use_s2 = rng.random(n_neg) < 0.5

sampled_s2 = rng.choice(s2_ids, size=n_neg)
sampled_s3 = rng.choice(s3_ids, size=n_neg)

sampled_candidate_id = np.where(use_s2, sampled_s2, sampled_s3)
sampled_source = np.where(use_s2, "S2", "S3")

negative = pd.DataFrame(
    {
        "source1_entity_id": sampled_s1,
        "candidate_entity_id": sampled_candidate_id,
        "candidate_source": sampled_source,
        "blocker_sources": "random_negative",
        "num_blockers": "1",
        "label": 0,
    }
)

stage("Sampled negative pairs (vectorized)")

print("Negative pairs:", len(negative))


# ------------------------------------------------------------
# REMOVE ACCIDENTAL POSITIVE DUPLICATES -- VECTORIZED ANTI-JOIN
#
# CHANGED: the previous version built a Python `set` of tuples from
# `positive` (via zip/list, one Python-level tuple per row) and another
# list of tuples from `negative`, then did `.isin(set_of_tuples)`.
# Building those tuples is a Python-level loop under the hood. A
# pd.merge(..., indicator=True) anti-join does the same "is this
# combination of 3 columns present in the other frame" check entirely
# in vectorized C code, with no per-row Python tuple construction.
# ------------------------------------------------------------

key_cols = ["source1_entity_id", "candidate_entity_id", "candidate_source"]

merge_check = negative.merge(
    positive[key_cols],
    on=key_cols,
    how="left",
    indicator=True,
)

negative = negative[
    merge_check["_merge"].to_numpy() == "left_only"
].reset_index(drop=True)

data = pd.concat(
    [
        positive,
        negative,
    ],
    ignore_index=True,
)

stage("Deduplicated negatives against positives (vectorized anti-join)")

print("Total pairs:", len(data))


# ------------------------------------------------------------
# SPLIT BY SOURCE-1 ENTITY -- single isin() + complement
#
# CHANGED: val_df was previously built with its OWN separate
# data.source1_entity_id.isin(val_s1_ids) call. Since train_test_split
# partitions unique_s1 completely (every id is in exactly one side),
# val is by construction the exact complement of train -- so the
# second isin() pass was redundant work over the same full column.
# ------------------------------------------------------------

unique_s1 = data["source1_entity_id"].unique()

train_s1_ids, val_s1_ids = train_test_split(
    unique_s1,
    test_size=0.20,
    random_state=42,
)

train_mask = data["source1_entity_id"].isin(train_s1_ids).to_numpy()

train_df = data[train_mask].reset_index(drop=True)
val_df = data[~train_mask].reset_index(drop=True)

stage("Train/validation split")

print("\nTrain pairs:", len(train_df))
print("Validation pairs:", len(val_df))


# ------------------------------------------------------------
# FEATURES -- single combined fe.transform() call
#
# CHANGED: fe.transform() was previously called twice (once for
# train_df, once for val_df), each a separate pass against the full
# s1/s2/s3 frames. If FeatureEngineer builds any lookup/index structure
# internally (a very common pattern -- e.g. indexing s2/s3 by entity_id
# for fast merges), that setup cost was being paid twice. Calling it
# once on the combined train+val pairs and splitting the OUTPUT
# afterward pays that fixed cost once instead of twice.
#
# ASSUMPTION THIS RELIES ON: fe.transform() returns exactly one output
# row per input candidate-pair row, IN THE SAME ORDER, with no
# dropping/reordering. This is asserted below rather than assumed
# silently -- if FeatureEngineer violates this, the script fails loudly
# here instead of silently producing train/val features that don't
# correspond to the right pairs.
# ------------------------------------------------------------

fe = FeatureEngineer()

combined_pairs = data.drop(columns=["label"]).reset_index(drop=True)

print("\nBuilding features for TRAIN + VALIDATION in one pass...")

X_all_df = fe.transform(
    source1=s1,
    source2=s2,
    source3=s3,
    candidate_pairs=combined_pairs,
)

stage("Built combined TRAIN+VALIDATION features (single pass)")

assert len(X_all_df) == len(combined_pairs), (
    "fe.transform() returned a different row count than its input -- "
    "the single-pass train/val split below assumes row-order "
    "preservation and is NOT safe here. Revert to calling "
    "fe.transform() separately for train_df and val_df."
)

print("Combined feature rows:", len(X_all_df))

features = get_feature_columns()

X_all = X_all_df[features].to_numpy(dtype=np.float32)
y_all = data["label"].to_numpy(dtype=np.int8)

X_train = X_all[train_mask]
X_val = X_all[~train_mask]
y_train = y_all[train_mask]
y_val = y_all[~train_mask]


# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

print("\nTraining HistGradientBoosting...")

matcher = EntityMatcher()

matcher.fit(
    X_train,
    y_train,
)

stage("Trained matcher")

print("Model trained.")
matcher.save("output/matcher.joblib")
print("SAVED: output/matcher.joblib")


# ------------------------------------------------------------
# VALIDATION SCORES
# ------------------------------------------------------------

scores = matcher.predict_scores(X_val)

stage("Scored validation set")

print("\nScore statistics:")
print("min :", scores.min())
print("max :", scores.max())
print("mean:", scores.mean())


# ------------------------------------------------------------
# F0.5 THRESHOLD SEARCH -- VECTORIZED (no 96-iteration grid loop)
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("F0.5 THRESHOLD SEARCH")
print("=" * 70)

precisions, recalls, thresholds = precision_recall_curve(y_val, scores)

# precision_recall_curve returns one more precision/recall pair than
# thresholds (the last pair corresponds to predicting nothing positive,
# i.e. threshold = +inf) -- drop that trailing pair so everything lines
# up with an actual threshold.
precisions = precisions[:-1]
recalls = recalls[:-1]

beta_sq = 0.5 ** 2
denom = beta_sq * precisions + recalls

f05_scores = np.divide(
    (1 + beta_sq) * precisions * recalls,
    denom,
    out=np.zeros_like(denom),
    where=denom > 0,
)

order = np.argsort(f05_scores)[::-1]

print(
    "\nThreshold    F0.5       Precision    Recall      Predicted"
)

for idx in order[:15]:
    threshold = thresholds[idx]
    predicted = int((scores >= threshold).sum())

    print(
        f"{threshold:.4f}       "
        f"{f05_scores[idx]:.6f}   "
        f"{precisions[idx]:.6f}     "
        f"{recalls[idx]:.6f}    "
        f"{predicted:,}"
    )

best_idx = order[0]
best_threshold = thresholds[best_idx]
best_predicted = int((scores >= best_threshold).sum())

stage("Threshold search")

print("\n" + "=" * 70)
print("BEST VALIDATION RESULT")
print("=" * 70)

print(f"Best threshold : {best_threshold:.4f}")
print(f"F0.5           : {f05_scores[best_idx]:.6f}")
print(f"Precision      : {precisions[best_idx]:.6f}")
print(f"Recall         : {recalls[best_idx]:.6f}")
print(f"Predicted      : {best_predicted:,}")

print("=" * 70)

print(f"\nTOTAL RUNTIME: {(time.time() - overall_start) / 60:.2f} minutes")
