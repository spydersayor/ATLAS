import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import pandas as pd

from src.matcher import EntityMatcher
from src.inference import InferencePipeline


TEST_S1 = "dataset/test/test_source1.tsv"
TEST_S2 = "dataset/test/test_source2.tsv"
TEST_S3 = "dataset/test/test_source3.tsv"
CANDIDATES = "output/candidate_pairs.tsv"
MATCHER_PATH = "output/matcher.joblib"
THRESHOLD_PATH = "output/threshold.txt"
OUTPUT_DIR = "output"
FALLBACK_THRESHOLD = 0.7933  # only used if THRESHOLD_PATH doesn't exist


def load_threshold() -> float:
    """
    Read the threshold the training run actually optimized and saved,
    rather than relying on a manually copy-pasted number here that can
    silently drift out of sync with whichever matcher.joblib gets
    loaded (e.g. after retraining with different data/config).
    """
    path = Path(THRESHOLD_PATH)
    if path.exists():
        value = float(path.read_text().strip())
        print(f"Using threshold from {THRESHOLD_PATH}: {value:.6f}")
        return value

    print(
        f"WARNING: {THRESHOLD_PATH} not found -- falling back to "
        f"hardcoded THRESHOLD={FALLBACK_THRESHOLD}. This may not match "
        f"the matcher currently at {MATCHER_PATH} if it was retrained "
        "since that number was recorded."
    )
    return FALLBACK_THRESHOLD


def require_file(path: str, hint: str = "") -> None:
    if not Path(path).exists():
        message = f"Required file not found: {path}"
        if hint:
            message += f"\n{hint}"
        raise FileNotFoundError(message)


def main() -> None:
    start = time.perf_counter()

    # ------------------------------------------------------------------
    # Fail fast, with an actionable message, before spending time
    # loading multi-million-row files that would otherwise be wasted
    # work if the matcher isn't there.
    # ------------------------------------------------------------------
    require_file(
        MATCHER_PATH,
        hint=(
            "This means no training run has saved a matcher here yet "
            "(e.g. via EntityMatcher.save() at the end of a training "
            "script). Run training first, or point MATCHER_PATH at "
            "wherever that run actually saved it."
        ),
    )
    require_file(CANDIDATES, hint="Run candidate generation first.")
    require_file(TEST_S1)
    require_file(TEST_S2)
    require_file(TEST_S3)

    print(f"Loading matcher: {MATCHER_PATH}")
    matcher = EntityMatcher.load(MATCHER_PATH)
    print(f"  done [{time.perf_counter() - start:.1f}s]")

    print(f"\nLoading test sources...")
    test_s1 = pd.read_csv(TEST_S1, sep="\t", dtype=str)
    test_s2 = pd.read_csv(TEST_S2, sep="\t", dtype=str)
    test_s3 = pd.read_csv(TEST_S3, sep="\t", dtype=str)
    print(f"  S1: {len(test_s1):,}")
    print(f"  S2: {len(test_s2):,}")
    print(f"  S3: {len(test_s3):,}")
    print(f"  done [{time.perf_counter() - start:.1f}s]")

    # ------------------------------------------------------------------
    # BUG: the original script read every column, including
    # num_blockers, as dtype=str. The candidate-pair contract (see
    # blocking.py's REQUIRED_CANDIDATE_COLUMNS / validate_candidate_pairs)
    # defines num_blockers as int64. If InferencePipeline or its scoring
    # logic does any numeric comparison/threshold on num_blockers
    # (e.g. "num_blockers >= 2"), a string dtype compares lexically, not
    # numerically -- "10" < "2" as strings, which silently produces
    # wrong results rather than an error. Explicit per-column dtypes
    # fix this and are also meaningfully faster/lighter than an
    # all-string parse at this many rows.
    # ------------------------------------------------------------------
    print(f"\nLoading candidate pairs: {CANDIDATES}")
    print(
        "  NOTE: if this file has ~75M rows (the full/loose blocking "
        "config on the full test set), this step alone can take "
        "several minutes and a large amount of RAM. If that's not "
        "intentional, regenerate candidate_pairs.tsv with a tighter "
        "BlockingConfig first."
    )

    candidates = pd.read_csv(
        CANDIDATES,
        sep="\t",
        dtype={
            "source1_entity_id": str,
            "candidate_entity_id": str,
            "candidate_source": str,
            "blocker_sources": str,
            "num_blockers": "int64",
        },
    )
    print(f"  Candidates: {len(candidates):,}")
    print(f"  done [{time.perf_counter() - start:.1f}s]")

    # ------------------------------------------------------------------
    # Minimal sanity check before handing this to InferencePipeline --
    # catches a truncated/corrupt candidate file with a clear message
    # instead of a confusing failure deep inside inference.py.
    # ------------------------------------------------------------------
    required_columns = {
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
        "blocker_sources",
        "num_blockers",
    }
    missing_columns = required_columns - set(candidates.columns)
    if missing_columns:
        raise RuntimeError(
            f"candidate_pairs.tsv is missing columns: {sorted(missing_columns)}"
        )
    if candidates.empty:
        raise RuntimeError("candidate_pairs.tsv has zero rows.")

    threshold = load_threshold()
    print(f"\nRunning inference (threshold={threshold:.6f})...")

    pipeline = InferencePipeline(
        matcher=matcher,
        threshold=threshold,
        conflict_gap=0.05,
    )

    pipeline.run(
        test_source1=test_s1,
        test_source2=test_s2,
        test_source3=test_s3,
        candidate_pairs=candidates,
        output_dir=OUTPUT_DIR,
    )

    total = time.perf_counter() - start
    print(f"\nDONE [{total:.1f}s total, {total / 60:.1f} min]")
    print(f"matching_results.tsv generated in {OUTPUT_DIR}/")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFAILED: {exc}")
        raise