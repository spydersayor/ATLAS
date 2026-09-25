import numpy as np

from src.matcher import EntityMatcher


def main():
    print("=" * 60)
    print("ATLAS MATCHER TEST")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Create synthetic feature data.
    #
    # First two features roughly represent similarity scores.
    # Third feature represents an exact-match indicator.
    # Fourth feature represents a combined similarity score.

    n_samples = 200

    X = rng.random((n_samples, 4))

    # Create realistic synthetic labels from the feature values.
    #
    # Strong similarity + exact agreement -> likely match.
    signal = (
        0.35 * X[:, 0]
        + 0.25 * X[:, 1]
        + 0.25 * X[:, 2]
        + 0.15 * X[:, 3]
    )

    y = (signal > 0.60).astype(np.int8)

    # Make sure both classes exist.
    assert np.sum(y == 1) > 0
    assert np.sum(y == 0) > 0

    matcher = EntityMatcher()

    matcher.fit(X, y)

    scores = matcher.predict_scores(X)

    print("\nScore statistics:")
    print(f"Minimum : {scores.min():.4f}")
    print(f"Maximum : {scores.max():.4f}")
    print(f"Mean    : {scores.mean():.4f}")

    predictions = matcher.predict(
        X,
        threshold=0.5,
    )

    print("\nFirst 20 predictions:")

    for index in range(20):
        print(
            f"Pair {index + 1:02d}: "
            f"score={scores[index]:.4f}, "
            f"prediction={predictions[index]}"
        )

    # -----------------------------
    # Assertions
    # -----------------------------

    assert len(scores) == n_samples

    assert len(predictions) == n_samples

    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)

    assert np.all(np.isin(predictions, [0, 1]))

    # Important:
    # The model should produce non-constant probabilities.
    assert np.std(scores) > 0.01

    print("\n[PASS] Matcher training")
    print("[PASS] Probability generation")
    print("[PASS] Binary prediction")
    print("[PASS] Probability range")
    print("[PASS] Non-constant model scores")

    print("=" * 60)
    print("ALL MATCHER TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()