"""Step 2: train the two baselines and save test accuracy to results/baseline_accuracy.csv."""
import pandas as pd
from sklearn.metrics import accuracy_score

from common import MODEL_NAMES, RESULTS, load_split, make_model, set_seeds


def main():
    set_seeds()
    X_train, X_test, y_train, y_test = load_split()
    print(f"train: {X_train.shape}, test: {X_test.shape}")

    rows = []
    for key, name in MODEL_NAMES.items():
        model = make_model(key).fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        rows.append({"model": name, "test_accuracy": round(acc, 4)})
        print(f"{name:<24} accuracy = {acc:.4f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "baseline_accuracy.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
