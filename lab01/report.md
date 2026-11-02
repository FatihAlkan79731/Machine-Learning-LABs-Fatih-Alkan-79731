# Lab 1 – Environment and First System Measurements

## 1. Goal

Set up a reproducible Python environment and measure the system cost of two baseline classifiers on the Breast Cancer Wisconsin (Diagnostic) dataset: a linear model and a tree ensemble. The measurements (training time, inference latency, model size, peak memory) are then compared with the Cloud, Edge, Mobile and TinyML resource budgets to decide where each model could be deployed.

## 2. Method

**Environment.** Isolated virtual environment (`.venv`) with Python 3.11.8 and the pinned packages in `requirements.txt`. `src/print_versions.py` prints Python, the platform and every package version, and compares them with the pinned versions. Output: `results/versions.txt`.

**Data.** `load_breast_cancer(return_X_y=True)`: 569 samples, 30 numeric features, 2 classes. 70/30 stratified split with `random_state=42` (398 train / 171 test). All seeds are fixed to 42.

**Models.**
- `LogisticRegression(max_iter=1000, random_state=42)`
- `RandomForestClassifier(n_estimators=100, random_state=42)`

No preprocessing was applied, so both models see the raw features as the manual specifies.

**Measurements** (`src/measure_cost.py`). Each model is measured in a separate Python process so the memory of one model does not affect the other.
- *Training time:* 1 warm-up fit, then 5 timed fits with `time.perf_counter`; median reported.
- *Inference latency:* `predict` on one test sample, 1 warm-up, then 100 timed calls; median reported.
- *Model size:* fitted model saved with `joblib.dump` to `results/model_<name>.joblib`; size in bytes and KB.
- *Peak memory:* a background thread samples `psutil.Process().memory_info().rss` every 1 ms during the training runs and during the inference runs. Peak RSS is the memory of the whole Python process, including the interpreter, NumPy and scikit-learn. The increase over the RSS before training is also reported, as the part caused by the model itself.

**Budget check** (`src/budget_fit.py`). A model fits a target when peak inference RSS, median latency and model size are all within the target's limits. For memory, only the upper bound of each range is checked against the model, because the lower bound describes the device, not the model.

All steps can be reproduced with `python lab01/src/run_all.py`.

## 3. Results

Hardware: *see `results/versions.txt`*.

### Accuracy (`results/baseline_accuracy.csv`)

| Model | Test accuracy |
|---|---|
| LogisticRegression | 0.9415 |
| RandomForestClassifier | 0.9357 |

### System cost (`results/system_cost.csv`)

| Model | Train time (ms, median of 5) | Inference latency (ms, median of 100) | Model size (bytes / KB) | Peak RSS train (MB) | Peak RSS inference (MB) | Memory increase from training (MB) |
|---|---|---|---|---|---|---|
| LogisticRegression | 122.11 | 0.129 | 1,103 / 1.08 | 158.75 | 158.75 | 1.34 |
| RandomForestClassifier | 204.64 | 7.800 | 290,921 / 284.10 | 158.23 | 158.51 | 0.71 |

### Deployment fit (`results/budget_fit.csv`)

| Target | Budget (memory / latency / size) | LogisticRegression | RandomForestClassifier |
|---|---|---|---|
| Cloud | ≥ 1 GB / ≤ 100 ms / ≤ 500 MB | **fits** | **fits** |
| Edge | 256–1024 MB / ≤ 50 ms / ≤ 50 MB | **fits** | **fits** |
| Mobile | 64–256 MB / ≤ 20 ms / ≤ 10 MB | **fits** (RSS 159 MB) | **fits** (RSS 159 MB) |
| TinyML | ≤ 256 KB / ≤ 10 ms / ≤ 100 KB | no – memory | no – memory and size |

**Deployment argument.**

- **Cloud and Edge:** both models are far inside every limit. The largest value is the Random Forest latency (7.8 ms), which is 6× below the Edge limit of 50 ms.
- **Mobile:** both models fit, but the margin is the memory, not the model. The ~159 MB process is almost entirely the Python runtime with NumPy and scikit-learn; the models themselves add only about 1 MB. That leaves less than 100 MB of headroom on a 256 MB device. The Random Forest latency of 7.8 ms is within the 20 ms limit, but it is 60× slower than Logistic Regression.
- **TinyML:** neither model can run in its current form. A 256 KB device cannot hold a Python interpreter, so the memory check fails for both. The Random Forest also fails on size alone (284 KB > 100 KB). Logistic Regression passes on size (1.1 KB) and latency: it is just 30 weights and a bias. Exported to plain C (a dot product plus a sigmoid), it would easily fit a microcontroller. The Random Forest would need fewer or shallower trees to get under 100 KB.

## 4. Conclusions

1. On this small tabular dataset the simple model is as good as the ensemble: Logistic Regression reaches 0.9415 test accuracy against 0.9357 for the Random Forest. At the same time it is about 260× smaller and 60× faster at inference. The extra cost of the forest buys nothing here.
2. At this scale, the runtime environment dominates the memory cost, not the model. Both processes peak at about 159 MB, while the models add only around 1 MB. Any deployment decision for Mobile or TinyML therefore depends on the runtime (Python vs. a compiled or exported model) as much as on the model.
3. Measurement method matters. Single fits and single predictions vary a lot because of warm-up, caching and the OS scheduler, so the median of repeated runs gives stable numbers. Logistic Regression also hit the 1000-iteration limit with a convergence warning because the features are not scaled. Training time and accuracy therefore depend on preprocessing that this lab deliberately left out.
