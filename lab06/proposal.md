# Individual Project Proposal

**Student:** Fatih Alkan  **Student ID:** 79731
**Course:** Machine Learning Systems  **Date:** November 2026

## Title

**Real-time traffic-sign recognition for a low-cost driver-assistance camera**

## 1. Problem

A dash camera or small robot car has to recognise traffic signs on a cheap CPU, next to other work and without a network connection. A cloud model is not an option (latency, connectivity, privacy), so the classifier must run on the device. The task is to classify a cropped image of a traffic sign into one of 43 sign classes (speed limits, priority, prohibitory, danger and mandatory signs). The sign detector is out of scope; the project starts from the cropped region of interest, as provided by the dataset.

## 2. Dataset

**GTSRB – German Traffic Sign Recognition Benchmark** (`torchvision.datasets.GTSRB`, free for research use).

| Property | Value |
|---|---|
| Classes | 43 |
| Images | 26,640 train / 12,630 test (torchvision split) |
| Image size | 15 × 15 to 250 × 250 px, RGB; resized to 32 × 32 (or 48 × 48) |
| Difficulties | strong class imbalance (≈ 10× between the most and least frequent class), motion blur, low light, partial occlusion, physical tracks of the same sign |

Validation: 10 % of the training set, split **by physical track** (all frames of one real sign stay in one split) to avoid near-duplicate leakage between train and validation. The data pipeline from Lab 4 (validation report, hashes, data card) is reused.

## 3. Success metrics

| Metric | Target |
|---|---|
| Top-1 accuracy on the official test set | ≥ 97 % |
| Macro-F1 (rare classes count equally) | ≥ 0.95 |
| Accuracy drop of the optimized model vs. the uncompressed teacher | ≤ 1.0 percentage point |

## 4. Deployment constraints

| Constraint | Limit | How it is measured |
|---|---|---|
| Latency | **≤ 50 ms** per image (batch 1), p95 | ONNX Runtime on a laptop CPU limited to 1 thread (to imitate an edge CPU); 20 warm-up + 500 timed runs |
| Model size | **≤ 5 MB** on disk | size of the exported `.onnx` file |
| Memory | ≤ 100 MB peak RSS of the inference process | `psutil` |
| Offline | no network calls at inference | — |

## 5. Plan and planned optimization

1. **Baseline / teacher.** A ResNet-style CNN (≈ 2–3 M parameters, ≈ 10 MB in float32) trained with the Lab 6 pipeline (YAML config, MLflow, checkpoints, profiling). Expected to meet the accuracy target but **not** the size target.
2. **Main optimization – knowledge distillation** (Lab 7): a compact student CNN with ≈ 150–300 k parameters is trained with α · CE + (1 − α) · T² · KL against the teacher (T = 4, α = 0.5 as a starting point), and compared with the same student trained without a teacher.
3. **Second step – post-training int8 quantization** (Lab 8) of the student, with a calibration set of 500 training images, exported to ONNX. Structured pruning is the fallback if latency is still too high.
4. **Evaluation.** Accuracy, macro-F1, per-class recall (rare signs), latency p50/p95, size and energy (CodeCarbon) for teacher, plain student, distilled student and quantized student; the final model is served with FastAPI (Lab 9–10).

## 6. Feasibility and risks

- Compute: GTSRB at 32 × 32 trains in minutes per epoch on a laptop CPU; the whole study fits in the course time.
- *Risk:* the student misses 97 % → use 48 × 48 input or a slightly wider student (still far below 5 MB).
- *Risk:* rare classes suffer most from compression → watch macro-F1 and per-class recall, use class-balanced sampling.
- *Ethics:* the model is a course prototype and not a safety-certified component; failure cases are reported openly.
