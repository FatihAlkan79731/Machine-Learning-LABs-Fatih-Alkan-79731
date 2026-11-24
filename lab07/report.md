# Lab 7 – Compression: Pruning and Distillation

**Student:** Fatih Alkan (79731)

## 1. Goal

Compress an oversized Fashion-MNIST CNN in two different ways and measure honestly what each method buys: **magnitude pruning** (unstructured – global and per layer – and structured filter pruning) and **knowledge distillation** into a 10× smaller student. The key question is not only how much accuracy survives, but whether a smaller *count* of weights also becomes a smaller *file* and a *faster* model on a real CPU.

Reading: Reddi, Ch. 9 *Efficient AI* (efficiency principles, the data / model / compute / energy pillars, system trade-offs) and Ch. 10 *Model Optimizations* (§10.3 optimization dimensions, §10.4 model compression: pruning and knowledge distillation).

## 2. Method

All steps run with `python lab07/src/run_all.py`. Fashion-MNIST is decoded once into normalised tensors (`Normalize(0.2860, 0.3530)`), with a 55k / 5k train / validation split (seed 42) and the official 10k test set. All training uses Adam and batch 128 on the CPU.

| Step | Script | Output |
|---|---|---|
| 1 | `src/models.py`, `src/baseline.py` | `results/baseline.csv`, `results/baseline_history.csv` |
| 2 | `src/prune_unstructured.py`, `src/prune_structured.py`, `src/plot_pruning.py` | `results/pruning_unstructured.csv`, `results/global_layer_sparsity.csv`, `results/pruning_structured.csv`, `results/sparsity_curves.png` |
| 3 | `src/latency_size.py` | `results/size_formats.csv`, `results/latency_sparse.csv`, `results/sparse_matmul_micro.csv`, `results/latency_size.png` |
| 4 | `src/distill.py` | `results/distillation*.csv`, `results/distillation.png` |
| 5 | `src/summary.py` | `results/efficiency_summary.csv/.md` |

### 2.1 Oversized baseline (step 1)

| Layer | Output | Parameters | FLOPs / image |
|---|---|---|---|
| Conv 1→32, 3×3, pad 1 + ReLU + MaxPool | 32 × 14 × 14 | 320 | 0.45 M |
| Conv 32→64, 3×3, pad 1 + ReLU + MaxPool | 64 × 7 × 7 | 18,496 | 7.23 M |
| Conv 64→128, 3×3, pad 1 + ReLU | 128 × 7 × 7 | 73,856 | 7.23 M |
| Linear 6272→144 + ReLU + Dropout 0.3 | 144 | 903,312 | 1.81 M |
| Linear 144→10 | 10 | 1,450 | 0.003 M |
| **Total** | | **997,434** | **16.7 M** |

About 1.0 M parameters, as the manual asks. Note the imbalance that matters later: **fc1 holds 90.6 % of the parameters but only 11 % of the FLOPs**, while the two 3×3 convolutions do 87 % of the work with 9 % of the weights. Training: 10 epochs, lr 1e-3. Recorded: test accuracy, parameters, size of the saved `state_dict`, FLOPs, latency at batch 1 (median of 200 calls after 20 warm-ups) and batch 256 (median of 30 after 5), and CodeCarbon energy.

### 2.2 Magnitude pruning (step 2)

**Unstructured.** For each target sparsity s ∈ {0, 0.25, 0.5, 0.75, 0.9, 0.95}, always starting from the trained baseline (one-shot pruning):

1. The weights of all Conv2d and Linear layers are ranked by |w| (biases are not pruned; they are 0.1 % of the parameters).
2. A binary mask zeroes the smallest weights:
   - **global** – one threshold, the k-th smallest |w| of the whole network with k = s · N. Layers whose weights are small in magnitude (in practice the huge fc1) lose more than s, and small, important layers lose less (`results/global_layer_sparsity.csv`).
   - **per layer** – each layer loses exactly its own fraction s.
3. Test accuracy is measured directly after pruning.
4. **Fine-tuning:** 3 epochs, Adam lr 3e-4, with the mask fixed. The mask is multiplied into the weights after every optimizer step, so a pruned weight can never come back (checked with an assertion after fine-tuning).

s = 0 is the control: the same 3 extra epochs without pruning, to separate the effect of pruning from the effect of training longer.

**Structured.** In each of conv1, conv2 and conv3 the 10 %, 25 % or 50 % filters with the smallest **L2 norm** (over in-channels × 3 × 3) are removed, together with everything that depends on them:
- the matching input channel of the next convolution;
- for conv3, the 49 inputs of fc1 that come from that channel (flatten order C × H × W).

The network is then **rebuilt as a smaller dense BigCNN** with the surviving weights copied in (`models.structured_prune`), and fine-tuned in the same way (3 epochs, lr 3e-4).

### 2.3 Latency and storage (step 3)

- **Storage formats** for every globally pruned model:
  - dense float32 `state_dict`;
  - dense + gzip;
  - CSR with int64 indices (PyTorch's default);
  - CSR with int32 indices;
  - non-zero values + a 1-bit mask (the cheapest way to say *where* the non-zeros are).
- **Latency** (batch 1 and 256) of:
  - the dense baseline;
  - the 90 % and 95 % globally pruned models stored dense (zeros are still multiplied);
  - the same models with fc1/fc2 executed as **sparse CSR matrix products** (`torch.sparse.mm`). PyTorch has no sparse convolution on the CPU, so the convolutions stay dense;
  - the structurally pruned models.
- **Micro-benchmark:** fc1 alone (144 × 6272), dense matmul vs CSR matmul at sparsity 0 … 0.99, batch 1 and 256, to find the break-even point.

### 2.4 Knowledge distillation (step 4)

- **Teacher:** the oversized baseline, frozen, in eval mode. Its logits for the 55k training images are computed once and stored next to the images in the dataset.
- **Student:** the same layout, much narrower, **103,750 parameters** (9.6× fewer) and 3.56 M FLOPs (4.7× fewer):

  | Layer | Parameters |
  |---|---|
  | Conv 1→16 + ReLU + MaxPool | 160 |
  | Conv 16→32 + ReLU + MaxPool | 4,640 |
  | Conv 32→48 + ReLU | 13,872 |
  | Linear 2352→36 + ReLU | 84,708 |
  | Linear 36→10 | 370 |

- **Loss** (from the manual), T = 4, α = 0.5:

  L = α · CE(student, labels) + (1 − α) · T² · KL( softmax(s/T) ‖ softmax(t/T) )

  The factor T² keeps the gradient of the soft term on the same scale as the CE term when T changes, because the gradients of the softened KL scale with 1/T².
- **KL direction.** The manual writes the KL with the student distribution first, KL(p_s ‖ p_t). This is the "reverse", mode-seeking direction: the student is punished for putting mass where the teacher has little. Hinton et al. (2015) use KL(p_t ‖ p_s), the forward, mass-covering direction, in which the teacher's full distribution (including its small probabilities for wrong classes, the "dark knowledge") is the target. This is also what `nn.KLDivLoss(log_softmax(s/T), softmax(t/T))` computes. The manual's form is the main experiment; the Hinton direction is run as an extra.
- **Runs:** 10 epochs, lr 1e-3, for *plain* (CE only, no teacher), *kd_manual* and *kd_hinton*. Every variant starts from the same initial student weights and sees the same batches. All three are repeated with seeds 42, 43 and 44, because the expected differences are a few tenths of a percent; mean ± standard deviation is reported. Also measured: size, FLOPs, latency, energy and the agreement of the student's test predictions with the teacher's.

### 2.5 Efficiency summary (step 5)

One table with the dense baseline, global 90 % pruning, structured 50 % pruning and the plain and distilled students, organised by the four pillars of Ch. 9. **Data:** training recipe, samples processed, dependence on another model. **Model:** parameters, non-zeros, dense and best sparse size, accuracy. **Compute:** FLOPs executed vs non-zero FLOPs, latency. **Energy:** CodeCarbon training and inference energy. Latency and inference energy are re-measured in one process so that all rows are comparable.

## 3. Results

> Machine and versions: `results/versions.txt`.

### 3.1 Baseline (step 1)

_pending local run_

### 3.2 Pruning (step 2)

_pending local run_

![Sparsity curves](results/sparsity_curves.png)

### 3.3 Latency and storage (step 3)

_pending local run_

![Latency and size](results/latency_size.png)

### 3.4 Distillation (step 4)

_pending local run_

![Distillation](results/distillation.png)

### 3.5 Efficiency summary (step 5)

_pending local run_

## 4. Conclusions

_pending local run_
