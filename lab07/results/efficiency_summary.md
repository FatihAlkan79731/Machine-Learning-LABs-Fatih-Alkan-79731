| Pillar | Metric | Dense baseline | Unstructured global 90 % | Structured 50 % filters | Student, no teacher | Student, distilled (T=4, α=0.5) |
|---|---|---|---|---|---|---|
| Data | data_training_recipe | 10 epochs, hard labels | baseline + 3 fine-tune epochs | baseline + 3 fine-tune epochs | 10 epochs, hard labels | 10 epochs, hard + teacher soft labels |
| Data | data_samples_processed | 550,000 | 715,000 | 715,000 | 550,000 | 605,000 |
| Data | data_depends_on | - | needs the trained baseline | needs the trained baseline | - | needs the trained teacher + 1 teacher pass |
| Model | model_params | 997,434 | 997,434 | 476,474 | 103,750 | 103,750 |
| Model | model_nonzero_params | 997,434 | 100,084 | 476,474 | 103,750 | 103,750 |
| Model | model_size_dense_MB | 3.808 | 3.808 | 1.821 | 0.3991 | 0.3991 |
| Model | model_size_best_sparse_MB | 3.928 | 0.5047 | 1.878 | 0.4121 | 0.4121 |
| Model | test_acc | 0.9231 | 0.9284 | 0.927 | 0.9056 | 0.8981 |
| Compute | compute_flops_executed | 16,711,488 | 16,711,488 | 4,744,512 | 3,556,944 | 3,556,944 |
| Compute | compute_flops_nonzero | 16,711,488 | 4,334,647 | 4,744,512 | 3,556,944 | 3,556,944 |
| Compute | compute_latency_b1_ms | 0.4654 | 0.4695 | 0.2817 | 0.2385 | 0.2418 |
| Compute | compute_latency_b256_ms | 80.33 | 83.25 | 23.16 | 21.83 | 22.42 |
| Energy | energy_training_kwh_total | 0.01326 | 0.01721 | 0.01496 | 0.00497 | 0.01827 |
| Energy | energy_inference_kwh_per_1k_images | 1.28e-05 | 1.31e-05 | 3.7e-06 | 3.49e-06 | 3.56e-06 |
