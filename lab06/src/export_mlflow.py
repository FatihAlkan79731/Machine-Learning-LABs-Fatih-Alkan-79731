"""Export every MLflow run of the Lab 6 experiment to results/mlflow_runs.csv.

The MLflow file store (lab06/mlruns) is not committed; this table is the committed copy
of the runs. Browse the full store with:  mlflow ui --backend-store-uri lab06/mlruns
"""
import mlflow

from config import RESULTS, ROOT, load_config


def main():
    cfg = load_config()
    mlflow.set_tracking_uri((ROOT / cfg.mlflow.tracking_uri).resolve().as_uri())
    runs = mlflow.search_runs(experiment_names=[cfg.experiment], order_by=["attributes.start_time ASC"])
    keep = ["run_id", "tags.mlflow.runName", "tags.step", "status", "start_time", "end_time",
            "params.data.batch_size", "params.data.num_workers", "params.optimizer.name",
            "params.optimizer.lr", "params.scheduler.name", "params.training.epochs",
            "params.training.grad_accum_steps",
            "metrics.train_loss", "metrics.val_loss", "metrics.val_acc", "metrics.test_acc",
            "metrics.throughput_samples_s", "metrics.peak_memory_mb", "metrics.epoch_s",
            "metrics.data_only_s", "metrics.share_data_loading", "metrics.share_forward",
            "metrics.share_backward", "metrics.share_optimizer_step"]
    runs = runs[[c for c in keep if c in runs.columns]]
    runs.to_csv(RESULTS / "mlflow_runs.csv", index=False, float_format="%.5g")
    print(f"{len(runs)} MLflow runs exported to results/mlflow_runs.csv")
    print(runs[[c for c in ["tags.mlflow.runName", "status", "metrics.val_acc", "metrics.test_acc"]
                if c in runs.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
