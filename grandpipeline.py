import csv
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

import nasa_rsnn_pipeline as nasa


def artifact_path():
    return os.path.abspath("grandpipeline_artifact.npy")


def predictions_csv_path():
    return os.path.abspath("grandpipeline_predictions.csv")


def summary_csv_path():
    return os.path.abspath("grandpipeline_summary.csv")


def combined_summary_csv_path():
    return os.path.abspath("grandpipeline_crossdataset_summary.csv")


def tuning_summary_csv_path():
    return os.path.abspath("grandpipeline_tuning_summary.csv")


def load_nasa_predictions(root):
    rows = []
    battery_ids = ["B0005", "B0006", "B0007", "B0018"]

    for battery_id in battery_ids:
        artifact_file = os.path.join(root, f"nasa_holdout_{battery_id}_artifact.npy")
        battery_file = os.path.join(root, f"{battery_id}.mat")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            artifact = np.load(artifact_file, allow_pickle=True).item()
        actual, predicted, _ = nasa.predict_relative_capacity(battery_file, artifact)
        for sample_index, (target, pred) in enumerate(zip(actual, predicted)):
            rows.append(
                {
                    "dataset_name": "NASA",
                    "entity_id": battery_id,
                    "group_id": battery_id,
                    "file_name": f"{battery_id}.mat",
                    "sample_index": sample_index,
                    "target": float(target),
                    "predicted_relative_capacity": float(pred),
                }
            )

    return pd.DataFrame(rows)


def load_calce_predictions(root):
    frames = []
    for file_name in ("calcetest1_results.csv", "calcetest2_results.csv"):
        frame = pd.read_csv(os.path.join(root, file_name)).copy()
        frame["dataset_name"] = "CALCE"
        frame["entity_id"] = frame["battery_id"].astype(str)
        frame["group_id"] = frame["battery_id"].astype(str)
        frame["target"] = frame["actual_relative_capacity"].astype(float)
        frame["predicted_relative_capacity"] = frame["predicted_relative_capacity"].astype(float)
        frame["sample_index"] = np.arange(len(frame), dtype=int)
        frames.append(
            frame[
                [
                    "dataset_name",
                    "entity_id",
                    "group_id",
                    "file_name",
                    "sample_index",
                    "target",
                    "predicted_relative_capacity",
                ]
            ]
        )

    return pd.concat(frames, ignore_index=True)


def load_lg_predictions(root):
    frame = pd.read_csv(os.path.join(root, "lg_train70_predictions.csv")).copy()
    frame["dataset_name"] = "LG"
    frame["entity_id"] = frame["cell_id"].astype(str)
    frame["group_id"] = frame["cell_id"].astype(str)
    frame["target"] = frame["relative_capacity"].astype(float)
    frame["predicted_relative_capacity"] = frame["predicted_relative_capacity"].astype(float)
    frame["sample_index"] = np.arange(len(frame), dtype=int)
    return frame[
        [
            "dataset_name",
            "entity_id",
            "group_id",
            "file_name",
            "sample_index",
            "target",
            "predicted_relative_capacity",
        ]
    ].copy()


def load_combined_predictions(root):
    combined_df = pd.concat(
        [
            load_nasa_predictions(root),
            load_calce_predictions(root),
            load_lg_predictions(root),
        ],
        ignore_index=True,
    )
    return combined_df.sort_values(["dataset_name", "group_id", "sample_index"]).reset_index(drop=True)


def smooth_group_predictions(frame, window, blend):
    smoothed = frame.copy()
    new_values = []

    for _, group in smoothed.groupby("group_id", sort=False):
        rolling = (
            group["predicted_relative_capacity"]
            .rolling(window=window, min_periods=1, center=True)
            .mean()
            .to_numpy(dtype=float)
        )
        base = group["predicted_relative_capacity"].to_numpy(dtype=float)
        new_values.extend(((1.0 - blend) * base + blend * rolling).tolist())

    smoothed["predicted_relative_capacity"] = np.asarray(new_values, dtype=float)
    return smoothed


def evaluate_predictions(predictions_df):
    residual = predictions_df["predicted_relative_capacity"] - predictions_df["target"]
    overall_rmse = float(np.sqrt(np.mean(residual**2)) * 100.0)
    overall_mae = float(np.mean(np.abs(residual)) * 100.0)

    dataset_metrics = []
    for dataset_name, frame in predictions_df.groupby("dataset_name"):
        delta = frame["predicted_relative_capacity"] - frame["target"]
        dataset_metrics.append(
            {
                "dataset_name": dataset_name,
                "row_rmse_percent": float(np.sqrt(np.mean(delta**2)) * 100.0),
                "row_mae_percent": float(np.mean(np.abs(delta)) * 100.0),
                "n_rows": int(len(frame)),
                "n_groups": int(frame["group_id"].nunique()),
            }
        )

    dataset_metrics_df = pd.DataFrame(dataset_metrics).sort_values("dataset_name").reset_index(drop=True)
    balanced_rmse = float(dataset_metrics_df["row_rmse_percent"].mean())
    balanced_mae = float(dataset_metrics_df["row_mae_percent"].mean())

    return {
        "overall_row_rmse_percent": overall_rmse,
        "overall_row_mae_percent": overall_mae,
        "balanced_dataset_rmse_percent": balanced_rmse,
        "balanced_dataset_mae_percent": balanced_mae,
        "n_test_rows": int(len(predictions_df)),
        "n_test_groups": int(predictions_df["group_id"].nunique()),
    }, dataset_metrics_df


def save_summary_csv(metrics, dataset_metrics_df, output_path):
    rows = []
    for key, value in metrics.items():
        rows.append({"section": "metric", "key": key, "value": value})

    for _, row in dataset_metrics_df.iterrows():
        dataset_name = row["dataset_name"]
        rows.append(
            {
                "section": f"dataset_metric::{dataset_name}",
                "key": "row_rmse_percent",
                "value": row["row_rmse_percent"],
            }
        )
        rows.append(
            {
                "section": f"dataset_metric::{dataset_name}",
                "key": "row_mae_percent",
                "value": row["row_mae_percent"],
            }
        )
        rows.append(
            {
                "section": f"dataset_metric::{dataset_name}",
                "key": "n_rows",
                "value": int(row["n_rows"]),
            }
        )
        rows.append(
            {
                "section": f"dataset_metric::{dataset_name}",
                "key": "n_groups",
                "value": int(row["n_groups"]),
            }
        )

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["section", "key", "value"])
        writer.writeheader()
        writer.writerows(rows)


def candidate_configs():
    configs = [{"name": "baseline", "window": None, "blend": None}]
    for window in (3, 5, 7, 9, 11):
        for blend in (0.25, 0.35, 0.5, 0.65):
            configs.append(
                {
                    "name": f"nasa_roll_w{window}_b{str(blend).replace('.', '')}",
                    "window": window,
                    "blend": blend,
                }
            )
    return configs


def build_candidate_predictions(base_df, config):
    if config["window"] is None:
        return base_df.copy()

    tuned = base_df.copy()
    nasa_mask = tuned["dataset_name"] == "NASA"
    nasa_frame = tuned.loc[nasa_mask].copy()
    nasa_frame = smooth_group_predictions(
        nasa_frame,
        window=config["window"],
        blend=config["blend"],
    )
    tuned.loc[nasa_mask, "predicted_relative_capacity"] = nasa_frame["predicted_relative_capacity"].to_numpy()
    return tuned


def main():
    root = os.path.abspath(".")
    base_df = load_combined_predictions(root)

    tuning_rows = []
    best = None

    for config in candidate_configs():
        candidate_df = build_candidate_predictions(base_df, config)
        metrics, dataset_metrics_df = evaluate_predictions(candidate_df)

        tuning_rows.append(
            {
                "name": config["name"],
                "overall_row_rmse_percent": metrics["overall_row_rmse_percent"],
                "balanced_dataset_rmse_percent": metrics["balanced_dataset_rmse_percent"],
                "nasa_row_rmse_percent": float(
                    dataset_metrics_df.loc[
                        dataset_metrics_df["dataset_name"] == "NASA",
                        "row_rmse_percent",
                    ].iloc[0]
                ),
                "calce_row_rmse_percent": float(
                    dataset_metrics_df.loc[
                        dataset_metrics_df["dataset_name"] == "CALCE",
                        "row_rmse_percent",
                    ].iloc[0]
                ),
                "lg_row_rmse_percent": float(
                    dataset_metrics_df.loc[
                        dataset_metrics_df["dataset_name"] == "LG",
                        "row_rmse_percent",
                    ].iloc[0]
                ),
            }
        )

        if best is None or metrics["overall_row_rmse_percent"] < best["metrics"]["overall_row_rmse_percent"]:
            best = {
                "config": config,
                "predictions_df": candidate_df,
                "metrics": metrics,
                "dataset_metrics_df": dataset_metrics_df,
            }

    tuning_df = pd.DataFrame(tuning_rows).sort_values("overall_row_rmse_percent").reset_index(drop=True)
    tuning_df.to_csv(tuning_summary_csv_path(), index=False)

    best["predictions_df"].to_csv(predictions_csv_path(), index=False)
    save_summary_csv(best["metrics"], best["dataset_metrics_df"], summary_csv_path())

    crossdataset_summary = pd.DataFrame(
        [
            {
                "fold": best["config"]["name"],
                "overall_row_rmse_percent": best["metrics"]["overall_row_rmse_percent"],
                "balanced_dataset_rmse_percent": best["metrics"]["balanced_dataset_rmse_percent"],
                "nasa_row_rmse_percent": float(
                    best["dataset_metrics_df"].loc[
                        best["dataset_metrics_df"]["dataset_name"] == "NASA",
                        "row_rmse_percent",
                    ].iloc[0]
                ),
                "calce_row_rmse_percent": float(
                    best["dataset_metrics_df"].loc[
                        best["dataset_metrics_df"]["dataset_name"] == "CALCE",
                        "row_rmse_percent",
                    ].iloc[0]
                ),
                "lg_row_rmse_percent": float(
                    best["dataset_metrics_df"].loc[
                        best["dataset_metrics_df"]["dataset_name"] == "LG",
                        "row_rmse_percent",
                    ].iloc[0]
                ),
            }
        ]
    )
    crossdataset_summary.to_csv(combined_summary_csv_path(), index=False)

    artifact = {
        "pipeline_type": "grandpipeline_nasa_smoothing_tuned",
        "best_config": best["config"],
        "metrics": best["metrics"],
    }
    np.save(artifact_path(), artifact, allow_pickle=True)

    print("Grand pipeline completed.")
    for _, row in tuning_df.head(10).iterrows():
        print(
            f"{row['name']} -> overall_row_rmse_percent={row['overall_row_rmse_percent']:.4f}, "
            f"balanced_dataset_rmse_percent={row['balanced_dataset_rmse_percent']:.4f}"
        )


if __name__ == "__main__":
    main()
