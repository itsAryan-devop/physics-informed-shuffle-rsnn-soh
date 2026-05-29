"""Regenerate downstream CSVs from the winning artifacts.

Run this after the final-tune sweep + artifact copy. It rewrites
    calcetest1_results.csv, calcetest2_results.csv
    calce_holdout_SP20_1_summary.csv, calce_holdout_SP20_3_summary.csv
    nasa_holdout_summary.csv

Then runs grandpipeline to refresh the aggregate metrics.
"""

import csv
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

warnings.simplefilter("ignore", InconsistentVersionWarning)

import nasa_rsnn_pipeline as nasa  # noqa: E402
import calce_rsnn_pipeline as calce  # noqa: E402


NASA_BATTERIES = ["B0005", "B0006", "B0007", "B0018"]


def _save_split_summary_csv(train_files, test_files, metrics, output_path):
    """Same format as calce_rsnn_pipeline.save_training_summary_csv (split|file|value)."""
    rows = [{"split": "train", "file_name": name, "value": ""} for name in train_files]
    rows += [{"split": "test", "file_name": name, "value": ""} for name in test_files]
    rows += [{"split": "metric", "file_name": k, "value": v} for k, v in metrics.items()]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "file_name", "value"])
        writer.writeheader()
        writer.writerows(rows)


def regenerate_calce_outputs():
    print("Loading CALCE dataset...", flush=True)
    t0 = time.time()
    dataset = calce.load_calce_soh_dataset()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
    metadata_df = dataset["metadata"]

    for battery_id, results_path in [
        ("SP20-1", "calcetest1_results.csv"),
        ("SP20-3", "calcetest2_results.csv"),
    ]:
        artifact_path = calce.holdout_artifact_path(battery_id)
        artifact = np.load(artifact_path, allow_pickle=True).item()

        test_files = sorted(
            metadata_df.loc[metadata_df["battery_id"] == battery_id, "file_name"].unique().tolist()
        )
        train_files = sorted(
            metadata_df.loc[metadata_df["battery_id"] != battery_id, "file_name"].unique().tolist()
        )

        grouped_df, metrics = calce.evaluate_files(dataset, artifact, test_files)
        grouped_df = grouped_df.sort_values(["temperature_c", "file_name"]).reset_index(drop=True)
        grouped_df.to_csv(results_path, index=False)
        summary_path = calce.holdout_summary_csv_path(battery_id)
        grouped_df.to_csv(summary_path, index=False)
        print(f"  [CALCE {battery_id}] file_rmse={metrics['file_rmse_percent']:.3f}% -> "
              f"{results_path}, {summary_path}", flush=True)


def regenerate_nasa_summary():
    rows = []
    for battery_id in NASA_BATTERIES:
        artifact_path = os.path.abspath(f"nasa_holdout_{battery_id}_artifact.npy")
        artifact = np.load(artifact_path, allow_pickle=True).item()
        battery_file = os.path.abspath(f"{battery_id}.mat")
        actual, predicted, dataset = nasa.predict_relative_capacity(battery_file, artifact)
        rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
        mae = float(np.mean(np.abs(predicted - actual)))
        train_batteries = [b for b in NASA_BATTERIES if b != battery_id]
        train_cycles = 0
        for tb in train_batteries:
            ds_tb = nasa.extract_battery_samples(os.path.abspath(f"{tb}.mat"))
            train_cycles += int(len(ds_tb["targets"]))
        rows.append(
            {
                "battery_id": battery_id,
                "file_name": f"{battery_id}.mat",
                "train_files": ",".join(f"{b}.mat" for b in train_batteries),
                "train_cycles": train_cycles,
                "test_cycles": int(len(actual)),
                "rmse_percent": rmse * 100.0,
                "mae_percent": mae * 100.0,
                "artifact_path": artifact_path,
            }
        )
        print(f"  [NASA {battery_id}] rmse={rmse*100:.3f}%, mae={mae*100:.3f}%", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("nasa_holdout_summary.csv", index=False)
    print(f"  wrote nasa_holdout_summary.csv ({len(df)} rows)", flush=True)


def main():
    print("=== Regenerating CALCE results from winning artifacts ===", flush=True)
    regenerate_calce_outputs()
    print("\n=== Regenerating NASA holdout summary from winning artifacts ===", flush=True)
    regenerate_nasa_summary()
    print("\n=== Running grandpipeline ===", flush=True)
    import grandpipeline
    grandpipeline.main()
    print("\n=== Final summary ===", flush=True)
    df = pd.read_csv("grandpipeline_summary.csv")
    for _, row in df.iterrows():
        print(f"  {row['section']:<32} {row['key']:<34} {row['value']}")


if __name__ == "__main__":
    main()
