import csv
import glob
import os

import numpy as np

from nasa_rsnn_pipeline import (
    RIDGE_ALPHA,
    artifact_path_for_battery,
    evaluate_holdout,
    save_artifact,
)


SUMMARY_CSV = "nasa_holdout_summary.csv"


def main(seed=0, ridge_alpha=RIDGE_ALPHA):
    files = sorted(glob.glob("B*.mat"))
    if not files:
        raise RuntimeError("No B*.mat files found in the current folder.")

    print("Found NASA batteries:", files)
    print("Running leave-one-battery-out training with the RSNN reservoir + lightweight health features.")

    rows = []
    for held_out_file in files:
        train_files = [path for path in files if path != held_out_file]
        artifact, metrics = evaluate_holdout(
            train_files=train_files,
            held_out_file=held_out_file,
            seed=seed,
            ridge_alpha=ridge_alpha,
        )

        artifact_path = artifact_path_for_battery(held_out_file)
        save_artifact(artifact, artifact_path)

        metrics["artifact_path"] = artifact_path
        rows.append(metrics)

        print(
            f"Held-out {metrics['battery_id']} | "
            f"RMSE {metrics['rmse_percent']:.2f}% | "
            f"MAE {metrics['mae_percent']:.2f}% | "
            f"saved {artifact_path}"
        )

    avg_rmse = float(np.mean([row["rmse_percent"] for row in rows]))
    avg_mae = float(np.mean([row["mae_percent"] for row in rows]))

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "battery_id",
                "file_name",
                "train_files",
                "train_cycles",
                "test_cycles",
                "rmse_percent",
                "mae_percent",
                "artifact_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            row_to_write = dict(row)
            row_to_write["train_files"] = ",".join(row_to_write["train_files"])
            writer.writerow(row_to_write)

    print("\n===== NASA HOLD-OUT SUMMARY =====")
    for row in rows:
        print(
            f"{row['battery_id']}: RMSE {row['rmse_percent']:.2f}% | "
            f"MAE {row['mae_percent']:.2f}%"
        )
    print(f"Average RMSE: {avg_rmse:.2f}%")
    print(f"Average MAE : {avg_mae:.2f}%")
    print(f"Saved summary: {os.path.abspath(SUMMARY_CSV)}")

    return rows


if __name__ == "__main__":
    main(seed=0, ridge_alpha=RIDGE_ALPHA)
