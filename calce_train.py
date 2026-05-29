import os

from calce_rsnn_pipeline import (
    INCLUDED_BATTERIES,
    evaluate_files,
    fit_artifact,
    holdout_artifact_path,
    holdout_summary_csv_path,
    load_calce_soh_dataset,
    save_artifact,
    save_grouped_predictions_csv,
    save_training_summary_csv,
    split_files_70_30,
    training_artifact_path,
    training_summary_csv_path,
)


def main(seed=7):
    dataset = load_calce_soh_dataset()
    metadata_df = dataset["metadata"]

    unique_files = metadata_df[["battery_id", "file_name", "temperature_c"]].drop_duplicates()
    print("Loaded CALCE SOH files:")
    print(unique_files.sort_values(["battery_id", "temperature_c", "file_name"]).to_string(index=False))
    print(f"\nTotal labeled segment samples: {len(metadata_df)}")
    print(f"Unique files used for SOH: {len(unique_files)}")

    train_files, test_files = split_files_70_30(metadata_df, seed=seed)
    print("\n===== 70/30 FILE SPLIT =====")
    print("Train files:", train_files)
    print("Test files :", test_files)

    artifact = fit_artifact(dataset, train_files, seed=seed)
    save_artifact(artifact, training_artifact_path())

    grouped_test_df, metrics = evaluate_files(dataset, artifact, test_files)
    save_grouped_predictions_csv(grouped_test_df, os.path.abspath("calce_train70_predictions.csv"))
    save_training_summary_csv(train_files, test_files, metrics, training_summary_csv_path())

    print("\n===== 70/30 TEST RESULTS =====")
    print(grouped_test_df.to_string(index=False))
    print(f"File-level RMSE: {metrics['file_rmse_percent']:.2f}%")
    print(f"File-level MAE : {metrics['file_mae_percent']:.2f}%")
    print(f"Saved training artifact: {training_artifact_path()}")
    print(f"Saved split summary    : {training_summary_csv_path()}")

    print("\n===== SAMPLE HOLD-OUT ARTIFACTS =====")
    for battery_id in INCLUDED_BATTERIES:
        battery_test_files = sorted(
            metadata_df.loc[metadata_df["battery_id"] == battery_id, "file_name"].unique().tolist()
        )
        battery_train_files = sorted(
            metadata_df.loc[metadata_df["battery_id"] != battery_id, "file_name"].unique().tolist()
        )

        holdout_artifact = fit_artifact(dataset, battery_train_files, seed=seed)
        artifact_path = holdout_artifact_path(battery_id)
        save_artifact(holdout_artifact, artifact_path)

        holdout_df, holdout_metrics = evaluate_files(dataset, holdout_artifact, battery_test_files)
        summary_path = holdout_summary_csv_path(battery_id)
        save_grouped_predictions_csv(holdout_df, summary_path)

        print(
            f"{battery_id} hold-out | RMSE {holdout_metrics['file_rmse_percent']:.2f}% | "
            f"MAE {holdout_metrics['file_mae_percent']:.2f}% | "
            f"saved {artifact_path}"
        )

    return grouped_test_df


if __name__ == "__main__":
    main(seed=7)
