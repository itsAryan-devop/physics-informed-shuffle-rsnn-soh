import os

from lg_rsnn_pipeline import (
    evaluate_cells,
    fit_artifact,
    load_lg_dataset,
    predictions_csv_path,
    save_artifact,
    save_predictions_csv,
    save_training_summary_csv,
    split_cells_70_30,
    training_artifact_path,
    training_summary_csv_path,
)


def main(seed=7):
    dataset = load_lg_dataset()
    metadata_df = dataset["metadata"]
    unique_cells = metadata_df[["dataset_name", "cell_id", "sequence_code"]].drop_duplicates()

    print("Loaded LG diagnostic full-cycle samples:")
    print(unique_cells.sort_values(["dataset_name", "cell_id"]).to_string(index=False))
    print(f"\nTotal labeled rows: {len(metadata_df)}")
    print(f"Unique cells used: {len(unique_cells)}")

    train_cells, test_cells = split_cells_70_30(metadata_df, seed=seed)
    print("\n===== 70/30 CELL SPLIT =====")
    print("Train cells:", train_cells)
    print("Test cells :", test_cells)

    artifact = fit_artifact(dataset, train_cells, seed=seed)
    save_artifact(artifact, training_artifact_path())

    predictions_df, metrics, dataset_metrics_df = evaluate_cells(dataset, artifact, test_cells)
    save_predictions_csv(predictions_df, predictions_csv_path())
    save_training_summary_csv(
        train_cells=train_cells,
        test_cells=test_cells,
        metrics=metrics,
        dataset_metrics_df=dataset_metrics_df,
        output_path=training_summary_csv_path(),
    )

    print("\n===== TEST RESULTS =====")
    print(predictions_df.head(30).to_string(index=False))
    print(f"\nRow-level RMSE: {metrics['row_rmse_percent']:.2f}%")
    print(f"Row-level MAE : {metrics['row_mae_percent']:.2f}%")
    print(f"Test cells    : {metrics['n_cells']}")
    print(f"Test rows     : {metrics['n_rows']}")

    print("\n===== DATASET BREAKDOWN =====")
    print(dataset_metrics_df.to_string(index=False))

    print(f"\nSaved training artifact: {training_artifact_path()}")
    print(f"Saved predictions CSV : {predictions_csv_path()}")
    print(f"Saved summary CSV     : {training_summary_csv_path()}")


if __name__ == "__main__":
    main(seed=7)
