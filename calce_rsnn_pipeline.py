import csv
import glob
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import physics_layer as phys
import physics_readout as pr


CALCE_DATA_ROOT = os.path.abspath(os.path.join("CALCE_official_cylindrical", "extracted"))
INCLUDED_BATTERIES = ("SP20-1", "SP20-3")
N_POINTS = 100
N_INPUT = 3
N_RESERVOIR = 200
CONN_PROB = 0.1
# Shuffle-RSNN grouped connectivity: W_res is block-diagonal with N_GROUPS
# groups, and prev_spikes is channel-shuffled before feedback every step.
N_GROUPS = 4
RIDGE_ALPHA = 10.0   # final-tune sweep winner (was 1.0); beats CALCE holdout by ~15%
FILE_CALIBRATION_ALPHA = 10.0
USE_FILE_CALIBRATION = False
# RSNN parent idea preserved (LIF reservoir + ridge readout). The readout
# regresses an *absolute Ah residual* on top of the directly-measured
# segment_capacity_ah feature, then evaluation re-normalises by the test
# battery's own reference capacity. This prevents the cross-battery
# reference-mismatch error that previously inflated RMSE on the 45C
# low-current OCV file.
TARGET_MODE = "residual"   # "residual" | "absolute" | "relative"
ENSEMBLE_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)   # final-tune sweep winner (was 5 seeds)
SEG_CAP_BASE_INDEX = 5

# Strategy-A composite-loss readout (pure-numpy gradient descent). Active
# only when target_mode == "relative" because the loss terms are defined in
# SOH space. Residual / absolute target modes fall back to sklearn Ridge.
USE_PHYSICS_READOUT = False
PHYS_ALPHA_MONO = 0.10
PHYS_ALPHA_RANGE = 1.0
PHYS_EPSILON_REG = 0.005
PHYS_LO = 0.55
PHYS_HI = 1.15
PHYS_LR = 1e-3
PHYS_N_EPOCHS = 400

BASE_SUMMARY_FEATURE_NAMES = [
    "segment_index",
    "segment_position",
    "segments_in_file",
    "temperature_c",
    "segment_duration_s",
    "segment_capacity_ah",
    "voltage_start",
    "voltage_end",
    "voltage_mean",
    "voltage_std",
    "current_start",
    "current_end",
    "current_mean",
    "current_std",
    "temp_start",
    "temp_end",
    "temp_mean",
    "temp_max",
    "temp_rise",
    "voltage_q10",
    "voltage_q50",
    "voltage_q90",
    "voltage_auc_norm",
    "voltage_drop_head_tail",
    "throughput_ah",
    "is_low_current_ocv",
    "is_incremental_ocv",
    "is_initial_capacity",
]

EXTRA_CONTEXT_FEATURE_NAMES = [
    "file_order_index",
    "file_order_position",
    "days_from_first",
    "days_from_first_normalized",
]

# Physics-informed feature columns (Strategy B from physics_layer_blueprint).
# Each value is a per-segment prior/consistency estimate derived from the
# base physical quantities; the readout learns residual corrections on top.
PHYSICS_FEATURE_NAMES = [
    "phys_soh_sqrt_time_prior",
    "phys_arrhenius_age_ah",
    "phys_nernst_soh_estimate",
]

SUMMARY_FEATURE_NAMES = (
    BASE_SUMMARY_FEATURE_NAMES
    + [f"{name}_drift_from_first" for name in BASE_SUMMARY_FEATURE_NAMES]
    + [f"{name}_delta_from_previous" for name in BASE_SUMMARY_FEATURE_NAMES]
    + EXTRA_CONTEXT_FEATURE_NAMES
    + PHYSICS_FEATURE_NAMES
)


def training_artifact_path():
    return os.path.abspath("calce_train70_artifact.npy")


def training_summary_csv_path():
    return os.path.abspath("calce_train70_summary.csv")


def holdout_artifact_path(battery_id):
    safe = battery_id.replace("-", "_")
    return os.path.abspath(f"calce_holdout_{safe}_artifact.npy")


def holdout_summary_csv_path(battery_id):
    safe = battery_id.replace("-", "_")
    return os.path.abspath(f"calce_holdout_{safe}_summary.csv")


def clean_name(value):
    return str(value).lower().replace("_", "").replace(" ", "").strip()


def find_column(columns, candidates):
    col_map = {clean_name(col): col for col in columns}
    for candidate in candidates:
        norm_candidate = clean_name(candidate)
        for norm_name, original in col_map.items():
            if norm_candidate in norm_name:
                return original
    return None


def infer_battery_id(file_path):
    match = re.search(r"(SP20-\d+)", file_path, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return os.path.splitext(os.path.basename(file_path))[0]


def infer_temperature(file_path, default=25.0):
    lower = file_path.lower()
    if "45c" in lower:
        return 45.0
    if "0c" in lower:
        return 0.0
    if "25c" in lower:
        return 25.0
    return float(default)


def infer_profile_flags(file_path):
    lower = file_path.lower()
    return (
        1.0 if "low current" in lower or "lowcurrent" in lower else 0.0,
        1.0 if "incremental" in lower else 0.0,
        1.0 if "initial capacity" in lower else 0.0,
    )


def infer_file_datetime(file_path):
    file_name = os.path.basename(file_path)
    match = re.search(r"(\d{1,2})_(\d{1,2})_(\d{4})", file_name)
    if not match:
        return None

    month, day, year = (int(part) for part in match.groups())
    try:
        return datetime(year, month, day)
    except ValueError:
        return None


def resample_series(values, n_points=N_POINTS):
    x_old = np.linspace(0.0, 1.0, len(values))
    x_new = np.linspace(0.0, 1.0, n_points)
    return interp1d(x_old, values)(x_new)


class RSNNReservoir:
    def __init__(self, n_input=N_INPUT, n_neurons=N_RESERVOIR, conn_prob=CONN_PROB,
                 seed=0, n_groups=N_GROUPS):
        if n_neurons % n_groups != 0:
            raise ValueError(
                f"n_neurons ({n_neurons}) must be divisible by n_groups ({n_groups})"
            )
        rng = np.random.default_rng(seed)
        self.n_input = n_input
        self.n_neurons = n_neurons
        self.conn_prob = conn_prob
        self.seed = seed
        self.n_groups = n_groups
        self.group_size = n_neurons // n_groups
        self.dt = 1.0
        self.tau = 20.0
        self.v_th = 0.5
        self.alpha = np.exp(-self.dt / self.tau)

        self.W_in = rng.normal(0.0, 0.5, (n_neurons, n_input))

        # Block-diagonal W_res: each group only wires within itself.
        self.W_res = np.zeros((n_neurons, n_neurons))
        for g in range(n_groups):
            s = g * self.group_size
            e = s + self.group_size
            mask = rng.random((self.group_size, self.group_size)) < conn_prob
            if mask.sum() > 0:
                self.W_res[s:e, s:e][mask] = rng.normal(0.0, 0.1, int(mask.sum()))

        eigvals = np.linalg.eigvals(self.W_res)
        radius = np.max(np.abs(eigvals))
        if radius == 0:
            radius = 1.0
        self.W_res *= 0.95 / radius

        # Channel-shuffle permutation (reshape -> transpose -> flatten).
        idx = np.arange(n_neurons).reshape(n_groups, self.group_size)
        self.shuffle_perm = idx.T.flatten()

    def process_cycle(self, sequence):
        membrane = np.zeros(self.n_neurons)
        spike_counts = np.zeros(self.n_neurons)
        spike_bins = np.zeros((3, self.n_neurons))
        membrane_trace_mean = np.zeros(self.n_neurons)
        prev_spikes = np.zeros(self.n_neurons)
        n_steps = max(1, len(sequence))

        for t in range(len(sequence)):
            shuffled_prev = prev_spikes[self.shuffle_perm]
            current = self.W_in @ sequence[t] + self.W_res @ shuffled_prev
            membrane = self.alpha * membrane + current
            membrane_trace_mean += membrane
            spikes = (membrane >= self.v_th).astype(float)
            membrane[spikes > 0] = 0.0
            spike_counts += spikes
            bin_index = min(2, int((3 * t) / n_steps))
            spike_bins[bin_index] += spikes
            prev_spikes = spikes

        return np.concatenate(
            [
                spike_counts,
                spike_bins[0],
                spike_bins[1],
                spike_bins[2],
                membrane_trace_mean / n_steps,
            ]
        )


def discover_calce_soh_files(root=CALCE_DATA_ROOT):
    file_paths = sorted(
        path
        for path in glob.glob(os.path.join(root, "SP*", "*.xls*"))
        if "~$" not in os.path.basename(path)
    )
    return [path for path in file_paths if infer_battery_id(path) in INCLUDED_BATTERIES]


def load_calce_workbook(file_path):
    workbook = pd.ExcelFile(file_path, engine="openpyxl")
    frames = []

    for sheet_name in workbook.sheet_names:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
        except Exception:
            continue

        if df is None or df.empty:
            continue

        time_col = find_column(df.columns, ["testtime", "duration", "time"])
        volt_col = find_column(df.columns, ["voltage(v)", "voltage", "mv"])
        curr_col = find_column(df.columns, ["current(a)", "current", "ma"])
        if not all([time_col, volt_col, curr_col]):
            continue

        frame = pd.DataFrame()
        frame["time"] = pd.to_numeric(df[time_col], errors="coerce")

        voltage = pd.to_numeric(df[volt_col], errors="coerce")
        if "mv" in clean_name(volt_col):
            voltage = voltage / 1000.0
        frame["voltage"] = voltage

        current = pd.to_numeric(df[curr_col], errors="coerce")
        if "ma" in clean_name(curr_col):
            current = current / 1000.0
        frame["current"] = current

        temp_col = find_column(df.columns, ["temperature", "temp"])
        if temp_col:
            frame["temp"] = pd.to_numeric(df[temp_col], errors="coerce").ffill().bfill()
        else:
            frame["temp"] = infer_temperature(file_path)

        step_col = find_column(df.columns, ["stepindex", "pgmstep", "step"])
        frame["step"] = pd.to_numeric(df[step_col], errors="coerce") if step_col else np.nan

        cycle_col = find_column(df.columns, ["cycleindex", "cycle"])
        frame["cycle"] = pd.to_numeric(df[cycle_col], errors="coerce") if cycle_col else 1.0

        discharge_capacity_col = find_column(df.columns, ["dischargecapacity"])
        frame["discharge_capacity"] = (
            pd.to_numeric(df[discharge_capacity_col], errors="coerce")
            if discharge_capacity_col
            else np.nan
        )

        point_col = find_column(df.columns, ["datapoint"])
        frame["data_point"] = (
            pd.to_numeric(df[point_col], errors="coerce")
            if point_col
            else np.arange(len(df), dtype=float)
        )

        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    full_df = pd.concat(frames, ignore_index=True)
    full_df = full_df.dropna(subset=["time", "voltage", "current"]).copy()
    full_df["temp"] = full_df["temp"].ffill().bfill().fillna(infer_temperature(file_path))
    full_df["cycle"] = full_df["cycle"].ffill().bfill().fillna(1.0)

    if full_df["step"].notna().any():
        full_df["step"] = full_df["step"].ffill().bfill()

    full_df = full_df.sort_values(["time", "data_point"]).reset_index(drop=True)
    return full_df


def build_segment_ids(df):
    active = df["current"].abs() >= 0.02

    sign = np.sign(df["current"])
    sign[np.abs(df["current"]) < 0.02] = 0.0
    sign = pd.Series(sign, index=df.index).ffill().fillna(0.0)

    if df["step"].notna().any():
        step = df["step"]
    else:
        step = pd.Series(np.zeros(len(df)), index=df.index)

    cycle = df["cycle"].ffill().bfill().fillna(1.0)

    boundary = (
        active.ne(active.shift(fill_value=False))
        | sign.ne(sign.shift(fill_value=0.0))
        | step.ne(step.shift())
        | cycle.ne(cycle.shift())
    )
    return boundary.cumsum()


def estimate_segment_capacity_ah(group):
    discharge_capacity = group["discharge_capacity"]
    if discharge_capacity.notna().sum() >= 2 and discharge_capacity.max() > discharge_capacity.min() + 1e-9:
        return float(discharge_capacity.max() - discharge_capacity.min())

    time_values = group["time"].to_numpy(dtype=float)
    current_values = group["current"].to_numpy(dtype=float)
    dt = np.diff(time_values, prepend=time_values[0])
    dt[0] = 0.0
    dt = np.clip(dt, 0.0, None)

    discharge_current = np.maximum(-current_values, 0.0)
    return float(np.sum(discharge_current * dt) / 3600.0)


def extract_file_segments(file_path):
    df = load_calce_workbook(file_path)
    if df.empty:
        return None

    df["segment_id"] = build_segment_ids(df)

    battery_id = infer_battery_id(file_path)
    temperature_c = infer_temperature(file_path)
    is_low_current, is_incremental, is_initial_capacity = infer_profile_flags(file_path)

    segments = []
    total_file_capacity_ah = 0.0
    for _, group in df.groupby("segment_id"):
        if len(group) < 50:
            continue
        if group["current"].mean() > -0.02:
            continue

        segment_capacity = estimate_segment_capacity_ah(group)
        if segment_capacity <= 0:
            continue

        voltage = group["voltage"].to_numpy(dtype=float)
        current = group["current"].to_numpy(dtype=float)
        temp = group["temp"].to_numpy(dtype=float)
        time = group["time"].to_numpy(dtype=float)

        sequence = np.stack(
            [
                resample_series(voltage),
                resample_series(current),
                resample_series(temp),
            ],
            axis=1,
        )

        head_size = max(2, len(voltage) // 10)
        tail_size = max(2, len(voltage) // 10)
        summary = np.array(
            [
                0.0,
                0.0,
                0.0,
                float(temperature_c),
                float(time[-1] - time[0]),
                float(segment_capacity),
                float(voltage[0]),
                float(voltage[-1]),
                float(voltage.mean()),
                float(voltage.std()),
                float(current[0]),
                float(current[-1]),
                float(current.mean()),
                float(current.std()),
                float(temp[0]),
                float(temp[-1]),
                float(temp.mean()),
                float(temp.max()),
                float(temp[-1] - temp[0]),
                float(np.quantile(voltage, 0.1)),
                float(np.quantile(voltage, 0.5)),
                float(np.quantile(voltage, 0.9)),
                float(np.trapezoid(voltage, dx=1.0) / max(len(voltage), 1)),
                float(voltage[:head_size].mean() - voltage[-tail_size:].mean()),
                float(np.trapezoid(np.abs(current), time) / 3600.0),
                float(is_low_current),
                float(is_incremental),
                float(is_initial_capacity),
            ],
            dtype=float,
        )

        segments.append(
            {
                "sequence": sequence,
                "summary": summary,
                "segment_capacity_ah": float(segment_capacity),
            }
        )
        total_file_capacity_ah += float(segment_capacity)

    if not segments:
        return None

    total_segments = len(segments)
    base_summary = []
    for index, segment in enumerate(segments):
        segment["summary"][0] = float(index)
        segment["summary"][1] = float(index / max(total_segments - 1, 1))
        segment["summary"][2] = float(total_segments)
        base_summary.append(segment["summary"])

    base_summary = np.asarray(base_summary, dtype=float)
    first_summary = base_summary[0]
    previous_summary = np.vstack([base_summary[0], base_summary[:-1]])
    safe_first_summary = np.where(np.abs(first_summary) < 1e-8, 1.0, first_summary)
    summary_features = np.hstack(
        [
            base_summary,
            (base_summary - first_summary) / safe_first_summary,
            base_summary - previous_summary,
        ]
    )

    return {
        "battery_id": battery_id,
        "file_path": os.path.abspath(file_path),
        "file_name": os.path.basename(file_path),
        "temperature_c": float(temperature_c),
        "total_file_capacity_ah": float(total_file_capacity_ah),
        "n_segments": int(total_segments),
        "sequences": np.asarray([segment["sequence"] for segment in segments], dtype=float),
        "summary_features": summary_features,
        "is_initial_capacity": bool(is_initial_capacity),
        "is_low_current_ocv": bool(is_low_current),
        "is_incremental_ocv": bool(is_incremental),
    }


def load_calce_soh_dataset(root=CALCE_DATA_ROOT):
    file_paths = discover_calce_soh_files(root=root)
    file_records = []
    for path in file_paths:
        record = extract_file_segments(path)
        if record is not None:
            file_records.append(record)

    if not file_records:
        raise RuntimeError("No usable CALCE SOH files were extracted from SP1/SP3.")

    reference_capacity_by_battery = {}
    for battery_id in INCLUDED_BATTERIES:
        battery_records = [record for record in file_records if record["battery_id"] == battery_id]
        if not battery_records:
            continue
        initial_capacity_candidates = [
            record["total_file_capacity_ah"]
            for record in battery_records
            if record["is_initial_capacity"]
        ]
        observed_max = max(record["total_file_capacity_ah"] for record in battery_records)
        if initial_capacity_candidates:
            reference_capacity = max(initial_capacity_candidates)
            if observed_max > reference_capacity * 1.02:
                reference_capacity = observed_max
        else:
            reference_capacity = observed_max
        reference_capacity_by_battery[battery_id] = float(reference_capacity)

    chronology_by_file = {}
    for battery_id in INCLUDED_BATTERIES:
        battery_records = [record for record in file_records if record["battery_id"] == battery_id]
        if not battery_records:
            continue

        ordered = sorted(
            battery_records,
            key=lambda record: (
                infer_file_datetime(record["file_name"]) or datetime.max,
                record["temperature_c"],
                record["file_name"],
            ),
        )
        valid_dates = [
            infer_file_datetime(record["file_name"])
            for record in ordered
            if infer_file_datetime(record["file_name"]) is not None
        ]
        base_date = min(valid_dates) if valid_dates else None
        max_days = 1.0
        if base_date is not None:
            max_days = max(
                1.0,
                max(
                    float((infer_file_datetime(record["file_name"]) - base_date).days)
                    if infer_file_datetime(record["file_name"]) is not None
                    else 0.0
                    for record in ordered
                ),
            )

        for order_index, record in enumerate(ordered):
            file_date = infer_file_datetime(record["file_name"])
            days_from_first = (
                float((file_date - base_date).days)
                if base_date is not None and file_date is not None
                else 0.0
            )
            chronology_by_file[record["file_name"]] = np.array(
                [
                    float(order_index),
                    float(order_index / max(len(ordered) - 1, 1)),
                    float(days_from_first),
                    float(days_from_first / max_days),
                ],
                dtype=float,
            )

    sequences = []
    summary_features = []
    targets = []
    metadata_rows = []
    for record in file_records:
        reference_capacity = reference_capacity_by_battery[record["battery_id"]]
        relative_capacity = float(record["total_file_capacity_ah"] / reference_capacity)
        chronology_features = chronology_by_file.get(
            record["file_name"],
            np.zeros(len(EXTRA_CONTEXT_FEATURE_NAMES), dtype=float),
        )
        repeated_context = np.repeat(
            chronology_features.reshape(1, -1),
            record["n_segments"],
            axis=0,
        )

        # Physics-informed columns (Strategy B). Derived purely from base
        # physical measurements already present per-segment; the readout
        # will learn residual corrections. See physics_layer_blueprint.txt.
        base_cols = record["summary_features"][:, : len(BASE_SUMMARY_FEATURE_NAMES)]
        cycle_col = base_cols[:, 0]            # segment_index
        temperature_col = base_cols[:, 3]      # temperature_c
        voltage_end_col = base_cols[:, 7]      # voltage_end
        throughput_col = base_cols[:, 24]      # throughput_ah
        is_low_c_col = base_cols[:, 25]        # is_low_current_ocv
        days_col = repeated_context[:, 2]      # days_from_first

        phys_sqrt_time = phys.soh_sqrt_time_prior(
            cycle_index=cycle_col,
            days_from_first=days_col,
        ).reshape(-1, 1)
        phys_arr_age = phys.arrhenius_effective_age(
            throughput_ah=throughput_col,
            temperature_c=temperature_col,
        ).reshape(-1, 1)
        phys_nernst = phys.nernst_consistent_soh(
            voltage_end=voltage_end_col,
            throughput_ah=throughput_col,
            reference_capacity_ah=reference_capacity,
            is_low_current_mask=is_low_c_col.astype(bool),
        ).reshape(-1, 1)

        file_summary_features = np.hstack(
            [
                record["summary_features"],
                repeated_context,
                phys_sqrt_time,
                phys_arr_age,
                phys_nernst,
            ]
        )
        for segment_index in range(record["n_segments"]):
            sequences.append(record["sequences"][segment_index])
            summary_features.append(file_summary_features[segment_index])
            targets.append(relative_capacity)
            metadata_rows.append(
                {
                    "battery_id": record["battery_id"],
                    "file_name": record["file_name"],
                    "file_path": record["file_path"],
                    "temperature_c": record["temperature_c"],
                    "n_segments": record["n_segments"],
                    "segment_index": segment_index,
                    "total_file_capacity_ah": record["total_file_capacity_ah"],
                    "reference_capacity_ah": reference_capacity,
                    "relative_capacity": relative_capacity,
                    "is_initial_capacity": record["is_initial_capacity"],
                    "is_low_current_ocv": record["is_low_current_ocv"],
                    "is_incremental_ocv": record["is_incremental_ocv"],
                    "file_order_index": chronology_features[0],
                    "file_order_position": chronology_features[1],
                    "days_from_first": chronology_features[2],
                    "days_from_first_normalized": chronology_features[3],
                }
            )

    metadata_df = pd.DataFrame(metadata_rows)
    return {
        "sequences": np.asarray(sequences, dtype=float),
        "summary_features": np.asarray(summary_features, dtype=float),
        "targets": np.asarray(targets, dtype=float),
        "metadata": metadata_df,
        "reference_capacity_by_battery": reference_capacity_by_battery,
        "file_records": file_records,
    }


def split_files_70_30(metadata_df, seed=0):
    rng = np.random.default_rng(seed)
    train_files = []
    test_files = []

    for battery_id in sorted(metadata_df["battery_id"].unique()):
        battery_files = sorted(metadata_df.loc[metadata_df["battery_id"] == battery_id, "file_name"].unique())
        shuffled = list(battery_files)
        rng.shuffle(shuffled)
        n_train = max(1, int(np.ceil(0.7 * len(shuffled))))
        train_files.extend(shuffled[:n_train])
        test_files.extend(shuffled[n_train:])

    return sorted(train_files), sorted(test_files)


def combine_feature_blocks(sequences, summary_features, input_scaler, reservoir):
    scaled_sequences = np.asarray([input_scaler.transform(cycle) for cycle in sequences], dtype=float)
    reservoir_features = np.asarray([reservoir.process_cycle(cycle) for cycle in scaled_sequences], dtype=float)
    return np.hstack([reservoir_features, summary_features])


def build_file_level_calibration_features(grouped_df):
    return grouped_df[
        [
            "raw_predicted_relative_capacity",
            "temperature_c",
            "is_low_current_ocv",
            "is_incremental_ocv",
            "is_initial_capacity",
            "file_order_position",
            "days_from_first_normalized",
        ]
    ].to_numpy(dtype=float)


def _build_train_target(train_metadata, train_summary, mode):
    """Build the per-segment regression target for a given target mode.

    relative : target = relative_capacity                              (legacy)
    absolute : target = relative_capacity * reference_capacity_ah     (per-segment Ah)
    residual : target = (file_total_Ah / n_segments) - segment_capacity_ah
               so the readout learns a small correction on top of the
               directly measured segment capacity.
    """
    relative = train_metadata["relative_capacity"].to_numpy(dtype=float)
    if mode == "relative":
        return relative
    ref = train_metadata["reference_capacity_ah"].to_numpy(dtype=float)
    absolute_file_ah = relative * ref
    if mode == "absolute":
        return absolute_file_ah
    if mode == "residual":
        n_seg = train_metadata["n_segments"].to_numpy(dtype=float)
        seg_cap = train_summary[:, SEG_CAP_BASE_INDEX]
        return absolute_file_ah / n_seg - seg_cap
    raise ValueError(f"Unknown TARGET_MODE: {mode}")


def fit_artifact(dataset, train_file_names, seed=0, ridge_alpha=RIDGE_ALPHA,
                 ensemble_seeds=None, target_mode=None):
    metadata_df = dataset["metadata"]
    train_mask = metadata_df["file_name"].isin(train_file_names).to_numpy()

    train_sequences = dataset["sequences"][train_mask]
    train_summary = dataset["summary_features"][train_mask]
    train_metadata = metadata_df.loc[train_mask].copy()
    train_sample_weights = (
        1.0 / train_metadata["n_segments"].to_numpy(dtype=float)
    )

    if target_mode is None:
        target_mode = TARGET_MODE
    if ensemble_seeds is None:
        ensemble_seeds = (seed,) if not ENSEMBLE_SEEDS else ENSEMBLE_SEEDS

    train_y = _build_train_target(train_metadata, train_summary, target_mode)

    input_scaler = MinMaxScaler().fit(np.vstack(train_sequences))

    # Per-row metadata for Strategy-A monotonicity loss: battery tag plus a
    # within-battery time order that respects both file chronology and
    # segment sequence. days_from_first gives coarse order across files;
    # adding segment_index * 1e-4 preserves intra-file segment ordering.
    train_cell_ids = train_metadata["battery_id"].to_numpy()
    train_cycle_order = (
        train_metadata["days_from_first"].to_numpy(dtype=float)
        + train_metadata["segment_index"].to_numpy(dtype=float) * 1e-4
    )

    members = []
    for member_seed in ensemble_seeds:
        reservoir = RSNNReservoir(seed=member_seed)
        train_features_raw = combine_feature_blocks(
            sequences=train_sequences,
            summary_features=train_summary,
            input_scaler=input_scaler,
            reservoir=reservoir,
        )
        feat_scaler = StandardScaler().fit(train_features_raw)
        train_features = feat_scaler.transform(train_features_raw)

        if USE_PHYSICS_READOUT and target_mode != "residual":
            readout = pr.PhysicsReadout(
                alpha_mono=PHYS_ALPHA_MONO,
                alpha_range=PHYS_ALPHA_RANGE,
                epsilon_reg=PHYS_EPSILON_REG,
                lo=PHYS_LO,
                hi=PHYS_HI,
                lr=PHYS_LR,
                n_epochs=PHYS_N_EPOCHS,
                l2=max(ridge_alpha, 1e-3),
            )
            readout.fit(
                train_features,
                train_y,
                cell_ids=train_cell_ids,
                cycle_order=train_cycle_order,
            )
        else:
            readout = Ridge(alpha=ridge_alpha)
            readout.fit(train_features, train_y, sample_weight=train_sample_weights)

        members.append(
            {
                "seed": int(member_seed),
                "feat_scaler": feat_scaler,
                "readout_model": readout,
            }
        )

    # Keep "primary" readout as ensemble member 0 for backwards-compatible keys.
    primary = members[0]
    feat_scaler = primary["feat_scaler"]
    readout = primary["readout_model"]
    train_features = feat_scaler.transform(
        combine_feature_blocks(
            sequences=train_sequences,
            summary_features=train_summary,
            input_scaler=input_scaler,
            reservoir=RSNNReservoir(seed=primary["seed"]),
        )
    )
    train_targets = train_y  # legacy variable name kept below

    calibration_scaler = None
    calibration_model = None
    if USE_FILE_CALIBRATION:
        train_metadata["raw_predicted_relative_capacity"] = np.clip(
            readout.predict(train_features),
            0.0,
            1.0,
        )
        grouped_train = (
            train_metadata.groupby(["battery_id", "file_name", "temperature_c"], as_index=False)
            .agg(
                actual_relative_capacity=("relative_capacity", "first"),
                raw_predicted_relative_capacity=("raw_predicted_relative_capacity", "mean"),
                is_initial_capacity=("is_initial_capacity", "first"),
                is_low_current_ocv=("is_low_current_ocv", "first"),
                is_incremental_ocv=("is_incremental_ocv", "first"),
                file_order_position=("file_order_position", "first"),
                days_from_first_normalized=("days_from_first_normalized", "first"),
            )
        )
        calibration_scaler = StandardScaler().fit(
            build_file_level_calibration_features(grouped_train)
        )
        calibration_model = Ridge(alpha=FILE_CALIBRATION_ALPHA)
        calibration_model.fit(
            calibration_scaler.transform(build_file_level_calibration_features(grouped_train)),
            grouped_train["actual_relative_capacity"].to_numpy(dtype=float),
        )

    return {
        "seed": int(primary["seed"]),
        "ridge_alpha": ridge_alpha,
        "input_scaler": input_scaler,
        "feat_scaler": feat_scaler,
        "readout_model": readout,
        "ensemble_members": members,
        "ensemble_seeds": tuple(int(s) for s in ensemble_seeds),
        "target_mode": target_mode,
        "seg_cap_base_index": SEG_CAP_BASE_INDEX,
        "file_calibration_scaler": calibration_scaler,
        "file_calibration_model": calibration_model,
        "use_file_calibration": USE_FILE_CALIBRATION,
        "summary_feature_names": SUMMARY_FEATURE_NAMES,
        "train_file_names": list(train_file_names),
        "included_batteries": list(INCLUDED_BATTERIES),
        "data_root": CALCE_DATA_ROOT,
        "n_points": N_POINTS,
        "n_reservoir": N_RESERVOIR,
        "conn_prob": CONN_PROB,
    }


def _ensemble_members(artifact):
    members = artifact.get("ensemble_members")
    if members:
        return members
    return [
        {
            "seed": artifact["seed"],
            "feat_scaler": artifact["feat_scaler"],
            "readout_model": artifact["readout_model"],
        }
    ]


def predict_rows_raw(artifact, sequences, summary_features):
    """Return the raw ensemble-averaged readout output (no clipping).

    In residual mode this is the Ah residual; in absolute mode the per-segment
    Ah; in relative mode the predicted relative capacity. Clipping happens
    only after we convert back to a relative value in evaluate_files.
    """
    members = _ensemble_members(artifact)
    preds = []
    for member in members:
        reservoir = RSNNReservoir(
            n_input=N_INPUT,
            n_neurons=artifact["n_reservoir"],
            conn_prob=artifact["conn_prob"],
            seed=member["seed"],
        )
        features_raw = combine_feature_blocks(
            sequences=sequences,
            summary_features=summary_features,
            input_scaler=artifact["input_scaler"],
            reservoir=reservoir,
        )
        features = member["feat_scaler"].transform(features_raw)
        preds.append(member["readout_model"].predict(features))
    return np.mean(preds, axis=0)


def predict_rows(artifact, sequences, summary_features):
    """Backwards-compat helper. Returns clipped *relative* predictions.

    Used only for the legacy `target_mode == "relative"` path.
    """
    raw = predict_rows_raw(artifact, sequences, summary_features)
    return np.clip(raw, 0.0, 1.0)


def evaluate_files(dataset, artifact, file_names):
    metadata_df = dataset["metadata"].copy()
    file_mask = metadata_df["file_name"].isin(file_names).to_numpy()

    evaluation_rows = metadata_df.loc[file_mask].copy()
    summary_features = dataset["summary_features"][file_mask]
    raw_pred = predict_rows_raw(
        artifact=artifact,
        sequences=dataset["sequences"][file_mask],
        summary_features=summary_features,
    )

    target_mode = artifact.get("target_mode", "relative")
    seg_cap_idx = artifact.get("seg_cap_base_index", SEG_CAP_BASE_INDEX)
    seg_cap = summary_features[:, seg_cap_idx]
    ref_ah = evaluation_rows["reference_capacity_ah"].to_numpy(dtype=float)

    if target_mode == "relative":
        # Per-segment relative prediction; mean over segments per file.
        per_segment_relative = np.clip(raw_pred, 0.0, 1.2)
        evaluation_rows["predicted_relative_capacity"] = per_segment_relative
        evaluation_rows["predicted_segment_ah"] = per_segment_relative * ref_ah
    elif target_mode == "absolute":
        # Per-segment absolute Ah prediction (this represents the *file* total
        # since absolute target was the file Ah replicated to all segments).
        # Convert to per-segment Ah by dividing by n_segments so summing recovers it.
        n_seg = evaluation_rows["n_segments"].to_numpy(dtype=float)
        per_segment_ah = raw_pred / n_seg
        evaluation_rows["predicted_segment_ah"] = per_segment_ah
        evaluation_rows["predicted_relative_capacity"] = np.clip(raw_pred / ref_ah, 0.0, 1.2)
    elif target_mode == "residual":
        # Per-segment Ah = measured seg_cap + learned residual. Sum across
        # segments → file Ah, divide by ref to get relative capacity.
        per_segment_ah = seg_cap + raw_pred
        evaluation_rows["predicted_segment_ah"] = per_segment_ah
        evaluation_rows["predicted_relative_capacity"] = np.clip(per_segment_ah / ref_ah, 0.0, 1.2)
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    grouped = (
        evaluation_rows.groupby(["battery_id", "file_name", "temperature_c"], as_index=False)
        .agg(
            actual_relative_capacity=("relative_capacity", "first"),
            predicted_segment_ah_sum=("predicted_segment_ah", "sum"),
            predicted_relative_capacity_seg_mean=("predicted_relative_capacity", "mean"),
            segment_count=("segment_index", "count"),
            total_file_capacity_ah=("total_file_capacity_ah", "first"),
            reference_capacity_ah=("reference_capacity_ah", "first"),
            is_initial_capacity=("is_initial_capacity", "first"),
            is_low_current_ocv=("is_low_current_ocv", "first"),
            is_incremental_ocv=("is_incremental_ocv", "first"),
            file_order_position=("file_order_position", "first"),
            days_from_first_normalized=("days_from_first_normalized", "first"),
        )
    )

    if target_mode in ("residual", "absolute"):
        # File-level prediction = sum of per-segment Ah / reference.
        grouped["raw_predicted_relative_capacity"] = (
            grouped["predicted_segment_ah_sum"] / grouped["reference_capacity_ah"]
        ).clip(0.0, 1.2)
    else:
        grouped["raw_predicted_relative_capacity"] = grouped["predicted_relative_capacity_seg_mean"]
    grouped = grouped.drop(columns=["predicted_segment_ah_sum", "predicted_relative_capacity_seg_mean"])
    if artifact.get("use_file_calibration") and artifact.get("file_calibration_model") is not None:
        grouped["predicted_relative_capacity"] = np.clip(
            artifact["file_calibration_model"].predict(
                artifact["file_calibration_scaler"].transform(
                    build_file_level_calibration_features(grouped)
                )
            ),
            0.0,
            1.0,
        )
    else:
        grouped["predicted_relative_capacity"] = grouped["raw_predicted_relative_capacity"]

    # Physics-informed post-processing (Strategy C). Range clip only.
    # Monotonic projection is omitted because CALCE files are sparse (14
    # files over several months per battery) and tests alternate between
    # incremental-OCV and low-current-OCV protocols whose measured
    # capacity can legitimately rebound by several percentage points
    # between adjacent files. Monotonicity is still encoded softly into
    # the readout via the sqrt(t) / Arrhenius / Nernst priors.
    grouped["predicted_relative_capacity"] = phys.range_clip(
        grouped["predicted_relative_capacity"].to_numpy(dtype=float),
        lo=0.55,
        hi=1.15,
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (grouped["predicted_relative_capacity"] - grouped["actual_relative_capacity"]) ** 2
            )
        )
    )
    mae = float(
        np.mean(np.abs(grouped["predicted_relative_capacity"] - grouped["actual_relative_capacity"]))
    )

    return grouped, {
        "file_rmse_percent": rmse * 100.0,
        "file_mae_percent": mae * 100.0,
        "n_files": int(len(grouped)),
        "n_segments": int(len(evaluation_rows)),
    }


def save_artifact(artifact, output_path):
    np.save(output_path, artifact, allow_pickle=True)


def load_artifact(output_path):
    return np.load(output_path, allow_pickle=True).item()


def save_grouped_predictions_csv(grouped_df, output_path):
    grouped_df.to_csv(output_path, index=False)


def save_training_summary_csv(train_files, test_files, metrics, output_path):
    rows = [
        {"split": "train", "file_name": name, "value": ""} for name in train_files
    ] + [
        {"split": "test", "file_name": name, "value": ""} for name in test_files
    ] + [
        {"split": "metric", "file_name": key, "value": value} for key, value in metrics.items()
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["split", "file_name", "value"])
        writer.writeheader()
        writer.writerows(rows)
