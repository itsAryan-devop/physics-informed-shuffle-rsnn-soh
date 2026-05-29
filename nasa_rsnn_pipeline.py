import os

import numpy as np
import scipy.io
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

import physics_readout as pr
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import physics_layer as phys


N_POINTS = 100
N_INPUT = 3
N_RESERVOIR = 200
CONN_PROB = 0.1
# Shuffle-RSNN: grouped recurrent connectivity + channel-shuffle of
# previous spikes. W_res is block-diagonal (each of N_GROUPS groups only
# wires within itself, cutting synaptic connections by ~1/N_GROUPS), and
# at every timestep prev_spikes is permuted via reshape-transpose-flatten
# so each group's block receives a mix of spikes from all other groups.
N_GROUPS = 4
RIDGE_ALPHA = 50.0
# RSNN parent preserved (LIF reservoir + ridge readout). The readout learns
# an absolute-Ah residual on top of the per-cycle discharge_throughput_ah
# feature. At evaluation we add the throughput back and divide by the
# held-out battery's first-cycle capacity to reconstruct relative capacity.
TARGET_MODE = "residual"          # "residual" | "relative"
ENSEMBLE_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)   # final-tune sweep winner (was 5 seeds)
THROUGHPUT_BASE_INDEX = 15        # index of discharge_throughput_ah in BASE_SUMMARY_FEATURE_NAMES

# ---------------------------------------------------------------------------
# Strategy-A physics readout (composite-loss gradient-descent linear layer).
# Disabled by default because NASA trains in residual-Ah space where mono /
# range losses need to be routed through throughput reconstruction (future
# work). Enabling this flag falls back to ridge when target_mode=="residual"
# and to physics readout otherwise.
# ---------------------------------------------------------------------------
USE_PHYSICS_READOUT = False
PHYS_ALPHA_MONO = 0.10
PHYS_ALPHA_RANGE = 1.0
PHYS_EPSILON_REG = 0.005
PHYS_LO = 0.55
PHYS_HI = 1.15
PHYS_LR = 1e-3
PHYS_N_EPOCHS = 400

BASE_SUMMARY_FEATURE_NAMES = [
    "cycle_index",
    "discharge_duration_s",
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
    "voltage_drop_head_tail",
    "discharge_throughput_ah",
    "impedance_re",
    "impedance_rct",
    "rectified_impedance_mean",
]

PHYSICS_FEATURE_NAMES = ["phys_soh_ecm_prior", "phys_soh_sqrt_time_prior"]

SUMMARY_FEATURE_NAMES = (
    BASE_SUMMARY_FEATURE_NAMES
    + [f"{name}_drift_from_first" for name in BASE_SUMMARY_FEATURE_NAMES]
    + [f"{name}_delta_from_previous" for name in BASE_SUMMARY_FEATURE_NAMES]
    + PHYSICS_FEATURE_NAMES
)


def battery_name_from_path(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]


def artifact_path_for_battery(file_path_or_battery_id):
    battery_id = battery_name_from_path(file_path_or_battery_id)
    return os.path.abspath(f"nasa_holdout_{battery_id}_artifact.npy")


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
        prev_spikes = np.zeros(self.n_neurons)

        for t in range(len(sequence)):
            shuffled_prev = prev_spikes[self.shuffle_perm]
            current = self.W_in @ sequence[t] + self.W_res @ shuffled_prev
            membrane = self.alpha * membrane + current
            spikes = (membrane >= self.v_th).astype(float)
            membrane[spikes > 0] = 0.0
            spike_counts += spikes
            prev_spikes = spikes

        return spike_counts


def extract_battery_samples(file_path):
    mat = scipy.io.loadmat(file_path)
    battery_id = [key for key in mat.keys() if not key.startswith("__")][-1]
    cycles = mat[battery_id][0, 0]["cycle"]

    first_re = 0.0
    first_rct = 0.0
    first_rectified_impedance_mean = 0.0
    for index in range(cycles.shape[1]):
        cycle = cycles[0, index]
        if str(cycle["type"][0]) == "impedance":
            c_data = cycle["data"][0, 0]
            rectified = np.abs(np.asarray(c_data["Rectified_Impedance"]).reshape(-1).astype(complex))
            first_re = float(c_data["Re"][0][0])
            first_rct = float(c_data["Rct"][0][0])
            first_rectified_impedance_mean = float(rectified.mean())
            break

    last_re = np.nan
    last_rct = np.nan
    last_rectified_impedance_mean = np.nan
    first_capacity = None
    sequences = []
    base_summary_features = []
    targets = []

    discharge_index = 0
    for index in range(cycles.shape[1]):
        cycle = cycles[0, index]
        cycle_type = str(cycle["type"][0])
        c_data = cycle["data"][0, 0]

        if cycle_type == "impedance":
            rectified = np.abs(np.asarray(c_data["Rectified_Impedance"]).reshape(-1).astype(complex))
            last_re = float(c_data["Re"][0][0])
            last_rct = float(c_data["Rct"][0][0])
            last_rectified_impedance_mean = float(rectified.mean())
            continue

        if cycle_type != "discharge":
            continue

        voltage = np.asarray(c_data["Voltage_measured"][0], dtype=float)
        current = np.asarray(c_data["Current_measured"][0], dtype=float)
        temp = np.asarray(c_data["Temperature_measured"][0], dtype=float)
        time = np.asarray(c_data["Time"][0], dtype=float)
        capacity = float(c_data["Capacity"][0][0])

        if first_capacity is None:
            first_capacity = capacity

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
        re_value = last_re if np.isfinite(last_re) else first_re
        rct_value = last_rct if np.isfinite(last_rct) else first_rct
        rectified_mean = (
            last_rectified_impedance_mean
            if np.isfinite(last_rectified_impedance_mean)
            else first_rectified_impedance_mean
        )

        summary = np.array(
            [
                discharge_index,
                float(time[-1] - time[0]),
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
                float(voltage[:head_size].mean() - voltage[-tail_size:].mean()),
                float(np.trapezoid(np.abs(current), time) / 3600.0),
                float(re_value),
                float(rct_value),
                float(rectified_mean),
            ],
            dtype=float,
        )

        sequences.append(sequence)
        base_summary_features.append(summary)
        targets.append(capacity / float(first_capacity))
        discharge_index += 1

    base_summary_features = np.asarray(base_summary_features, dtype=float)
    first_summary = base_summary_features[0]
    previous_summary = np.vstack([base_summary_features[0], base_summary_features[:-1]])
    safe_first_summary = np.where(np.abs(first_summary) < 1e-8, 1.0, first_summary)

    # Physics-informed priors: impedance-driven ECM SOH estimate and a
    # simple sqrt(t)+linear-cycle calendar prior. They appear as two extra
    # columns the ridge readout can weight freely. See physics_layer.py.
    phys_ecm = phys.soh_ecm_prior(
        impedance_re=base_summary_features[:, 16],
        impedance_rct=base_summary_features[:, 17],
    ).reshape(-1, 1)
    phys_sqrt = phys.soh_sqrt_time_prior(
        cycle_index=base_summary_features[:, 0]
    ).reshape(-1, 1)

    summary_features = np.hstack(
        [
            base_summary_features,
            (base_summary_features - first_summary) / safe_first_summary,
            base_summary_features - previous_summary,
            phys_ecm,
            phys_sqrt,
        ]
    )

    return {
        "battery_id": battery_id,
        "file_path": os.path.abspath(file_path),
        "sequences": np.asarray(sequences, dtype=float),
        "summary_features": np.asarray(summary_features, dtype=float),
        "targets": np.asarray(targets, dtype=float),
        "first_capacity": float(first_capacity) if first_capacity is not None else 0.0,
    }


def combine_feature_blocks(sequences, summary_features, input_scaler, reservoir):
    scaled_sequences = np.asarray([input_scaler.transform(cycle) for cycle in sequences], dtype=float)
    reservoir_features = np.asarray([reservoir.process_cycle(cycle) for cycle in scaled_sequences], dtype=float)
    return np.hstack([reservoir_features, summary_features])


def _build_train_residual(train_sets):
    """Per-cycle absolute-Ah residual target: true_capacity_Ah - throughput_Ah."""
    pieces = []
    for dataset in train_sets:
        tgt = dataset["targets"]
        first_capacity = dataset["first_capacity"]
        throughput = dataset["summary_features"][:, THROUGHPUT_BASE_INDEX]
        absolute_capacity_ah = tgt * first_capacity
        pieces.append(absolute_capacity_ah - throughput)
    return np.concatenate(pieces)


def fit_holdout_artifact(train_files, held_out_file, seed=0, ridge_alpha=RIDGE_ALPHA,
                         ensemble_seeds=None, target_mode=None):
    train_sets = [extract_battery_samples(path) for path in train_files]
    held_out_set = extract_battery_samples(held_out_file)

    train_sequences = np.vstack([dataset["sequences"] for dataset in train_sets])
    train_summary = np.vstack([dataset["summary_features"] for dataset in train_sets])
    train_targets_relative = np.concatenate([dataset["targets"] for dataset in train_sets])

    if target_mode is None:
        target_mode = TARGET_MODE
    if ensemble_seeds is None:
        ensemble_seeds = ENSEMBLE_SEEDS if ENSEMBLE_SEEDS else (seed,)

    if target_mode == "residual":
        train_targets = _build_train_residual(train_sets)
    elif target_mode == "relative":
        train_targets = train_targets_relative
    else:
        raise ValueError(f"Unknown TARGET_MODE: {target_mode}")

    input_scaler = MinMaxScaler().fit(np.vstack(train_sequences))

    # Build per-row metadata for Strategy-A monotonicity loss: battery id
    # tags rows to their cell, cycle_index gives the time order within
    # each cell. ``physics_readout`` uses the forward-difference pairs
    # ((k, k+1) within the same cell) as the mono penalty.
    train_cell_ids = np.concatenate(
        [np.full(len(dataset["targets"]), dataset["battery_id"]) for dataset in train_sets]
    )
    train_cycle_order = train_summary[:, 0]

    members = []
    for member_seed in ensemble_seeds:
        reservoir = RSNNReservoir(seed=int(member_seed))
        train_features_raw = combine_feature_blocks(
            sequences=train_sequences,
            summary_features=train_summary,
            input_scaler=input_scaler,
            reservoir=reservoir,
        )
        feat_scaler = StandardScaler().fit(train_features_raw)
        train_features = feat_scaler.transform(train_features_raw)

        if USE_PHYSICS_READOUT:
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
            # For residual-Ah target the output is an absolute Ah
            # correction, not a [0,1] SOH — disable range term then.
            if target_mode == "residual":
                readout.alpha_range = 0.0
                readout.alpha_mono = 0.0  # mono applies to SOH, not residual
            readout.fit(
                train_features,
                train_targets,
                cell_ids=train_cell_ids,
                cycle_order=train_cycle_order,
            )
        else:
            readout = Ridge(alpha=ridge_alpha)
            readout.fit(train_features, train_targets)

        members.append(
            {"seed": int(member_seed), "feat_scaler": feat_scaler, "readout_model": readout}
        )

    primary = members[0]

    artifact = {
        "held_out_battery": held_out_set["battery_id"],
        "train_batteries": [dataset["battery_id"] for dataset in train_sets],
        "train_files": [os.path.basename(path) for path in train_files],
        "seed": int(primary["seed"]),
        "ridge_alpha": ridge_alpha,
        "input_scaler": input_scaler,
        "feat_scaler": primary["feat_scaler"],
        "readout_model": primary["readout_model"],
        "ensemble_members": members,
        "ensemble_seeds": tuple(int(s) for s in ensemble_seeds),
        "target_mode": target_mode,
        "throughput_base_index": THROUGHPUT_BASE_INDEX,
        "summary_feature_names": SUMMARY_FEATURE_NAMES,
        "n_points": N_POINTS,
        "n_reservoir": N_RESERVOIR,
        "conn_prob": CONN_PROB,
    }
    return artifact, held_out_set


def predict_relative_capacity(file_path, artifact):
    dataset = extract_battery_samples(file_path)

    target_mode = artifact.get("target_mode", "relative")
    throughput_base_index = artifact.get("throughput_base_index", THROUGHPUT_BASE_INDEX)
    members = artifact.get("ensemble_members")

    if not members:
        members = [
            {
                "seed": artifact["seed"],
                "feat_scaler": artifact["feat_scaler"],
                "readout_model": artifact["readout_model"],
            }
        ]

    raw_preds = []
    for member in members:
        reservoir = RSNNReservoir(
            n_input=N_INPUT,
            n_neurons=artifact["n_reservoir"],
            conn_prob=artifact["conn_prob"],
            seed=int(member["seed"]),
        )
        features_raw = combine_feature_blocks(
            sequences=dataset["sequences"],
            summary_features=dataset["summary_features"],
            input_scaler=artifact["input_scaler"],
            reservoir=reservoir,
        )
        features = member["feat_scaler"].transform(features_raw)
        raw_preds.append(member["readout_model"].predict(features))

    raw_mean = np.mean(np.asarray(raw_preds, dtype=float), axis=0)

    if target_mode == "residual":
        throughput_ah = dataset["summary_features"][:, throughput_base_index]
        predicted_absolute_ah = throughput_ah + raw_mean
        first_capacity = dataset["first_capacity"]
        if first_capacity <= 0:
            first_capacity = 1.0
        predictions = predicted_absolute_ah / first_capacity
    else:
        predictions = raw_mean

    # Physics post-processing (Strategy C):
    #   1. Coulomb-counting clamp — absolute-Ah reconstruction cannot
    #      diverge from measured throughput by more than `tol`. This
    #      enforces conservation of charge at inference.
    #   2. Hard range clip to the physically plausible SOH band.
    # Monotonic projection was tested and turned off because it squashes
    # NASA's legitimate capacity-regeneration rebounds (>3pp between
    # successive discharges); monotone structure still biases the readout
    # softly via the sqrt(t) + ECM priors.
    if target_mode == "residual":
        throughput_ah = dataset["summary_features"][:, throughput_base_index]
        first_capacity = dataset["first_capacity"]
        if first_capacity <= 0:
            first_capacity = 1.0
        predictions = phys.coulomb_clamp(
            y_hat=predictions,
            throughput_ah=throughput_ah,
            reference_capacity_ah=first_capacity,
            tol=0.08,
        )
    predictions = phys.range_clip(predictions, lo=0.55, hi=1.15)
    return dataset["targets"], predictions, dataset


def evaluate_holdout(train_files, held_out_file, seed=0, ridge_alpha=RIDGE_ALPHA):
    artifact, held_out_set = fit_holdout_artifact(
        train_files=train_files,
        held_out_file=held_out_file,
        seed=seed,
        ridge_alpha=ridge_alpha,
    )
    actual, predicted, _ = predict_relative_capacity(held_out_file, artifact)
    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    mae = float(np.mean(np.abs(predicted - actual)))
    metrics = {
        "battery_id": held_out_set["battery_id"],
        "file_name": os.path.basename(held_out_file),
        "train_files": [os.path.basename(path) for path in train_files],
        "train_cycles": int(sum(len(extract_battery_samples(path)["targets"]) for path in train_files)),
        "test_cycles": int(len(actual)),
        "rmse_percent": rmse * 100.0,
        "mae_percent": mae * 100.0,
    }
    return artifact, metrics


def save_artifact(artifact, output_path):
    np.save(output_path, artifact, allow_pickle=True)


def load_artifact(output_path):
    return np.load(output_path, allow_pickle=True).item()
