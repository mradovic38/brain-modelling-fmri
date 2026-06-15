import os
from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np

HCP_DIR = "data/hcp_task"
FIG_DIR = "figs"

N_PARCELS = 360
TR = 0.72
RUNS = ["LR", "RL"]
HRF_SHIFT_SEC = 5.0  # shift trial windows to peak of canonical HRF
HRF_SHIFT_TR = int(round(HRF_SHIFT_SEC / TR))
# Fixed window length (in seconds) applied to every trial to remove the
# trial-duration confound that otherwise lets the decoder cheat on tasks
# where conditions have very different block lengths (LANGUAGE, EMOTION).
FIXED_WINDOW_SEC = 8.0
FIXED_WINDOW_TR = int(round(FIXED_WINDOW_SEC / TR))

# 2AFC label mapping per task: class 0 vs class 1.
# Conditions are aggregated (e.g. MOTOR left = lf OR lh).
TASK_BINARY = {
    "MOTOR":      {"0": ["lf", "lh"],          "1": ["rf", "rh"],          "name0": "left",   "name1": "right"},
    "WM":         {"0": ["0bk_body", "0bk_faces", "0bk_places", "0bk_tools"],
                   "1": ["2bk_body", "2bk_faces", "2bk_places", "2bk_tools"],
                   "name0": "0-back", "name1": "2-back"},
    "EMOTION":    {"0": ["neut"],              "1": ["fear"],              "name0": "neut",   "name1": "fear"},
    "GAMBLING":   {"0": ["loss"],              "1": ["win"],               "name0": "loss",   "name1": "win"},
    "LANGUAGE":   {"0": ["story"],             "1": ["math"],              "name0": "story",  "name1": "math"},
    "RELATIONAL": {"0": ["match"],             "1": ["relation"],          "name0": "match",  "name1": "relation"},
    "SOCIAL":     {"0": ["rnd"],               "1": ["mental"],           "name0": "rnd",    "name1": "mental"},
}

TASKS = list(TASK_BINARY.keys())

# Behavioural keys to look for in Stats.txt: (key_substring, scale_to_seconds)
# Different tasks store RT differently; we try several patterns.
RT_KEYS = {
    "MOTOR":      [],  # no RT
    "WM":         [],  # not in this NMA Stats.txt slice
    "EMOTION":    [("Median Face RT", 1e-3), ("Median Shape RT", 1e-3)],
    "GAMBLING":   [("Mean RT", 1e-3)],
    "LANGUAGE":   [("Story RT", 1e-3), ("Math RT", 1e-3)],
    "RELATIONAL": [("Control RT", 1e-3), ("Relational RT", 1e-3)],
    "SOCIAL":     [("Mean RT Mental", 1e-3), ("Mean RT Random", 1e-3)],
}
ACC_KEYS = {
    "MOTOR":      [],
    "WM":         [],
    "EMOTION":    [("Face Accuracy", 1.0), ("Shape Accuracy", 1.0)],
    "GAMBLING":   [],  # no accuracy concept
    "LANGUAGE":   [("Story ACC", 1.0), ("Math ACC", 1.0)],
    "RELATIONAL": [("Control ACC", 1.0), ("Relational ACC", 1.0)],
    "SOCIAL":     [("ACC Mental", 1.0), ("ACC Random", 1.0)],
}


def load_single_timeseries(subject: str, experiment: str, run: int, remove_mean: bool = True) -> np.ndarray:
    bold_run = RUNS[run]
    path = f"{HCP_DIR}/subjects/{subject}/{experiment}/tfMRI_{experiment}_{bold_run}/data.npy"
    ts = np.load(path)
    if remove_mean:
        ts -= ts.mean(axis=1, keepdims=True)
    return ts


def load_condition_trial_frames(subject: str, experiment: str, run: int, cond: str,
                                 fixed_window: bool = True) -> List[np.ndarray]:
    """Return per-trial frame indices.
    If fixed_window=True, every trial gets exactly FIXED_WINDOW_TR frames,
    starting at onset + HRF shift. This removes the trial-duration confound.
    """
    ev_file = f"{HCP_DIR}/subjects/{subject}/{experiment}/tfMRI_{experiment}_{RUNS[run]}/EVs/{cond}.txt"
    if not os.path.isfile(ev_file) or os.path.getsize(ev_file) == 0:
        return []
    arr = np.loadtxt(ev_file, ndmin=2)
    if arr.size == 0:
        return []
    onsets, durations = arr[:, 0], arr[:, 1]
    frames = []
    for onset, dur in zip(onsets, durations):
        start = int(np.floor(onset / TR)) + HRF_SHIFT_TR
        if fixed_window:
            n = FIXED_WINDOW_TR
        else:
            n = max(1, int(np.ceil(dur / TR)))
        # Skip trials shorter than the fixed window so we don't pad with frames
        # that fall outside the trial.
        if fixed_window and dur < FIXED_WINDOW_SEC:
            continue
        frames.append(np.arange(start, start + n))
    return frames


@dataclass
class TrialDataset:
    X: np.ndarray              # (n_trials, 360)
    y: np.ndarray              # (n_trials,) in {0,1}
    subject: np.ndarray        # (n_trials,) subject id
    task: np.ndarray           # (n_trials,) task name
    run: np.ndarray            # (n_trials,)


def build_dataset(subjects: List[str]) -> TrialDataset:
    Xs, ys, subs, tasks, runs = [], [], [], [], []
    for si, sid in enumerate(subjects):
        if si % 10 == 0:
            print(f"  loading subject {si+1}/{len(subjects)}: {sid}")
        for task in TASKS:
            for run in (0, 1):
                try:
                    ts = load_single_timeseries(sid, task, run, remove_mean=False)
                except FileNotFoundError:
                    continue
                # z-score per parcel within this run, for this subject
                ts = (ts - ts.mean(axis=1, keepdims=True)) / (ts.std(axis=1, keepdims=True) + 1e-8)
                T = ts.shape[1]
                for label_str in ("0", "1"):
                    for cond in TASK_BINARY[task][label_str]:
                        frames_list = load_condition_trial_frames(sid, task, run, cond)
                        for frames in frames_list:
                            frames = frames[(frames >= 0) & (frames < T)]
                            if frames.size == 0:
                                continue
                            x = ts[:, frames].mean(axis=1)  # (360,)
                            Xs.append(x)
                            ys.append(int(label_str))
                            subs.append(sid)
                            tasks.append(task)
                            runs.append(run)
    X = np.stack(Xs).astype(np.float32)
    return TrialDataset(
        X=X,
        y=np.array(ys, dtype=np.int64),
        subject=np.array(subs),
        task=np.array(tasks),
        run=np.array(runs, dtype=np.int64),
    )


def parse_stats(subject: str, experiment: str, run: int) -> Dict[str, float]:
    f = f"{HCP_DIR}/subjects/{subject}/{experiment}/tfMRI_{experiment}_{RUNS[run]}/EVs/Stats.txt"
    out = {}  # type: Dict[str, float]
    if not os.path.isfile(f):
        return out
    with open(f) as fh:
        for line in fh:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            try:
                out[k.strip()] = float(v.strip())
            except ValueError:
                pass
    return out