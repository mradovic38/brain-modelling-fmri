import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from helpers import (
    HCP_DIR, RUNS,
    TASK_BINARY, TASKS,
    load_single_timeseries, load_condition_trial_frames,
    build_dataset,
)


def get_subject_trials(subject: str, task: str, run: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build the test set for one (subject, task, run).

    Returns:
    X : (n_trials, 360) brain-state vectors
    y : (n_trials,) true binary labels
    descriptions : list of human-readable trial descriptors
    """
    ts = load_single_timeseries(subject, task, run, remove_mean=False)
    ts = (ts - ts.mean(axis=1, keepdims=True)) / (ts.std(axis=1, keepdims=True) + 1e-8)
    T = ts.shape[1]

    X, y, desc = [], [], []
    for label_str in ("0", "1"):
        for cond in TASK_BINARY[task][label_str]:
            frames_list = load_condition_trial_frames(subject, task, run, cond)
            for k, frames in enumerate(frames_list):
                frames = frames[(frames >= 0) & (frames < T)]
                if frames.size == 0:
                    continue
                X.append(ts[:, frames].mean(axis=1))
                y.append(int(label_str))
                desc.append(f"{cond}#{k}")
    if not X:
        return np.zeros((0, 360)), np.array([], dtype=int), []
    return np.stack(X).astype(np.float32), np.array(y, dtype=int), desc


def train_decoder(ds, train_task: str, holdout_subject: str, C: float = 1.0):
    """
    Train logistic regression on `train_task`, excluding `holdout_subject`.
    """
    m = (ds.task == train_task) & (ds.subject != holdout_subject)
    if m.sum() == 0:
        raise RuntimeError(f"No training data found for task={train_task}")
    if len(np.unique(ds.y[m])) < 2:
        raise RuntimeError(f"Training set for {train_task} has only one class")
    sc = StandardScaler().fit(ds.X[m])
    clf = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=500)
    clf.fit(sc.transform(ds.X[m]), ds.y[m])
    return clf, sc, int(m.sum())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=TASKS, help="Task to predict on (test task).")
    p.add_argument("--train-task", default=None, choices=TASKS,
                   help="Task to train the decoder on (default: same as --task = within-task).")
    p.add_argument("--subject", required=True, help="Subject ID, e.g. 100307.")
    p.add_argument("--run", type=int, choices=[0, 1], required=True, help="Run: 0=LR, 1=RL.")
    p.add_argument("--cond", default=None, help="Filter to a single condition name (e.g. rh, fear, math).")
    p.add_argument("--trial", type=int, default=None, help="Filter to a single trial index within the condition.")
    p.add_argument("--C", type=float, default=1.0, help="Ridge inverse-regularisation strength.")
    return p.parse_args()


def main():
    """
    Run a prediction on a single HCP trial (or a small set of trials).

    Trains a within-task ridge logistic-regression decoder, holding the chosen
    subject out of training, then predicts the chosen subject's trials and prints
    per-trial results + summary metrics.

    Examples
    --------
    # Predict one trial: subject 100307, MOTOR task, run 0 (LR), first 'rh' trial
    python predict_trial.py --task MOTOR --subject 100307 --run 0 --cond rh --trial 0

    # Predict all trials of one run for a subject:
    python predict_trial.py --task RELATIONAL --subject 100307 --run 1

    # Cross-task transfer prediction: train on WM, predict LANGUAGE trials
    python predict_trial.py --task LANGUAGE --train-task WM --subject 100307 --run 0
    """
    args = parse_args()
    train_task = args.train_task or args.task
    transfer = train_task != args.task

    print(f"\n  task to predict on : {args.task}  (subject {args.subject}, run {RUNS[args.run]})")
    print(f"  decoder trained on : {train_task}" + ("  [cross-task transfer]" if transfer else "  [within-task]"))
    print(f"  label mapping      : 0 = {TASK_BINARY[args.task]['name0']:>10s}    "
          f"1 = {TASK_BINARY[args.task]['name1']}")
    if transfer:
        print(f"  decoder labels     : 0 = {TASK_BINARY[train_task]['name0']:>10s}    "
              f"1 = {TASK_BINARY[train_task]['name1']}")

    ### Load full dataset for training (excludes the held-out subject)
    subjects = np.loadtxt(os.path.join(HCP_DIR, "subjects_list.txt"), dtype=str).tolist()
    if args.subject not in subjects:
        print(f"\nERROR: subject {args.subject} not in subjects_list.txt", file=sys.stderr)
        sys.exit(1)

    print("\n  building training dataset (this takes ~1 min)...")
    ds = build_dataset(subjects)
    clf, scaler, n_train = train_decoder(ds, train_task, args.subject, C=args.C)
    print(f"  trained on {n_train} trials from {len(subjects) - 1} other subjects")

    ### Build the test trials for the held-out subject
    X_test, y_test, desc = get_subject_trials(args.subject, args.task, args.run)
    if X_test.shape[0] == 0:
        print(f"\nERROR: no trials found for {args.subject} {args.task} run {args.run}", file=sys.stderr)
        sys.exit(1)

    # Optional filtering to one condition / one trial
    keep = np.ones(len(desc), dtype=bool)
    if args.cond is not None:
        keep &= np.array([d.split("#")[0] == args.cond for d in desc])
    if args.trial is not None:
        # apply trial index within the per-condition stream
        cond_counts = {}
        idx_keep = np.zeros(len(desc), dtype=bool)
        for i, d in enumerate(desc):
            c = d.split("#")[0]
            cond_counts[c] = cond_counts.get(c, -1) + 1
            if cond_counts[c] == args.trial and (args.cond is None or c == args.cond):
                idx_keep[i] = True
        keep &= idx_keep
    X_test, y_test, desc = X_test[keep], y_test[keep], [d for d, k in zip(desc, keep) if k]
    if len(desc) == 0:
        print("\nERROR: filter selected zero trials", file=sys.stderr)
        sys.exit(1)

    ### Predict
    proba = clf.predict_proba(scaler.transform(X_test))[:, 1]
    pred = (proba >= 0.5).astype(int)
    correct = (pred == y_test)

    # log-odds = drift-rate proxy
    p_clip = np.clip(proba, 1e-6, 1 - 1e-6)
    logit = np.log(p_clip / (1 - p_clip))

    ### Print results
    print("\n" + "=" * 78)
    print(f"{'trial':<15s} {'true':>6s} {'pred':>6s} {'P(class=1)':>12s} {'log-odds':>10s} {'correct':>9s}")
    print("-" * 78)
    name0 = TASK_BINARY[args.task]["name0"]
    name1 = TASK_BINARY[args.task]["name1"]
    for d, yt, yp, pr, lo, ok in zip(desc, y_test, pred, proba, logit, correct):
        t_name = name1 if yt == 1 else name0
        p_name = name1 if yp == 1 else name0
        mark = "✓" if ok else "✗"
        print(f"{d:<15s} {t_name:>6s} {p_name:>6s} {pr:>12.3f} {lo:>+10.3f} {mark:>9s}")
    print("=" * 78)

    n = len(desc)
    acc = correct.mean()
    mean_conf = np.where(y_test == 1, proba, 1 - proba).mean()
    print(f"\nSummary  ({n} trial{'s' if n > 1 else ''}):")
    print(f"  accuracy                : {acc:.3f}   ({correct.sum()}/{n})")
    print(f"  mean P(true class)      : {mean_conf:.3f}")
    print(f"  mean |log-odds|         : {np.abs(logit).mean():.3f}   (decoder confidence / drift-rate proxy)")
    if transfer:
        print(f"  NOTE: predictions use the {train_task} axis; class names above refer to the {args.task} mapping.")


if __name__ == "__main__":
    main()
