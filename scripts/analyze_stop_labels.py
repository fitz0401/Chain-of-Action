#!/usr/bin/env python
"""
Validate stop-label stability: for each task × segment-index, plot the
cross-demo σ²(k) curve (variance of SE(3)-normalised EE position at offset k
from the keyframe) and mark the auto-detected inflection / stop point.

Run this BEFORE training BIP to confirm the boundary signal is clean:

  python scripts/analyze_stop_labels.py \\
      --tasks stack_wine open_drawer reach_target push_button sweep_to_dustpan

If inflection points are inconsistent across segment indices, or the curves
have no clear elbow, the cross-demo variance is not a reliable signal.
The script also prints a summary table so you can spot outliers quickly.

Output: plots/stop_labels/<task>.png  (one subplot per segment-index)
"""
import os, sys, pickle, argparse
from collections import defaultdict

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, ".")


# ── demo loading ──────────────────────────────────────────────────────────────

def _load_pkl_demos(dataset_root: str, task: str, n_demos: int):
    ep_dir = os.path.join(dataset_root, "train", task, "variation0", "episodes")
    demos = []
    for ep in range(n_demos):
        path = os.path.join(ep_dir, f"episode{ep}", "low_dim_obs.pkl")
        if not os.path.exists(path):
            break
        with open(path, "rb") as f:
            demos.append(pickle.load(f))
    return demos


# ── segmentation ──────────────────────────────────────────────────────────────

def _gripper_only_keyframes(demo):
    """Return keyframe indices triggered by gripper state changes only."""
    kfs, prev = [], demo[0].gripper_open
    for i, obs in enumerate(demo):
        if obs.gripper_open != prev or i == len(demo) - 1:
            kfs.append(i)
            prev = obs.gripper_open
    return kfs


def _segment(demos):
    """Split each demo into gripper-change segments; return (segs, seg_indices)."""
    segs, seg_idx = [], []
    for demo in demos:
        kfs = _gripper_only_keyframes(demo)
        if kfs[0] < 10 and len(kfs) > 1:
            kfs[0] = 0
        elif kfs[0] >= 10:
            kfs = [0] + kfs
        for i in range(len(kfs) - 1):
            segs.append(demo[kfs[i]:kfs[i + 1] + 1])
            seg_idx.append(i)
    return segs, seg_idx


# ── SE(3) normalisation ───────────────────────────────────────────────────────

def _local_ee_pos(seg):
    """EE xyz in keyframe SE(3) frame.  seg: list[Observation]. Returns (T, 3)."""
    poses = np.array([obs.gripper_pose for obs in seg], dtype=np.float32)  # (T, 7) xyz+xyzw
    R_kf = R.from_quat(poses[-1, 3:7])                                    # xyzw
    return R_kf.inv().apply(poses[:, :3] - poses[-1, :3])                 # (T, 3)


# ── variance curve & inflection ───────────────────────────────────────────────

def sigma2_curve(local_pos_list):
    """σ²(k) for k=0..max_k across demos of the same segment-index."""
    max_k = min(len(p) for p in local_pos_list) - 1
    return np.array([
        np.stack([p[-1 - k] for p in local_pos_list]).var(axis=0).sum()
        for k in range(max_k + 1)
    ])


def find_stop_k(curve: np.ndarray) -> int:
    """
    Auto-detect the inflection point in σ²(k).

    Tries PELT (ruptures) first; falls back to 2nd-order difference peak.
    Returns k as an integer offset from the keyframe.
    """
    n = len(curve)
    if n < 4:
        return max(1, n // 2)

    # PELT changepoint (optional dependency)
    try:
        import ruptures as rpt
        bkps = rpt.Pelt(model="rbf", min_size=2, jump=1).fit(
            curve.reshape(-1, 1)).predict(pen=0.3)
        if bkps and 0 < bkps[0] < n:
            return int(bkps[0])
    except ImportError:
        pass

    # 2nd-order difference peak  (standard finite-difference approximation of d²σ²/dk²)
    d2 = np.diff(curve, n=2)
    return min(int(np.argmax(d2)) + 2, n - 1)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+",
                    default=["stack_wine", "open_drawer", "reach_target",
                             "push_button", "sweep_to_dustpan"])
    ap.add_argument("--demos",        type=int, default=100,
                    help="Maximum number of train demos to load per task")
    ap.add_argument("--dataset-root", default="data/rlbench")
    ap.add_argument("--out-dir",      default="plots/stop_labels")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {}

    for task in args.tasks:
        print(f"\n{'='*64}\nTask: {task}")
        raw = _load_pkl_demos(args.dataset_root, task, args.demos)
        if not raw:
            print(f"  No demos found in {args.dataset_root}/train/{task}/")
            continue
        print(f"  Loaded {len(raw)} demos")

        segs, seg_idx = _segment(raw)

        groups = defaultdict(list)
        for seg, n in zip(segs, seg_idx):
            try:
                groups[n].append(_local_ee_pos(seg))
            except Exception as e:
                print(f"  Warning: seg {n} skipped — {e}")

        n_seg = len(groups)
        fig, axes = plt.subplots(1, n_seg, figsize=(5 * n_seg, 4), squeeze=False)
        fig.suptitle(f"σ²(k)  —  {task}  ({len(raw)} demos)", fontsize=12)

        task_summary = {}
        for seg_n, ax in zip(sorted(groups), axes[0]):
            members = groups[seg_n]
            if len(members) < 3:
                ax.set_title(f"seg {seg_n}  (only {len(members)} demos — skip)")
                continue

            curve  = sigma2_curve(members)
            k_stop = find_stop_k(curve)
            task_summary[seg_n] = k_stop

            ks = np.arange(len(curve))
            ax.plot(ks, curve, "b-o", ms=3, label="σ²(k)")
            ax.fill_between(ks, curve, alpha=0.15, color="blue")
            ax.axvline(k_stop, color="r", ls="--", lw=1.5, label=f"stop_k={k_stop}")
            ax.set_xlabel("k  (steps before keyframe)")
            ax.set_ylabel("σ²  (m²)")
            ax.set_title(f"seg {seg_n}  |  {len(members)} demos  |  stop_k={k_stop}")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

            seg_lens = [len(p) for p in members]
            print(f"  seg {seg_n}: demos={len(members)}, "
                  f"σ²_max={curve.max():.5f}, σ²@stop={curve[k_stop]:.5f}, "
                  f"stop_k={k_stop}/{len(curve)-1}, "
                  f"seg_len[min/mean/max]={min(seg_lens)}/"
                  f"{int(np.mean(seg_lens))}/{max(seg_lens)}")

        plt.tight_layout()
        out_path = os.path.join(args.out_dir, f"{task}.png")
        plt.savefig(out_path, dpi=130); plt.close()
        print(f"  → {out_path}")
        summary[task] = task_summary

    print(f"\n{'='*64}")
    print("Summary (stop_k per task × segment-index):")
    for task, d in summary.items():
        print(f"  {task}: {d}")
    print(f"\nPlots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
