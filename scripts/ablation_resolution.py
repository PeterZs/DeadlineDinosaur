"""2x2 ablation: resolution schedule {const, freq} x position-LR hold {off, on}.

  A  const + no hold   -> vanilla LiteGS schedule
  B  freq  + hold      -> DeadlineDinosaur (challenge config)
  C  freq  + no hold   -> resolution scheduling alone
  D  const + hold      -> LR hold alone (same hold length as B on that scene)
  E  B + tile-cache fix -> DeadlineDinosaur with the stale tile-list bug fixed

Every arm uses the challenge config (7000 iterations, 1M target primitives,
~60 s wall-clock cut-off in the trainer). Arms are interleaved within each
repeat to avoid drift bias. Resumable: finished runs are skipped.
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess

BASE = ("--sh_degree 3 --source_type colmap --target_primitives 1000000 --iterations 7000 "
        "--position_lr_max_steps 7000 --position_lr_final 0.000016 --densification_interval 2 --eval")

ENV = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID")  # CUDA_VISIBLE_DEVICES set from --gpu


def run(cmd, log):
    with open(log, "w") as f:
        p = subprocess.run(cmd, shell=True, env=ENV, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode


def find_metrics(model_path):
    hits = glob.glob(os.path.join(model_path, "point_cloud", "*", "training_metrics.json"))
    return json.load(open(hits[0])) if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--scenes", nargs="+", required=True, help="name:image_dir, e.g. garden:images_4")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--python", default="python")
    ap.add_argument("--gpu", default="0", help="nvidia-smi (PCI bus order) index")
    ap.add_argument("--arms", nargs="+", default=["B", "A", "C", "D"], help="subset to run; keep B before D")
    args = ap.parse_args()
    ENV["CUDA_VISIBLE_DEVICES"] = args.gpu
    gpu_name = subprocess.run(["nvidia-smi", "-i", args.gpu, "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    print(f"[GPU] {args.gpu}: {gpu_name}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "results.csv")
    fields = ["scene", "arm", "rep", "psnr", "time", "status", "global_step", "n_gaussians", "decay_from_iter", "gpu"]
    done = set()
    if os.path.exists(csv_path):
        for r in csv.DictReader(open(csv_path)):
            done.add((r["scene"], r["arm"], r["rep"]))
    else:
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fields).writeheader()

    for rep in range(args.reps):
        for spec in args.scenes:
            name, images = spec.split(":")
            src = os.path.join(args.data, name)
            hold_iter = None
            for arm in args.arms:  # B first: D needs B's hold length
                mp = os.path.join(args.out, name, f"{arm}_rep{rep}")
                if (name, arm, str(rep)) in done:
                    if arm == "B":
                        hold_iter = find_metrics(mp)["decay_from_iter"]
                    continue
                extra = {"A": "--resolution_mode const",
                         "B": "--resolution_mode freq",
                         "C": "--resolution_mode freq --lr_decay_from 1",
                         "D": f"--resolution_mode const --lr_decay_from {hold_iter}",
                         "E": "--resolution_mode freq --fix_tile_cache"}[arm]
                os.makedirs(mp, exist_ok=True)
                rc = run(f"{args.python} example_train.py -s {src} -m {mp} -i {images} {BASE} {extra}",
                         os.path.join(mp, "train.log"))
                m = find_metrics(mp)
                if rc != 0 or m is None:
                    print(f"[FAIL] {name} {arm} rep{rep}: train rc={rc}", flush=True)
                    continue
                if arm == "B":
                    hold_iter = m["decay_from_iter"]
                run(f"{args.python} example_metrics.py -s {src} -m {mp} -i {images} --sh_degree 3 --eval",
                    os.path.join(mp, "eval.log"))
                found = re.findall(r"PSNR :\s+([\d.]+)", open(os.path.join(mp, "eval.log")).read())
                row = {"scene": name, "arm": arm, "rep": rep, "psnr": found[-1] if found else "",
                       "time": round(m["time"], 2), "status": m["status"], "global_step": m.get("global_step"),
                       "n_gaussians": m.get("n_gaussians"), "decay_from_iter": m.get("decay_from_iter"), "gpu": gpu_name}
                with open(csv_path, "a", newline="") as f:
                    csv.DictWriter(f, fields).writerow(row)
                print(f"[DONE] {row}", flush=True)


if __name__ == "__main__":
    main()
