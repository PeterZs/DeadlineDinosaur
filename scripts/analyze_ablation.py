"""Summarise results.csv from ablation_resolution.py.

Arms: A const/no-hold, B freq/hold, C freq/no-hold, D const/hold.
Effects are paired within (scene, rep):
  resolution effect = mean of (C - A) and (B - D)
  LR-hold effect    = mean of (D - A) and (B - C)
"""
import sys

import pandas as pd

df = pd.read_csv(sys.argv[1])
df = df[pd.to_numeric(df["psnr"], errors="coerce").notna()].astype({"psnr": float})

print(f"GPU: {', '.join(df['gpu'].unique())}   runs: {len(df)}\n")

summary = df.groupby(["scene", "arm"]).agg(psnr_mean=("psnr", "mean"), psnr_std=("psnr", "std"),
                                           n=("psnr", "size"), iters=("global_step", "mean"),
                                           time=("time", "mean"))
print("Per scene / arm (mean over repeats)")
print(summary.round(2).to_string(), "\n")

table = df.pivot_table(index="scene", columns="arm", values="psnr", aggfunc="mean")
table.loc["MEAN"] = table.mean()
print("PSNR by arm (A=vanilla LiteGS, B=DeadlineDinosaur, C=freq only, D=LR hold only)")
print(table.round(2).to_string(), "\n")

w = df.pivot_table(index=["scene", "rep"], columns="arm", values="psnr")
eff = pd.DataFrame({
    "B-A (ours vs vanilla)": w["B"] - w["A"],
    "resolution effect": ((w["C"] - w["A"]) + (w["B"] - w["D"])) / 2,
    "LR-hold effect": ((w["D"] - w["A"]) + (w["B"] - w["C"])) / 2,
    "interaction": ((w["B"] - w["D"]) - (w["C"] - w["A"])) / 2,
})
per_scene = eff.groupby("scene").agg(["mean", "std"])
print("Paired effects in dB, per scene (mean, std over repeats)")
print(per_scene.round(2).to_string(), "\n")
overall = eff.agg(["mean", "std", "count"]).T
overall["sem"] = overall["std"] / overall["count"] ** 0.5
print("Paired effects in dB, all scenes x repeats")
print(overall.round(3).to_string(), "\n")

it = df.pivot_table(index="scene", columns="arm", values="global_step", aggfunc="mean")
print("Mean iterations completed within the 60 s budget")
print(it.round(0).to_string())
