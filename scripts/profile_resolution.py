"""Time one training iteration at different render downsampling scales.

Answers: in the LiteGS backend, how much of an iteration's cost actually
shrinks when rendering at 1/s resolution? Timed at several primitive counts,
since the low-resolution phase of the schedule happens early in training,
when the model is still small.

Usage:
  python scripts/profile_resolution.py -s <scene> -i images_4 --ply <trained point_cloud.ply> --out result.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import fused_ssim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deadlinedino
import deadlinedino.config
from deadlinedino import render, scene, io_manager
from deadlinedino.data import CameraFrameDataset
from deadlinedino.training import optimizer as gs_optimizer


def build_params(ply_path, keep_frac, cluster_size, sh_degree):
    xyz, scale, rot, sh_0, sh_rest, opacity = io_manager.load_ply(ply_path, sh_degree)
    tensors = [torch.tensor(t, dtype=torch.float32, device="cuda") for t in (xyz, scale, rot, sh_0, sh_rest, opacity)]
    n = tensors[0].shape[-1]
    if keep_frac < 1.0:
        g = torch.Generator(device="cuda").manual_seed(0)
        idx = torch.randperm(n, device="cuda", generator=g)[: int(n * keep_frac)]
        tensors = [t[..., idx].contiguous() for t in tensors]
    tensors = scene.point.spatial_refine(False, None, *tensors)
    tensors = scene.cluster.cluster_points(cluster_size, *tensors)
    return [torch.nn.Parameter(t.contiguous()) for t in tensors]


def ev():
    return torch.cuda.Event(enable_timing=True)


def time_scale(params, opt, loader_items, s, n_iters, warmup, op, pp, sh_degree):
    xyz, scale, rot, sh_0, sh_rest, opacity = params
    with torch.no_grad():  # as in the trainer, AABBs are computed once per spatial refine, not per iteration
        cluster_origin, cluster_extend = scene.cluster.get_cluster_AABB(xyz, scale.exp(), torch.nn.functional.normalize(rot, dim=0))
    stages = {"gt_downsample": [], "render_fwd": [], "loss_bwd": [], "opt_step": [], "total": []}
    for it in range(warmup + n_iters):
        view_matrix, proj_matrix, frustumplane, gt = loader_items[it % len(loader_items)]
        e0, ed, e1, e2, e3 = ev(), ev(), ev(), ev(), ev()
        e0.record()
        gt_s = gt
        if s > 1:  # same per-iteration GT resize as the trainer
            gt_s = torch.nn.functional.interpolate(gt, scale_factor=1.0 / s, mode="bilinear",
                                                   recompute_scale_factor=True, antialias=True)
        ed.record()
        visible_chunkid, cx, cs, cr, c0, cre, co = render.render_preprocess(
            cluster_origin, cluster_extend, frustumplane, xyz, scale, rot, sh_0, sh_rest, opacity, op, pp)
        img, _, _, _, primitive_visible = render.render(view_matrix, proj_matrix, cx, cs, cr, c0, cre, co,
                                                        sh_degree, gt_s.shape[2:], pp)
        e1.record()
        img_b = img.unsqueeze(0)
        loss = 0.8 * torch.abs(img_b - gt_s).mean() + 0.2 * (1 - fused_ssim.fused_ssim(img_b, gt_s))
        loss.backward()
        e2.record()
        opt.step(visible_chunkid, primitive_visible)
        opt.zero_grad(set_to_none=True)
        e3.record()
        torch.cuda.synchronize()
        if it >= warmup:
            stages["gt_downsample"].append(e0.elapsed_time(ed))
            stages["render_fwd"].append(ed.elapsed_time(e1))
            stages["loss_bwd"].append(e1.elapsed_time(e2))
            stages["opt_step"].append(e2.elapsed_time(e3))
            stages["total"].append(e0.elapsed_time(e3))
    return {k: float(np.median(v)) for k, v in stages.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("-i", "--images", default="images_4")
    parser.add_argument("--ply", required=True)
    parser.add_argument("--scales", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--fracs", nargs="+", type=float, default=[0.1, 0.3, 1.0])
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    lp, op, pp, dp = deadlinedino.config.get_default_arg()
    sh_degree = 3
    cameras_info, frames, _, _ = io_manager.load_colmap_result(args.source_path, args.images)
    for f in frames:
        f.load_image(-1)
    train_frames = [f for i, f in enumerate(frames) if i % 8 != 0]
    ds = CameraFrameDataset(cameras_info, train_frames, -1, True)
    items = []
    for i in range(len(ds)):
        v, p, fp, img, _ = ds[i]
        items.append((v.unsqueeze(0), p.unsqueeze(0), fp.unsqueeze(0), img.unsqueeze(0).float() / 255.0))
    _, norm_radius = ds.get_norm()
    H, W = items[0][3].shape[2:]

    results = {"scene": args.source_path, "full_res": [int(H), int(W)], "gpu": torch.cuda.get_device_name(0), "runs": []}
    for frac in args.fracs:
        params = build_params(args.ply, frac, pp.cluster_size, sh_degree)
        n_gs = int(params[0].shape[-1] * params[0].shape[-2])
        opt, _ = gs_optimizer.get_optimizer(*params, norm_radius, op, pp)
        for s in args.scales:
            r = time_scale(params, opt, items, s, args.iters, args.warmup, op, pp, sh_degree)
            r.update({"scale": s, "n_gaussians": n_gs, "frac": frac})
            results["runs"].append(r)
            print(f"N={n_gs:>8d} scale=1/{s}: total {r['total']:.2f} ms "
                  f"(gt_resize {r['gt_downsample']:.2f}, render {r['render_fwd']:.2f}, loss+bwd {r['loss_bwd']:.2f}, opt {r['opt_step']:.2f})", flush=True)
        del params, opt
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
