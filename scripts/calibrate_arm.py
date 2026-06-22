# calibrate_arm.py — headless FK reachability/capability map for the Ridgeback-Franka arm.
#
# No ML: arm kinematics is deterministic, so we DON'T learn a model — we sample it. This samples
# arm joint configs within their (soft) limits, reads the EE via forward kinematics, and records
# the reach radius (shoulder->hand) + manipulability = sqrt(det(Jp·Jpᵀ)) (the singularity sensor).
# It aggregates that into r_min/r_max for the radial workspace clamp (drive_env_v2 --reach_r/rmin)
# plus a manip-vs-radius table. --optimize_home runs a tiny CEM search for a home pose that reaches
# a forward-down target with high manipulability (a principled replacement for guessing joint angles).
#
# Headless, NO camera/render -> fast on the Blackwell RTX 5050 (no SDP path). This replaces the
# slow manual teleop sweep (drive_env_v2 --calib): thousands of samples in ~minutes, objective.
#
# AppLauncher owned here (see bugs_errors/2026-05-15_double-applaunch-crash.md).
#
# Usage:
#   conda activate isaaclab
#   python scripts/calibrate_arm.py --samples 4000
#   python scripts/calibrate_arm.py --optimize_home --target 0.55 0.0 0.25

"""Headless FK reachability + manipulability map for the Ridgeback-Franka arm (no ML)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="FK reachability/manipulability calibration")
parser.add_argument("--samples", type=int, default=4000, help="random arm configs to sample.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--joint_margin", type=float, default=0.95,
                    help="sample within this fraction of each joint's hard USD range (centred).")
parser.add_argument("--manip_keep", type=float, default=0.5,
                    help="for the r_min/r_max fit, keep samples with manip >= this fraction of the "
                         "max manip (the well-conditioned 'dexterous core').")
parser.add_argument("--out", type=str, default="calib/reach_map.csv",
                    help="CSV of all samples (radius, manip, ee xyz), relative to project root.")
parser.add_argument("--optimize_home", action="store_true",
                    help="also run a CEM search for a home pose reaching --target with high manip.")
parser.add_argument("--optimize_only", action="store_true",
                    help="skip the reachability map (no CSV) and ONLY run the home-pose CEM — fast "
                         "re-run when you just want to retune the home pose. Implies --optimize_home.")
parser.add_argument("--target", type=float, nargs=3, default=[0.55, 0.0, 0.25],
                    help="EE base-frame target (m) for --optimize_home: forward-down grasp-ready pose.")
parser.add_argument("--view", action="store_true",
                    help="show the Isaac GUI and build a colored REACHABLE-WORKSPACE point cloud "
                         "(one sphere per sampled EE, red=low / green=high manipulability) so you SEE "
                         "what it maps instead of running blind headless. Slower — use fewer --samples "
                         "(e.g. 1500). The window stays open at the end to orbit/inspect.")
parser.add_argument("--auto", action="store_true",
                    help="FULL auto-calibration: verify joint limits + per-joint sweep + global home "
                         "(CEM) + r_min/r_max, then WRITE --calib_out. The env reads that file on "
                         "start, so the robot uses the calibration immediately. Pair with --view to watch.")
parser.add_argument("--per_joint_steps", type=int, default=25,
                    help="samples per joint in the --auto per-joint sweep.")
parser.add_argument("--calib_out", type=str, default="calib/arm_calib.yaml",
                    help="where --auto writes the applied calibration (the env reads this on start).")
parser.add_argument("--envs", type=int, default=64,
                    help="parallel envs for --auto (batched FK): one sim.step evaluates this many "
                         "configs at once (~100x faster than 1-env). Lower if you hit 8GB VRAM OOM.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = not args_cli.view   # --view -> GUI; default headless (fast)
args_cli.enable_cameras = False         # no onboard camera -> no SDP path

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from env.warehouse_scene import RIDGEBACK_FRANKA_CFG  # noqa: E402

ARM_RE = "panda_joint.*"
EE_BODY = "panda_hand"
SHOULDER = "panda_link0"   # arm mount (rigid on the chassis) — reach radius is measured from here


@configclass
class _ArmSceneCfg(InteractiveSceneCfg):
    """Bare scene: ground + light + robot. No box/rack/camera (FK only)."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0)
    )
    robot = RIDGEBACK_FRANKA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _jac_index(robot, arm_ids, ee_idx) -> tuple[list[int], int]:
    """Resolve Jacobian column ids + EE row, handling the welded-but-floating base (see drive_robot)."""
    jac = robot.root_physx_view.get_jacobians()
    floating = jac.shape[-1] == robot.num_joints + 6
    jids = [i + 6 for i in arm_ids] if floating else list(arm_ids)
    eidx = ee_idx if floating else ee_idx - 1
    return jids, eidx


def _measure(sim, scene, robot, arm_ids, ee_idx, sh_idx, jids, eidx, q, render=False
             ) -> tuple[float, float, list]:
    """Write arm config q (1,7), settle, return (reach_radius_m, manipulability, ee_xyz)."""
    z = torch.zeros_like(q)
    robot.write_joint_state_to_sim(q, z, joint_ids=arm_ids)
    robot.set_joint_position_target(q, joint_ids=arm_ids)
    for _ in range(2):  # settle (arm gravity disabled -> holds at q)
        scene.write_data_to_sim()
        sim.step(render=render)
        scene.update(dt=sim.get_physics_dt())
    ee, sh = robot.data.body_pos_w[:, ee_idx], robot.data.body_pos_w[:, sh_idx]
    radius = float(torch.norm(ee - sh, dim=-1)[0])
    Jp = robot.root_physx_view.get_jacobians()[:, eidx, :, jids][:, 0:3, :]  # (1,3,7) position rows
    manip = float(torch.sqrt(torch.det(Jp @ Jp.transpose(1, 2)).clamp_min(0.0))[0])
    return radius, manip, ee[0].tolist()


def _fit_radii(rows: list, keep_frac: float) -> tuple[float, float]:
    """Fit r_min/r_max from the well-conditioned core (manip >= keep_frac * max manip)."""
    ms = torch.tensor([m for _, m, _ in rows])
    rs = torch.tensor([r for r, _, _ in rows])
    keep = ms >= keep_frac * float(ms.max())
    core = rs[keep] if int(keep.sum()) >= 20 else rs
    s = core.sort().values
    return float(s[int(0.02 * len(s))]), float(s[int(0.98 * len(s))])


def _cloud_markers() -> "VisualizationMarkers":
    """Sphere-cloud marker set with 3 colors (manipulability buckets: red/yellow/green)."""
    cfg = VisualizationMarkersCfg(prim_path="/Visuals/reach_cloud", markers={
        "lo": sim_utils.SphereCfg(radius=0.012,
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.15, 0.15))),
        "mid": sim_utils.SphereCfg(radius=0.012,
                                   visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.8, 0.1))),
        "hi": sim_utils.SphereCfg(radius=0.012,
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.8, 0.2))),
    })
    return VisualizationMarkers(cfg)


def _show_cloud(markers, rows: list) -> None:
    """Draw a marker at every sampled EE, colored by manipulability tercile. Best-effort (never raises)."""
    try:
        pts = torch.tensor([ee for _, _, ee in rows])
        ms = torch.tensor([m for _, m, _ in rows])
        lo_t, hi_t = ms.quantile(0.33), ms.quantile(0.66)
        idx = torch.where(ms > hi_t, 2, torch.where(ms > lo_t, 1, 0))
        markers.visualize(translations=pts, marker_indices=idx)
    except Exception as exc:  # noqa: BLE001 — viz is a nicety, don't kill the run
        print(f"[calib] cloud draw skipped ({exc})")


def _print_table(rows: list) -> None:
    """Print a manip-vs-radius table (0.05 m bins) so the dexterous band is visible."""
    print("radius_bin   n   manip(median)")
    b = 0.30
    while b < 1.20:
        sel = [m for r, m, _ in rows if b <= r < b + 0.05]
        if sel:
            sel.sort()
            print(f"  {b:.2f}-{b + 0.05:.2f}  {len(sel):5d}   {sel[len(sel) // 2]:.4f}")
        b += 0.05


def _optimize_home(measure_q, lo, hi, target, iters=15, pop=48, elite=8, init=None):
    """Tiny CEM: find arm q maximizing manip - 5*||ee - target|| within [lo, hi]. Returns (q, meta).

    `init` (1,7 or 7) seeds the search mean (e.g. the per-joint sweep result) for faster convergence.
    """
    lo1, hi1 = lo.squeeze(0), hi.squeeze(0)
    mean = init.squeeze(0).clone() if init is not None else 0.5 * (lo1 + hi1)
    std = 0.3 * (hi1 - lo1)
    tgt = torch.tensor(target)
    best_q, best_s, best_meta = None, -1e9, None
    for it in range(iters):
        pops = (mean + std * torch.randn(pop, mean.numel())).clamp(lo1, hi1)
        scores = []
        for i in range(pop):
            r, m, ee = measure_q(pops[i:i + 1])
            d = float(torch.norm(torch.tensor(ee) - tgt))
            s = m - 5.0 * d
            scores.append(s)
            if s > best_s:
                best_s, best_q, best_meta = s, pops[i].clone(), (r, m, d, ee)
        idx = torch.tensor(scores).topk(elite).indices
        e = pops[idx]
        mean, std = e.mean(0), e.std(0) + 1e-3
        print(f"  iter {it + 1}/{iters}: best manip={best_meta[1]:.4f} dist_to_target={best_meta[2]:.3f}m")
    return best_q, best_meta


def _verify_limits(robot, arm_ids) -> dict:
    """Per-joint [min, max] (rad) from the USD — exact sim limits (no encoder offset to calibrate)."""
    jl = robot.data.joint_pos_limits[:, arm_ids][0]   # (7,2)
    return {robot.joint_names[i]: [round(float(jl[k, 0]), 4), round(float(jl[k, 1]), 4)]
            for k, i in enumerate(arm_ids)}


def _score(measure_q, q, target) -> tuple[float, tuple]:
    """Scalar home-pose objective: manipulability - 5*||ee - target||. Higher = better."""
    r, m, ee = measure_q(q)
    d = float(torch.norm(torch.tensor(ee) - torch.tensor(target)))
    return m - 5.0 * d, (r, m, d, ee)


def _per_joint_sweep(score_q, lo, hi, neutral, steps) -> torch.Tensor:
    """Move EACH joint across its range (others at `neutral`); set each to its best-scoring value.

    This is the requested per-joint sweep ('gerakin tiap joint, cari titik terbaiknya'). Joints
    COUPLE, so this is only the SEED — the global home is refined by CEM afterward. With --view you
    see one joint articulate at a time. Returns (1,7) seed config.
    """
    base = neutral.squeeze(0).clone()
    lo1, hi1 = lo.squeeze(0), hi.squeeze(0)
    for j in range(base.numel()):
        grid = torch.linspace(float(lo1[j]), float(hi1[j]), steps)
        best_v, best_s = float(base[j]), -1e9
        for v in grid:
            q = base.clone()
            q[j] = v
            s = score_q(q.unsqueeze(0))
            if s > best_s:
                best_s, best_v = s, float(v)
        base[j] = best_v
        print(f"  joint{j + 1}: best @ {best_v:+.3f} rad (score {best_s:+.3f})")
    return base.unsqueeze(0)


def _write_calib(path: Path, home, limits: dict, r_min: float, r_max: float, meta: tuple) -> None:
    """Write the applied calibration YAML the env reads on start (auto-generated; don't hand-edit)."""
    names = [f"panda_joint{i}" for i in range(1, 8)]
    lines = ["# AUTO-GENERATED by scripts/calibrate_arm.py --auto. Re-run to regenerate; do not hand-edit.",
             "# Read on start by env/warehouse_scene (home pose) + scripts/drive_env_v2 (radial clamp).",
             "arm_calib:",
             "  home_joint_pos:"]
    lines += [f"    {nm}: {round(float(v), 4)}" for nm, v in zip(names, home.squeeze(0).tolist())]
    lines += [f"  reach_rmin: {round(r_min, 3)}",
              f"  reach_r: {round(r_max, 3)}",
              f"  manip_at_home: {round(meta[1], 5)}",
              f"  dist_to_target_m: {round(meta[2], 4)}",
              "  joint_limits_rad:"]
    lines += [f"    {nm}: [{lim[0]}, {lim[1]}]" for nm, lim in limits.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _measure_batch(sim, scene, robot, arm_ids, ee_idx, sh_idx, jids, eidx, q):
    """Batched FK: q (N,7) -> (radius (N,), manip (N,), ee_local (N,3)). One sim.step for ALL envs."""
    z = torch.zeros_like(q)
    robot.write_joint_state_to_sim(q, z, joint_ids=arm_ids)
    robot.set_joint_position_target(q, joint_ids=arm_ids)
    for _ in range(2):  # settle (arm gravity disabled -> holds at q)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt=sim.get_physics_dt())
    ee, sh = robot.data.body_pos_w[:, ee_idx], robot.data.body_pos_w[:, sh_idx]
    radius = torch.norm(ee - sh, dim=-1)                                       # (N,)
    Jp = robot.root_physx_view.get_jacobians()[:, eidx, :, jids][:, 0:3, :]    # (N,3,7)
    manip = torch.sqrt(torch.det(Jp @ Jp.transpose(1, 2)).clamp_min(0.0))      # (N,)
    return radius, manip, ee - scene.env_origins


def _fit_radii_t(radii: torch.Tensor, manips: torch.Tensor, keep_frac: float) -> tuple[float, float]:
    """Tensor version of _fit_radii: r_min/r_max from the well-conditioned core (manip >= keep*max)."""
    keep = manips >= keep_frac * float(manips.max())
    core = radii[keep] if int(keep.sum()) >= 20 else radii
    s = core.sort().values
    return float(s[int(0.02 * len(s))]), float(s[int(0.98 * len(s))])


def _run_auto_batched(sim, scene, robot, arm_ids, ee_idx, sh_idx, jids, eidx, target, n_map, pj_steps):
    """Batched auto pipeline (parallel envs): map -> per-joint sweep -> CEM home -> verify limits.

    One sim.step evaluates `scene.num_envs` configs, so the whole calibration is ~tens of steps
    instead of thousands. Headless only (no per-env render). Returns (home(1,7), meta, r_min, r_max, limits).
    """
    n = scene.num_envs
    dev = robot.device
    jl = robot.data.joint_pos_limits[:, arm_ids][0]                # (7,2), same across envs
    half = 0.5 * (jl[:, 1] - jl[:, 0]) * float(args_cli.joint_margin)
    mid = 0.5 * (jl[:, 0] + jl[:, 1])
    lo, hi = mid - half, mid + half                                # (7,)
    tgt = torch.tensor(target, device=dev)

    print(f"[auto] 1/4 reachability map (~{n_map} samples, batched x{n} envs)...")
    radii, manips = [], []
    done = 0
    while done < n_map:
        q = lo + torch.rand(n, 7, device=dev) * (hi - lo)
        r, m, _ = _measure_batch(sim, scene, robot, arm_ids, ee_idx, sh_idx, jids, eidx, q)
        radii.append(r)
        manips.append(m)
        done += n
    r_min, r_max = _fit_radii_t(torch.cat(radii), torch.cat(manips), 0.7)
    r_max *= 0.93   # stability margin: clamp INSIDE the dexterous core, not at its optimistic edge

    print("[auto] 2/4 per-joint sweep (each joint moved individually, batched)...")
    neutral = 0.5 * (lo + hi)
    seed = neutral.clone()
    steps = min(pj_steps, n)
    for j in range(7):
        grid = torch.linspace(float(lo[j]), float(hi[j]), steps, device=dev)
        q = neutral.unsqueeze(0).repeat(n, 1)
        q[:steps, j] = grid
        r, m, ee = _measure_batch(sim, scene, robot, arm_ids, ee_idx, sh_idx, jids, eidx, q)
        score = m[:steps] - 5.0 * torch.norm(ee[:steps] - tgt, dim=-1)
        b = int(score.argmax())
        seed[j] = grid[b]
        print(f"  joint{j + 1}: best @ {float(grid[b]):+.3f} rad (score {float(score[b]):+.3f})")

    print(f"[auto] 3/4 global home refine (CEM, pop={n}, seeded)...")
    mean, std = seed.clone(), 0.3 * (hi - lo)
    best_q, best_s, best_meta = None, -1e9, None
    for it in range(15):
        pops = (mean + std * torch.randn(n, 7, device=dev)).clamp(lo, hi)
        r, m, ee = _measure_batch(sim, scene, robot, arm_ids, ee_idx, sh_idx, jids, eidx, pops)
        d = torch.norm(ee - tgt, dim=-1)
        margin = (torch.minimum(pops - lo, hi - pops) / (hi - lo)).min(dim=1).values  # joint headroom
        scores = m - 5.0 * d + 0.5 * margin   # reach target + manip, BUT avoid a near-limit home pose
        b = int(scores.argmax())
        if float(scores[b]) > best_s:
            best_s, best_q = float(scores[b]), pops[b].clone()
            best_meta = (float(r[b]), float(m[b]), float(d[b]), ee[b].tolist())
        elite = pops[scores.topk(min(8, n)).indices]
        mean, std = elite.mean(0), elite.std(0) + 1e-3
        print(f"  iter {it + 1}/15: best manip={best_meta[1]:.4f} dist_to_target={best_meta[2]:.3f}m")

    print("[auto] 4/4 verifying joint limits (USD)...")
    return best_q.unsqueeze(0), best_meta, r_min, r_max, _verify_limits(robot, arm_ids)


def main() -> None:
    """Sample the arm workspace via FK, fit r_min/r_max, print the map, optionally optimize home."""
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    n_envs = args_cli.envs if args_cli.auto else 1   # --auto = batched parallel envs (fast)
    scene = InteractiveScene(_ArmSceneCfg(num_envs=n_envs, env_spacing=4.0))
    sim.reset()
    robot = scene["robot"]

    arm_ids, _ = robot.find_joints(ARM_RE, preserve_order=True)
    ee_idx = robot.body_names.index(EE_BODY)
    sh_idx = robot.body_names.index(SHOULDER)
    jids, eidx = _jac_index(robot, arm_ids, ee_idx)
    jlim = robot.data.joint_pos_limits[:, arm_ids]            # (1,7,2)
    mid = 0.5 * (jlim[..., 0] + jlim[..., 1])
    half = 0.5 * (jlim[..., 1] - jlim[..., 0]) * float(args_cli.joint_margin)
    lo, hi = mid - half, mid + half                          # (1,7) soft limits

    markers = None
    if args_cli.view:
        try:
            markers = _cloud_markers()
        except Exception as exc:  # noqa: BLE001 — markers are a nicety, never block the run
            print(f"[calib] marker cloud unavailable ({exc}); GUI view only.")
    rnd = bool(args_cli.view)

    torch.manual_seed(args_cli.seed)
    rows = None
    if args_cli.auto:
        home, meta, r_min, r_max, limits = _run_auto_batched(
            sim, scene, robot, arm_ids, ee_idx, sh_idx, jids, eidx,
            args_cli.target, min(args_cli.samples, 2000), args_cli.per_joint_steps)
        cpath = PROJECT_ROOT / args_cli.calib_out
        _write_calib(cpath, home, limits, r_min, r_max, meta)
        _names = [f"panda_joint{i}" for i in range(1, 8)]
        print(f"\n[auto] CALIBRATION WRITTEN -> {cpath} (the robot reads it on start):")
        for _nm, _v in zip(_names, home.squeeze(0).tolist()):
            print(f"    {_nm}: {_v:+.3f}")
        print(f"    reach_rmin={r_min:.2f} reach_r={r_max:.2f} manip_at_home={meta[1]:.4f} "
              f"dist_to_target={meta[2]:.3f}m")
        return   # batched auto done (skip the single-env map/optimize/view blocks)
    if not args_cli.optimize_only and not args_cli.auto:
        print(f"[calib] sampling {args_cli.samples} arm configs (FK)"
              f"{' + live cloud' if args_cli.view else ', headless'}...")
        rows = []
        for k in range(args_cli.samples):
            q = lo + torch.rand_like(lo) * (hi - lo)
            rows.append(_measure(sim, scene, robot, arm_ids, ee_idx, sh_idx, jids, eidx, q, render=rnd))
            if (k + 1) % 500 == 0:
                print(f"  {k + 1}/{args_cli.samples}")
            if markers is not None and (k + 1) % 200 == 0:
                _show_cloud(markers, rows)
        out = PROJECT_ROOT / args_cli.out
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["radius_m", "manipulability", "ee_x", "ee_y", "ee_z"])
            for r, m, ee in rows:
                w.writerow([round(r, 4), round(m, 6), *[round(v, 4) for v in ee]])
        r_min, r_max = _fit_radii(rows, float(args_cli.manip_keep))
        _print_table(rows)
        print(f"\n[calib] samples written -> {out}")
        print(f"[calib] FITTED radial clamp: --reach_rmin {r_min:.2f} --reach_r {r_max:.2f}  "
              f"(dexterous core, manip >= {args_cli.manip_keep:.0%} of max)")

    if (args_cli.optimize_home or args_cli.optimize_only) and not args_cli.auto:
        print(f"\n[calib] optimizing home pose toward EE target {args_cli.target} (CEM)...")
        measure_q = lambda q: _measure(sim, scene, robot, arm_ids, ee_idx, sh_idx, jids, eidx, q, render=rnd)
        q, meta = _optimize_home(measure_q, lo, hi, args_cli.target)
        names = [f"panda_joint{i}" for i in range(1, 8)]
        print("[calib] suggested home_joint_pos (radians) — VERIFY in sim before trusting:")
        for nm, v in zip(names, q.tolist()):
            print(f"    {nm}: {v:+.3f}")
        r, m, d, ee = meta
        print(f"[calib] reached ee={[round(v, 3) for v in ee]} radius={r:.3f} manip={m:.4f} "
              f"dist_to_target={d:.3f}m")

    if args_cli.view:
        if markers is not None and rows is not None:
            _show_cloud(markers, rows)
        print("[calib] GUI open: orbit the reachable cloud (red=low manip, green=high). "
              "Close the window to exit.")
        while simulation_app.is_running():
            sim.step(render=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
