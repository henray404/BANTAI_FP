# Isaac Lab Tutorial — Warehouse Environment
> Panduan untuk Person 1: Environment & Integration  
> Project: Text-Conditioned World Model untuk Warehouse Robot  
> Setup: Windows 11 · RTX 5050 · Isaac Lab 5.x · Python 3.11

---

## Daftar Isi

1. [Konsep Dasar Isaac Lab](#1-konsep-dasar-isaac-lab)
2. [Struktur Folder Isaac Lab](#2-struktur-folder-isaac-lab)
3. [Anatomi Script Isaac Lab](#3-anatomi-script-isaac-lab)
4. [Empat Config Class yang Harus Dibuat](#4-empat-config-class-yang-harus-dibuat)
5. [Cara Spawn Objek](#5-cara-spawn-objek)
6. [Cara Load Robot](#6-cara-load-robot)
7. [Template Warehouse Environment](#7-template-warehouse-environment)
8. [Cara Jalankan & Debug](#8-cara-jalankan--debug)
9. [Cheatsheet Command](#9-cheatsheet-command)
10. [Troubleshooting Error Umum](#10-troubleshooting-error-umum)

---

## 1. Konsep Dasar Isaac Lab

### Tiga hal yang harus selalu diingat

**1. AppLauncher selalu pertama**

Isaac Sim harus dinyalakan dulu sebelum import apapun.
Kalau urutannya salah → error.

```python
# BENAR — AppLauncher di atas semua
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Baru setelah ini boleh import lain
import torch
import isaaclab.envs.mdp as mdp
```

**2. Semua pakai @configclass**

Di Isaac Lab, objek tidak dibuat langsung — selalu lewat config dulu.

```python
# SALAH — jangan begini
robot = Robot(pos=[0,0,0])

# BENAR — selalu pakai config
@configclass
class MyRobotCfg:
    prim_path = "/World/Robot"
    pos = [0, 0, 0]
```

Kenapa? Supaya Isaac Lab bisa parallelkan ribuan environment sekaligus di GPU.

**3. Simulation step manual**

Tidak ada loop otomatis. Kamu yang kontrol kapan simulasi maju.

```python
while simulation_app.is_running():
    sim.step()    # maju 1 langkah waktu
    sim.render()  # render ke viewport
```

---

## 2. Struktur Folder Isaac Lab

```
C:\IsaacLab\
├── scripts\                    ← semua script yang bisa dijalankan
│   ├── tutorials\              ← tutorial resmi, pelajari urut
│   │   ├── 00_sim\             ← dasar: buat scene kosong, spawn objek
│   │   ├── 01_assets\          ← robot: load, gerakkan, baca sensor
│   │   └── 03_envs\            ← environment RL: obs, action, reward
│   ├── demos\                  ← demo robot yang sudah jadi
│   └── reinforcement_learning\ ← training script
│
├── source\
│   ├── isaaclab\               ← library utama Isaac Lab
│   │   └── isaaclab\
│   │       ├── envs\           ← base class environment
│   │       ├── assets\         ← robot, object loader
│   │       ├── sim\            ← simulator utils
│   │       └── managers\       ← obs, action, reward manager
│   │
│   └── isaaclab_assets\        ← robot yang sudah tersedia
│       └── isaaclab_assets\
│           └── robots\         ← config robot siap pakai
│               ├── unitree\    ← Go1, Go2, B2, dll
│               ├── franka\     ← Franka arm
│               └── jetbot\     ← wheeled robot (mirip AMR kita)
│
└── apps\                       ← launcher config (jangan diubah)
```

### Folder project kita nanti

```
C:\warehouse_robot\             ← folder project kita
├── warehouse_env.py            ← environment utama (kamu yang bikin)
├── warehouse_scene.py          ← scene config
├── test_env.py                 ← test script
└── README.md
```

---

## 3. Anatomi Script Isaac Lab

Setiap script Isaac Lab punya struktur yang sama.
Berikut template yang bisa kamu salin:

```python
# ─────────────────────────────────────────────────────
# BAGIAN 1: LAUNCH SIMULATOR (SELALU DI ATAS)
# ─────────────────────────────────────────────────────
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─────────────────────────────────────────────────────
# BAGIAN 2: IMPORT (setelah simulator jalan)
# ─────────────────────────────────────────────────────
import torch
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.utils import configclass

# ─────────────────────────────────────────────────────
# BAGIAN 3: DEFINISI CONFIG (scene, obs, action, event)
# ─────────────────────────────────────────────────────
@configclass
class MyEnvCfg(ManagerBasedEnvCfg):
    ...

# ─────────────────────────────────────────────────────
# BAGIAN 4: MAIN FUNCTION
# ─────────────────────────────────────────────────────
def main():
    env = ManagerBasedEnv(cfg=MyEnvCfg())
    obs, _ = env.reset()

    while simulation_app.is_running():
        action = ...
        obs, _ = env.step(action)

    env.close()

# ─────────────────────────────────────────────────────
# BAGIAN 5: ENTRY POINT
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
    simulation_app.close()
```

---

## 4. Empat Config Class yang Harus Dibuat

Setiap environment di Isaac Lab terdiri dari **4 config class**.
Ini adalah inti yang perlu kamu pahami.

```
ManagerBasedEnvCfg
├── SceneCfg        → apa saja yang ada di scene (robot, lantai, item)
├── ActionsCfg      → apa yang robot bisa lakukan
├── ObservationsCfg → apa yang robot bisa lihat/rasakan
└── EventCfg        → apa yang terjadi saat reset/startup
```

### 4.1 SceneCfg — Isi Scene

```python
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

@configclass
class WarehouseSceneCfg(InteractiveSceneCfg):
    # Ground plane (lantai)
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg()
    )

    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0)
    )

    # Robot AMR (akan diisi nanti)
    robot: ArticulationCfg = AMR_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    # {ENV_REGEX_NS} = placeholder untuk path tiap environment
    # Isaac Lab otomatis ganti ini untuk setiap env paralel
```

### 4.2 ActionsCfg — Aksi Robot

```python
@configclass
class ActionsCfg:
    # Untuk AMR beroda: kontrol kecepatan roda
    wheel_velocity = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["left_wheel_joint", "right_wheel_joint"],
        scale=1.0,
    )
```

`asset_name="robot"` harus sama dengan nama di SceneCfg.

### 4.3 ObservationsCfg — Pengamatan Robot

```python
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Posisi robot di world
        base_pos = ObsTerm(func=mdp.base_pos_w)
        # Kecepatan robot
        base_vel = ObsTerm(func=mdp.base_lin_vel_w)
        # Orientasi robot
        base_yaw = ObsTerm(func=mdp.base_euler_xyz)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
```

Setiap `ObsTerm` = satu baris angka yang robot "lihat".
`concatenate_terms = True` = semua digabung jadi satu tensor.

### 4.4 EventCfg — Reset & Randomisasi

```python
@configclass
class EventCfg:
    # Terjadi setiap episode reset — robot kembali ke posisi awal
    reset_robot = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",   # "reset" = tiap episode, "startup" = sekali saja
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.5, 0.5),   # random x dalam range ini
                "y": (-0.5, 0.5),   # random y
                "yaw": (-3.14, 3.14) # random orientasi
            },
            "velocity_range": {},
        }
    )
```

---

## 5. Cara Spawn Objek

Referensi: `scripts\tutorials\00_sim\spawn_prims.py`

### Spawn ground plane

```python
cfg = sim_utils.GroundPlaneCfg()
cfg.func("/World/GroundPlane", cfg)
```

### Spawn box berwarna

```python
cfg = sim_utils.CuboidCfg(
    size=(0.3, 0.3, 0.3),              # panjang, lebar, tinggi (meter)
    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    visual_material=sim_utils.PreviewSurfaceCfg(
        diffuse_color=(1.0, 0.0, 0.0)  # RGB: merah = fragile
    ),
)
cfg.func("/World/Items/fragile_box", cfg, translation=(1.0, 0.0, 0.15))
#                                                       x     y     z
```

### Warna untuk kategori item warehouse

```python
# Merah = fragile
diffuse_color=(1.0, 0.0, 0.0)

# Hijau = regular
diffuse_color=(0.0, 1.0, 0.0)

# Biru = heavy
diffuse_color=(0.0, 0.0, 1.0)

# Kuning = zone A di lantai
diffuse_color=(1.0, 1.0, 0.0)
```

### Spawn multiple objek dengan loop

```python
item_positions = [
    (1.0, 0.0, 0.15),   # item 1
    (2.0, 1.0, 0.15),   # item 2
    (3.0, -1.0, 0.15),  # item 3
]

for i, pos in enumerate(item_positions):
    cfg = sim_utils.CuboidCfg(
        size=(0.3, 0.3, 0.3),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.0, 0.0)
        ),
    )
    cfg.func(f"/World/Items/box_{i}", cfg, translation=pos)
```

---

## 6. Cara Load Robot

Referensi: `scripts\tutorials\01_assets\run_articulation.py`

### Pakai robot yang sudah ada di Isaac Lab

```python
# Jetbot — wheeled robot dengan kamera (mirip AMR kita)
from isaaclab_assets.robots.jetbot import JETBOT_CFG

robot_cfg = JETBOT_CFG.replace(prim_path="/World/Robot")
robot = Articulation(robot_cfg)
```

### Akses data robot

```python
# Posisi robot di world
pos = robot.data.root_pos_w        # shape: (num_envs, 3)

# Orientasi robot (quaternion)
quat = robot.data.root_quat_w      # shape: (num_envs, 4)

# Kecepatan linear
vel = robot.data.root_lin_vel_w    # shape: (num_envs, 3)

# Posisi tiap joint
joint_pos = robot.data.joint_pos   # shape: (num_envs, n_joints)
```

### Kasih perintah gerak ke robot

```python
# Kasih kecepatan ke roda
wheel_vel = torch.tensor([[1.0, 1.0]])  # kiri, kanan
robot.set_joint_velocity_target(wheel_vel)
```

---

## 7. Template Warehouse Environment

Ini template lengkap yang bisa kamu jadikan starting point.
Simpan sebagai `C:\warehouse_robot\warehouse_env.py`.

```python
# ─────────────────────────────────────────────────────────────────────
# warehouse_env.py — Template Warehouse Environment
# ─────────────────────────────────────────────────────────────────────

"""Launch Isaac Sim Simulator first."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Warehouse Robot Environment")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# Import robot — ganti ini dengan AMR kalau sudah ada
from isaaclab_assets.robots.jetbot import JETBOT_CFG


# ─────────────────────────────────────────────────────────────────────
# 1. SCENE CONFIG — apa yang ada di environment
# ─────────────────────────────────────────────────────────────────────
@configclass
class WarehouseSceneCfg(InteractiveSceneCfg):
    """Scene warehouse sederhana."""

    # Lantai
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg()
    )

    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 1.0))
    )

    # Robot AMR (Jetbot sebagai placeholder)
    robot: ArticulationCfg = JETBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )


# ─────────────────────────────────────────────────────────────────────
# 2. ACTIONS CONFIG — apa yang robot bisa lakukan
# ─────────────────────────────────────────────────────────────────────
@configclass
class ActionsCfg:
    """Aksi robot: kontrol kecepatan roda."""
    wheel_velocity = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["left_wheel_joint", "right_wheel_joint"],
        scale=1.0,
    )


# ─────────────────────────────────────────────────────────────────────
# 3. OBSERVATIONS CONFIG — apa yang robot lihat
# ─────────────────────────────────────────────────────────────────────
@configclass
class ObservationsCfg:
    """Observasi robot."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Posisi robot (x, y, z)
        base_pos = ObsTerm(func=mdp.base_pos_w)
        # Kecepatan linear (vx, vy, vz)
        base_vel = ObsTerm(func=mdp.base_lin_vel_w)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ─────────────────────────────────────────────────────────────────────
# 4. EVENTS CONFIG — apa yang terjadi saat reset
# ─────────────────────────────────────────────────────────────────────
@configclass
class EventCfg:
    """Reset robot ke posisi awal tiap episode."""
    reset_robot = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
            "velocity_range": {},
        }
    )


# ─────────────────────────────────────────────────────────────────────
# 5. MAIN ENV CONFIG — gabungkan semua
# ─────────────────────────────────────────────────────────────────────
@configclass
class WarehouseEnvCfg(ManagerBasedEnvCfg):
    """Config environment warehouse."""
    scene        = WarehouseSceneCfg(num_envs=4, env_spacing=4.0)
    actions      = ActionsCfg()
    observations = ObservationsCfg()
    events       = EventCfg()

    def __post_init__(self):
        self.decimation = 4          # update setiap 4 sim step
        self.sim.dt = 0.005          # 200Hz physics
        self.viewer.eye = [5.0, 5.0, 5.0]
        self.viewer.lookat = [0.0, 0.0, 0.0]


# ─────────────────────────────────────────────────────────────────────
# 6. MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────
def main():
    # Buat environment
    env_cfg = WarehouseEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedEnv(cfg=env_cfg)

    # Reset
    obs, _ = env.reset()
    print(f"[INFO] Environment siap. Obs shape: {obs['policy'].shape}")

    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # Reset setiap 200 step
            if count % 200 == 0:
                obs, _ = env.reset()
                print(f"[INFO] Reset. Step: {count}")

            # Aksi random (nanti diganti dengan policy)
            action = torch.randn_like(env.action_manager.action)
            obs, _ = env.step(action)

            # Print posisi robot env pertama
            pos = obs["policy"][0][:3]
            print(f"[Step {count}] Robot pos: {pos.numpy()}")

            count += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
```

### Cara jalankan template ini:

```powershell
cd C:\warehouse_robot
python warehouse_env.py --num_envs 4
```

---

## 8. Cara Jalankan & Debug

### Jalankan script

```powershell
# Dari folder IsaacLab
python C:\IsaacLab\scripts\tutorials\03_envs\create_cartpole_base_env.py --num_envs 4

# Jalankan headless (tanpa window, lebih cepat untuk training)
python warehouse_env.py --num_envs 4 --headless
```

### Navigasi viewport

| Aksi | Cara |
|------|------|
| Rotate view | Klik kanan + drag |
| Zoom | Scroll mouse |
| Pan | Middle click + drag |
| Focus ke objek | F (setelah klik objek di Stage) |
| Fly mode | Klik kanan tahan + WASD |

### Print untuk debug

```python
# Cek shape tensor
print(obs["policy"].shape)    # (num_envs, obs_dim)

# Cek nilai satu environment
print(obs["policy"][0])       # env pertama

# Cek posisi robot
print(env.scene["robot"].data.root_pos_w)
```

---

## 9. Cheatsheet Command

```powershell
# Aktifkan environment conda
conda activate isaaclab

# Jalankan tutorial
python C:\IsaacLab\scripts\tutorials\00_sim\create_empty.py
python C:\IsaacLab\scripts\tutorials\00_sim\spawn_prims.py
python C:\IsaacLab\scripts\tutorials\01_assets\run_articulation.py
python C:\IsaacLab\scripts\tutorials\03_envs\create_cartpole_base_env.py --num_envs 4

# Cek GPU
nvidia-smi

# Cek PyTorch GPU
python -c "import torch; print(torch.cuda.is_available())"

# List environment yang tersedia di Isaac Lab
python -c "import gymnasium; print(gymnasium.envs.registry.keys())"

# Stop script yang sedang jalan
Ctrl + C
```

---

## 10. Troubleshooting Error Umum

### Error: `conda not recognized`
```
Penyebab: PATH conda belum diset
Fix:
  conda init powershell
  → tutup dan buka PowerShell baru
```

### Error: `ImportError: DLL load failed (h5py)`
```
Penyebab: versi h5py tidak kompatibel
Fix:
  pip uninstall h5py -y
  pip install h5py==3.11.0
```

### Error: `No module named 'isaaclab'`
```
Penyebab: conda env belum aktif atau Isaac Lab belum install
Fix:
  conda activate isaaclab
  cd C:\IsaacLab
  .\isaaclab.bat --install
```

### Error: `Windows fatal exception: code 0xc0000139`
```
Penyebab: DLL conflict — biasanya h5py atau numpy versi salah
Fix:
  pip uninstall h5py numpy -y
  pip install h5py==3.11.0 numpy==1.26.4
```

### Viewport hitam / kosong
```
Penyebab: scene memang kosong (normal untuk create_empty.py)
           atau renderer belum init
Fix:
  - Tunggu 30 detik, biasanya muncul sendiri
  - Coba klik di viewport lalu tekan F
  - Cek apakah ada objek di Stage panel kanan
```

### Isaac Lab jalan sangat lambat / lag
```
Penyebab: terlalu banyak num_envs untuk VRAM 8GB
Fix:
  Kurangi --num_envs, mulai dari 4
  Untuk training nanti pakai --headless
```

---

## Langkah Selanjutnya

Setelah semua tutorial selesai, urutan pengerjaan warehouse environment:

```
[ ] Tutorial 00_sim selesai   → paham spawn objek
[ ] Tutorial 01_assets selesai → paham load robot
[ ] Tutorial 03_envs selesai  → paham structure env

[ ] Bikin WarehouseSceneCfg   → lantai + lighting + robot
[ ] Tambah item berwarna      → merah/hijau/biru di atas scene
[ ] Tambah zona tujuan        → plane berwarna di lantai
[ ] Define ActionsCfg         → wheel velocity untuk AMR
[ ] Define ObservationsCfg    → posisi + kamera
[ ] Define EventCfg           → reset posisi robot & item
[ ] Test dengan random policy → pastikan env.step() jalan
[ ] Wrap ke Gymnasium API     → siap dipakai Person 2, 3, 4
```

---

*Dibuat untuk project Final Project Deep Learning — Warehouse Robot*  
*Isaac Lab 5.x · Windows 11 · RTX 5050 · Python 3.11*
