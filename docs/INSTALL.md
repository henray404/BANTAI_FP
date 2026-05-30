# INSTALL — Warehouse Robot Environment (BANTAI_FP)

Full setup tutorial: from a clean Windows machine to a running Isaac Lab warehouse env.
Follow steps in order. Each step has a verify command — do not continue until it passes.

> **Who this is for:** teammates (P2–P5) cloning this project for the first time.
> **Time:** ~60–90 min (most of it is the Isaac Sim download, ~10 GB).

---

## 0. Target versions (what you end up with)

These are the exact versions running on the reference machine. Match them — Isaac Sim is
version-sensitive and the RTX 50-series (Blackwell) needs the CUDA 12.8 PyTorch build.

| Component | Version | Notes |
|---|---|---|
| OS | Windows 11 | |
| GPU | NVIDIA RTX 5050 8 GB (Blackwell) | any RTX works; 8 GB drives `num_envs` |
| NVIDIA driver | CUDA 12.8-capable | `nvidia-smi` must show CUDA ≥ 12.8 |
| Miniconda | latest | conda package manager |
| Python | 3.11 | Isaac Sim 5.x requires 3.11 exactly |
| Isaac Sim | 5.1.0.0 | installed via NVIDIA pip index |
| Isaac Lab | 2.3.2 (repo) / pkg 0.54.3 | cloned to `C:\IsaacLab`, editable install |
| PyTorch | 2.7.0+cu128 | torchvision 0.22.0+cu128 |
| numpy | 1.26.0 | |
| gymnasium | 1.2.1 | |

Conda env name used throughout: **`isaaclab`**.

---

## 1. Prerequisites (install once)

### 1.1 NVIDIA driver
Download the latest **Game Ready / Studio** driver from nvidia.com for your GPU. After install,
open a terminal:

```powershell
nvidia-smi
```
Top-right must read `CUDA Version: 12.8` (or higher). If lower, update the driver — Blackwell
(RTX 50-series) will not run the sim correctly without it.

### 1.2 Git
Install Git for Windows: https://git-scm.com/download/win . Verify:
```powershell
git --version
```

### 1.3 Miniconda
Install Miniconda (Python 3.11) from https://docs.conda.io/en/latest/miniconda.html .
Use the **"Just Me"** install. After install, open a **new** "Anaconda Prompt" (or PowerShell with
conda initialized) and verify:
```powershell
conda --version
```

---

## 2. Clone Isaac Lab

Isaac Lab lives **outside** this project, at `C:\IsaacLab` (referenced by `CLAUDE.md`).

```powershell
cd C:\
git clone https://github.com/isaac-sim/IsaacLab.git
cd C:\IsaacLab
```

Verify the launcher script exists:
```powershell
Test-Path C:\IsaacLab\isaaclab.bat   # must print True
```

---

## 3. Create the conda environment

From the Isaac Lab repo root (`C:\IsaacLab`):

```powershell
conda create -n isaaclab python=3.11
conda activate isaaclab
python -m pip install --upgrade pip
```

Verify the prompt now shows `(isaaclab)` and:
```powershell
python --version    # Python 3.11.x
```

> Every command from here on assumes **`conda activate isaaclab`** is active.

---

## 4. Install Isaac Sim 5.1

Isaac Sim 5.x is installed as pip packages from NVIDIA's index (no separate Omniverse Launcher
needed). This downloads ~10 GB — be patient.

```powershell
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

Verify:
```powershell
python -c "import isaacsim; print('isaacsim OK')"
```

---

## 5. Install PyTorch (CUDA 12.8 build)

The RTX 5050 is Blackwell (compute capability sm_120). It needs the **cu128** PyTorch wheels —
older CUDA builds fail with "no kernel image available". Install the exact pinned versions:

```powershell
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA is visible to torch:
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expected: `2.7.0+cu128 True NVIDIA GeForce RTX 5050` (name varies by GPU). If `is_available()` is
`False`, your driver/CUDA is the problem — go back to step 1.1.

---

## 6. Install Isaac Lab extensions

From `C:\IsaacLab` with the env active:

```powershell
cd C:\IsaacLab
.\isaaclab.bat --install
```

This installs all `isaaclab*` extensions editable from `C:\IsaacLab\source\`. Verify:
```powershell
python -c "import isaaclab; print('isaaclab', isaaclab.__version__)"
```

### Smoke test (optional but recommended)
Confirm the simulator launches end-to-end before touching this project:
```powershell
cd C:\IsaacLab
.\isaaclab.bat -p scripts\tutorials\00_sim\create_empty.py
```
A window opens with an empty stage, then closes. If this works, Isaac Sim + Isaac Lab are healthy.

---

## 7. Clone this project

The project lives separately from Isaac Lab.

```powershell
cd C:\Users\<you>\Documents\KULIAH\sem_4   # or wherever you keep it
git clone https://github.com/henray404/BANTAI_FP.git
cd BANTAI_FP
```

---

## 8. Copy the NVIDIA warehouse assets

The 3D assets (~500 MB) are **not in git** (`assets/` is gitignored — too large). Copy them from
the NVIDIA Digital Twin pack. You need the pack `Warehouse_NVD@10013` downloaded locally.

> Adjust `$src` to wherever your pack is. `$dst` is this project's `assets/` folder.

```powershell
$src = "C:\Users\<you>\Downloads\Warehouse_NVD@10013\Assets\DigitalTwin\Assets\Warehouse"
$dst = "C:\Users\<you>\Documents\KULIAH\sem_4\BANTAI_FP\assets"

# Rack: copy only L01 (the folder has 18 variants ~700 MB; the env uses one)
New-Item -ItemType Directory -Force "$dst\Shelving\Racks\Rack_L" | Out-Null
Copy-Item "$src\Shelving\Racks\Rack_L\Rack_L01_PR_NVD_01.usd" "$dst\Shelving\Racks\Rack_L\" -Force

# Boxes (size-coded categories) — copy whole folder so materials/textures come along
Copy-Item "$src\Shipping\Cardboard_Boxes\Cube_A" "$dst\Shipping\Cardboard_Boxes\Cube_A" -Recurse -Force

# Props
Copy-Item "$src\Equipment\Forklifts\Forklift_A" "$dst\Equipment\Forklifts\Forklift_A" -Recurse -Force
Copy-Item "$src\Shipping\Cardboard_Boxes_on_Pallet\Pallet_Asm_A" "$dst\Shipping\Cardboard_Boxes_on_Pallet\Pallet_Asm_A" -Recurse -Force
Copy-Item "$src\Shipping\Pallets\Plastic\Economy_A" "$dst\Shipping\Pallets\Plastic\Economy_A" -Recurse -Force
Copy-Item "$src\Safety\Cones\Heavy-Duty_Traffic" "$dst\Safety\Cones\Heavy-Duty_Traffic" -Recurse -Force
Copy-Item "$src\Safety\Floor_Signs\Warning_A" "$dst\Safety\Floor_Signs\Warning_A" -Recurse -Force
```

Verify the rack and a box landed:
```powershell
Test-Path "$dst\Shelving\Racks\Rack_L\Rack_L01_PR_NVD_01.usd"                     # True
Test-Path "$dst\Shipping\Cardboard_Boxes\Cube_A\CubeBox_A03_21cm_PR_NVD_01.usd"   # True
```

> **Note:** rack/forklift external MDL materials are not copied (they live in a 1.2 GB SubUSDs
> folder). Geometry and collision load fine; surfaces render pink/default. Non-fatal for training.

---

## 9. Verify the project

### 9.1 Pure unit tests (no Isaac Sim — fast)
```powershell
cd C:\Users\<you>\Documents\KULIAH\sem_4\BANTAI_FP
pytest tests/test_layout_grid.py -v
```
All tests pass = the island/rack layout math is correct. This runs without a GPU.

### 9.2 Measure the rack (sets shelf height)
Rack_L01's exact size is unknown until measured. Run the standalone viewer:
```powershell
python asset_sandbox/scripts/explore_rack.py
```
Read the printed X/Y/Z footprint (meters). Then in `env/warehouse_scene.py`:
- set `RACK_SHELF_Z` to where boxes should sit on the shelf,
- set `ISLAND_RACK_DX` larger than the rack's X footprint (so racks in an island don't overlap).

### 9.3 Interface contract test (needs Isaac Sim)
```powershell
python tests/test_env.py --num_envs 1
```
Confirms the obs dict shapes, action space, reward, and termination match the contract in
`CLAUDE.md`.

---

## 10. Run the environment

```powershell
# Visual debug — windowed, 1 env (watch the robot)
python scripts/run_env.py --num_envs 1 --steps 99999

# Headless — 2 envs (for training; VRAM-safe on 8 GB)
python scripts/run_env.py --num_envs 2 --headless --steps 99999

# Visual inspection of the full warehouse scene
python asset_sandbox/scripts/explore_scene.py
```

If these run without crashing and you see the warehouse, **setup is complete.**

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` → False | driver < CUDA 12.8, or wrong torch build | update driver (step 1.1); reinstall torch with `--index-url .../cu128` (step 5) |
| "no kernel image is available for execution" | CPU/old-CUDA torch on Blackwell GPU | must be `torch==2.7.0+cu128` (step 5) |
| Sim crashes on camera / segfault at startup | `CameraCfg` crashes on RTX 50-series | env already uses `TiledCameraCfg` — don't switch back to `CameraCfg` |
| Racks/boxes appear 100× too large | NVIDIA assets authored in cm; Isaac Lab `UsdFileCfg` has no auto unit conversion | every USD prop uses `scale=(0.01, 0.01, 0.01)` — keep it |
| Out-of-memory / OOM during run | 8 GB VRAM + 30 m room + props | keep `--num_envs 2`; lower to 1 if needed; try 4 only after a clean OOM-free run |
| Racks/forklift render pink | external MDL materials not copied | expected, non-fatal (step 8 note) |
| `python -c "import isaacsim"` fails | step 4 incomplete or wrong env | `conda activate isaaclab`; rerun step 4 |
| `isaaclab.bat` not found | not in `C:\IsaacLab` | `cd C:\IsaacLab` first |

---

## Reference

- Isaac Lab pip install (official): https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
- Project context & file map: `CLAUDE.md`
- Environment design doc: `docs/environment.md`
- Known bugs: `bugs_errors/`
- Config reference: `configs/env_config.yaml`
