# INSTALL (Linux / Ubuntu) — Warehouse Robot Environment (BANTAI_FP)

From a clean Ubuntu machine to a running Isaac Lab warehouse env + DreamerV3 training.
Follow steps in order; each has a verify command — don't continue until it passes.

> **Windows version:** `docs/setup/INSTALL.md`. This file is the Linux twin.
> **2-GPU run procedure:** after setup, see `docs/setup/TRAINING_2GPU.md` (§5–7).
> **Target machine here:** desktop with 2× RTX 2080 Ti (Turing). No Blackwell driver pin.

---

## 0. Target versions

| Component | Version | Notes |
|---|---|---|
| OS | Ubuntu 22.04 LTS (or 20.04) | Isaac Sim 5.x officially supports these |
| GPU | NVIDIA RTX 2080 Ti ×2 (Turing sm_75) | 11 GB each; one training process per GPU |
| NVIDIA driver | CUDA 12.8-capable (e.g. 590.x) | `nvidia-smi` → `CUDA Version: 12.8`+ |
| Miniconda | latest | |
| Python | 3.11 | Isaac Sim 5.x requires exactly 3.11 |
| Isaac Sim | 5.1.0.0 | pip install from NVIDIA index |
| Isaac Lab | repo `main`, editable | cloned to `~/IsaacLab` |
| PyTorch | 2.7.0+cu128 | torchvision 0.22.0+cu128 (cu128 wheels include sm_75) |

Conda env name throughout: **`isaaclab`**.

---

## 1. System prerequisites

### 1.1 NVIDIA driver
Install a CUDA 12.8-capable driver (590.x is fine on Turing — the "avoid 591.x"
warning in CLAUDE.md is **Blackwell-only**, irrelevant here):
```bash
sudo ubuntu-drivers autoinstall      # or install a specific recent driver
sudo reboot
nvidia-smi                           # top-right must read CUDA Version: 12.8 or higher
```
`nvidia-smi` must list **both** 2080 Ti cards. If CUDA < 12.8, update the driver.

### 1.2 Build + render libs (Isaac Sim needs these)
```bash
sudo apt update
sudo apt install -y git build-essential libgl1 libglib2.0-0 libxrandr2 \
    libxinerama1 libxcursor1 libxi6 vulkan-tools
vulkaninfo | head -n 5     # must print a Vulkan instance, not an error
```

### 1.3 Miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh        # accept, let it init your shell
exec $SHELL                                    # reload shell
conda --version
```

---

## 2. Clone Isaac Lab

Isaac Lab lives **outside** this project, at `~/IsaacLab`.
```bash
cd ~
git clone https://github.com/isaac-sim/IsaacLab.git
cd ~/IsaacLab
test -f isaaclab.sh && echo "launcher OK"
```

---

## 3. Create the conda env

```bash
conda create -n isaaclab python=3.11 -y
conda activate isaaclab
python -m pip install --upgrade pip
python --version            # 3.11.x
```
> Every command below assumes **`conda activate isaaclab`** is active.

---

## 4. Install Isaac Sim 5.1

~10 GB download from NVIDIA's pip index:
```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
python -c "import isaacsim; print('isaacsim OK')"
```

---

## 5. Install PyTorch (CUDA 12.8 build)

cu128 wheels include Turing (sm_75) kernels — same build as the Windows reference:
```bash
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```
Expected: `2.7.0+cu128 True 2`. If `False`, the driver/CUDA is wrong — back to §1.1.

---

## 6. Install Isaac Lab extensions

```bash
cd ~/IsaacLab
./isaaclab.sh --install
python -c "import isaaclab; print('isaaclab', isaaclab.__version__)"
```

Smoke test (headless — no display needed):
```bash
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
```
Runs and exits cleanly = Isaac Sim + Isaac Lab healthy.

---

## 7. Clone this project

```bash
cd ~                                  # or wherever you keep projects
git clone https://github.com/henray404/BANTAI_FP.git
cd BANTAI_FP
git checkout feat/env-pickup-migration     # branch with the DreamerV3 pickup fix
```

---

## 8. Install the ML / DreamerV3 deps

```bash
pip install -r requirements-ml.txt
```
Pulls vendored NM512 deps (`gym==0.22`, `ruamel.yaml`, `einops`, `moviepy`) + baselines
(`stable-baselines3`, `tensorboard`, `wandb`).

**Guard the torch pin** (some deps may try to move it):
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# MUST still print: 2.7.0+cu128 True
```
If torch changed, reinstall §5 with `--no-deps`.

---

## 9. Copy the NVIDIA warehouse assets

Assets (~0.5 GB) are gitignored. You need the `Warehouse_NVD@10013` pack locally.
Adjust `SRC` to where your pack is; `DST` is this project's `assets/`.
```bash
SRC=~/Downloads/Warehouse_NVD@10013/Assets/DigitalTwin/Assets/Warehouse
DST=~/BANTAI_FP/assets

mkdir -p "$DST/Shelving/Racks/Rack_L"
cp "$SRC/Shelving/Racks/Rack_L/Rack_L01_PR_NVD_01.usd" "$DST/Shelving/Racks/Rack_L/"

cp -r "$SRC/Shipping/Cardboard_Boxes/Cube_A"                         "$DST/Shipping/Cardboard_Boxes/Cube_A"
cp -r "$SRC/Equipment/Forklifts/Forklift_A"                          "$DST/Equipment/Forklifts/Forklift_A"
cp -r "$SRC/Shipping/Cardboard_Boxes_on_Pallet/Pallet_Asm_A"        "$DST/Shipping/Cardboard_Boxes_on_Pallet/Pallet_Asm_A"
cp -r "$SRC/Shipping/Pallets/Plastic/Economy_A"                     "$DST/Shipping/Pallets/Plastic/Economy_A"
cp -r "$SRC/Safety/Cones/Heavy-Duty_Traffic"                        "$DST/Safety/Cones/Heavy-Duty_Traffic"
cp -r "$SRC/Safety/Floor_Signs/Warning_A"                           "$DST/Safety/Floor_Signs/Warning_A"

test -f "$DST/Shelving/Racks/Rack_L/Rack_L01_PR_NVD_01.usd" && echo "rack OK"
```
> External MDL materials aren't copied (1.2 GB SubUSDs). Geometry/collision load fine;
> surfaces render pink. Non-fatal for training.

---

## 10. Verify the project

```bash
# Pure CPU — no Isaac
pytest tests/test_layout_grid.py tests/test_obs_adapter.py -v

# DreamerV3 config + wrapper build (no GPU)
python -c "import sys; sys.path.insert(0,'.'); from models.dreamerv3.config import build_config; c=build_config(); print('task',c.task)"

# Full env contract (needs Isaac, 1 GPU)
python tests/test_env.py --num_envs 1
```
All green → ready to train.

---

## 11. Train DreamerV3

Single GPU first (confirm it clears prefill → `Simulate agent` → `[1000] ...`):
```bash
python scripts/train_dreamer.py --headless --steps 200000 \
  --logdir training/results/dreamerv3_seed0 2>&1 | tee training/results/run_seed0.log
```

Two seeds in parallel — one per GPU (VRAM is not pooled). Two terminals:
```bash
# Terminal A
CUDA_VISIBLE_DEVICES=0 python scripts/train_dreamer.py --headless --seed 0 --steps 200000 \
  --logdir training/results/dreamerv3_seed0 2>&1 | tee training/results/run_seed0.log

# Terminal B (start ~1 min after A so sims don't boot simultaneously)
CUDA_VISIBLE_DEVICES=1 python scripts/train_dreamer.py --headless --seed 1 --steps 200000 \
  --logdir training/results/dreamerv3_seed1 2>&1 | tee training/results/run_seed1.log
```
Verify both cards busy: `nvidia-smi` (a python process on GPU 0 and GPU 1).

Read results: `tensorboard --logdir training/results` → http://localhost:6006 .
What "good" looks like + metric meanings: `docs/setup/TRAINING_2GPU.md` §7.

---

## 12. Troubleshooting (Linux)

| Symptom | Cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` False | driver < CUDA 12.8 or wrong torch build | update driver (§1.1); reinstall torch cu128 (§5) |
| `vulkaninfo` errors / sim won't render | Vulkan/driver libs missing | reinstall driver; `sudo apt install vulkan-tools libvulkan1` |
| `ImportError: libGL.so.1` | render libs missing | `sudo apt install libgl1 libglib2.0-0` (§1.2) |
| `./isaaclab.sh: Permission denied` | not executable | `chmod +x isaaclab.sh` |
| Run B uses GPU 0 too | `CUDA_VISIBLE_DEVICES` not prefixed | put it inline, same line as `python ...` |
| Racks/boxes render pink | MDL materials not copied | expected, non-fatal (§9) |
| Zombie python after a run | Isaac `close()` hang | `pkill -9 python` between runs (not while training) |

---

## Reference
- Windows install: `docs/setup/INSTALL.md`
- 2-GPU run + reading results: `docs/setup/TRAINING_2GPU.md`
- Project map: `CLAUDE.md` · Env design: `docs/specs/environment.md`
- Official Isaac Lab pip install: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
