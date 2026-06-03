# perception/detection/ — Object Detection (YOLOv8)

> **Owner:** P3 (Object Detection)
> **Deadline:** Minggu 5–6 (11–24 Juni) — dataset render + model trained + SLOPE integration

---

## Tujuan

Deteksi dan klasifikasi box per kategori (fragile/regular/heavy) dari onboard camera 64×64 RGB
menggunakan YOLOv8. Output dipakai untuk **Category-Aware SLOPE** reward shaping.

---

## Tugas Minggu 4 (4–10 Juni) — Persiapan (bisa mulai tanpa camera)

- [ ] Siapkan pipeline labeling dari size-coded boxes
- [ ] Rancang quantile reward head (SLOPE)
- [ ] Setup YOLOv8 environment (ultralytics)
- [ ] Tentukan label categories: `fragile` (21cm), `regular` (32cm), `heavy` (52cm)

## Tugas Minggu 5 (11–17 Juni) — Dataset + Training

- [ ] Render dataset dari env camera (butuh camera jalan — sudah RESOLVED ✅)
- [ ] Label per kategori box (size-coded: kecil/sedang/besar)
- [ ] Train YOLOv8 model
- [ ] Evaluasi detection accuracy

## Tugas Minggu 6 (18–24 Juni) — SLOPE Integration

- [ ] **Category-Aware SLOPE** (novelty): reward shaping per kategori
- [ ] SLOPE generic dulu, lalu kondisikan per kategori
- [ ] Integrasi ke training pipeline

---

## Info Scene (dari `env/warehouse_scene.py`)

### Box Specs (54 boxes total, 18 racks × 3 shelf levels)

| Kategori | Size | Warna (brown shade) | Massa | Zone Tujuan |
|---|---|---|---|---|
| **fragile** | 21 cm | Light brown (0.85, 0.70, 0.45) | 2 kg | zone_A (orange) |
| **regular** | 32 cm | Medium brown (0.70, 0.52, 0.28) | 6 kg | zone_B (cyan) |
| **heavy** | 52 cm | Dark brown (0.50, 0.37, 0.18) | 12 kg | zone_C (purple) |

### Camera Specs

| Parameter | Value |
|---|---|
| Resolution | 64 × 64 RGB |
| Mount | Front-top Ridgeback base (~0.55m height) |
| HFOV | ~60° (focal_length 18mm) |
| Format | float [0,1], CHW (3, 64, 64) |

### Dataset Rendering

```python
# Cara ambil frame dari env:
from env.warehouse_env import WarehouseGymEnv, WarehouseEnvCfg

cfg = WarehouseEnvCfg()
env = WarehouseGymEnv(cfg=cfg)
obs, _ = env.reset()
rgb = obs["pixels"]  # Tensor(1, 3, 64, 64), float [0,1]

# Atau via render():
frame = env.render()  # numpy uint8 (64, 64, 3) — HWC
```

---

## Struktur File yang Diharapkan

```
perception/detection/
├── README.md           # file ini
├── __init__.py
├── dataset/            # rendered + labeled dataset
│   ├── images/
│   └── labels/
├── render_dataset.py   # script render frames dari env
├── train.py            # YOLOv8 training script
├── model.py            # model wrapper / inference
├── slope.py            # SLOPE reward shaping (Category-Aware)
└── config.yaml         # detection hyperparameters
```

---

## Koordinasi

- **P1 (Henry):** env sudah bisa render camera — pakai `WarehouseGymEnv` untuk dataset
- **P2 (DreamerV3):** SLOPE output masuk sebagai auxiliary reward signal
- **P5 (Training):** integrasi SLOPE ke training loop
