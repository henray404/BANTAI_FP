# Research — Isaac Lab di Cloud + Tekan VRAM di bawah 8GB

*Generated: 2026-06-08 · Sumber: 12 · Confidence: High untuk angka resmi/forum, Medium untuk harga cloud (geser mingguan)*

> Konteks proyek: RTX 5050 8GB (Blackwell), task pick-place Franka camera-based RL, num_envs=1.
> Pertanyaan: (1) run Isaac Lab di cloud, (2) cara tekan VRAM, (3) apakah 8GB Blackwell sanggup.

---

## Executive Summary

8GB **di bawah spec resmi** Isaac Sim 5.1 (minimum **16GB VRAM / RTX 4080**). Bisa jalan untuk scene ringan + non-camera, tapi task camera-based manipulation (mirip punyamu) makan **~9.3–9.6GB di num_envs=1** menurut data resmi NVIDIA SkillGen → besar kemungkinan **OOM di 8GB**. Plus ada bug Blackwell aktif (TiledCamera hang, issue #4951) — workaround-nya pakai `Camera` standar, yang **kebetulan sudah kamu lakukan** (driver 580.88 + camera ON). Jadi 8GB-mu *barely* jalan sekarang, tapi headroom nyaris nol — sekali naik resolusi/box/env langsung mentok.

Rekomendasi praktis: **sewa cloud GPU RTX 4090 24GB** (~$0.30–0.40/jam vast.ai, ~$0.34–0.69 RunPod) untuk training serius. 24GB hilangkan semua kekhawatiran VRAM untuk 1–beberapa env. Lokal 8GB cukup buat dev/debug logic, bukan training panjang.

---

## 1. Cloud Options

### Tier GPU yang cocok Isaac Sim
Isaac Sim butuh **RTX GPU** (ray tracing). Cloud yang punya RTX consumer/datacenter:
- **RTX 4090 24GB** — sweet spot. Cukup untuk Isaac Lab camera RL num_envs kecil-menengah. Termurah.
- **RTX 5090 32GB** — Blackwell, lebih VRAM, tapi kena bug Blackwell yang sama (sm_120) kalau pakai TiledCamera di Isaac Sim 5.1.
- **A100 40/80GB** — kalau mau scale num_envs banyak. Overkill untuk 1 env.
- **L40S 48GB** — datacenter RTX, bagus untuk rendering banyak kamera.
- Hindari A100/H100 *hanya* kalau butuh banyak env paralel; untuk 1 env, 4090 lebih murah & cukup.

### Harga (per GPU-jam, Mei 2026 — verifikasi ulang, geser mingguan)

| GPU | Vast.ai (spot/market) | RunPod Community | RunPod Secure | Lambda |
|---|---|---|---|---|
| RTX 4090 24GB | **$0.27–0.45** | $0.34–0.39 | $0.69 | tidak ada |
| RTX 5090 32GB | ~$0.53 | — | $0.99 | tidak ada |
| L40S 48GB | ~$0.53 | — | $0.86 | — |
| A100 80GB | $0.67–1.10 | $1.19–1.64 | $1.89–2.21 | $1.29–2.49 |

Sumber harga: [RentGPU](https://rentgpu.org/articles/lambda-vs-runpod-vast-ai-comparison), [RunAIHome](https://runaihome.com/blog/cloud-gpu-pricing-runpod-vast-lambda-2026/), [Klymentiev](https://klymentiev.com/blog/runpod-vs-lambda-vs-vast), [TechPlained](https://www.techplained.com/best-gpu-cloud-ai-training).

### Pilih provider mana
- **Vast.ai** — termurah absolut (P2P marketplace). RTX 4090 ~$0.27–0.37/jam. Catatan: host bisa interrupt → **wajib checkpoint** training tiap 15–20 menit. Filter host reliability score ≥0.95. Cocok budget <$100/bulan.
- **RunPod Community** — UX paling bersih, template Docker, spin-up <30 detik, per-second billing. RTX 4090 ~$0.34/jam. Default paling aman buat solo/riset.
- **Lambda** — tidak punya RTX consumer (cuma A100/H100). Skip untuk kasus ini.
- Hyperscaler (AWS g5/g6, GCP, Azure) — 4–5× lebih mahal dari specialized cloud. AWS p5 8×H100 = $98/jam. Skip kecuali ada kredit kampus.

### Setup container/headless (Isaac Sim Docker)
- Image resmi: `nvcr.io/nvidia/isaac-sim` (NGC). Isaac Lab punya Dockerfile sendiri di repo.
- Wajib: GPU dengan driver NVIDIA + nvidia-container-toolkit di host.
- Headless: jalankan dengan `--headless --enable_cameras` (camera RL butuh enable_cameras eksplisit).
- EULA: set env `OMNI_KIT_ACCEPT_EULA=YES` kalau headless, else kit tak mau start.
- Asset caching: aktifkan (asset di AWS S3, loading lama tiap run kalau tak di-cache).
- ⚠️ Provider P2P (Vast.ai) — pilih host yang sudah expose Docker/CUDA 12.x; Isaac Sim image besar (~20GB+), perhitungkan storage + download time.

---

## 2. Teknik Tekan VRAM (Isaac Lab spesifik)

Urut dari paling berdampak:

1. **`--headless`** — matikan viewport GUI. Wajib di cloud/training. Catatan: headless `python.sh` tetap render viewport by default → set `disable_viewport_updates=True` di SimulationApp config.
2. **num_envs serendah mungkin** — penyebab OOM #1. Untuk 8GB lokal, num_envs=1. Rendering memory scale ~linear dengan env. (Solusi resmi NVIDIA untuk OOM: "reduce the number of environments".)
3. **Resolusi kamera kecil** — kamu sudah 64×64 (bagus, sudah minimal praktis). Satu image 800×600 32-bit = ~2MB; 64×64 jauh lebih kecil.
4. **TiledCamera, bukan Camera** — tiled render semua env dalam 1 pass GPU, jauh lebih hemat untuk banyak env (laptop RTX 4070 bisa >512 tiled cam). **TAPI** lihat §3 — di Blackwell ada bug, kamu terpaksa pakai Camera standar.
5. **headless.rendering.kit flags** — matikan fitur RTX mahal (file `isaaclab.python.headless.rendering.kit`):
   ```kit
   rtx.translucency.enabled = false
   rtx.reflections.enabled = false
   rtx.indirectDiffuse.enabled = false
   rtx.raytracing.cached.enabled = false
   rtx.ambientOcclusion.enabled = false
   rtx.directLighting.sampledLighting.samplesPerPixel = 1
   renderer.multiGpu.maxGpuCount = 1
   ```
6. **DLSS mode = Performance (0)** — kurangi VRAM + render time, turunkan kualitas. Default "Auto" bisa pilih Quality di resolusi rendah → boros.
7. **Texture Streaming Budget** — turunkan dari default 0.6 (60% GPU mem). Set `/rtx-transient/resourcemanager/texturestreaming/memoryBudget`. (Catatan: matikan total texture streaming malah NAIKKAN VRAM — turunkan budget-nya, jangan disable.)
8. **Scene lebih ringan** — kurangi objek. Kamu sudah turun 54→18 box (bagus). Tiap rigid body + collision + texture makan VRAM.
9. **GPU pipeline, bukan CPU** — pastikan physics di GPU (lebih cepat); tapi kalau VRAM mentok, `--device cpu` untuk physics bisa offload sebagian (SkillGen contoh pakai `--device cpu` untuk kurangi beban).
10. **Kurangi data_types per kamera** — kalau cuma butuh RGB, jangan render depth+segmentation sekaligus.
11. **Physics substeps / decimation** — kamu decimation=20. Substep tinggi = lebih banyak state, tapi dampak VRAM kecil dibanding rendering.

### Tool ukur kapasitas
`scripts/benchmarks/benchmark_cameras.py` + install `pynvml` → `--autotune` cari num_envs/num_cameras maksimum sampai threshold GPU mem. Pakai ini untuk ukur batas pasti 8GB-mu sebelum training.
[Docs: estimate how many cameras](https://isaac-sim.github.io/IsaacLab/main/source/how-to/estimate_how_many_cameras_can_run.html).

---

## 3. Bisakah RTX 5050 8GB (Blackwell) untuk camera-based RL?

**Verdict: secara teknis jalan untuk kasusmu sekarang, tapi di bawah spec & headroom ~nol.**

### Fakta keras
- **Spec resmi Isaac Sim 5.1: minimum 16GB VRAM (RTX 4080).** 8GB tidak ada di tabel = di bawah supported. "GPUs with less than 16GB VRAM may be insufficient to run a complex scene." [Requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
- **Tapi** minimum = "GPU terendah yang dites NVIDIA", bukan hard limit teknis. Isaac Sim *lama* sebut RTX 3070 8GB sebagai minimum. Jadi 8GB bukan mustahil — cuma sempit. [Discussion #423](https://github.com/isaac-sim/IsaacSim/discussions/423)
- **Data konkret bahaya:** NVIDIA SkillGen (Franka cube-stacking, mirip task-mu) = **~9.3–9.6GB di num_envs=1**. Minimum rekomendasi SkillGen **≥24GB**. Task manipulasi camera-based di 1 env sudah >8GB. [SkillGen docs](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/skillgen.html)
- **Bug Blackwell aktif:** TiledCamera **hang selamanya** (100% CPU, no output) di RTX 5090 sm_120, Isaac Sim 5.1, issue **OPEN**. Akar: `omni.replicator` tiled rendering di Blackwell. **Workaround resmi reporter: pakai `Camera` standar, bukan TiledCamera** (RGBA → slice `[...,:3]`). [Issue #4951](https://github.com/isaac-sim/IsaacLab/issues/4951)

### Relevan ke proyekmu
Kamu **sudah** kena & lewati varian masalah Blackwell ini (camera SDP crash → fix driver downgrade 580.88, pakai Camera bukan TiledCamera, `test_env.py` ALL PASS camera ON). Jadi 8GB-mu jalan **sekarang** karena: scene ringan (18 box), 64×64, 1 env, Camera standar. Tapi:
- Naik num_envs >1 → hampir pasti OOM.
- Active arm IK + grasp attach + replay buffer pixel (500K frame 64×64×3 ≈ 24GB di disk/memmap, sebagian di VRAM) → tekanan tambahan.
- DreamerV3 (P2) + actor-critic (P3) di GPU yang sama saat training → model network + imajinasi rollout rebut VRAM dengan sim.

**Kesimpulan:** 8GB cukup untuk **dev env + verifikasi logic + debug** (yang sedang kamu kerjakan, num_envs=1). **Tidak cukup** untuk training DreamerV3 camera-based serius bareng sim di mesin yang sama. Saat masuk fase training (P2/P3 mulai), **pindah ke cloud RTX 4090 24GB** atau A100.

---

## Key Takeaways (actionable)

1. **Lokal 8GB = dev/debug only.** Lanjut pakai untuk migrasi env + `test_env.py` num_envs=1. Jangan paksa training penuh.
2. **Training → cloud RTX 4090 24GB.** RunPod Community (~$0.34/jam, UX enak) untuk run stabil; Vast.ai (~$0.30/jam) kalau budget ketat + checkpoint rajin. 24GB = aman total untuk 1–beberapa env.
3. **Pakai `Camera` bukan `TiledCamera`** selama di Blackwell (5050/5090) sampai issue #4951 fix. Kamu sudah benar.
4. **Ukur batas dulu:** jalankan `benchmark_cameras.py --autotune` + pynvml untuk tahu num_envs maks 8GB-mu sebelum buang waktu.
5. **Hemat VRAM lokal:** `--headless --enable_cameras`, `disable_viewport_updates=True`, DLSS Performance, turunkan texture streaming budget, matikan rtx reflections/translucency/AO via kit flags.
6. **Cloud bukan Blackwell:** sewa RTX 4090 (Ada, bukan sm_120) → hindari sekaligus bug Blackwell DAN keterbatasan VRAM. Dobel untung.
7. **Budget estimasi:** training eksperimen 5 konfig × 3 seed, anggap ~200–400 GPU-jam total → ~$70–160 di Vast.ai 4090, ~$70–140 di RunPod Community. Murah.

---

## Sumber
1. [Isaac Sim 5.1 Requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html) — min 16GB/RTX 4080, 8GB di bawah spec.
2. [IsaacLab Issue #4951](https://github.com/isaac-sim/IsaacLab/issues/4951) — TiledCamera hang di Blackwell sm_120, workaround Camera standar.
3. [SkillGen docs](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/skillgen.html) — Franka manip ~9.3GB/1env, rekomendasi ≥24GB.
4. [Estimate cameras / benchmark_cameras.py](https://isaac-sim.github.io/IsaacLab/main/source/how-to/estimate_how_many_cameras_can_run.html) — autotune kapasitas kamera.
5. [Camera core concepts](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/camera.html) — tiled rendering, bandwidth, 512 cam di 4090.
6. [Isaac Sim Performance Handbook](https://docs.robotsfan.com/isaacsim/5.0.0/reference_material/sim_performance_optimization_handbook.html) — texture streaming budget, DLSS, disable viewport.
7. [headless.rendering.kit](https://github.com/isaac-sim/IsaacLab/blob/d94504bc/apps/isaacsim_4_5/isaaclab.python.headless.rendering.kit) — rtx flags hemat VRAM.
8. [IsaacLab Issue #604](https://github.com/isaac-sim/IsaacLab/issues/604) — scaling env dengan kamera, breakpoint.
9. [Computation cost training cameras (forum)](https://forums.developer.nvidia.com/t/isaaclab-computation-cost-of-training-with-cameras/359423) — RTX 4070 cuma 64 env tanpa render.
10. [RentGPU](https://rentgpu.org/articles/lambda-vs-runpod-vast-ai-comparison), [RunAIHome](https://runaihome.com/blog/cloud-gpu-pricing-runpod-vast-lambda-2026/), [Klymentiev](https://klymentiev.com/blog/runpod-vs-lambda-vs-vast), [TechPlained](https://www.techplained.com/best-gpu-cloud-ai-training) — harga cloud GPU 2026.
11. [torchrl IsaacLab guide](https://docs.pytorch.org/rl/stable/reference/generated/knowledge_base/ISAACLAB.html) — TiledCamera setup, enable_cameras, num_envs.
12. [IsaacSim Discussion #423](https://github.com/isaac-sim/IsaacSim/discussions/423) — minimum spec = lowest tested, bukan hard limit.

## Methodology
4 query (cloud pricing, VRAM reduction, 8GB requirements, Blackwell OOM) + 2 deep-fetch (req resmi, issue #4951). ~12 sumber unik. Gap: harga cloud geser mingguan — verifikasi live sebelum sewa. Angka VRAM proyekmu belum diukur langsung — pakai `benchmark_cameras.py` untuk pasti.
