# Referensi Paper & GitHub — Text-Conditioned World Model Warehouse Robot

**Project**: Text-Conditioned World Model for Visual Category-Aware Warehouse Robot  
**Last updated**: 2026-05-27

---

## Tier 1 — Langsung Relevan (Score 5/5)

### 1. LED-WM: Language-aware Encoder for Dreamer World Model (2025)
- **Paper**: https://arxiv.org/pdf/2511.22904
- **Code**: Belum tersedia
- **Relevansi**: Language conditioning langsung di DreamerV3 via cross-attention. Architecture persis seperti pipeline kita (RSSM + CLIP text embedding). Ini referensi utama untuk Person 4 wiring CLIP ke DreamerV3.
- **Baca**: Section 3 — cara grounding language ke latent RSSM.

### 2. DreamerNav: Learning-based Autonomous Navigation in Dynamic Indoor Environments (2025)
- **Paper**: https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1655171/full
- **Code**: Belum tersedia
- **Relevansi**: DreamerV3 + **Isaac Sim** + wheeled robot navigation. Simulator sama persis. Curriculum training + occupancy map auxiliary. Pipeline paling mirip secara end-to-end.

### 3. Mastering Diverse Domains through World Models — DreamerV3 (2023)
- **Paper**: https://arxiv.org/abs/2301.04104
- **Code (JAX official)**: https://github.com/danijar/dreamerv3
- **Code (PyTorch, dipakai Person 2)**: https://github.com/NM512/dreamerv3-torch
- **Relevansi**: Paper dasar DreamerV3. RSSM architecture, categorical latents, symlog normalization, fixed hyperparameters across 150+ tasks.

---

## Tier 2 — Sangat Relevan (Score 4/5)

### 4. RoboDreamer: Learning Compositional World Models for Robot Imagination (2024)
- **Paper**: https://proceedings.mlr.press/v235/zhou24f.html
- **Code**: Belum tersedia
- **Relevansi**: Language-conditioned world model untuk robotics. Faktoriasi instruksi ke primitif → generalisasi ke kombinasi objek-aksi baru. Ditest di RT-X robot dataset.

### 5. LS-Imagine: Long Short-Term Imagination with DreamerV3 (2024)
- **Paper**: https://arxiv.org/abs/2410.03618
- **Relevansi**: DreamerV3 + MineCLIP untuk long-horizon goal conditioning. Menggunakan CLIP embedding sebagai goal signal — pola sama dengan goal_emb kita. Ada code.

### 6. RLVR-World: Training World Models with Reinforcement Learning (2025)
- **Paper**: https://arxiv.org/abs/2505.13934
- **Code**: https://github.com/thuml/RLVR-World
- **Relevansi**: Training world models via RL dengan language goals. Bagus sebagai referensi reward signal design. Ada code.

### 7. Dreamwalker: Mental Planning for Continuous Vision-Language Navigation (2023)
- **Paper**: https://arxiv.org/abs/2308.07498
- **Relevansi**: World model yang plan di latent space sebelum eksekusi nyata. Relevan untuk planning delivery route.

---

## Tier 3 — Relevan Sebagian (Score 3/5)

### 8. LM-Nav: Robotic Navigation with Large Pre-Trained Models (2023)
- **Paper**: https://proceedings.mlr.press/v205/shah23b.html
- **Relevansi**: Kombinasi CLIP + GPT-3 untuk navigasi robot tanpa finetuning. CLIP dipakai grounding landmark dari bahasa.

### 9. TransDreamer: Reinforcement Learning with Transformer World Models (2022)
- **Paper**: https://arxiv.org/abs/2202.09481
- **Relevansi**: Ganti GRU di RSSM dengan transformer. Lebih baik untuk long-horizon memory — relevan jika robot perlu navigasi jarak jauh di warehouse.

### 10. VLA-MBPO: World Model-based RL for Vision-Language-Action Models (2026)
- **Paper**: https://arxiv.org/abs/2603.20607
- **Relevansi**: Model-based RL dengan language conditioning untuk robot manipulation. Multi-view consistency. Ditest di LIBERO + robot nyata.

---

## GitHub Repos — PyTorch DreamerV3

| Repo | Stars | Keterangan |
|------|-------|------------|
| https://github.com/NM512/dreamerv3-torch | 828 | Paling mature, **dipakai Person 2** |
| https://github.com/A-SHOJAEI/dreamerv3-robotic-control | - | Sudah dioptimasi untuk robot control |
| https://github.com/DrunkJin/dreamer-from-scratch | - | Paling bersih, educational, ada penjelasan symlog & twohot |
| https://github.com/danijar/dreamerv3 | 3K+ | Official JAX, maintained oleh author |

---

## Arsitektur Pipeline (dari LED-WM + DreamerNav)

```
RGB obs (64x64x3)
        │
        ▼
   CNN Encoder
        │
        ▼
   RSSM (h_t, z_t) ◄──── cross-attention ◄──── CLIP text embedding (512-dim)
        │                                              ▲
        ▼                                              │
   Actor-Critic                              "deliver heavy box to zone A"
        │
        ▼
   wheel velocity [left, right]
```

- **LED-WM** → cara fuse CLIP ke RSSM (Person 4)
- **DreamerNav** → cara jalanin DreamerV3 di Isaac Sim (Person 1, sudah done)
- **NM512/dreamerv3-torch** → base implementation (Person 2)

---

## Pembagian Baca per Person

| Person | Paper Prioritas |
|--------|----------------|
| Person 1 (Env) | DreamerNav (#2) |
| Person 2 (DreamerV3) | DreamerV3 (#3), LS-Imagine (#5), TransDreamer (#9) |
| Person 3 (YOLOv8) | RoboDreamer (#4) — compositional object reasoning |
| Person 4 (CLIP) | LED-WM (#1) — **wajib baca section 3** |
