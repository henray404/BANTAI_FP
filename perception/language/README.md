# perception/language/ — Language Encoder (CLIP)

> **Owner:** P4 (Language Encoder)
> **Deadline:** Minggu 5 (11–17 Juni) — `goal_embedding()` wired + Visual HER design ready

---

## Tujuan

Wire frozen CLIP (ViT-B/32) text embedding ke `obs["goal_emb"]` di environment,
menggantikan placeholder zeros. Lalu implementasi **Visual HER** (novelty contribution).

---

## Tugas Minggu 4 (4–10 Juni) — CLIP Wiring (✅ bisa mulai tanpa nunggu camera)

- [ ] Implement `goal_embedding()` dengan CLIP frozen ViT-B/32
- [ ] Replace placeholder zeros di `env/warehouse_env.py` → fungsi `goal_embedding()` (line 89–91)
- [ ] Text instructions per zone (lihat mapping di bawah)
- [ ] Projection layer: 512-dim → 64-dim (referensi: LED-WM Section 3, lihat `docs/research/referensi.md`)
- [ ] Rancang relabel logic untuk Visual HER

## Tugas Minggu 5 (11–17 Juni) — Integration

- [ ] Integrasi CLIP embedding ke training loop (koordinasi P2 + P5)
- [ ] Verifikasi embedding quality (cosine similarity antar instruksi)

## Tugas Minggu 6 (18–24 Juni) — Visual HER (Novelty)

- [ ] **Visual HER**: relabel episode gagal berbasis kategori yang ter-approach
- [ ] Integrasi ke replay buffer / training pipeline

---

## Interface — Apa yang Perlu Diubah

### File: `env/warehouse_env.py` → fungsi `goal_embedding()`

```python
# CURRENT (line 89-91): placeholder zeros
def goal_embedding(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return placeholder goal embedding (zeros). Filled by Person 4 with CLIP."""
    return torch.zeros(env.num_envs, GOAL_EMB_DIM, device=env.device)

# TARGET: CLIP frozen embedding
def goal_embedding(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return CLIP text embedding for the current goal zone instruction."""
    # env.goal_pos → determine which zone → get text → CLIP encode
    # Return: Tensor(num_envs, 512)
    ...
```

### Text Instructions per Zone

| Zone | Color | Category | Text Instruction |
|---|---|---|---|
| zone_A | orange | fragile | "deliver small box to orange zone" |
| zone_B | cyan | regular | "deliver medium box to cyan zone" |
| zone_C | purple | heavy | "deliver large box to purple zone" |

Zone ↔ category mapping: `env/warehouse_scene.py` → `ZONE_ITEM_MAP`

### Projection Note

CLIP output = 512-dim. Terlalu besar untuk langsung masuk RSSM head.
Project ke **64-dim** sebelum feed ke policy head.
Referensi: LED-WM (2025) Section 3. Lihat `docs/research/referensi.md`.

---

## Struktur File yang Diharapkan

```
perception/language/
├── README.md           # file ini
├── __init__.py
├── clip_encoder.py     # CLIP frozen encoder (ViT-B/32) + caching
├── projection.py       # 512 → 64 dim linear projection
├── visual_her.py       # Visual HER relabeling logic
├── instructions.py     # text instruction templates per zone/category
└── config.yaml         # CLIP model name, projection dim, etc.
```

---

## Koordinasi

- **P1 (Henry):** modifikasi `goal_embedding()` di `env/warehouse_env.py` — diskusi dulu sebelum ubah
- **P2 (DreamerV3):** `goal_emb` masuk RSSM input — saat ini zeros, nanti 512-dim (or 64-dim projected)
- **P5 (Training):** Visual HER masuk replay buffer — koordinasi interface `buffer.relabel()`
