# Understanding
Dino v3 - self-supervised learning (SSL). Comes from the Meta family. 

# DINOv3 builds main innovations

- Massive Scale (Data + Model):
  - The data → Trained on billions of diverse images (natural, aerial, synthetic, etc.).
  - The model → Larger Vision Transformers (ViTs) — ViT-L, ViT-g, ViT-H sizes.(Need to search what is ViTs)
- Gram Anchoring 🧷 (The Key Technical Innovation):
  - It uses the Gram matrix — a mathematical way to measure relationships between features — to keep the feature space rich and diverse during training. (need to read about it)
- Post-hoc Flexibility Tricks:
  - After training, they add some light adjustments that make DINOv3 adaptable without re-training.
  - Can be aligned with text embeddings (like CLIP) using simple linear mappings — no joint training required
# Why It Matters
- Outperforms other self-supervised models (e.g., MAE, SimCLR, SwAV) and even some supervised ones