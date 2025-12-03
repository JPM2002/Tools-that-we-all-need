import argparse
import math
import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from transformers import AutoImageProcessor, AutoModel
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# A few permissive/public images (Wikimedia, small sizes)
SAMPLE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/5/50/VAN_CAT.png",            # cat (PNG with alpha)
    "https://upload.wikimedia.org/wikipedia/commons/5/5f/Alaskan_Malamute.jpg",    # dog
    "https://upload.wikimedia.org/wikipedia/commons/9/94/Golden_Gate_Bridge_0002.jpg",  # bridge
    "https://upload.wikimedia.org/wikipedia/commons/7/7d/Bicycle_Arounder.jpg",    # bike
    "https://upload.wikimedia.org/wikipedia/commons/5/5a/Matterhorn_from_Domh%C3%BCtte_-_2.jpg", # mountain
]

def ensure_out_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_images_from_dir(images_dir: Path, max_images: int = 64) -> Tuple[List[Image.Image], List[Path]]:
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        paths.extend(sorted(images_dir.glob(ext)))
        # also search one level deep
        paths.extend(sorted(images_dir.glob(f"*/*{ext[1:]}")))
    images = []
    keep_paths = []
    for p in paths[:max_images]:
        try:
            images.append(Image.open(p).convert("RGB"))
            keep_paths.append(p)
        except Exception as e:
            print(f"[warn] could not load {p}: {e}")
    return images, keep_paths

def download_with_headers(url: str, out_path: Path):
    import urllib.request
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; MiniDINOv3/1.0; +https://example.org)"},
    )
    with urllib.request.urlopen(req) as resp, open(out_path, "wb") as f:
        f.write(resp.read())

def maybe_download_samples(images_dir: Path, max_images: int = 8):
    images = []
    paths = []
    try:
        for i, url in enumerate(SAMPLE_URLS[:max_images]):
            out = images_dir / f"sample_{i}.jpg"
            if not out.exists():
                download_with_headers(url, out)
            try:
                images.append(Image.open(out).convert("RGB"))
                paths.append(out)
            except Exception as e:
                print(f"[warn] downloaded but could not open {out}: {e}")
        return images, paths
    except Exception as e:
        print("[warn] Could not download sample images:", e)
        return [], []

def generate_synthetic_images(images_dir: Path, n: int = 8, size: Tuple[int, int] = (384, 384)):
    """Always available fallback: create varied synthetic pictures so the demo runs offline."""
    images = []
    paths = []
    W, H = size
    try:
        font = None
        # try to pick a default PIL font (platform-independent)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        for i in range(n):
            img = Image.new("RGB", size, (255, 255, 255))
            draw = ImageDraw.Draw(img)

            # pattern type
            mode = i % 4

            if mode == 0:
                # checkerboard
                tile = 32
                for y in range(0, H, tile):
                    for x in range(0, W, tile):
                        if ((x // tile) + (y // tile)) % 2 == 0:
                            draw.rectangle([x, y, x + tile, y + tile], fill=(220, 220, 220))
                draw.text((10, 10), "checkerboard", fill=(10, 10, 10), font=font)

            elif mode == 1:
                # horizontal gradient
                for x in range(W):
                    c = int(255 * x / (W - 1))
                    draw.line([(x, 0), (x, H)], fill=(c, 100, 255 - c))
                draw.text((10, 10), "gradient", fill=(0, 0, 0), font=font)

            elif mode == 2:
                # shapes
                draw.rectangle([40, 40, W - 40, H - 40], outline=(0, 0, 0), width=4)
                draw.ellipse([80, 80, 220, 220], outline=(255, 0, 0), width=6)
                draw.rectangle([W - 260, 80, W - 80, 220], outline=(0, 128, 255), width=6)
                draw.polygon([(W//2, 260), (W//2 - 60, H - 60), (W//2 + 60, H - 60)], outline=(0, 180, 0), width=6)
                draw.text((10, 10), "shapes", fill=(0, 0, 0), font=font)

            else:
                # stripes + noise
                for y in range(0, H, 16):
                    color = (200, 230, 255) if (y // 16) % 2 == 0 else (235, 245, 255)
                    draw.rectangle([0, y, W, y + 16], fill=color)
                # sprinkle noise
                rng = np.random.default_rng(i)
                pts = rng.integers(0, min(W, H), size=(500, 2))
                for (x, y) in pts:
                    draw.point((x, y), fill=(rng.integers(0, 255), rng.integers(0, 255), rng.integers(0, 255)))
                draw.text((10, 10), "stripes+noise", fill=(0, 0, 0), font=font)

            out = images_dir / f"synthetic_{i}.jpg"
            img.save(out, quality=95)
            images.append(img)
            paths.append(out)
        print(f"[info] generated {len(images)} synthetic images.")
    except Exception as e:
        print("[warn] Could not generate synthetic images:", e)
    return images, paths

def get_model_and_processor(model_name: str, device: str):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval().to(device)
    return processor, model

@torch.no_grad()
def forward_batch(model, processor, pil_images: List[Image.Image], device: str):
    inputs = processor(images=pil_images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    pooled = getattr(outputs, "pooler_output", None)         # (N, D)
    tokens = getattr(outputs, "last_hidden_state", None)     # (N, T, D) for ViT
    return pooled, tokens, inputs["pixel_values"]

def grid_from_tokens(tokens: torch.Tensor) -> int:
    seq = tokens.shape[1] - 1  # exclude CLS
    g = int(round(math.sqrt(seq)))
    assert g * g == seq, f"Token count without CLS is not a perfect square: {seq}"
    return g

def make_patch_heatmap(patch_feats: torch.Tensor, query_xy: Tuple[int, int]) -> np.ndarray:
    H, W, D = patch_feats.shape
    qy, qx = query_xy
    q = patch_feats[qy, qx].unsqueeze(0)  # (1, D)
    flat = patch_feats.view(-1, D)        # (H*W, D)
    sim = torch.nn.functional.cosine_similarity(q, flat, dim=-1)  # (H*W,)
    sim = sim.view(H, W)
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
    return sim.cpu().numpy()

def overlay_heatmap_on_image(orig_pil: Image.Image, heatmap: np.ndarray, out_path: Path, title: str = ""):
    H_img, W_img = orig_pil.size[1], orig_pil.size[0]
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((W_img, H_img), resample=Image.BILINEAR)

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(orig_pil)
    plt.imshow(heatmap_img, alpha=0.45, interpolation="bilinear")
    plt.axis("off")
    if title:
        plt.title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def pca_2d(X: np.ndarray) -> np.ndarray:
    if X.shape[0] < 2:
        return np.zeros((X.shape[0], 2))
    return PCA(n_components=2, random_state=0).fit_transform(X)

def cosine_nn(embeds: np.ndarray) -> List[List[int]]:
    S = cosine_similarity(embeds)
    np.fill_diagonal(S, -1)  # ignore self
    nn_idx = np.argsort(-S, axis=1)  # descending
    return nn_idx.tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/dinov3-vits16-pretrain-lvd1689m",
                        help="HF model id (e.g., facebook/dinov3-vitl16-pretrain-lvd1689m).")
    parser.add_argument("--images_dir", type=str, default="images")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--max_images", type=int, default=24)
    parser.add_argument("--center_only", action="store_true",
                        help="Use center patch for heatmap only.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_out_dir(out_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] using device: {device}")

    images_dir = Path(args.images_dir)
    images, image_paths = load_images_from_dir(images_dir, max_images=args.max_images)

    if not images:
        print("[info] images/ empty; trying to download a few permissive samples…")
        images, image_paths = maybe_download_samples(images_dir, max_images=min(args.max_images, 8))

    if not images:
        print("[info] download blocked or offline; generating synthetic images so the demo can run…")
        images, image_paths = generate_synthetic_images(images_dir, n=min(args.max_images, 8))

    if not images:
        print("[error] no images available. Put some in the images/ folder.")
        sys.exit(1)

    print(f"[info] loading model: {args.model}")
    processor, model = get_model_and_processor(args.model, device)

    # forward in small batches to avoid OOM
    batch_size = 8
    all_pooled = []
    all_tokens = []
    prepped_images = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        pooled, tokens, pix = forward_batch(model, processor, batch, device)
        prepped_images.append(pix.cpu())
        if pooled is not None:
            all_pooled.append(pooled.cpu())
        if tokens is not None:
            all_tokens.append(tokens.cpu())

    pooled = torch.cat(all_pooled, dim=0) if all_pooled else None
    tokens = torch.cat(all_tokens, dim=0) if all_tokens else None
    pixel_values = torch.cat(prepped_images, dim=0)

    # --- 1) Global embedding viz (PCA 2D)
    if pooled is not None:
        X = pooled.numpy()
        X2 = pca_2d(X)
        fig = plt.figure(figsize=(6, 5))
        plt.scatter(X2[:, 0], X2[:, 1])
        for i, p in enumerate(image_paths):
            plt.text(X2[i, 0], X2[i, 1], p.stem, fontsize=7)
        plt.title("DINOv3 global embeddings (PCA)")
        plt.tight_layout()
        fig.savefig(out_dir / "embeds_2d.png", dpi=200)
        plt.close(fig)
        print(f"[ok] saved 2D embedding plot -> {out_dir/'embeds_2d.png'}")

        # --- 2) Nearest neighbors (cosine) for each image
        idxs = cosine_nn(X)
        with open(out_dir / "nn_report.txt", "w", encoding="utf-8") as f:
            for i, row in enumerate(idxs):
                nbrs = [image_paths[j].name for j in row[:3]]
                f.write(f"{image_paths[i].name} -> {nbrs}\n")
        print(f"[ok] saved NN report -> {out_dir/'nn_report.txt'}")
    else:
        print("[warn] pooled (global) output not provided by this backbone.")

    # --- 3) Dense feature heatmaps (ViT only)
    if tokens is None:
        print("[warn] last_hidden_state (tokens) not available; dense heatmaps need a ViT backbone (e.g., vits16).")
        return

    N, T, D = tokens.shape
    grid = grid_from_tokens(tokens)  # infer H=W
    print(f"[info] token grid: {grid} x {grid}")

    for i, (img, path) in enumerate(zip(images, image_paths)):
        tok = tokens[i]  # (T, D)
        patch = tok[1:]  # drop CLS -> (H*W, D)
        patch = patch.view(grid, grid, D)  # (H, W, D)

        qy = grid // 2
        qx = grid // 2
        title_suffix = f"center ({qy},{qx})"

        heat = make_patch_heatmap(patch, (qy, qx))
        out_path = Path(out_dir / f"heatmap_{path.stem}.png")
        overlay_heatmap_on_image(img, heat, out_path, title=f"{path.name} – {title_suffix}")
        print(f"[ok] saved heatmap -> {out_path}")

if __name__ == "__main__":
    main()
