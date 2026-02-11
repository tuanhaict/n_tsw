#!/usr/bin/env python
"""
Gradient flow on synthetic 28x28 grayscale digits
=================================================

Tensor per sample: [ 784 image | d_pos positional | 1 index ]  (all learnable)

Generation pipeline
-------------------
1. For i in 0 .. n_train-1:
       render digit str(i % 10) into a 28x28 PIL image
2. Flatten to 784-D vector in [0,1]
3. Concatenate optional positional encoding + fixed index scalar
4. Shuffle the first 784 features with a permutation seeded by --perm_seed
5. Add small Gaussian noise and make the *whole* tensor nn.Parameter

Command-line flags (most used)
------------------------------
--n_train     number of samples (default 512)
--n_plot      images per grid  (default 16)
--pos_type    none | sinusoid | learned
--pos_dim     size of positional block
--perm_seed   seed controlling the initial shuffle of image dims
"""

from __future__ import annotations
import argparse, random, time, math, numpy as np, torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
import sys
sys.path.append("./src")          # ensure src/ot.py etc. are importable

# ---------- Wasserstein losses ----------
from src.ot import SW
from src.TW_concurrent_lines import TWConcurrentLines, generate_trees_frames
from src.n_tsw import NTWConcurrentLines

# ---------- helpers ---------------------
def ensure_dirs(*ds): [Path(d).mkdir(parents=True, exist_ok=True) for d in ds]

def sinusoid(idx: torch.Tensor, dim: int, base: float = 10000.0):
    """Return an (n, dim) sinusoidal positional encoding (even dim)."""
    assert dim % 2 == 0
    freqs = base ** (-torch.arange(0, dim, 2, device=idx.device) / dim)
    angles = idx.float().unsqueeze(1) * freqs
    return torch.cat([torch.sin(angles), torch.cos(angles)], 1)

def build_twd(d, device):
    nlines, L_total = 2, 1000
    ntrees = L_total // nlines
    twd = TWConcurrentLines(ntrees=ntrees, nlines=nlines,
                            mass_division="distance_based",
                            device=device, p=1, delta=0.1)
    return twd, nlines, L_total, ntrees
def build_ntwd(d, device, noisy_mode=None, lambda_=0.0, p_noise=2):
    nlines, L_total = 2, 1000
    ntrees = L_total // nlines
    twd = NTWConcurrentLines(ntrees=ntrees, nlines=nlines,
                            mass_division="distance_based",
                            device=device, p=1, delta=0.1, noisy_mode=noisy_mode, lambda_=lambda_, p_noise=p_noise)
    return twd, nlines, L_total, ntrees

# ---------- synthetic digit renderer -----
def render_digit(digit: int, size: int = 28) -> np.ndarray:
    """Return a size×size float32 array in [0,1] showing the digit."""
    from PIL import Image, ImageDraw, ImageFont

    img  = Image.new("L", (size, size), color=0)          # black
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # -- measure text size (textbbox works in Pillow ≥ 8.0) --------------
    bbox = draw.textbbox((0, 0), str(digit), font=font)   # (x0,y0,x1,y1)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # -- center the digit -----------------------------------------------
    pos = ((size - w) / 2, (size - h) / 2)
    draw.text(pos, str(digit), fill=255, font=font)

    return np.asarray(img, dtype=np.float32) / 255.0       # (size,size) in [0,1]


# ---------- grid saver -------------------
def save_grid(vecs784: torch.Tensor, path: str):
    """vecs784: (k,784) tensor in [0,1]; save a grayscale grid PNG."""
    k = vecs784.shape[0]
    imgs = vecs784.view(k, 28, 28).cpu().numpy()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, cols = (k + 3) // 4, min(4, k)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.0, rows*2.0), dpi=100)
    axes = axes.flatten() if rows * cols > 1 else [axes]
    for i, im in enumerate(imgs):
        axes[i].imshow(im, cmap="gray")
        axes[i].axis("off")
    for j in range(len(imgs), len(axes)):
        axes[j].axis("off")
    
    # Modify tight_layout to set all padding parameters to 0.
    # This includes padding between the figure edge and subplots (pad),
    # and padding between adjacent subplots (h_pad for height, w_pad for width).
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    
    # Save the figure, using bbox_inches='tight' to crop to the drawn content,
    # and pad_inches=0 to ensure no extra padding is added during saving.
    fig.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# ---------- core run ---------------------
def run_once(args, loss_type, lr, seed, device):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)

    # ---------- build synthetic Y -------------------
    n_tot = args.n_train
    idx_t = torch.arange(n_tot, device=device)

    imgs_np = np.stack([render_digit(int(i)) for i in range(n_tot)])  # (n,28,28)
    Y_img   = torch.tensor(imgs_np, device=device).view(n_tot, -1)         # (n,784)

    # positional block
    if args.pos_type == "none":
        pos_Y = torch.empty(n_tot, 0, device=device)
    elif args.pos_type == "sinusoid":
        pos_Y = sinusoid(idx_t, args.pos_dim)
    elif args.pos_type == "learned":
        embed = torch.nn.Embedding(n_tot, args.pos_dim, device=device)
        torch.nn.init.normal_(embed.weight, std=0.02)
        pos_Y = embed(idx_t).detach()
    else:
        raise ValueError("pos_type")

    idx_col = (idx_t.float() / n_tot).unsqueeze(1)              # (n,1) fixed in Y
    Y_full  = torch.cat([Y_img, pos_Y, idx_col], 1)             # (n, D)
    n_tot, D = Y_full.shape

    # ------------------------------------------------------------
    #  Initialisation: ONLY the index column is learnable
    # ------------------------------------------------------------

    # 1. reproducible shuffle of the image part (optional, keeps stats)
    # g     = torch.Generator(device=device)
    # g.manual_seed(args.perm_seed)
    # perm  = torch.randperm(n_tot, generator=g, device=device)
    # Y_img_shuffled = Y_img[perm]                 # (n, 784)

    # # 2. fixed part = shuffled image + positional block
    # fixed_part = torch.cat([Y_img_shuffled, pos_Y], dim=1)   # (n, D-1)   no grad

    # idx_init = torch.zeros_like(idx_col)
    # idx_param = torch.nn.Parameter(idx_init)                 # (n, 1)     learnable

    # # 4. helper that rebuilds the full tensor on-the-fly
    # def X_full():
    #     return torch.cat([fixed_part, idx_param], dim=1)     # (n, D)

    # # 5. optimiser touches ONLY the index column
    # optim = torch.optim.Adam([idx_param], lr=lr)

    X_full = torch.nn.Parameter(torch.rand_like(Y_full) / n_tot)
    perm = torch.arange(n_tot, device=device)
    optim = torch.optim.Adam([X_full], lr=lr)

    # ---------- loss objects --------------------------------------------
    twd, nlines, L_total, ntrees = build_twd(D, device)
    ntwd, nlines, L_total, ntrees = build_ntwd(D, device, noisy_mode=args.noisy_mode, lambda_=args.lambda_, p_noise=args.p_noise)
    def loss_fn(step):
        if loss_type == "sw":
            return SW(X_full, Y_full, L=L_total, p=2, device=device)
        prog  = step / args.n_steps
        kappa = 0.0001
        common = dict(ntrees=ntrees, nlines=nlines, d=D, std=0.001,
                      device=device, kappa=kappa if loss_type.endswith("_rp") else None,
                      X=X_full.detach(), Y=Y_full.detach())
        if loss_type.startswith("n_tsw"):
            th, ic = generate_trees_frames(mean=Y_full.mean(0),
                                           gen_mode="random_path" if loss_type.endswith("_rp")
                                           else "gaussian_raw", **common)
            return ntwd(X_full, Y_full, th, ic)
        if loss_type.startswith("fw_"):
            th, ic = generate_trees_frames(intercept_mode="geometric_median",
                                           gen_mode="random_path" if loss_type.endswith("_rp")
                                           else "gaussian_raw", **common)
        else:   
            th, ic = generate_trees_frames(mean=Y_full.mean(0),
                                           gen_mode="random_path" if loss_type.endswith("_rp")
                                           else "gaussian_raw", **common)
        return twd(X_full, Y_full, th, ic)

    # ---------- bookkeeping ---------------------------------------------
    ensure_dirs("imgs", "saved", "logs")
    tag    = f"images_{loss_type}_lr{lr:g}_seed{seed}"
    n_plot = min(args.n_plot, n_tot)

    save_grid(Y_img[:n_plot], f"imgs/{tag}_GT.png")

    traj, dists, times = [], [], []
    t0 = time.time()

    for step in range(args.n_steps):
        if step in args.print_steps:
            order = torch.argsort(X_full[:, -1])
            dist = torch.linalg.vector_norm(X_full[order][:, :784] - Y_full[:, :784]).item()
            elapsed = time.time() - t0
            print(f"{loss_type:8s} lr={lr:<6g} seed={seed} "
                  f"step {step+1:4d}/{args.n_steps} dist {dist:.4e} t {elapsed:.1f}s")
            traj.append(X_full.detach().cpu().numpy()); dists.append(dist); times.append(elapsed)

            save_grid(X_full[order][:n_plot, :784].detach(), f"imgs/{tag}_step{step:04d}.png")

        optim.zero_grad(); loss_fn(step).backward(); optim.step()

    traj.append(Y_full.cpu().numpy())
    np.save(f"saved/{tag}_points.npy", np.stack(traj))
    np.savetxt(f"logs/{tag}_distances.txt", np.array(dists), delimiter=",")
    np.savetxt(f"logs/{tag}_times.txt",     np.array(times), delimiter=",")

# ---------- CLI & entry point ----------
def parse_args():
    p = argparse.ArgumentParser("Gradient flow on synthetic 28x28 digits")
    p.add_argument("--n_train", type=int, default=16)
    p.add_argument("--n_plot",  type=int, default=16)
    p.add_argument("--n_steps", type=int, default=100)
    p.add_argument("--print_steps", type=int, nargs="*", default=[0,19,39,59,79,99])
    p.add_argument("--losses", nargs="*", default=["sw"])
    p.add_argument("--lrs", type=float, nargs="*", default=[1e-3])
    p.add_argument("--seeds", type=int, nargs="*", default=[1])
    p.add_argument("--pos_type", choices=["none", "sinusoid", "learned"], default="none")
    p.add_argument("--pos_dim", type=int, default=8)
    p.add_argument("--perm_seed", type=int, default=0,
                   help="Seed for the initial shuffle of image features")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--noisy_mode", type=str, default=None,
                   help="Type of noise regularization for n-TSW: None | interval | ball")
    p.add_argument("--lambda_", type=float, default=0.0,
                   help="Regularization strength for n-TSW")
    p.add_argument("--p_noise", type=int, default=2,
                   help="Dual norm exponent for noise regularization in n-TSW")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}") if args.gpu >= 0 and torch.cuda.is_available() \
             else torch.device("cpu")
    for loss in args.losses:
        for lr in args.lrs:
            for seed in args.seeds:
                run_once(args, loss, lr, seed, device)

if __name__ == "__main__":
    main()
