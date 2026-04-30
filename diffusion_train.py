"""
Train a small time-conditioned diffusion model on the half-moons distribution
and export weights/norm_params in the format consumed by `pytorch_net` in
load_networks.jl.

Parameterization: predict-x_0 (the model's output is an estimate of the clean
sample, in the same coordinate space as the input). This is one of the two
standard DDPM parameterizations (Ho et al. 2020); we choose it so that fixed
points of the network correspond to "points the network believes are already
clean", which is the manifold-projection question we care about.

Forward process (linear-sigma):
    x_t = x_0 + sigma(t) * eps,   eps ~ N(0, I)
    sigma(t) = SIGMA_MIN + t * (SIGMA_MAX - SIGMA_MIN),    t in [0, 1]

Loss:
    MSE(f([x_t, t]), x_0)

Architecture: FFReLUNet [3, 16, 16, 16, 16, 16, 2] (5 hidden layers, width 16).
Inputs: [x1, x2, t]. Output: predicted x_0 in R^2.
"""

import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


# ---------- Config ----------
SEED = 0
N_SAMPLES = 8000
DATA_NOISE = 0.05
LAYER_SIZES = [3, 16, 16, 16, 16, 16, 2]   # [in_dim, hidden..., out_dim]
SIGMA_MIN = 0.01
# SIGMA_MAX must be large enough that at t=1 the forward-process input
# x_t = x_0 + SIGMA_MAX * eps is noise-dominated (sigma >> data std). Otherwise
# reverse-process sampling from N(0, SIGMA_MAX^2 I) feeds the model inputs
# nothing like its training distribution. Half-moons span ~[-1.5, 1.5] so we
# pick SIGMA_MAX = 2.0.
SIGMA_MAX = 2.0
T_MIN = 0.01
T_MAX = 1.0
BATCH_SIZE = 128
N_EPOCHS = 500
LR = 1e-3
OUT_DIR = "models/diffusion"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- Data ----------
def load_halfmoons(n_samples=N_SAMPLES, noise=DATA_NOISE, seed=SEED):
    """Generate half-moons centered at origin, scaled to fit in roughly [-2, 2]^2."""
    X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    X = X.astype(np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    return X


class HalfMoonsDataset(TorchDataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i]


# ---------- Model ----------
class FFReLUNet(nn.Module):
    """Feed-forward ReLU MLP. Identical structure to pytorch_.py:28-61."""

    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        layers = []
        for i in range(len(shape) - 1):
            layers.append(nn.Linear(shape[i], shape[i + 1]))
            if i != len(shape) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


# ---------- Forward (noising) process ----------
def sigma_of_t(t):
    return SIGMA_MIN + t * (SIGMA_MAX - SIGMA_MIN)


def make_noisy_batch(x0):
    """Sample t, eps; return (x_t_with_t, x0) where x_t_with_t = [x_t, t]."""
    n = x0.shape[0]
    t = torch.rand(n, 1, device=x0.device) * (T_MAX - T_MIN) + T_MIN
    sig = sigma_of_t(t)
    eps = torch.randn_like(x0)
    x_t = x0 + sig * eps
    inp = torch.cat([x_t, t], dim=1)
    return inp, x0


# ---------- Reverse (sampling) process: deterministic DDIM-style ----------
@torch.no_grad()
def sample(model, n_samples=1000, n_steps=50, device=DEVICE):
    """Deterministic reverse process. x_{t'} = x_hat0 + (sigma(t')/sigma(t)) * (x_t - x_hat0)."""
    model.eval()
    ts = torch.linspace(T_MAX, T_MIN, n_steps + 1, device=device)
    x = torch.randn(n_samples, 2, device=device) * SIGMA_MAX
    for i in range(n_steps):
        t_now, t_next = ts[i], ts[i + 1]
        sig_now, sig_next = sigma_of_t(t_now), sigma_of_t(t_next)
        t_col = torch.full((n_samples, 1), t_now.item(), device=device)
        inp = torch.cat([x, t_col], dim=1)
        x_hat0 = model(inp)
        x = x_hat0 + (sig_next / sig_now) * (x - x_hat0)
    return x.cpu().numpy()


# ---------- Train ----------
def train(model, loader, n_epochs=N_EPOCHS, lr=LR):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []
    for epoch in range(n_epochs):
        epoch_loss, n_batches = 0.0, 0
        for x0 in loader:
            x0 = x0.to(DEVICE)
            inp, target = make_noisy_batch(x0)
            pred = model(inp)
            loss = loss_fn(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)
        if epoch == 0 or (epoch + 1) % 20 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch + 1:4d} / {n_epochs}    loss = {avg:.6f}")
    return losses


# ---------- Export ----------
def export_weights(model, out_dir, layer_sizes):
    """Save weights in pytorch_net's expected npz format.

    pytorch_net (load_networks.jl:144-177) expects:
      weights.npz: keys arr_0, arr_1, ..., arr_(2L-1) where
                   arr_(2k)   = layer k weight matrix W_k
                   arr_(2k+1) = layer k bias vector b_k
      norm_params.npz: X_mean, X_std, Y_mean, Y_std (per-feature), layer_sizes
    """
    os.makedirs(out_dir, exist_ok=True)
    weight_arrays = []
    for name, param in model.named_parameters():
        weight_arrays.append(param.detach().cpu().numpy())
    np.savez(os.path.join(out_dir, "weights.npz"), *weight_arrays)

    in_dim = layer_sizes[0]
    out_dim = layer_sizes[-1]
    np.savez(
        os.path.join(out_dir, "norm_params.npz"),
        X_mean=np.zeros(in_dim, dtype=np.float64),
        X_std=np.ones(in_dim, dtype=np.float64),
        Y_mean=np.zeros(out_dim, dtype=np.float64),
        Y_std=np.ones(out_dim, dtype=np.float64),
        layer_sizes=np.asarray(layer_sizes, dtype=np.int64),
    )


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Device: {DEVICE}")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading half-moons...")
    X = load_halfmoons()
    print(f"  X shape = {X.shape}, range = [{X.min():.3f}, {X.max():.3f}]")
    np.save(os.path.join(OUT_DIR, "training_data.npy"), X)

    loader = DataLoader(HalfMoonsDataset(X), batch_size=BATCH_SIZE, shuffle=True)

    print(f"Building FFReLUNet({LAYER_SIZES})...")
    model = FFReLUNet(LAYER_SIZES).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params} parameters")

    print(f"Training for {N_EPOCHS} epochs...")
    train(model, loader)

    print("Exporting weights...")
    export_weights(model, OUT_DIR, LAYER_SIZES)
    print(f"  wrote {OUT_DIR}/weights.npz")
    print(f"  wrote {OUT_DIR}/norm_params.npz")
    print(f"  wrote {OUT_DIR}/training_data.npy")

    print("Generating reverse-process samples for sanity check...")
    samples = sample(model, n_samples=1000, n_steps=50)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(X[:, 0], X[:, 1], s=8, alpha=0.4, label="training data")
    ax.scatter(samples[:, 0], samples[:, 1], s=8, alpha=0.4, label="reverse-process samples")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(f"Half-moons diffusion ({N_EPOCHS} epochs, predict-x0)")
    fig_path = os.path.join(OUT_DIR, "samples_sanity.png")
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"  wrote {fig_path}")


if __name__ == "__main__":
    main()
