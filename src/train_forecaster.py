"""Entraîne un modèle de prévision d'humidité (LSTM / GRU / Transformer).

  python -m src.train_forecaster --source sim --model lstm --seeds 0
  python -m src.train_forecaster --source sim --model transformer --seeds 0,1,2 --wandb
  python -m src.train_forecaster --source era5 --model lstm          # validation externe
Sorties : models/forecaster_<source>_<model><tag>_s<seed>.pt  +  results/forecast_<source>_<model><tag>.csv
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import config as C
from .dataset import load_source
from .metrics import horizon_table
from .models import build_model


def predict(model, X, device, bs=2048):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).to(device)
            outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs)


def run(args, seed, data, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    F = data["train"]["X"].shape[2]
    model = build_model(args.model, F, data["H"], args.hidden, args.layers, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    loss_fn = nn.HuberLoss(delta=1.0)

    tr = TensorDataset(torch.from_numpy(data["train"]["X"]), torch.from_numpy(data["train"]["Y_delta"]))
    dl = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=True)
    Xv = torch.from_numpy(data["val"]["X"]).to(device)
    Yv = torch.from_numpy(data["val"]["Y_delta"]).to(device)

    best, bad, best_state = 1e9, 0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        tl, n = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(xb)
            n += len(xb)
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xv), Yv).item()
        sched.step(vl)
        print(f"  [{args.model} s{seed}] epoch {epoch:03d}  train={tl / n:.5f}  val={vl:.5f}  lr={opt.param_groups[0]['lr']:.1e}", flush=True)
        if args.wandb:
            import wandb
            wandb.log({"epoch": epoch, "train_loss": tl / n, "val_loss": vl})
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("  early stopping")
                break

    model.load_state_dict(best_state)
    pred_delta = predict(model, data["test"]["X"], device)
    pred_abs = data["test"]["sw_now"][:, None] + pred_delta * data["sw_std"]
    tab = horizon_table(data["test"]["Y_raw"], pred_abs, f"{args.model}{args.tag}")
    tab["seed"] = seed
    ckpt = {"state_dict": best_state, "model": args.model, "features": data["features"], "mean": data["mean"],
            "std": data["std"], "sw_std": data["sw_std"], "K": data["K"], "H": data["H"],
            "hidden": args.hidden, "layers": args.layers, "dropout": args.dropout, "val_loss": best}
    path = C.MODELS / f"forecaster_{args.source}_{args.model}{args.tag}_s{seed}.pt"
    torch.save(ckpt, path)
    print("  ->", path)
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="sim", choices=["sim", "era5"])
    ap.add_argument("--model", default="lstm", choices=["lstm", "gru", "transformer"])
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--window", type=int, default=C.WINDOW)
    ap.add_argument("--horizon", type=int, default=C.HORIZON)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--tag", default="", help="suffixe pour ne pas écraser les fichiers (ex: _h128)")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Appareil :", device)
    data = load_source(args.source, args.window, args.horizon)
    print({k: data[k]["X"].shape for k in ("train", "val", "test")}, "features:", data["features"])

    if args.wandb:
        import wandb
        wandb.init(project="irrigation-agent", name=f"fc_{args.source}_{args.model}", config=vars(args))
    tabs = [run(args, int(s), data, device) for s in args.seeds.split(",")]
    allt = pd.concat(tabs)
    allt.to_csv(C.RESULTS / f"forecast_{args.source}_{args.model}{args.tag}.csv", index=False)
    summ = allt.groupby(["model", "horizon"])[["rmse", "mae", "r2", "kge"]].agg(["mean", "std"]).round(4)
    print(summ)


if __name__ == "__main__":
    main()
