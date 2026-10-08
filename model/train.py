"""Huan luyen MLP phan loai so 10..20.

  python train.py                       # dung config.json, ensemble.seeds trong config
  python train.py --seeds 0             # chi 1 seed
  python train.py --set train.lr=0.001 model.hidden=[1024,512] train.epochs=40   # ghi de tham so

Dau ra trong runs/<name>/: model_seed<k>.pt (trong so + config + chuan hoa), history_seed<k>.csv, metrics.json
"""
import argparse
import copy
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from augment_torch import augment  # noqa: E402
from data_utils import load_train, load_val, to_tensor  # noqa: E402
from mlp import LABEL_OFFSET, build_model  # noqa: E402


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def apply_overrides(cfg, items):
    """items: ['train.lr=0.001', 'model.hidden=[512,256]'] -> sua cfg tai cho."""
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        d = cfg
        ks = k.split(".")
        for kk in ks[:-1]:
            d = d.setdefault(kk, {})
        d[ks[-1]] = v
    return cfg


@torch.no_grad()
def accuracy(model, X, y, bs=4096):
    model.eval()
    correct = 0
    for i in range(0, len(X), bs):
        correct += (model(X[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
    return correct / len(X)


def primary_score(accs):
    """Diem chon model: neu co val 'real' (tu viet) thi lay trung binh real, nguoc lai trung binh synthetic."""
    real = [v for k, v in accs.items() if k.startswith("real:")]
    syn = [v for k, v in accs.items() if k.startswith("synthetic:")]
    use = real or syn
    return float(np.mean(use)) if use else 0.0


def train_one(cfg, seed, out_dir, base, verbose=True, data_cache=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(cfg.get("num_threads", torch.get_num_threads()))
    tcfg = cfg["train"]

    if data_cache is None:
        Xtr, ytr, desc = load_train(cfg, base, seed)
        val = {k: (to_tensor(X), torch.from_numpy(y - LABEL_OFFSET).long()) for k, (X, y) in load_val(cfg, base).items()}
    else:
        Xtr, ytr, desc, val = data_cache
    Xt = to_tensor(Xtr)
    yt = torch.from_numpy(ytr - LABEL_OFFSET).long()
    if verbose:
        print(f"[seed {seed}] train {len(Xt)} mau tu {desc}; val: {list(val)}")

    mean, std = Xt.mean().item(), Xt.std().item()
    model = build_model(cfg["model"], mean, std)
    n_params = sum(p.numel() for p in model.parameters())

    params = model.parameters()
    if tcfg.get("optimizer", "adamw") == "sgd":
        opt = torch.optim.SGD(params, lr=tcfg["lr"], momentum=0.9, nesterov=True, weight_decay=tcfg.get("weight_decay", 1e-4))
    elif tcfg["optimizer"] == "adam":
        opt = torch.optim.Adam(params, lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.0))
    else:
        opt = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 1e-4))

    bs, epochs = tcfg.get("batch_size", 256), tcfg.get("epochs", 30)
    steps_per_epoch = math.ceil(len(Xt) / bs)
    total_steps, warm = epochs * steps_per_epoch, tcfg.get("warmup_epochs", 1) * steps_per_epoch

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        if tcfg.get("scheduler", "cosine") == "constant":
            return 1.0
        t = (step - warm) / max(total_steps - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * t)) * (1 - tcfg.get("min_lr_frac", 0.01)) + tcfg.get("min_lr_frac", 0.01)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=tcfg.get("label_smoothing", 0.0))
    g = torch.Generator().manual_seed(seed)
    best, best_state, best_ep, bad = -1.0, None, -1, 0
    hist = []
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        tot_loss, n = 0.0, 0
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            xb, yb = Xt[idx], yt[idx]
            xb = augment(xb, cfg.get("augment"), g)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            if tcfg.get("grad_clip"):
                nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            opt.step()
            sched.step()
            tot_loss += loss.item() * len(idx)
            n += len(idx)
        accs = {k: accuracy(model, X, y) for k, (X, y) in val.items()}
        score = primary_score(accs)
        hist.append({"epoch": ep + 1, "loss": tot_loss / n, "score": score, "lr": opt.param_groups[0]["lr"], **accs})
        if verbose:
            print(f"  ep {ep + 1:3d} loss {tot_loss / n:.4f} score {score:.4f} " +
                  " ".join(f"{k.split(':')[0][:3]}.{k.split(':')[-1].split('/')[0][:3]}={v:.3f}" for k, v in accs.items()) +
                  f" ({time.time() - t0:.0f}s)")
        if score > best:
            best, best_ep, bad = score, ep + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if tcfg.get("patience") and bad >= tcfg["patience"]:
                if verbose:
                    print(f"  early stop (khong cai thien {bad} epoch)")
                break
    if not val:  # khong co val: lay model cuoi
        best_state, best_ep = copy.deepcopy(model.state_dict()), len(hist)
    model.load_state_dict(best_state)
    final_accs = {k: accuracy(model, X, y) for k, (X, y) in val.items()}
    path = os.path.join(out_dir, f"model_seed{seed}.pt")
    torch.save({"state_dict": best_state, "config": cfg, "mean": mean, "std": std, "seed": seed,
                "best_epoch": best_ep, "val": final_accs}, path)
    with open(os.path.join(out_dir, f"history_seed{seed}.csv"), "w", newline="", encoding="utf-8") as f:
        keys = sorted({k for h in hist for k in h}, key=lambda k: (k not in ("epoch", "loss", "score", "lr"), k))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(hist)
    if verbose:
        print(f"[seed {seed}] best epoch {best_ep}, score {best:.4f}, params {n_params:,} -> {path}")
    return {"seed": seed, "best_epoch": best_ep, "score": best, "val": final_accs, "params": n_params, "path": path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--name")
    ap.add_argument("--set", nargs="*", help="ghi de tham so: train.lr=0.001 model.hidden=[512,256]")
    a = ap.parse_args()
    cfg = apply_overrides(load_config(a.config), a.set)
    base = cfg.get("base_dir") or os.path.dirname(os.path.abspath(a.config))
    name = a.name or cfg.get("name", "run")
    out_dir = os.path.join(HERE, "runs", name)
    seeds = a.seeds if a.seeds else cfg.get("ensemble", {}).get("seeds", [0])
    results = [train_one(cfg, s, out_dir, base) for s in seeds]
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "results": results}, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print("diem trung binh cac seed:", np.mean([r["score"] for r in results]))
    print("ket qua o", out_dir)


if __name__ == "__main__":
    main()
