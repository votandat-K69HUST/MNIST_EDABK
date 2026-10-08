"""Engine sinh dataset so 10..20 theo config JSON. Moi thu muc phuong phap chi la 1 config + generate.py goi vao day.

Chay:  python generate.py [--config config.json] [--n-train N] [--n-val N] [--seed S] [--workers W] [--out DIR]

Dau ra (out/<name>/):
    train.npz, val.npz : X uint8 (N,28,28), y (N,) in 10..20, va metadata (src_left, src_right, writer,
                         gap, thick, lig, layout) de phan tich loi
    config_used.json   : config da dung (de tai tao)
Tai tao: cung config + cung seed + cung du lieu goc => cung ket qua (khong phu thuoc so workers).
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import io_utils
from .compose import compose_uint8
from .features import style_features_batch
from .pairing import sample_pair, split_pools

LAYOUT_IDS = {"aspect": 0, "stretch": 1}
CHUNK = 1000
_STATE = {}


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_config_dir"] = os.path.dirname(os.path.abspath(path))
    return cfg


def _features_for(cfg, source):
    kind = cfg["source"]["kind"]
    raw_dir = cfg["source"].get("raw_dir", io_utils.DEFAULT_RAW_DIR)
    cache = os.path.join(raw_dir, f"features_{kind}.npy")
    if os.path.exists(cache):
        f = np.load(cache)
        if len(f) == len(source["digit"]):
            return f
    print(f"[features] tinh dac trung phong cach cho {len(source['digit'])} anh ({kind}) ...")
    f = style_features_batch(source["X"])
    # chuan hoa z-score de cac chieu co thang do ngang nhau
    f = (f - f.mean(0)) / (f.std(0) + 1e-9)
    os.makedirs(raw_dir, exist_ok=True)
    np.save(cache, f)
    return f


def _build_state(cfg):
    scfg = cfg["source"]
    source = io_utils.load_source(scfg["kind"], scfg.get("raw_dir", io_utils.DEFAULT_RAW_DIR),
                                  scfg.get("self_path"))
    feats = _features_for(cfg, source) if cfg["pairing"].get("mode") == "style_matched" else None
    sp = cfg.get("split", {})
    train_pool, val_pool = split_pools(source, sp.get("val_fraction", 0.1), sp.get("split_seed", 123), feats)
    return {"cfg": cfg, "source": source, "pools": {"train": train_pool, "val": val_pool}}


def _init_worker(cfg):
    _STATE.update(_build_state(cfg))


def _gen_chunk(args):
    split, labels, seed_seq = args
    cfg, source, pool = _STATE["cfg"], _STATE["source"], _STATE["pools"][split]
    rng = np.random.default_rng(seed_seq)
    n = len(labels)
    X = np.zeros((n, 28, 28), dtype=np.uint8)
    meta = {k: np.zeros(n, dtype=dt) for k, dt in
            [("src_left", np.int64), ("src_right", np.int64), ("writer", np.int64),
             ("gap", np.float32), ("thick", np.int8), ("lig", np.int8), ("layout", np.int8)]}
    for i, lab in enumerate(labels):
        il, ir = sample_pair(pool, int(lab), rng, cfg["pairing"])
        X[i], info = compose_uint8(source["X"][il], source["X"][ir], rng, cfg)
        meta["src_left"][i], meta["src_right"][i] = il, ir
        meta["writer"][i] = source["writer"][il]
        meta["gap"][i], meta["thick"][i] = info["gap"], info["thick"]
        meta["lig"][i], meta["layout"][i] = info["lig"], LAYOUT_IDS[info["layout"]]
    return X, labels, meta


def generate_split(cfg, split, n, seed, workers, executor):
    # nhan can bang giua 11 lop, xao tron
    rng = np.random.default_rng([seed, 0 if split == "train" else 1])
    labels = (np.arange(n) % 11 + 10).astype(np.int64)
    rng.shuffle(labels)
    chunks = [labels[i:i + CHUNK] for i in range(0, n, CHUNK)]
    seeds = np.random.SeedSequence([seed, 0 if split == "train" else 1]).spawn(len(chunks))
    jobs = [(split, c, s) for c, s in zip(chunks, seeds)]
    if executor is None:
        results = [_gen_chunk(j) for j in jobs]
    else:
        results = list(executor.map(_gen_chunk, jobs))
    X = np.concatenate([r[0] for r in results])
    y = np.concatenate([r[1] for r in results])
    meta = {k: np.concatenate([r[2][k] for r in results]) for k in results[0][2]}
    return X, y, meta


def run(cfg, n_train=None, n_val=None, seed=None, workers=None, out_root=None):
    seed = cfg.get("seed", 0) if seed is None else seed
    n_train = cfg.get("n_train", 55000) if n_train is None else n_train
    n_val = cfg.get("n_val", 5500) if n_val is None else n_val
    workers = cfg.get("workers", 1) if workers is None else workers
    out_dir = os.path.join(out_root or io_utils.DEFAULT_OUT_DIR, cfg["name"])
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    _STATE.update(_build_state(cfg))  # chinh process: cache features, kiem tra du lieu
    executor = ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(cfg,)) if workers > 1 else None
    try:
        for split, n in (("train", n_train), ("val", n_val)):
            if n <= 0:
                continue
            X, y, meta = generate_split(cfg, split, n, seed, workers, executor)
            io_utils.save_dataset(os.path.join(out_dir, f"{split}.npz"), X, y, **meta)
            print(f"[{cfg['name']}] {split}: {len(X)} anh -> {out_dir}")
    finally:
        if executor:
            executor.shutdown()
    used = {k: v for k, v in cfg.items() if not k.startswith("_")}
    used.update({"seed": seed, "n_train": n_train, "n_val": n_val})
    with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(used, f, indent=2, ensure_ascii=False)
    print(f"xong trong {time.time() - t0:.1f}s")
    return out_dir


def cli(default_config):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=default_config)
    ap.add_argument("--n-train", type=int)
    ap.add_argument("--n-val", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--out", help="thu muc goc cho output (mac dinh data/output)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    # duong dan raw_dir tuong doi duoc tinh theo thu muc config
    rd = cfg["source"].get("raw_dir")
    if rd and not os.path.isabs(rd):
        cfg["source"]["raw_dir"] = os.path.normpath(os.path.join(cfg["_config_dir"], rd))
    sp = cfg["source"].get("self_path")
    if sp and not os.path.isabs(sp):
        cfg["source"]["self_path"] = os.path.normpath(os.path.join(cfg["_config_dir"], sp))
    run(cfg, a.n_train, a.n_val, a.seed, a.workers, a.out)

