"""Thong ke TONG HOP (so lieu vo huong) cua mot dataset: dung de so sanh du lieu tu sinh voi test.csv
va chinh config (layout, do day net, ...) cho khop phan phoi.

  python tools/stats.py output/m04_ligature_strokes/train.npz
  python tools/stats.py path/to/test.csv                       # chi in cac so tong hop
  python tools/stats.py output/m04.../train.npz --compare path/to/test.csv

TUAN THU LUAT CUOC THI: script chi tinh cac con so tong hop (trung binh, phan vi) tren toan bo tap.
No KHONG ve / luu / hien thi bat ky anh nao tu test.csv va khong in thong tin theo tung mau.
Cac chi so: ty le pixel muc, trung binh pixel, bounding box (rong/cao/ti le), ty le pixel xam trung gian,
dinh sang, vi tri trong tam (com).
"""
import argparse

import numpy as np


def load_any(path):
    if path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(path)
        X = df.drop(columns=[c for c in ("id", "label") if c in df.columns]).values.astype(np.float32)
        return X.reshape(-1, 28, 28), None
    z = np.load(path)
    return z["X"].astype(np.float32), (z["y"] if "y" in z.files else None)


def summarize(X):
    N = len(X)
    ink = (X >= 128)
    out = {}
    out["N"] = N
    out["mean_pixel (0-255)"] = X.mean()
    out["ink_fraction (>=128)"] = ink.mean()
    out["mid_gray_fraction (20..235)"] = ((X > 20) & (X < 235)).mean()
    out["peak_mean"] = X.reshape(N, -1).max(1).mean()
    rows = ink.any(2)
    cols = ink.any(1)
    h = rows.sum(1)
    w = cols.sum(1)
    out["bbox_height_mean"] = h.mean()
    out["bbox_width_mean"] = w.mean()
    out["bbox_aspect_w/h_mean"] = (w / np.maximum(h, 1)).mean()
    for p in (10, 50, 90):
        out[f"bbox_width_p{p}"] = np.percentile(w, p)
        out[f"bbox_height_p{p}"] = np.percentile(h, p)
    tot = X.reshape(N, -1).sum(1) + 1e-9
    rr, cc = np.mgrid[0:28, 0:28]
    com_r = (X * rr).reshape(N, -1).sum(1) / tot
    com_c = (X * cc).reshape(N, -1).sum(1) / tot
    out["com_row_mean"], out["com_col_mean"] = com_r.mean(), com_c.mean()
    out["com_row_std"], out["com_col_std"] = com_r.std(), com_c.std()
    # tam bounding box (do lech cho biet can theo bbox hay theo trong tam)
    r0 = ink.any(2).argmax(1)
    r1 = 27 - ink.any(2)[:, ::-1].argmax(1)
    c0 = ink.any(1).argmax(1)
    c1 = 27 - ink.any(1)[:, ::-1].argmax(1)
    out["bbox_center_row_std"] = ((r0 + r1) / 2.0).std()
    out["bbox_center_col_std"] = ((c0 + c1) / 2.0).std()
    out["bbox_row_start_mean"], out["bbox_row_end_mean"] = r0.mean(), r1.mean()
    out["bbox_col_start_mean"], out["bbox_col_end_mean"] = c0.mean(), c1.mean()
    out["bbox_width_std"], out["bbox_height_std"] = w.std(), h.std()
    out["ink_pixels_mean (>=128)"] = ink.reshape(N, -1).sum(1).mean()
    out["col_ink_profile (7 bins)"] = np.round(ink.mean(1).reshape(N, 7, 4).mean((0, 2)), 3)
    out["row_ink_profile (7 bins)"] = np.round(ink.mean(2).reshape(N, 7, 4).mean((0, 2)), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--compare")
    a = ap.parse_args()
    X, y = load_any(a.path)
    s1 = summarize(X)
    s2 = summarize(load_any(a.compare)[0]) if a.compare else None
    print(f"{'metric':32s} {a.path.split('/')[-1][:24]:>26s}" + (f" {a.compare.split('/')[-1][:24]:>26s}" if s2 else ""))
    for k, v in s1.items():
        fmt = lambda t: np.array2string(t, precision=3) if isinstance(t, np.ndarray) else f"{float(t):.3f}"  # noqa: E731
        print(f"{k:32s} {fmt(v):>26s}" + (f" {fmt(s2[k]):>26s}" if s2 else ""))
    if y is not None:
        print("class counts:", dict(zip(*np.unique(y, return_counts=True))))


if __name__ == "__main__":
    main()
