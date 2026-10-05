"""Cong cu tu viet tay bang chuot/but cam ung (tkinter) de tao du lieu THAT.

Hai che do:
  --mode number : hien nhan 10..20, ban viet ca so hai chu so (vung ve rong). Dung de validate / train them.
  --mode digit  : hien chu so 0..9, ban viet tung chu so rieng. Dung cho pairing=same_writer voi phong
                  cach CUA BAN (build_dataset.py --mode digits -> output/self_digits.npz).

Phim tat: Enter = luu va sang mau tiep; Esc / C = xoa; S = bo qua mau.
Anh luu nguyen canvas (nen den, net trang) vao samples/<writer>/<mode>/<nhan>_<id>.png
=> co the xu ly lai voi bo cuc khac ma khong can viet lai.

Chay: python collect.py --writer ten_ban --mode number --per-label 30
Meo: viet nhieu kieu (noi net / khong noi, nghieng khac nhau, to / nho), nho nguoi khac viet giup
      (moi nguoi 1 --writer rieng) de tap val da dang.
"""
import argparse
import os
import random
import time
import tkinter as tk

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))


class App:
    def __init__(self, a):
        self.a = a
        self.labels = (list(range(10, 21)) if a.mode == "number" else list(range(10))) * a.per_label
        random.Random(a.seed).shuffle(self.labels)
        self.i = 0
        self.w, self.h = (560, 280) if a.mode == "number" else (280, 280)
        self.out_dir = os.path.join(HERE, "samples", a.writer, a.mode)
        os.makedirs(self.out_dir, exist_ok=True)
        self.root = tk.Tk()
        self.root.title("Thu thap du lieu tu viet")
        self.lbl = tk.Label(self.root, font=("Arial", 28, "bold"))
        self.lbl.pack()
        self.info = tk.Label(self.root, font=("Arial", 11))
        self.info.pack()
        self.canvas = tk.Canvas(self.root, width=self.w, height=self.h, bg="black", cursor="cross")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonPress-1>", self.start)
        self.root.bind("<Return>", lambda e: self.save())
        self.root.bind("<Escape>", lambda e: self.clear())
        self.root.bind("c", lambda e: self.clear())
        self.root.bind("s", lambda e: self.next())
        self.clear()
        self.refresh()

    def clear(self):
        self.canvas.delete("all")
        self.img = Image.new("L", (self.w, self.h), 0)
        self.dr = ImageDraw.Draw(self.img)
        self.last = None
        self.empty = True

    def start(self, e):
        self.last = (e.x, e.y)

    def draw(self, e):
        if self.last is None:
            self.last = (e.x, e.y)
        r = self.a.brush
        self.canvas.create_line(*self.last, e.x, e.y, fill="white", width=r, capstyle=tk.ROUND, smooth=True)
        self.dr.line([*self.last, e.x, e.y], fill=255, width=r)
        self.dr.ellipse([e.x - r / 2, e.y - r / 2, e.x + r / 2, e.y + r / 2], fill=255)
        self.last = (e.x, e.y)
        self.empty = False

    def refresh(self):
        if self.i >= len(self.labels):
            self.lbl.config(text="Xong! Dong cua so.")
            return
        self.lbl.config(text=f"Hay viet: {self.labels[self.i]}")
        self.info.config(text=f"{self.i}/{len(self.labels)} | Enter=luu  Esc/C=xoa  S=bo qua")

    def save(self):
        if self.empty or self.i >= len(self.labels):
            return
        name = f"{self.labels[self.i]}_{int(time.time() * 1000)}.png"
        self.img.save(os.path.join(self.out_dir, name))
        self.next()

    def next(self):
        self.i += 1
        self.clear()
        self.refresh()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--writer", required=True, help="ten/ID nguoi viet (dung de chia train/val theo nguoi)")
    ap.add_argument("--mode", choices=["number", "digit"], default="number")
    ap.add_argument("--per-label", type=int, default=20)
    ap.add_argument("--brush", type=int, default=16, help="do day net (pixel tren canvas)")
    ap.add_argument("--seed", type=int, default=0)
    App(ap.parse_args()).root.mainloop()


if __name__ == "__main__":
    main()
