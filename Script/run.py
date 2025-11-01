from __future__ import annotations
import os, sys, subprocess, json
from pathlib import Path
from typing import Dict, List
import time
import numpy as np
from ultralytics import YOLO

#CONFIG
MODEL_PATH     = r"D:\CODE\AI\TomatoLeafDisease\runs\classify\tomato_v13\weights\best.pt"
SOURCE_DIR     = r"D:\CODE\AI\TomatoLeafDisease\test_images"                                      
RECURSIVE      = True                                                        
IMG_SIZE       = 224                                                         
DEVICE         = ""                                                          
TOPK           = 3                                                           
SAVE_CSV_PATH  = r""            
SUPPORT_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

#main                                             
def ensure_package(pkg: str):
    try:
        __import__(pkg)
    except ImportError:
        print(f"[SETUP] Installing {pkg} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-U", pkg], check=True)
for _pkg in ("ultralytics", "numpy"):
    ensure_package(_pkg)


def find_model_if_missing(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_file():
        return p
    print(f"[INFO] MODEL_PATH not found: {p}. Trying auto-search for 'best.pt' or 'last.pt' ...")
    candidates = []
    if not candidates:
        raise FileNotFoundError("Không tìm thấy model. Hãy chỉnh MODEL_PATH chính xác.")
    best = max(candidates, key=lambda x: x.stat().st_mtime)
    print(f"[INFO] Auto-selected model: {best}")
    return best

def list_images(root: Path, recursive: bool) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Source folder not found: {root}")
    if recursive:
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORT_EXTS]
    else:
        files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in SUPPORT_EXTS]
    return sorted(files)

def topk_from_probs(probs, names: Dict[int, str], k: int = 5):
    scores = probs.data.detach().cpu().numpy().astype(float)
    idxs = np.argsort(scores)[::-1][:k]
    out = []
    for i in idxs:
        out.append({
            "class_id": int(i),
            "class_name": names.get(int(i), str(i)),
            "prob": float(scores[i])
        })
    return out

def main():
    t0 = time.time()

    model_path = find_model_if_missing(MODEL_PATH)
    source = Path(SOURCE_DIR)

    files = list_images(source, RECURSIVE)
    if not files:
        print(f"[WARN] Không tìm thấy ảnh trong: {source} (recursive={RECURSIVE})")
        sys.exit(0)

    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(str(model_path))
    names = model.names if hasattr(model, "names") else {}

    print(f"[INFO] Classifying {len(files)} images (imgsz={IMG_SIZE}, device='{DEVICE or 'auto'}')\n")
    results = model(
        source=[str(p) for p in files],
        imgsz=IMG_SIZE,
        device=DEVICE if DEVICE else None,
        stream=True,
        verbose=False,
    )

    rows = []
    ok, fail = 0, 0
    for res in results:
        path = Path(getattr(res, "path", ""))
        if not hasattr(res, "probs") or res.probs is None:
            print(f"[SKIP] No probs for {path.name}")
            fail += 1
            continue

        top1_id = int(res.probs.top1)
        try:
            top1_conf = float(res.probs.top1conf.item())
        except Exception:
            top1_conf = float(res.probs.data.max().item())

        top1_name = names.get(top1_id, str(top1_id))

        line = f"{path.name} -> {top1_name} ({top1_conf*100:.2f}%)"
        if TOPK and TOPK > 1:
            tk = topk_from_probs(res.probs, names, k=TOPK)
            extras = ", ".join([f"{t['class_name']}({t['prob']*100:.1f}%)" for t in tk])
            line += f" | top{TOPK}: {extras}"
        print(line)

        rows.append({
            "path": str(path),
            "top1_class_id": top1_id,
            "top1_class_name": top1_name,
            "top1_conf": round(top1_conf, 6),
            "topk": json.dumps(topk_from_probs(res.probs, names, k=max(1, TOPK))),
        })
        ok += 1
    if SAVE_CSV_PATH:
        import csv
        out_csv = Path(SAVE_CSV_PATH)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path","top1_class_id","top1_class_name","top1_conf","topk"])
            w.writeheader()
            w.writerows(rows)
        print(f"\n[SAVED] CSV -> {out_csv}")

    dt = time.time() - t0
    print(f"\n[SUMMARY] done: {ok}, skipped: {fail}, time: {dt:.2f}s")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERROR]", e)
        sys.exit(1)
