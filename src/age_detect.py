#!/usr/bin/env python3
"""Stima dell'età da immagini (evoluzione del tutorial PyImageSearch).

Due modalità:
  * immagini intere  -> YuNet (OpenCV) trova i volti, poi classificatore di età
  * --crops          -> le immagini sono già crop di volti, si salta il detector

Il classificatore è un modello Hugging Face (default: ViT fine-tuned su
FairFace/age bins). Qualsiasi modello `image-classification` con etichette
di età funziona via --model (es. SigLIP2 fine-tuned).
"""
from __future__ import annotations

import argparse
import csv
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
DEFAULT_MODEL = "nateraw/vit-age-classifier"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_yunet(path: Path, score: float) -> cv2.FaceDetectorYN:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Scarico YuNet -> {path}")
        urllib.request.urlretrieve(YUNET_URL, path)
    return cv2.FaceDetectorYN.create(str(path), "", (320, 320), score, 0.3, 5000)


def detect_faces(det: cv2.FaceDetectorYN, bgr: np.ndarray, margin: float):
    """Restituisce [(x1, y1, x2, y2, score)] con margine attorno al volto."""
    h, w = bgr.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(bgr)
    out = []
    for f in faces if faces is not None else []:
        x, y, bw, bh = f[:4]
        m = margin * max(bw, bh)
        out.append((
            int(max(0, x - m)), int(max(0, y - m)),
            int(min(w, x + bw + m)), int(min(h, y + bh + m)),
            float(f[-1]),
        ))
    return out


@torch.inference_mode()
def predict(model, proc, device, crops: list[Image.Image], topk: int = 3):
    inputs = proc(images=crops, return_tensors="pt").to(device)
    probs = model(**inputs).logits.softmax(-1).cpu()
    id2label = model.config.id2label
    res = []
    for p in probs:
        v, i = p.topk(min(topk, len(p)))
        res.append([(id2label[int(k)], float(s)) for s, k in zip(v, i)])
    return res


def list_images(path: Path):
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMG_EXT)
    return [path]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="immagine o cartella")
    ap.add_argument("--crops", action="store_true",
                    help="gli input sono già crop di volti (niente detection)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="modello HF di classificazione età")
    ap.add_argument("--yunet", type=Path, default=Path("models/yunet.onnx"))
    ap.add_argument("--det-score", type=float, default=0.7)
    ap.add_argument("--margin", type=float, default=0.15, help="margine sul bbox (frazione)")
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--no-draw", action="store_true", help="non salvare immagini annotate")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    proc = AutoImageProcessor.from_pretrained(a.model)
    model = AutoModelForImageClassification.from_pretrained(a.model).to(device).eval()
    det = None if a.crops else get_yunet(a.yunet, a.det_score)

    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for img_path in list_images(a.input):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"! illeggibile: {img_path}")
            continue
        h, w = bgr.shape[:2]
        boxes = [(0, 0, w, h, 1.0)] if a.crops else detect_faces(det, bgr, a.margin)
        if not boxes:
            print(f"{img_path.name}: nessun volto")
            continue

        pil = [Image.fromarray(cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
               for x1, y1, x2, y2, _ in boxes]
        preds = []
        for i in range(0, len(pil), a.batch):
            preds += predict(model, proc, device, pil[i:i + a.batch])

        for (x1, y1, x2, y2, s), pr in zip(boxes, preds):
            label, conf = pr[0]
            print(f"{img_path.name}: {label} ({conf:.0%})  box=({x1},{y1},{x2},{y2})")
            rows.append([str(img_path), x1, y1, x2, y2, f"{s:.3f}", label, f"{conf:.4f}"])
            if not a.crops and not a.no_draw:
                cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(bgr, f"{label} {conf:.0%}", (x1, max(15, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if not a.crops and not a.no_draw:
            cv2.imwrite(str(a.out / img_path.name), bgr)

    with open(a.out / "results.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "x1", "y1", "x2", "y2", "det_score", "age_bin", "age_conf"])
        wr.writerows(rows)
    print(f"\nRisultati: {a.out / 'results.csv'}")


if __name__ == "__main__":
    main()