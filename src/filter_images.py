#!/usr/bin/env python3
"""Filtra le immagini: tiene quelle a colori e senza volti di minorenni.

Input:
  --meta     tabella dei metadati (colonne `file`, `image_path`, `crop_file`, ...)
  --ages     results.csv di age_detect.py (colonne file, age_bin, age_conf[, p_minor])
Le due tabelle si collegano sul nome file (basename).

Una riga è tenuta solo se TUTTE queste condizioni sono vere:
  * color_dev_frac_in <= --max-color-dev (frazione di deviazione di colore)
  * ha almeno un volto stimato (altrimenti non si può escludere un minore)
  * nessun volto ha fascia in --minor-bins, né p_minor >= --max-p-minor
Con --by-frame un volto minorenne scarta anche tutte le altre righe dello stesso frame.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

DEFAULT_MINOR_BINS = ["0-2", "3-9", "10-19"]


def frame_id(name: str) -> str:
    m = re.match(r"(frame_\d+)", name)
    return m.group(1) if m else name


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--ages", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("output/filtered"))
    ap.add_argument("--max-color-dev", type=float, default=0.0,
                    help="scarta se color_dev_frac_in > soglia (default 0: solo valori nulli)")
    ap.add_argument("--minor-bins", nargs="+", default=DEFAULT_MINOR_BINS)
    ap.add_argument("--max-p-minor", type=float, default=0.2,
                    help="scarta se p_minor >= soglia (ignorato se la colonna manca)")
    ap.add_argument("--by-frame", action="store_true",
                    help="un minorenne scarta tutto il frame, non solo la riga")
    ap.add_argument("--keep-no-face", action="store_true",
                    help="tieni anche le righe senza volto stimato (sconsigliato)")
    a = ap.parse_args()

    meta = pd.read_csv(a.meta)
    ages = pd.read_csv(a.ages)
    ages["key"] = ages["file"].map(lambda p: Path(p).name)
    meta["key"] = meta["file"].map(lambda p: Path(p).name)

    # stato per immagine: peggior caso fra i volti stimati
    ages["is_minor_bin"] = ages["age_bin"].astype(str).str.replace(" ", "").isin(
        [b.replace(" ", "") for b in a.minor_bins])
    ages["is_minor_p"] = ages["p_minor"].ge(a.max_p_minor) if "p_minor" in ages else False
    ages["is_minor"] = ages["is_minor_bin"] | ages["is_minor_p"]
    per_img = ages.groupby("key").agg(n_est=("key", "size"), minor=("is_minor", "any"),
                                      p_minor_max=("p_minor", "max") if "p_minor" in ages else ("key", "size"))
    df = meta.merge(per_img, on="key", how="left")
    df["n_est"] = df["n_est"].fillna(0).astype(int)
    df["minor"] = df["minor"].fillna(False).astype(bool)

    if a.by_frame:
        df["frame"] = df["key"].map(frame_id)
        df["minor"] = df.groupby("frame")["minor"].transform("any")

    reason = pd.Series("", index=df.index)
    reason[df.color_dev_frac_in.isna()] = "color_dev_frac_in mancante"
    reason[(reason == "") & (df.color_dev_frac_in > a.max_color_dev)] = "deviazione di colore"
    reason[(reason == "") & df.minor] = "possibile minorenne"
    if not a.keep_no_face:
        reason[(reason == "") & (df.n_est == 0)] = "nessun volto stimato"
    df["reason"] = reason

    a.out.mkdir(parents=True, exist_ok=True)
    keep, drop = df[df.reason == ""], df[df.reason != ""]
    keep.to_csv(a.out / "kept.csv", index=False)
    drop.to_csv(a.out / "discarded.csv", index=False)
    print(f"totale {len(df)} | tenute {len(keep)} | scartate {len(drop)}")
    print(drop.reason.value_counts().to_string())
    print(f"-> {a.out}/kept.csv, discarded.csv")


if __name__ == "__main__":
    main()
