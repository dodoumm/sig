"""Reproduce the validation score of this submission (PESQ-WB, SNR; clean used only here).

    python eval_validation.py --data <folder containing validation/samples.csv> [--workers 8]
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from denoiser import enhance


def _score(job):
    root, r = job
    from pesq import pesq
    clean, sr = sf.read(root / r["clean_path"], dtype="float64")
    noisy, _ = sf.read(root / r["noisy_path"], dtype="float64")
    out = []
    for name, sig in (("noisy_input", noisy), ("ours", enhance(noisy, sr))):
        out.append({"segment_id": r["segment_id"], "method": name,
                    "noise_category": r["noise_category"], "target_snr_db": r["target_snr_db"],
                    "pesq": pesq(sr, clean, sig, "wb"),
                    "snr_db": 10 * np.log10(np.sum(clean ** 2) / np.sum((clean - sig) ** 2))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="noise_reduction_challenge data root")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    a = ap.parse_args()
    root = Path(a.data)
    df = pd.read_csv(root / a.split / "samples.csv")
    jobs = [(root, r) for r in df.to_dict("records")]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [x for chunk in ex.map(_score, jobs, chunksize=4) for x in chunk]
    det = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    det.to_csv(out / f"{a.split}_eval_per_file.csv", index=False, encoding="utf-8-sig")
    summ = det.groupby("method", sort=False)[["pesq", "snr_db"]].mean()
    summ.to_csv(out / f"{a.split}_eval_summary.csv", encoding="utf-8-sig")
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
