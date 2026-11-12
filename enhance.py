"""Batch denoising: every WAV in input/ -> same-name WAV in output/.

    python enhance.py                      # ./input -> ./output (relative to this file)
    python enhance.py --input IN --output OUT

The output keeps the input sample rate, channel count, number of samples and
sample format (subtype). Multichannel files are processed channel by channel.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

from denoiser import Config, enhance

HERE = Path(__file__).resolve().parent


def config_for(sr):
    """Keep the same window/hop duration (64 ms / 8 ms) at any sample rate."""
    base = Config()
    if sr == 16000:
        return base
    scale = sr / 16000
    return replace(base, n_fft=max(16, int(round(base.n_fft * scale))),
                   hop=max(2, int(round(base.hop * scale))))


def process_file(src, dst):
    info = sf.info(src)
    x, sr = sf.read(src, dtype="float64", always_2d=True)
    cfg = config_for(sr)
    y = np.stack([enhance(x[:, c], sr, cfg) for c in range(x.shape[1])], axis=1)
    if info.subtype.startswith("PCM") or info.subtype in ("ULAW", "ALAW"):
        y = np.clip(y, -1.0, 1.0)
    if info.channels == 1:
        y = y[:, 0]
    sf.write(dst, y, sr, subtype=info.subtype, format=info.format)
    out = sf.info(dst)
    assert (out.samplerate, out.channels, out.frames) == (info.samplerate, info.channels, info.frames)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(HERE / "input"))
    ap.add_argument("--output", default=str(HERE / "output"))
    a = ap.parse_args()
    src_dir, dst_dir = Path(a.input), Path(a.output)
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() == ".wav")
    if not files:
        raise SystemExit(f"No .wav files in {src_dir.resolve()}")
    for i, f in enumerate(files, 1):
        process_file(f, dst_dir / f.name)
        print(f"[{i}/{len(files)}] {f.name}", flush=True)
    print(f"Done: {len(files)} files -> {dst_dir.resolve()}")


if __name__ == "__main__":
    main()
