"""Traditional (non-ML) speech denoiser — 2nd version.

Pipeline (per file, noisy input only; clean speech is never used):
  1. STFT (Hann, 64 ms window, 8 ms hop @16 kHz; scaled for other sample rates)
  2. Noise PSD: quietest 10 % of frames -> offline SPP-weighted refinement
     -> slow SPP-based tracking (Gerkmann & Hendriks, 2012). No smoothing over
     frequency, so tonal noise lines (hum, alarms, engines) are kept sharp.
  3. A priori SNR: decision-directed (Ephraim & Malah, 1984), run forward and backward
     in time and combined (offline, removes the onset/offset lag)
  4. Gain: MMSE log-spectral amplitude (Ephraim & Malah, 1985)
  5. Harmonic regeneration (Plapous, Marro & Scalart, 2006): a half-wave rectified copy
     of the first output restores voiced harmonics; the a priori SNR is re-estimated
     from it and the LSA gain recomputed
  6. Frame-level speech-absence attenuation (-3 dB in frames whose estimated SNR is low)
  7. ISTFT (overlap-add), output length = input length

Parameters were selected on a fixed 320-file subset of TRAIN (dev/tune.py);
VALIDATION was used only to check the result.
"""
from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.special import exp1


@dataclass(frozen=True)
class Config:
    n_fft: int = 1024                # 64 ms @ 16 kHz
    hop: int = 128                   # 8 ms @ 16 kHz
    # noise estimation
    init_fraction: float = 0.10      # quietest fraction of frames for the first estimate
    refine_iters: int = 1            # offline SPP-weighted refinement iterations
    spp_xi_h1_db: float = 15.0       # fixed a priori SNR under speech presence (SPP model)
    spp_prior: float = 0.5           # prior speech presence probability
    noise_alpha: float = 0.99        # noise tracking smoothing (slow: short, mostly stationary clips)
    spp_alpha: float = 0.9           # SPP smoothing used for stagnation control
    # gain
    dd_alpha: float = 0.98           # decision-directed smoothing
    xi_min_db: float = -22.0         # lower bound of the a priori SNR
    gain_floor_db: float = -25.0     # lower bound of the spectral gain
    # harmonic regeneration
    hrnr_rho: float = 0.5            # weight of the first-stage estimate (1.0 = HRNR off)
    # frame-level speech-absence attenuation
    vad_atten_db: float = -3.0       # extra attenuation in speech-absent frames (0 = off)
    vad_theta_db: float = 0.0        # frame SNR (dB) where speech presence = 0.5
    vad_width_db: float = 2.0        # slope of the sigmoid


def _stft(x, cfg, sr):
    return signal.stft(x, fs=sr, window="hann", nperseg=cfg.n_fft,
                       noverlap=cfg.n_fft - cfg.hop, nfft=cfg.n_fft,
                       boundary="zeros", padded=True)


def _istft(z, n, cfg, sr):
    _, y = signal.istft(z, fs=sr, window="hann", nperseg=cfg.n_fft,
                        noverlap=cfg.n_fft - cfg.hop, nfft=cfg.n_fft,
                        input_onesided=True, boundary=True)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y[:n]


# ----------------------------------------------------------------------------- noise PSD
def _inner(n_frames):
    """Frame indices excluding the zero-padded boundary frames."""
    return np.arange(2, n_frames - 2) if n_frames > 8 else np.arange(n_frames)


def initial_noise(power, cfg):
    """Mean power of the quietest frames (ranked by total frame energy)."""
    inner = _inner(power.shape[1])
    energy = power[:, inner].mean(axis=0)
    count = max(1, int(np.ceil(cfg.init_fraction * len(inner))))
    sel = inner[np.argsort(energy, kind="stable")[:count]]
    return np.maximum(power[:, sel].mean(axis=1), 1e-12)


def speech_presence(power, noise, cfg):
    """Posterior speech-presence probability with a fixed a priori SNR under H1."""
    xi_h1 = 10 ** (cfg.spp_xi_h1_db / 10)
    prior_ratio = (1 - cfg.spp_prior) / cfg.spp_prior
    gamma = power / noise
    return 1.0 / (1.0 + prior_ratio * (1 + xi_h1)
                  * np.exp(-np.minimum(gamma * xi_h1 / (1 + xi_h1), 50)))


def refine_noise(power, noise, cfg):
    """Offline: whole-file average of the power weighted by P(speech absent)."""
    pw = power[:, _inner(power.shape[1])]
    for _ in range(cfg.refine_iters):
        w = 1.0 - speech_presence(pw, noise[:, None], cfg)
        est = (w * pw).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-6)
        noise = np.maximum(np.where(w.sum(axis=1) > 2.0, est, noise), 1e-12)
    return noise


def track_noise(power, cfg):
    """SPP-based MMSE noise PSD tracking (Gerkmann & Hendriks, IEEE TASLP 2012)."""
    noise = refine_noise(power, initial_noise(power, cfg), cfg)
    p_bar = np.zeros(power.shape[0])
    out = np.empty_like(power)
    for m in range(power.shape[1]):
        y2 = power[:, m]
        p = speech_presence(y2, noise, cfg)
        p_bar = cfg.spp_alpha * p_bar + (1 - cfg.spp_alpha) * p
        p = np.where(p_bar > 0.99, np.minimum(p, 0.99), p)      # avoid stagnation
        noise = np.maximum(cfg.noise_alpha * noise + (1 - cfg.noise_alpha)
                           * ((1 - p) * y2 + p * noise), 1e-12)
        out[:, m] = noise
    return out


# ----------------------------------------------------------------------------- gain
def lsa_gain(xi, gamma, cfg):
    """MMSE log-spectral amplitude gain (Ephraim & Malah 1985), bounded."""
    v = np.maximum(xi * gamma / (1 + xi), 1e-10)
    g = xi / (1 + xi) * np.exp(0.5 * exp1(v))
    return np.clip(g, 10 ** (cfg.gain_floor_db / 20), 1.0)


def decision_directed(gamma, cfg):
    """A priori SNR, decision-directed estimate (Ephraim & Malah 1984) with LSA feedback."""
    xi_min = 10 ** (cfg.xi_min_db / 10)
    ml = np.maximum(gamma - 1, 0)
    out = np.empty_like(gamma)
    prev = None
    for m in range(gamma.shape[1]):
        xi = ml[:, m] if prev is None else cfg.dd_alpha * prev + (1 - cfg.dd_alpha) * ml[:, m]
        xi = np.maximum(xi, xi_min)
        out[:, m] = xi
        prev = lsa_gain(xi, gamma[:, m], cfg) ** 2 * gamma[:, m]   # |S_hat|^2 / noise
    return out


def enhance(x, sr=16000, cfg=Config()):
    """Denoise a 1-D waveform. Returns a float64 array of the same length."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < cfg.n_fft:
        return x.copy()
    _, _, z = _stft(x, cfg, sr)
    power = np.abs(z) ** 2
    noise = track_noise(power, cfg)
    gamma = np.minimum(power / noise, 1e4)
    xi_min = 10 ** (cfg.xi_min_db / 10)

    # stage 1: bidirectional decision-directed a priori SNR + LSA gain
    xi = np.sqrt(decision_directed(gamma, cfg) * decision_directed(gamma[:, ::-1], cfg)[:, ::-1])
    gains = lsa_gain(xi, gamma, cfg)

    # stage 2: harmonic regeneration
    if cfg.hrnr_rho < 1.0:
        _, _, zh = _stft(np.maximum(_istft(gains * z, n, cfg, sr), 0.0), cfg, sr)
        xi_h = (cfg.hrnr_rho * gains ** 2 * power + (1 - cfg.hrnr_rho) * np.abs(zh) ** 2) / noise
        gains = lsa_gain(np.maximum(xi_h, xi_min), gamma, cfg)

    # stage 3: frame-level speech-absence attenuation
    if cfg.vad_atten_db < 0:
        band = slice(int(len(power) * 0.0125), int(len(power) * 0.5))      # 100 Hz - 4 kHz
        frame_snr = (gains[band] ** 2 * power[band]).sum(0) / noise[band].sum(0)
        snr_db = 10 * np.log10(np.maximum(frame_snr, 1e-10))
        p = 1.0 / (1.0 + np.exp(-(snr_db - cfg.vad_theta_db) / cfg.vad_width_db))
        gains = gains * (p + (1 - p) * 10 ** (cfg.vad_atten_db / 20))[None, :]

    return _istft(gains * z, n, cfg, sr)
