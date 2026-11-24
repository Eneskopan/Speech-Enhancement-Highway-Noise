# Single-Channel Speech Enhancement Under Highway Noise - Emin Enes Kopan (220301021)
# Pipeline: recording -> band-pass (4 filters) -> noise estimate (Welch + tracking)
#           -> enhancement (spectral subtraction vs. log-MMSE) -> ISTFT -> evaluation

# %% 0) SETTINGS
import os, sys, subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from scipy.special import exp1

INPUT_WAV, FS = "kayit.wav", 16000                  # recording, processing rate (Hz)
WP, WS, RP, AS = [300, 3400], [150, 4000], 1.0, 40.0  # passband, stopband (Hz), ripple, attenuation (dB)
NOISE_SEC, NPERSEG, NOVERLAP = 3.0, 512, 384        # speech-free start (s); 32 ms Hann frames, 75 % overlap
ALPHA, BETA, GMIN_DB = 1.5, 0.05, -30               # spectral subtraction; traffic level in pauses (dB)
COLORS = {"Hamming": "#2a78d6", "Kaiser": "#1baf7a", "Butterworth": "#eb6834", "Chebyshev I": "#9c2f6e"}
if "__file__" in globals(): os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("output", exist_ok=True); os.makedirs("figures", exist_ok=True)

# %% 1) LOAD: mono, normalize, 48 kHz -> 16 kHz
fs_in, x = wavfile.read(INPUT_WAV)
x = x.astype(np.float64)
x = x.mean(axis=1) if x.ndim == 2 else x
x = signal.resample_poly(x / np.max(np.abs(x)), FS, fs_in)
t = np.arange(len(x)) / FS
print(f"Loaded {INPUT_WAV}: {len(x) / FS:.1f} s, {fs_in} Hz -> {FS} Hz")

# %% 2) FILTER DESIGN - one specification, four filters
f_ax = np.linspace(0, FS / 2, 8001)
pb, sb = (f_ax >= WP[0]) & (f_ax <= WP[1]), (f_ax <= WS[0]) | (f_ax >= WS[1])
H = lambda c: signal.freqz(c, 1, worN=f_ax, fs=FS)[1] if c.ndim == 1 else signal.sosfreqz(c, worN=f_ax, fs=FS)[1]
dB = lambda v: 20 * np.log10(np.abs(v) + 1e-12)
dw, fc = 150, [225, 3700]                           # narrowest transition band, FIR cutoffs (Hz)

h_ham = signal.firwin(int(np.ceil(3.3 * FS / dw)) | 1, fc, window="hamming", pass_zero=False, fs=FS)
n_kai, beta = signal.kaiserord(AS, dw / (FS / 2))
n_kai |= 1
while True:                                         # Kaiser formula is an estimate: lengthen until spec is met
    h_kai = signal.firwin(n_kai, fc, window=("kaiser", beta), pass_zero=False, fs=FS)
    if dB(H(h_kai))[pb].min() >= -RP and dB(H(h_kai))[sb].max() <= -AS: break
    n_kai += 2
n_b, wn_b = signal.buttord(WP, WS, RP, AS, fs=FS)
n_c, wn_c = signal.cheb1ord(WP, WS, RP, AS, fs=FS)
filters = {"Hamming": h_ham, "Kaiser": h_kai,
           "Butterworth": signal.butter(n_b, wn_b, btype="band", output="sos", fs=FS),
           "Chebyshev I": signal.cheby1(n_c, RP, wn_c, btype="band", output="sos", fs=FS)}

info = {}
print(f"\n1) FILTER DESIGN  ({WP[0]}-{WP[1]} Hz, ripple <= {RP:.0f} dB, >= {AS:.0f} dB stopband)")
print(f"   {'Filter':17s}{'Order':>6s}{'Mult/sample':>13s}   Delay")
for name, c in filters.items():
    fir = c.ndim == 1
    gd = (np.full(pb.sum(), (len(c) - 1) / 2) if fir else
          sum(signal.group_delay((s[:3], s[3:]), w=f_ax[pb], fs=FS)[1] for s in c)) / FS * 1e3
    info[name] = {"mult": len(c) if fir else 5 * len(c), "gd": gd}   # one 2nd-order section = 5 mult.
    delay = f"{gd[0]:.1f} ms (constant)" if fir else f"{gd.min():.1f}-{gd.max():.1f} ms"
    print(f"   {name + (' FIR' if fir else ' IIR'):17s}{len(c) - 1 if fir else 2 * len(c):6d}"
          f"{info[name]['mult']:13d}   {delay}")
a_tf = signal.butter(n_b, wn_b, btype="band", output="ba", fs=FS)[1]  # Butterworth as ONE polynomial
print(f"   Note: Butterworth as one (b, a) polynomial has {np.sum(np.abs(np.roots(a_tf)) >= 1)} poles"
      f" outside the unit circle (unstable) -> implemented as second-order sections (SOS)")

# %% 3) FILTERING - FIR: 'same' removes the constant delay, IIR: cascade of second-order sections
filtered = {n: np.convolve(x, c, mode="same") if c.ndim == 1 else signal.sosfilt(c, x) for n, c in filters.items()}

# %% 4-5) NOISE ESTIMATE AND SPEECH ENHANCEMENT
stft = lambda y: signal.stft(y, FS, window="hann", nperseg=NPERSEG, noverlap=NOVERLAP)
istft = lambda S, n: signal.istft(S, FS, window="hann", nperseg=NPERSEG, noverlap=NOVERLAP)[1][:n]

def spectral_subtraction(y):
    """Basic method: subtract a fixed noise magnitude (from the speech-free start), keep a floor."""
    f, tt, Y = stft(y)
    noise_mag = np.sqrt(np.mean(np.abs(Y[:, tt < NOISE_SEC]) ** 2, axis=1))
    S = np.maximum(np.abs(Y) - ALPHA * noise_mag[:, None], BETA * np.abs(Y))
    return istft(S * np.exp(1j * np.angle(Y)), len(y))

def enhance(y):
    """Improved method: noise tracking + log-MMSE gain (Ephraim-Malah) + strong suppression in pauses."""
    f, tt, Y = stft(y)
    P = np.abs(Y) ** 2
    noise = P[:, tt < NOISE_SEC].mean(axis=1) + 1e-12            # Welch average of the speech-free start
    G, SPP, N = np.zeros_like(P), np.zeros_like(P), np.zeros_like(P)
    p_smooth, G_prev, gam_prev = np.full(len(f), 0.5), np.ones(len(f)), np.ones(len(f))
    for k in range(P.shape[1]):
        # noise tracking: update the noise only where speech is unlikely (traffic is not constant)
        post = np.minimum(P[:, k] / noise, 50)
        p = 1 / (1 + (1 + 10 ** 1.5) * np.exp(-post * 10 ** 1.5 / (1 + 10 ** 1.5)))  # speech presence prob.
        p_smooth = 0.9 * p_smooth + 0.1 * p
        p = np.where(p_smooth > 0.99, np.minimum(p, 0.99), p)
        noise = 0.8 * noise + 0.2 * ((1 - p) * P[:, k] + p * noise) + 1e-12
        # log-MMSE gain with decision-directed a priori SNR
        gam = np.minimum(P[:, k] / noise, 1e4)
        xi = np.maximum(0.98 * G_prev ** 2 * gam_prev + 0.02 * np.maximum(gam - 1, 0), 10 ** -2.5)
        G[:, k] = np.minimum(xi / (1 + xi) * np.exp(0.5 * exp1(np.maximum(xi * gam / (1 + xi), 1e-8))), 1)
        G_prev, gam_prev, SPP[:, k], N[:, k] = G[:, k], gam, p, noise
    # pause detection per frame (0 = pause, 1 = speech) with a short hangover
    frame = np.clip((SPP[(f >= WP[0]) & (f <= WP[1])].mean(axis=0) - 0.2) / 0.2, 0, 1)
    hold = frame.copy()
    for k in range(1, len(frame)):
        hold[k] = max(frame[k], 0.9 * hold[k - 1] + 0.1 * frame[k])
    p = hold[None, :] * (0.7 + 0.3 * signal.convolve(SPP, np.ones((3, 1)) / 3, mode="same"))
    gmin = 10 ** (GMIN_DB / 20)
    return istft((np.maximum(G, gmin) ** p * gmin ** (1 - p)) * Y, len(y)), N   # speech: log-MMSE, pauses: gmin

results = {n: enhance(y) for n, y in filtered.items()}
enhanced = {n: r[0] for n, r in results.items()}

# %% 6) EVALUATION - estimated SNR: speech frames vs. pause frames (same frames for every signal)
f_s, t_s, Z = stft(filtered["Kaiser"])
band = (f_s >= WP[0]) & (f_s <= WP[1])
frame_snr = 10 * np.log10(np.mean(np.abs(Z[band]) ** 2, axis=0) / np.mean(results["Kaiser"][1][band], axis=0))
speech, pause = frame_snr > 6, frame_snr < 1.5

def snr(y):
    p = np.mean(np.abs(stft(y)[2]) ** 2, axis=0)
    return 10 * np.log10((p[speech].mean() - p[pause].mean()) / p[pause].mean())

pause_db = lambda y: 10 * np.log10(np.mean(np.abs(stft(y)[2][band][:, pause]) ** 2)
                                   / np.mean(np.abs(Z[band][:, pause]) ** 2))   # traffic level change in pauses
snr_orig = snr(x)
for n in filters:
    info[n]["snr_f"], info[n]["snr_e"] = snr(filtered[n]), snr(enhanced[n])
best = max(v["snr_e"] for v in info.values())
winner = min((n for n in info if info[n]["snr_e"] >= best - 0.5), key=lambda n: info[n]["mult"])
final, basic = enhanced[winner], spectral_subtraction(filtered[winner])

print(f"\n2) ESTIMATED SNR        {'filter only':>12s}{'filter + enhancement':>23s}")
print(f"   {'Original':18s}{snr_orig:9.1f} dB")
for n, v in info.items():
    print(f"   {n:18s}{v['snr_f']:9.1f} dB{v['snr_e']:17.1f} dB")
print(f"\n   BEST FILTER: {winner} - same sound quality (within 0.5 dB) with the fewest operations"
      f" ({info[winner]['mult']} multiplications per sample)")
print(f"\n3) ENHANCEMENT METHOD ({winner} filter)   traffic in pauses   estimated SNR")
print(f"   Basic spectral subtraction       {pause_db(basic):8.1f} dB       {snr(basic):6.1f} dB")
print(f"   Improved (log-MMSE + tracking)   {pause_db(final):8.1f} dB       {snr(final):6.1f} dB")

# %% 7) PLOTS
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))                      # Figure 1 - filter responses
for n, c in filters.items():
    ax[0].plot(f_ax / 1e3, dB(H(c)), color=COLORS[n], lw=1.6, label=n)
    ax[1].plot(f_ax[pb] / 1e3, info[n]["gd"], color=COLORS[n], lw=1.6, label=n)
ax[0].axhline(-AS, color="0.3", ls="--", lw=1, label=f"-{AS:.0f} dB limit")
ax[0].set(ylim=(-80, 5), xlabel="Frequency (kHz)", ylabel="Magnitude (dB)", title="Magnitude response")
ax[1].set(xlabel="Frequency (kHz)", ylabel="Group delay (ms)", title="Group delay in the passband")
for a in ax: a.grid(alpha=0.3); a.legend()
fig.suptitle("Figure 1 - Four band-pass filters, one specification"); fig.tight_layout()
fig.savefig("figures/1_filter_responses.png", dpi=150)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))                      # Figure 2 - quality and cost
for a, key, label in [(ax[0], "snr_e", "Estimated SNR after enhancement (dB)"),
                      (ax[1], "mult", "Multiplications per sample")]:
    bars = a.bar(list(info), [v[key] for v in info.values()], color=list(COLORS.values()), width=0.6)
    a.bar_label(bars, labels=[f"{v[key]:.1f}" if key == "snr_e" else f"{v[key]}" for v in info.values()],
                padding=3)
    a.set(ylabel=label); a.grid(axis="y", alpha=0.3); a.set_axisbelow(True); a.margins(y=0.12)
ax[0].axhline(snr_orig, color="0.3", ls="--", lw=1)
ax[0].set_title(f"Quality: higher is better (dashed = original, {snr_orig:.1f} dB)")
ax[1].set_title("Cost: IIR is much cheaper")
fig.suptitle(f"Figure 2 - Filter comparison (best: {winner})"); fig.tight_layout()
fig.savefig("figures/2_filter_comparison.png", dpi=150)

seg = filtered[winner][: int(NOISE_SEC * FS)]                         # Figure 3 - Welch noise PSD
f_w, P_w = signal.welch(seg, FS, window="hann", nperseg=NPERSEG, noverlap=NOVERLAP, detrend=False)
f_1, P_1 = signal.periodogram(seg[len(seg) // 2: len(seg) // 2 + NPERSEG], FS, window="hann", detrend=False)
fig, a = plt.subplots(figsize=(9, 4.5))
a.plot(f_1 / 1e3, 10 * np.log10(P_1 + 1e-20), color="0.6", lw=1, label="one 32 ms frame (periodogram)")
a.plot(f_w / 1e3, 10 * np.log10(P_w + 1e-20), color="#d98e04", lw=2.2,
       label=f"Welch average of K = {1 + (len(seg) - NPERSEG) // (NPERSEG - NOVERLAP)} frames")
a.set(xlim=(0, 5), ylim=(-140, None), xlabel="Frequency (kHz)", ylabel="Power (dB/Hz)",
      title=f"Figure 3 - Traffic noise spectrum (first {NOISE_SEC:.0f} s, after band-pass)")
a.grid(alpha=0.3); a.legend(); fig.tight_layout()
fig.savefig("figures/3_noise_psd.png", dpi=150)

fig, ax = plt.subplots(5, 1, figsize=(13, 12), sharex=True)          # Figure 4 - before / after
ax[0].plot(t, x, color="0.65", lw=0.4, label="original")
ax[0].plot(t, final, color="#0e9a8a", lw=0.4, label="enhanced")
ax[0].set(ylabel="Amplitude", title="Waveform"); ax[0].legend(loc="upper right")
for a, (title, y) in zip(ax[1:], [("Original (noisy)", x), (f"Band-passed ({winner})", filtered[winner]),
                                  (f"Basic spectral subtraction (alpha = {ALPHA})", basic),
                                  ("Improved: log-MMSE + noise tracking (final)", final)]):
    f, tt, Zs = stft(y)
    im = a.pcolormesh(tt, f / 1e3, 20 * np.log10(np.abs(Zs) + 1e-10), shading="auto",
                      cmap="magma", vmin=-100, vmax=-20)
    a.set(ylabel="kHz", ylim=(0, 8), title=title)
ax[-1].set_xlabel("Time (s)")
fig.colorbar(im, ax=ax.tolist(), label="dB", shrink=0.7, anchor=(0.0, 0.0))
fig.suptitle("Figure 4 - Before and after")
fig.savefig("figures/4_before_after.png", dpi=150)

# %% 8) SAVE AND PLAY AUDIO
def save(name, y):
    wavfile.write(f"output/{name}", FS, (0.9 * y / np.max(np.abs(y)) * 32767).astype(np.int16))
    return f"output/{name}"

tracks = [("Original recording", save("1_original.wav", x)), ("Enhanced (final)", save("2_enhanced.wav", final))]
save("3_bandpassed_only.wav", filtered[winner]); save("4_basic_spectral_subtraction.wav", basic)
print(f"\nSaved audio to {os.path.abspath('output')}\nSaved figures to {os.path.abspath('figures')}")

if "--quiet" not in sys.argv:
    plt.show(block=False); plt.pause(0.5)
    print("\n4) LISTENING")
    for label, path in tracks:
        input(f"   Press Enter to play: {label} ")
        subprocess.run(["afplay", path])                                 # macOS audio player
    plt.show()

    
