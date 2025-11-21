import numpy as np
import matplotlib.pyplot as plt


gamma_c = 2.77e-4
wmin, wmax = -1e-3, 1e-3
n_w = 1801


tmax_ns = 50.0/gamma_c
nt = 8000  # increase for smoothness; ajuste si lent
taulist = np.linspace(0.0, tmax_ns, nt)
wlist = np.linspace(wmin, wmax, n_w)

# --- Test : cohérence entre wlist et freqs FFT ---
dt = taulist[1] - taulist[0]
N = len(taulist)
freqs_fft_cycles = np.fft.fftfreq(N, d=dt)        # cycles/ns
freqs_fft_rad = 2 * np.pi * freqs_fft_cycles      # rad/ns

# suppose que ton wlist vient des params.py
wlist_local = wlist  # copie ton axe tel qu'utilisé

# ---- Diagnostic ----
print("dt =", dt, "ns")
print("Length FFT =", N)
print("FFT frequency range:")
print("  cycles/ns : [{:.3g}, {:.3g}]".format(freqs_fft_cycles.min(), freqs_fft_cycles.max()))
print("  rad/ns     : [{:.3g}, {:.3g}]".format(freqs_fft_rad.min(), freqs_fft_rad.max()))
print("Your wlist range: [{:.3g}, {:.3g}]".format(wlist_local.min(), wlist_local.max()))

# Compare ratios
ratio_cycles = np.mean(np.diff(wlist_local)) / np.mean(np.diff(freqs_fft_cycles))
ratio_rad = np.mean(np.diff(wlist_local)) / np.mean(np.diff(freqs_fft_rad))
print(f"Ratio step (wlist / FFT_cycles) ≈ {ratio_cycles:.3f}")
print(f"Ratio step (wlist / FFT_rad) ≈ {ratio_rad:.3f}")

# Plot for visual check
plt.figure(figsize=(6,4))
plt.plot(freqs_fft_cycles, np.zeros_like(freqs_fft_cycles), 'b.', label='FFT axis (cycles/ns)')
plt.plot(freqs_fft_rad, np.zeros_like(freqs_fft_rad)+0.1, 'g.', label='FFT axis (rad/ns)')
plt.plot(wlist_local, np.zeros_like(wlist_local)+0.2, 'r.', label='wlist')
plt.legend()
plt.xlabel("frequency (ns⁻¹ units)")
plt.yticks([])
plt.title("Visual comparison of wlist vs FFT axes")
plt.show()
