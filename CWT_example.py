import matplotlib.pyplot as plt
import numpy as np
import pywt


def gaussian(x, x0, sigma):
    return np.exp(-np.power((x - x0) / sigma, 2.0) / 2.0)


def make_chirp(t, f_start, f_end, t1=1.0, method='linear'):
    if method == 'linear':
        k = (f_end - f_start) / t1
        instantaneous_freq = f_start + k * t
        phase = 2 * np.pi * (f_start * t + 0.5 * k * t ** 2)
    elif method == 'quadratic':
        k = (f_end - f_start) / (t1 ** 2)
        instantaneous_freq = f_start + k * t ** 2
        phase = 2 * np.pi * (f_start * t + (k / 3) * t ** 3)
    else:
        raise ValueError("Method must be 'linear' or 'quadratic'")
    chirp = np.sin(phase)
    return chirp, instantaneous_freq


# Generate time and chirps
time = np.linspace(0, 1, 1000)
chirp1, _ = make_chirp(time, 5, 50, method='linear')
chirp2, _ = make_chirp(time, 20, 100, method='linear')

# Composite signal
signal = chirp1 + 1.0 * chirp2
signal *= gaussian(time, 0.5, 0.2)


# CWT function — fixed argument name
def compute_cwt(sig, wavelet="cmor2.0-1.0", widths=np.geomspace(1, 2048, num=2000), sampling_period=1e-3):
    cwtmatr, freqs = pywt.cwt(sig, widths, wavelet, sampling_period=sampling_period)
    return np.abs(cwtmatr), freqs

# Compute CWTs
sampling_period = np.diff(time).mean()
cwt_chirp1, freqs = compute_cwt(chirp1, sampling_period=sampling_period)
cwt_composite, _ = compute_cwt(signal, sampling_period=sampling_period)

# Plot 2×2 grid
fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharex='col', sharey='row',
                        gridspec_kw={"height_ratios": [1, 2]})

# Top left: Chirp1 signal
axs[0, 0].plot(time, chirp1, color='teal')
axs[0, 0].set_ylabel("Signal")
axs[0, 0].set_title("Chirp Signal (5-50Hz)")

# Bottom left: Chirp1 scalogram
pcm1 = axs[1, 0].pcolormesh(time, freqs, cwt_chirp1, shading='auto', cmap='plasma')
axs[1, 0].set_ylabel("Frequency (Hz)")
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].set_ylim(ymax=120)
axs[1, 0].set_title("Chirp Scalogram")

# Top right: Composite signal
axs[0, 1].plot(time, signal, color='darkblue')
axs[0, 1].set_title("Composite Chirp Signal (5-50Hz and 20-100Hz)")

# Bottom right: Composite scalogram
pcm2 = axs[1, 1].pcolormesh(time, freqs, cwt_composite, shading='auto', cmap='plasma')
axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].set_title("Composite Scalogram")
axs[1, 1].set_ylim(ymax=120)

# Adjust layout manually to make space for colourbar
fig.subplots_adjust(right=0.88, hspace=0.4, wspace=0.3)

# Add standalone colourbar on the right
cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
fig.colorbar(pcm2, cax=cbar_ax, label="Magnitude")

# Save and show
plt.savefig("chirp_dual_scalogram.png", dpi=120)
plt.show()

