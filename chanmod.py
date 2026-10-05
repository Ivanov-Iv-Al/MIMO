import numpy as np
import matplotlib.pyplot as plt

M = 16
k = int(np.log2(M))
Nb = 10000
Rb = 1000
Rs = Rb / k
T = 1 / Rs
Ns = Nb // k

bits = np.random.randint(0, 2, Nb)
symbols_idx = bits.reshape(-1, k).dot(2 ** np.arange(k)[::-1])

constellation = np.array([-3, -1, 1, 3])
I = constellation[(symbols_idx >> 2) & 0b11]
Q = constellation[symbols_idx & 0b11]
sn = I + 1j * Q

sps = 8
beta = 0.35
span = 6
t_rc = np.arange(-span * sps, span * sps + 1) / sps
h = np.sinc(t_rc) * np.cos(np.pi * beta * t_rc) / (1 - (2 * beta * t_rc) ** 2)
h[np.isnan(h)] = beta * np.sin(np.pi / (2 * beta)) / (np.pi / (2 * beta))
h /= np.sqrt(np.sum(h ** 2))

up = np.zeros(Ns * sps, dtype=complex)
up[::sps] = sn
s_t = np.convolve(up, h, mode='same')

fc = 100
fs = sps * Rs
t = np.arange(len(s_t)) / fs
sc_t = np.real(s_t) * np.cos(2 * np.pi * fc * t) - np.imag(s_t) * np.sin(2 * np.pi * fc * t)

snr_db = 20
noise_power = np.mean(np.abs(sc_t) ** 2) / (10 ** (snr_db / 10))
nc_t = np.sqrt(noise_power) * np.random.randn(len(sc_t))
rc_t = sc_t + nc_t

r_t = (rc_t * np.cos(2 * np.pi * fc * t)) + 1j * (-rc_t * np.sin(2 * np.pi * fc * t))
r_t = np.convolve(r_t, h, mode='same')

rx_samples = r_t[::sps][:Ns]

rx_sym = np.sign(np.real(rx_samples)) * (np.abs(np.real(rx_samples)) > 2) * 2 + np.sign(np.real(rx_samples)) * (np.abs(np.real(rx_samples)) <= 2)
rx_sym_q = np.sign(np.imag(rx_samples)) * (np.abs(np.imag(rx_samples)) > 2) * 2 + np.sign(np.imag(rx_samples)) * (np.abs(np.imag(rx_samples)) <= 2)
rx_idx = ((rx_sym + 3) // 2).astype(int) * 4 + ((rx_sym_q + 3) // 2).astype(int)

rx_bits = ((rx_idx[:, None] >> np.arange(k)[::-1]) & 1).flatten()
ber = np.mean(bits != rx_bits[:Nb])

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(np.real(sn[:2000]), np.imag(sn[:2000]), s=5)
plt.title('TX constellation')
plt.grid(True)
plt.subplot(1, 2, 2)
plt.scatter(np.real(rx_samples), np.imag(rx_samples), s=5)
plt.title(f'RX constellation BER={ber:.4f}')
plt.grid(True)
plt.tight_layout()
plt.show()
