import numpy as np
import matplotlib.pyplot as plt

Nc = 64
N_pilot = 8
N_data = Nc - N_pilot
Ng = 16
N_sym_total = Ng + Nc

sto_true = 7
cfo_true = 0.25

pilot_indices = np.linspace(0, Nc - 1, N_pilot, dtype=int)
data_indices = np.delete(np.arange(Nc), pilot_indices)
pilot_value = 1 + 1j

h_channel = np.array([1.0, 0.4 + 0.31j, 0.1 - 0.2j])
H_channel_true = np.fft.fft(h_channel, n=Nc)

SNR_dB_range = np.arange(0, 15, 2)
ser_results = []
constellation_rx = []
last_H_estimated = None

def modulate_qpsk(num_symbols):
    bits = np.random.randint(0, 4, num_symbols)
    mapping = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
    return mapping[bits], bits

def demodulate_qpsk(symbols):
    bits = np.zeros(len(symbols), dtype=int)
    bits[(symbols.real < 0) & (symbols.imag >= 0)] = 1
    bits[(symbols.real < 0) & (symbols.imag < 0)] = 2
    bits[(symbols.real >= 0) & (symbols.imag < 0)] = 3
    return bits

for SNR_dB in SNR_dB_range:
    total_errors = 0
    total_symbols = 0

    for idx_iter in range(500):
        data_symbols, tx_bits = modulate_qpsk(N_data)

        ofdm_freq = np.zeros(Nc, dtype=complex)
        ofdm_freq[data_indices] = data_symbols
        ofdm_freq[pilot_indices] = pilot_value

        ofdm_time = np.fft.ifft(ofdm_freq, n=Nc)

        cp = ofdm_time[-Ng:]
        ofdm_tx = np.concatenate([cp, ofdm_time])

        ofdm_fading = np.convolve(ofdm_tx, h_channel)[:len(ofdm_tx)]

        n_axis = np.arange(len(ofdm_fading))
        ofdm_distorted = ofdm_fading * np.exp(1j * 2 * np.pi * n_axis * cfo_true / Nc)

        padding_front = np.zeros(sto_true, dtype=complex)
        padding_back = np.zeros(20, dtype=complex)
        ofdm_rx_stream = np.concatenate([padding_front, ofdm_distorted, padding_back])

        signal_power = np.mean(np.abs(ofdm_rx_stream)**2)
        snr_linear = 10 ** (SNR_dB / 10)
        noise_power = signal_power / snr_linear if signal_power > 0 else 1e-6
        noise = np.sqrt(noise_power / 2) * (np.random.randn(len(ofdm_rx_stream)) + 1j * np.random.randn(len(ofdm_rx_stream)))
        ofdm_rx_stream += noise

        L_stream = len(ofdm_rx_stream)
        cc = np.zeros(L_stream, dtype=complex)
        for n in range(Nc + Ng, L_stream - 5):
            w1 = ofdm_rx_stream[n - Ng + 1 : n + 1]
            w2 = ofdm_rx_stream[n - Nc - Ng + 1 : n - Nc + 1]
            num = np.sum(w1 * np.conjugate(w2))
            den = np.sqrt(np.sum(np.abs(w1)**2)) * np.sqrt(np.sum(np.abs(w2)**2))
            if den > 0:
                cc[n] = num / den

        n_max = np.argmax(np.abs(cc))
        m_est = n_max - Nc

        ofdm_rx_sync_time = ofdm_rx_stream[m_est : m_est + Nc]

        y_cp = ofdm_rx_stream[m_est - Ng : m_est]
        y_tail = ofdm_rx_stream[m_est + Nc - Ng : m_est + Nc]
        cfo_est = (1 / (2 * np.pi)) * np.angle(np.sum(np.conjugate(y_cp) * y_tail))

        n_corr = np.arange(len(ofdm_rx_sync_time))
        ofdm_rx_sync_time = ofdm_rx_sync_time * np.exp(-1j * 2 * np.pi * cfo_est / Nc * n_corr)

        ofdm_rx_freq = np.fft.fft(ofdm_rx_sync_time, n=Nc)

        H_LS_pilots = ofdm_rx_freq[pilot_indices] / pilot_value

        H_estimated = np.interp(np.arange(Nc), pilot_indices, H_LS_pilots)

        if SNR_dB == 12 and idx_iter == 0:
            last_H_estimated = H_estimated.copy()

        ofdm_rx_equalized = ofdm_rx_freq / H_estimated

        rx_data_symbols = ofdm_rx_equalized[data_indices]

        if SNR_dB == 8:
            constellation_rx.extend(rx_data_symbols)

        rx_bits = demodulate_qpsk(rx_data_symbols)
        total_errors += np.sum(tx_bits != rx_bits)
        total_symbols += N_data

    ser = total_errors / total_symbols
    ser_results.append(ser)

constellation_rx = np.array(constellation_rx)

plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
plt.semilogy(SNR_dB_range, ser_results, 'ro-', linewidth=2, label='SER (QPSK)')
plt.grid(True, which="both", linestyle="--")
plt.xlabel('SNR (dB)')
plt.ylabel('SER')
plt.title('SER')
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(constellation_rx.real, constellation_rx.imag, color='teal', s=2, alpha=0.6, label='Rx')
ideal_qpsk = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
plt.scatter(ideal_qpsk.real, ideal_qpsk.imag, color='red', marker='X', s=150, edgecolors='black', label='Ideal')
plt.grid(True, linestyle="--")
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.xlabel('I')
plt.ylabel('Q')
plt.title('Constellation (SNR = 8 dB)')
plt.axis('equal')
plt.legend()

plt.subplot(2, 2, (3, 4))
if last_H_estimated is not None:
    plt.plot(np.arange(Nc), np.abs(H_channel_true), 'b-', linewidth=2.5, label='Истинная АЧХ канала')
    plt.plot(np.arange(Nc), np.abs(last_H_estimated), 'r--', linewidth=2, label='Оцененная АЧХ (Интерполяция)')
    plt.scatter(pilot_indices, np.abs(last_H_estimated[pilot_indices]), color='black', s=60, zorder=5, label='Оценки на пилотах (LS)')
plt.grid(True, linestyle="--")
plt.xlabel('Индекс поднесущей (k)')
plt.ylabel('|H[k]|')
plt.title('Амплитудно-частотная характеристика (АЧХ) канала')
plt.xlim([0, Nc - 1])
plt.legend()

plt.tight_layout()
plt.show()

for snr, ser in zip(SNR_dB_range, ser_results):
    print(f"SNR: {snr:2d} dB | SER: {ser:.5f}")
