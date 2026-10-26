import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

fc = 2e9
c  = 3e8
lam = c / fc
F = 16
dx = lam / F
V = 10.0
N = 100
NFFT = 128
D = 1000.0
k = 2*np.pi / lam

n_idx = np.arange(N)
x = n_idx * dx
d = D - x
t = n_idx * (dx / V)
ts = dx / V
fs = 1.0 / ts

r1 = np.exp(-1j * k * d)

amp1 = np.abs(r1)
phase1 = np.unwrap(np.angle(r1))

R1 = np.fft.fftshift(np.fft.fft(r1, NFFT))
f_axis = (np.arange(-(NFFT//2), NFFT//2)) * fs / NFFT
dop1 = np.abs(R1)


fD_theory = V / lam

print(f"Длина волны λ = {lam:.4f} м")
print(f"Шаг Δx = {dx:.4f} м, ts = {ts:.6e} с, fs = {fs:.3e} Гц")
print(f"Амплитуда |r| = {amp1[0]:.3f} (постоянна), фаза меняется линейно")
print(f"Теоретическая доплеровская частота f_D ≈ V/λ = {fD_theory:.2f} Гц")
print(f"Пик спектра Доплера на частоте ≈ {f_axis[np.argmax(dop1)]:.2f} Гц")

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0,0].plot(t*1e3, amp1); ax[0,0].set_title('|r(t)| — амплитуда'); ax[0,0].set_xlabel('t, мс')
ax[0,1].plot(t*1e3, phase1); ax[0,1].set_title('arg r(t) — фаза'); ax[0,1].set_xlabel('t, мс')
ax[1,0].plot(t*1e3, np.real(r1)); ax[1,0].set_title('Re r(t)'); ax[1,0].set_xlabel('t, мс')
ax[1,1].plot(f_axis, dop1); ax[1,1].set_title('Спектр Доплера (|ДПФ|)'); ax[1,1].set_xlabel('f, Гц')
ax[1,1].axvline(fD_theory, color='r', ls='--', label=f'V/λ={fD_theory:.1f} Гц')
ax[1,1].legend()
plt.tight_layout(); plt.show()

rng = np.random.default_rng(42)

BS = np.array([1000.0, 1000.0])
MS0 = np.array([0.0, 0.0])

Nsc = 200
rad = rng.uniform(50, 300, Nsc)
ang = rng.uniform(0, 2*np.pi, Nsc)
SC = np.column_stack([MS0[0] + rad*np.cos(ang),
                      MS0[1] + rad*np.sin(ang)])

plt.figure()
plt.scatter(SC[:,0], SC[:,1], s=8, label='Рассеиватели (SC)')
plt.scatter(*BS, c='r', marker='^', s=150, label='BS')
plt.scatter(*MS0, c='k', marker='o', s=80, label='MS (старт)')
plt.axis('equal'); plt.legend(); plt.title('Исходный сценарий')
plt.xlabel('x, м'); plt.ylabel('y, м'); plt.grid(True); plt.show()

Ntr = 1000
V2 = 10.0

track_len = 300.0
x_tr = np.linspace(0, track_len, Ntr)
y_tr = np.zeros(Ntr)
MS = np.column_stack([x_tr, y_tr])
t2 = np.arange(Ntr) * (track_len/(Ntr-1)) / V2

plt.figure()
plt.scatter(SC[:,0], SC[:,1], s=8, label='SC')
plt.scatter(*BS, c='r', marker='^', s=150, label='BS')
plt.plot(MS[:,0], MS[:,1], 'k-', lw=2, label='трек MS')
plt.axis('equal'); plt.legend(); plt.title('Трек MS и рассеиватели')
plt.xlabel('x, м'); plt.ylabel('y, м'); plt.grid(True); plt.show()

dBSSC = np.linalg.norm(BS[None,:] - SC, axis=1)              # (Nsc,)
# dSCMS для каждой точки трека: (Nsc, Ntr)
dx_sc = SC[:,0][:,None] - MS[:,0][None,:]                    # (Nsc, Ntr)
dy_sc = SC[:,1][:,None] - MS[:,1][None,:]
dSCMS = np.sqrt(dx_sc**2 + dy_sc**2)
dBSSCMS = dBSSC[:,None] + dSCMS                              # (Nsc, Ntr)

phi0 = rng.uniform(0, 2*np.pi, Nsc)
r_paths = np.exp(1j*(phi0[:,None] - k*dBSSCMS))
r2 = r_paths.sum(axis=0) / np.sqrt(Nsc)                      # (Ntr,)

amp2 = np.abs(r2)
phase2 = np.unwrap(np.angle(r2))

NFFT2 = 1024
R2 = np.fft.fftshift(np.fft.fft(r2 - r2.mean(), NFFT2))
fs2 = 1.0 / (t2[1] - t2[0])
f2 = (np.arange(-(NFFT2//2), NFFT2//2)) * fs2 / NFFT2
dop2 = np.abs(R2)

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
ax[0,0].plot(t2, amp2); ax[0,0].set_title('|r(t)| — модуль (Релей)'); ax[0,0].set_xlabel('t, с')
ax[0,1].plot(t2, phase2); ax[0,1].set_title('Фаза r(t)'); ax[0,1].set_xlabel('t, с')
ax[1,0].plot(t2, np.real(r2)); ax[1,0].set_title('Re r(t)'); ax[1,0].set_xlabel('t, с')
ax[1,1].plot(f2, dop2); ax[1,1].set_title('Спектр Доплера'); ax[1,1].set_xlabel('f, Гц')
plt.tight_layout(); plt.show()

plt.figure()
plt.hist(amp2, bins=40, density=True, alpha=0.6, label='гистограмма |r|')
# Теоретическое распределение Релея: sigma = sqrt(E[|r|^2]/2)
sigma = np.sqrt(np.mean(amp2**2)/2)
xx = np.linspace(0, amp2.max()*1.2, 300)
plt.plot(xx, stats.rayleigh.pdf(xx, scale=sigma), 'r', lw=2, label='Рэлей (теория)')
plt.title('Гистограмма модуля |r| (нормирована)'); plt.legend(); plt.grid(True); plt.show()

plt.figure()
xs = np.sort(amp2); ys = np.arange(1, len(xs)+1)/len(xs)
plt.step(xs, ys, where='post', label='эмпирическая CDF')
plt.plot(xx, stats.rayleigh.cdf(xx, scale=sigma), 'r--', lw=2, label='Рэлей CDF')
plt.title('Функция распределения |r|'); plt.legend(); plt.grid(True); plt.show()

ac = np.correlate(r2 - r2.mean(), r2 - r2.mean(), mode='full')
ac = ac / ac.max()
lags = np.arange(-len(r2)+1, len(r2))
plt.figure()
plt.plot(lags, np.abs(ac))
plt.title('Автокорреляция принятого сигнала'); plt.xlabel('лаг, отсчеты'); plt.grid(True); plt.show()

print(f"Средняя мощность E|r|^2 = {np.mean(amp2**2):.4f}, σ_Рэлея = {sigma:.4f}")
print(f"Среднее |r| = {amp2.mean():.4f}, теория σ√(π/2) = {sigma*np.sqrt(np.pi/2):.4f}")
print(f"Дисперсия |r| = {amp2.var():.4f}, теория (2-π/2)σ² = {(2-np.pi/2)*sigma**2:.4f}")