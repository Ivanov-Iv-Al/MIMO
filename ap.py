import numpy as np
import matplotlib.pyplot as plt

f = 2e+9
lam = 299792458/f
d = lam / 2
theta_deg = np.linspace(-180, 180, 3601)
theta = np.deg2rad(theta_deg)

def steering_vector(N, theta, d, lam, theta0_deg=None):
    k = 2 * np.pi * d / lam
    delta = 0.0
    if theta0_deg is not None:
        delta = -k * np.sin(np.deg2rad(theta0_deg))
    n = np.arange(N)
    return np.exp(1j * (n[:, None] * k * np.sin(theta)[None, :] + n[:, None] * delta))

def af_sum(a, w=None):
    if w is None:
        w = np.ones(a.shape[0], dtype=complex)
    return w.conj() @ a

def af_formula(N, theta, d, lam, theta0_deg=None):
    k = 2 * np.pi * d / lam
    delta = 0.0
    if theta0_deg is not None:
        delta = -k * np.sin(np.deg2rad(theta0_deg))
    psi = k * np.sin(theta) + delta
    num = np.sin(N * psi / 2)
    den = np.sin(psi / 2)
    out = np.where(np.abs(den) < 1e-12, N, num / np.where(np.abs(den) < 1e-12, 1, den))
    return out

def element_pattern(theta, theta3db=70.0, Am=20.0):
    return -np.minimum(12 * (theta / theta3db) ** 2, Am)

def polar_plot(ax, theta, gain_lin, title):
    ax.plot(theta, gain_lin)
    ax.set_title(title)
    ax.grid(True)

N_list = [4, 8]

plt.figure(figsize=(14, 10))

for i, N in enumerate(N_list):
    a = steering_vector(N, theta, d, lam)
    af_num = np.array([af_sum(a[:, j]) for j in range(len(theta))])
    af_ref = af_formula(N, theta, d, lam)
    err = np.max(np.abs(af_num - af_ref))
    print(f"N={N}: max|AF_sum - AF_formula| = {err:.3e}")

    ax = plt.subplot(2, 2, i + 1, projection='polar')
    polar_plot(ax, theta, np.abs(af_num) ** 2, f"ДН АР N={N} (|AF|^2)")

    ax2 = plt.subplot(2, 2, i + 3, projection='polar')
    g = element_pattern(theta)
    g_lin = 10 ** (g / 10)
    g_aa = np.abs(af_num) ** 2 * g_lin
    polar_plot(ax2, theta, g_aa, f"ДН АР N={N} с ДН элемента")

plt.tight_layout()
plt.show()

theta0_list = [0, 30, -45]
plt.figure(figsize=(14, 10))
for i, theta0 in enumerate(theta0_list):
    for j, N in enumerate(N_list):
        a = steering_vector(N, theta, d, lam, theta0_deg=theta0)
        af_num = np.array([af_sum(a[:, k]) for k in range(len(theta))])
        af_norm = np.abs(af_num) ** 2 / N ** 2
        ax = plt.subplot(len(theta0_list), len(N_list), i * len(N_list) + j + 1, projection='polar')
        polar_plot(ax, theta, af_norm, f"N={N}, theta0={theta0}°")
plt.tight_layout()
plt.show()

Nv = 4
Nh = 8
theta_upa = np.deg2rad(np.linspace(0, 180, 361))
phi_upa = np.deg2rad(np.linspace(-180, 180, 721))
TH, PH = np.meshgrid(theta_upa, phi_upa)

def af_upa(Nv, Nh, TH, PH, theta0_deg=None, phi0_deg=None):
    dv = lam / 2
    dh = lam / 2
    kv = 2 * np.pi * dv / lam
    kh = 2 * np.pi * dh / lam
    mv = np.arange(Nv)
    nh = np.arange(Nh)
    deltav = 0.0
    deltah = 0.0
    if theta0_deg is not None:
        deltav = -kv * np.cos(np.deg2rad(theta0_deg))
    if phi0_deg is not None and theta0_deg is not None:
        deltah = -kh * np.sin(np.deg2rad(theta0_deg)) * np.sin(np.deg2rad(phi0_deg))
    AFv = np.zeros_like(TH, dtype=complex)
    AFh = np.zeros_like(TH, dtype=complex)
    for m in mv:
        AFv += np.exp(1j * (kv * m * np.cos(TH) + m * deltav))
    for n in nh:
        AFh += np.exp(1j * (kh * n * np.sin(TH) * np.sin(PH) + n * deltah))
    return AFv, AFh

def plot_upa(TH, PH, gain, title, step=4):
    THs = TH[::step, ::step]
    PHs = PH[::step, ::step]
    R = gain[::step, ::step]
    R = R / R.max()
    X = R * np.sin(THs) * np.cos(PHs)
    Y = R * np.sin(THs) * np.sin(PHs)
    Z = R * np.cos(THs)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0,
                    antialiased=False, shade=True)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_title(title)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.show()

AFv, AFh = af_upa(Nv, Nh, TH, PH)
g_elem = 10 ** (element_pattern(np.rad2deg(TH)) / 10)
G_upa = np.abs(AFv) ** 2 * np.abs(AFh) ** 2 * g_elem
G_upa_norm = G_upa / np.max(G_upa)
plot_upa(TH, PH, G_upa_norm, f"ДН UPA {Nv}x{Nh}, theta0=0, phi0=0")

AFv, AFh = af_upa(Nv, Nh, TH, PH, theta0_deg=60, phi0_deg=30)
G_upa = np.abs(AFv) ** 2 * np.abs(AFh) ** 2 * g_elem
G_upa_norm = G_upa / np.max(G_upa)
plot_upa(TH, PH, G_upa_norm, f"ДН UPA {Nv}x{Nh}, theta0=60°, phi0=30°")
