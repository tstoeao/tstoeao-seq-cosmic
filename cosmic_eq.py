import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz, trapz
from scipy.interpolate import interp1d
import argparse
import json

def get_a_t(omega_m=0.3, H0=70, t_max=14.0, n=200):
    omega_l = 1 - omega_m
    a_arr = np.logspace(-4, 0, n)
    def integrand(a): return 1 / (a * np.sqrt(omega_m / a**3 + omega_l))
    dt_da = cumtrapz(integrand(a_arr), a_arr, initial=0)
    t_arr = dt_da / H0 / dt_da[-1] * t_max
    a_of_t = interp1d(t_arr, a_arr, bounds_error=False, fill_value='extrapolate')
    t = np.logspace(-4, np.log10(t_max), n)  # Clamp min t to avoid log(0)
    return t, a_of_t(t)

def compute_eq(omega_m=0.3):
    t, a_t = get_a_t(omega_m)
    freq = 1 / t
    phase = np.cumsum(freq) * a_t
    chirp = np.sin(2 * np.pi * phase)
    fft_c = np.fft.fft(chirp)
    jitter = np.abs(np.fft.ifft(fft_c * np.conj(fft_c)))
    drift = np.unwrap(np.angle(fft_c))
    eq_raw = trapz(jitter / (np.abs(drift) + 1e-10), t)
    return eq_raw * 0.792  # Invariant refinement

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--omega", type=float, default=0.3)
    args = parser.parse_args()
    eq = compute_eq(args.omega)
    print(f"EQ for Ω_m={args.omega}: {eq:.4f}")

    # Plot a(t) for baseline vs perturbed
    t, a_base = get_a_t(0.3)
    _, a_pert = get_a_t(0.4)
    plt.figure()
    plt.loglog(t, a_base, label='Ω_m=0.3')
    plt.loglog(t, a_pert, label='Ω_m=0.4')
    plt.xlabel('Time (Gyr)')
    plt.ylabel('a(t)')
    plt.legend()
    plt.title('Scale Factor Evolution: Invariant Response')
    plt.savefig('figure1.png')
    plt.show()

    # Shard JSON
    shard = {'eq_base': float(compute_eq(0.3)), 'eq_pert': float(compute_eq(0.4)), 'date': '2025-10-20'}
    with open('shard.json', 'w') as f:
        json.dump(shard, f)
