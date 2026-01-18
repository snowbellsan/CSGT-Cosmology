import numpy as np
import pandas as pd
import os
import urllib.request
from scipy.integrate import quad
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 1. Physics Engine & Loaders
# =============================================================================
C_LIGHT = 299792.458
RD_FID = 147.09
Z_PEAK_FIXED = 0.7

def load_pantheon_final():
    url = "https://github.com/PantheonPlusSH0ES/DataRelease/raw/main/Pantheon%2B_Data/1_DISTANCES/Pantheon%2B_SH0ES.dat"
    fname = "Pantheon+SH0ES.dat"
    if not os.path.exists(fname):
        print("📡 Downloading Pantheon+ dataset...")
        urllib.request.urlretrieve(url, fname)
    
    df = pd.read_csv(fname, sep=r'\s+', engine='python', comment='#')
    z_col = next(c for c in df.columns if c.upper() in ['ZHD', 'ZHEL'])
    mu_col = next(c for c in df.columns if 'MU_SH0ES' in c.upper() or c.upper() == 'MU')
    err_col = next(c for c in df.columns if 'ERR' in c.upper())
    return df[[z_col, mu_col, err_col]].dropna().sort_values(z_col).reset_index(drop=True)

def w_z_csgt(z, A, sigma, w_off):
    return w_off + A * np.exp(-(z - Z_PEAK_FIXED)**2 / (2 * sigma**2))

def get_ez(z, A, sigma, w_off, Om):
    Ode = 1.0 - Om
    integrand = lambda zp: (1.0 + w_z_csgt(zp, A, sigma, w_off)) / (1.0 + zp)
    integral, _ = quad(integrand, 0, z)
    return np.sqrt(Om * (1 + z)**3 + Ode * np.exp(3.0 * integral))

def compute_mu_theory(z_array, A, sigma, w_off, Om, H0, M_fixed):
    z_grid = np.linspace(0, max(z_array)*1.05, 100)
    ez_inv = [1.0 / get_ez(zg, A, sigma, w_off, Om) for zg in z_grid]
    chi_grid = np.cumsum(np.concatenate(([0], np.diff(z_grid))) * ez_inv)
    interp_chi = interp1d(z_grid, chi_grid, kind='cubic')
    dl = (1 + z_array) * interp_chi(z_array) * (C_LIGHT / H0)
    return 5.0 * np.log10(np.maximum(dl, 1e-10)) + 25.0 + M_fixed

# =============================================================================
# 2. DESI DR2 Full Dataset (v2026 Reference)
# =============================================================================
DESI_DR2_FULL = [
    {'z': 0.142, 'dm_rd': 3.48, 'dm_err': 0.10, 'dh_rd': 27.21, 'dh_err': 1.10}, 
    {'z': 0.510, 'dm_rd': 12.77, 'dm_err': 0.19, 'dh_rd': 19.85, 'dh_err': 0.52}, 
    {'z': 0.706, 'dm_rd': 16.59, 'dm_err': 0.21, 'dh_rd': 17.54, 'dh_err': 0.41}, 
    {'z': 0.932, 'dm_rd': 21.05, 'dm_err': 0.33, 'dh_rd': 15.61, 'dh_err': 0.38}, 
    {'z': 1.112, 'dm_rd': 24.11, 'dm_err': 0.42, 'dh_rd': 14.32, 'dh_err': 0.45}, 
    {'z': 1.491, 'dm_rd': 29.58, 'dm_err': 0.61, 'dh_rd': 11.23, 'dh_err': 0.32}, 
    {'z': 2.330, 'dm_rd': 39.41, 'dm_err': 1.10, 'dh_rd': 8.52,  'dh_err': 0.25}  
]

def get_bao_full_chi2(params):
    A, sigma, w_off, Om, H0, _ = params
    chi2 = 0
    for data in DESI_DR2_FULL:
        z = data['z']
        ez = get_ez(z, A, sigma, w_off, Om)
        dh_theory = C_LIGHT / (H0 * ez)
        z_grid = np.linspace(0, z, 50)
        dm_theory = C_LIGHT / H0 * np.trapz([1.0/get_ez(zg, A, sigma, w_off, Om) for zg in z_grid], z_grid)
        chi2 += ((data['dm_rd'] - dm_theory/RD_FID)**2 / data['dm_err']**2)
        chi2 += ((data['dh_rd'] - dh_theory/RD_FID)**2 / data['dh_err']**2)
    return chi2

# =============================================================================
# 3. Objective & Solver
# =============================================================================
def final_joint_objective(params, z_sn, mu_sn, sig_sn):
    if not (0.2 < params[3] < 0.4 and 65 < params[4] < 80): return 1e18
    try:
        mu_th = compute_mu_theory(z_sn, *params)
        sig_int = 0.106
        chi2_sn = np.sum(((mu_sn - mu_th)**2) / (sig_sn**2 + sig_int**2))
        chi2_bao = get_bao_full_chi2(params)
        return chi2_sn + chi2_bao
    except:
        return 1e18

if __name__ == "__main__":
    df = load_pantheon_final()
    z_sn, mu_sn, sig_sn = df.iloc[:,0].values, df.iloc[:,1].values, df.iloc[:,2].values
    
    print("🚀 Initializing Ultra-Precision Joint Fit (Pantheon+ & DESI DR2 Full)...")
    bounds = [(0.1, 0.6), (0.2, 0.6), (-1.2, -0.8), (0.25, 0.35), (68, 76), (-0.05, 0.05)]
    
    res = differential_evolution(final_joint_objective, bounds, args=(z_sn, mu_sn, sig_sn), 
                                 popsize=15, maxiter=200, strategy='best1bin', disp=True)
    
    print("\n" + "⚔️"*30)
    print("   ULTIMATE COSMOLOGICAL CONVERGENCE")
    print("⚔️"*30)
    p = res.x
    print(f"Final Joint χ² : {res.fun:.2f}")
    print(f"Information Coupling (A) : {p[0]:.4f}")
    print(f"Hubble Constant (H0)     : {p[4]:.2f} km/s/Mpc")
    print(f"Matter Density (Om)      : {p[3]:.3f}")
    print(f"Reduced χ² (Total)       : {res.fun / (len(z_sn) + 14 - 6):.4f}")