import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d
import os
import urllib.request
import warnings

# 数値計算の安定化
warnings.filterwarnings('ignore')

# =============================================================================
# 1. Theoretical Constants & Data URL
# =============================================================================
PANTHEON_URL = "https://github.com/PantheonPlusSH0ES/DataRelease/raw/main/Pantheon%2B_Data/1_DISTANCES/Pantheon%2B_SH0ES.dat"
PANTHEON_FILE = "Pantheon+SH0ES.dat"
C_LIGHT = 299792.458  # km/s
Z_PEAK_FIXED = 0.7    # 理論論文 [cite: 37]

# =============================================================================
# 2. Advanced Data Loader (Auto-Detection)
# =============================================================================
def load_pantheon_plus():
    if not os.path.exists(PANTHEON_FILE):
        print("📡 Downloading Pantheon+ dataset...")
        urllib.request.urlretrieve(PANTHEON_URL, PANTHEON_FILE)
    
    df = pd.read_csv(PANTHEON_FILE, sep=r'\s+', engine='python', comment='#')
    
    # 理論に最適なカラムを抽出
    z_col = next((c for c in df.columns if c.upper() in ['ZHD', 'ZHEL']), None)
    mu_col = next((c for c in df.columns if 'MU_SH0ES' in c.upper() or c.upper() == 'MU'), None)
    err_col = next((c for c in df.columns if 'ERR' in c.upper()), None)
    
    sn_df = df[[z_col, mu_col, err_col]].copy()
    sn_df.columns = ['z', 'mu_obs', 'sigma_mu']
    sn_df = sn_df.apply(pd.to_numeric, errors='coerce').dropna()
    
    # 物理的カットオフ
    sn_df = sn_df[(sn_df['z'] > 0.01) & (sn_df['z'] < 2.3)]
    return sn_df.sort_values('z').reset_index(drop=True)

# =============================================================================
# 3. CSGT Physics Core
# =============================================================================
def w_z_csgt(z, A, sigma, w_off):
    """論文式(2): w(z) = w_off + A * exp(-(z-zp)^2 / 2sigma^2) """
    return w_off + A * np.exp(-(z - Z_PEAK_FIXED)**2 / (2 * sigma**2))

def get_h_ratio(z, A, sigma, w_off, Om):
    """論文式(5)に基づく E(z) = H(z)/H0 の計算 [cite: 66]"""
    Ode = 1.0 - Om
    # 状態方程式の積分項
    integrand = lambda zp: (1.0 + w_z_csgt(zp, A, sigma, w_off)) / (1.0 + zp)
    integral, _ = quad(integrand, 0, z)
    return np.sqrt(Om * (1 + z)**3 + Ode * np.exp(3.0 * integral))

def compute_mu_theory(z_array, A, sigma, w_off, Om, H0, M):
    """距離係数 μ = 5 log10(dL/Mpc) + 25 + M"""
    # 高速計算のための補間
    z_grid = np.linspace(0, max(z_array)*1.05, 100)
    ez_inv = [1.0 / get_h_ratio(zg, A, sigma, w_off, Om) for zg in z_grid]
    
    chi_grid = np.cumsum(np.concatenate(([0], np.diff(z_grid))) * ez_inv)
    interp_chi = interp1d(z_grid, chi_grid, kind='cubic')
    
    chi = interp_chi(z_array)
    dl = (1 + z_array) * chi * (C_LIGHT / H0)
    
    return 5.0 * np.log10(np.maximum(dl, 1e-10)) + 25.0 + M

# =============================================================================
# 4. Statistical Engine
# =============================================================================
def objective_function(params, z_obs, mu_obs, sigma_mu, model_type):
    if model_type == 'csgt':
        A, sigma, w_off, Om, H0, M = params
    else:
        Om, H0, M = params
        A, sigma, w_off = 0.0, 1.0, -1.0 # LCDM equivalent
        
    try:
        mu_th = compute_mu_theory(z_obs, A, sigma, w_off, Om, H0, M)
        chi2 = np.sum(((mu_obs - mu_th) / sigma_mu)**2)
        return chi2 if np.isfinite(chi2) else 1e20
    except:
        return 1e20

# =============================================================================
# 5. Main Analysis Execution
# =============================================================================
def main():
    sn_data = load_pantheon_plus()
    z, mu, sig = sn_data['z'].values, sn_data['mu_obs'].values, sn_data['sigma_mu'].values
    
    print(f"\n🚀 Analyzing {len(z)} Supernovae...")

    # Bounds: [A, sigma, w_off, Om, H0, M]
    # Mは-19.3付近の微調整パラメータとして設定
    bounds_csgt = [(0.0, 0.5), (0.01, 1.0), (-1.3, -0.8), (0.2, 0.4), (60, 80), (-0.1, 0.1)]
    bounds_lcdm = [(0.2, 0.4), (60, 80), (-0.1, 0.1)]

    print("\n--- Fitting CSGT Model ---")
    res_csgt = differential_evolution(objective_function, bounds_csgt, args=(z, mu, sig, 'csgt'), 
                                      popsize=15, maxiter=200, disp=True, polish=True)

    print("\n--- Fitting ΛCDM Model ---")
    res_lcdm = differential_evolution(objective_function, bounds_lcdm, args=(z, mu, sig, 'lcdm'), 
                                      popsize=15, maxiter=100, disp=True)

    # 結果表示
    d_chi2 = res_lcdm.fun - res_csgt.fun
    print("\n" + "="*40)
    print(f"RESULTS FOR MASTER")
    print("="*40)
    print(f"χ² (ΛCDM): {res_lcdm.fun:.2f}")
    print(f"χ² (CSGT): {res_csgt.fun:.2f}")
    print(f"Δχ²      : {d_chi2:.2f} (CSGT improvement)")
    
    p = res_csgt.x
    print(f"\nBest Fit CSGT Params:")
    print(f"H0: {p[4]:.2f}, Om: {p[3]:.3f}, w_off: {p[2]:.3f}, A: {p[0]:.4f}, sigma: {p[1]:.3f}")

    # Plotting
    z_plot = np.linspace(0.01, 2.3, 100)
    w_plot = w_z_csgt(z_plot, p[0], p[1], p[2])
    plt.figure(figsize=(8, 5))
    plt.plot(z_plot, w_plot, 'r-', label='CSGT w(z) Evolution')
    plt.axhline(-1, color='k', ls='--', alpha=0.5, label='ΛCDM (w=-1)')
    plt.axvline(Z_PEAK_FIXED, color='orange', ls=':', label='Information Peak (z=0.7)')
    plt.xlabel('Redshift z'); plt.ylabel('w(z)'); plt.legend(); plt.grid(alpha=0.3)
    plt.title(f"CSGT vs ΛCDM: Δχ² = {d_chi2:.2f}")
    plt.show()

if __name__ == "__main__":
    main()