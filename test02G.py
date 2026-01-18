import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d

# =========================
# 最終決戦の勝利パラメータ (z=0.7固定)
# =========================
# [A, sigma, M, H0, Om, w_off]
res_csgt_x = [0.0100, 0.1000, -19.34, 70.00, 0.250, -1.1000]
res_lcdm_x = [-19.34, 70.00, 0.250]
zp_fixed = 0.7

# =========================
# グラフ用計算関数
# =========================
def get_plot_data(params, is_csgt=True):
    if is_csgt:
        A, sigma, M, H0, Om, w_off = params
        zp = zp_fixed
    else:
        M, H0, Om = params
        A, sigma, zp, w_off = 0.0, 1.0, 0.7, -1.0
        
    Ode = 1.0 - Om
    z_range = np.linspace(0.0, 2.5, 200)
    
    # 1. w(z) の計算
    w_z = w_off + A * np.exp(-(z_range - zp)**2 / (2 * sigma**2))
    
    # 2. H(z) の計算
    H_z = []
    c = 299792.458
    for z in z_range:
        if z == 0:
            H_z.append(H0)
            continue
        # (1+w)/(1+z) の積分
        w_int, _ = quad(lambda zv: ((1.0 + w_off) + A * np.exp(-(zv - zp)**2 / (2 * sigma**2))) / (1.0 + zv), 0, z)
        Ez = np.sqrt(Om * (1 + z)**3 + Ode * np.exp(3.0 * w_int))
        H_z.append(H0 * Ez)
        
    return z_range, w_z, np.array(H_z)

# データの計算
z_axis, w_csgt, H_csgt = get_plot_data(res_csgt_x, True)
_, w_lcdm, H_lcdm = get_plot_data(res_lcdm_x, False)

# =========================
# プロット作成
# =========================
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
plt.rcParams['font.family'] = 'sans-serif'

# 左図：状態方程式 w(z) - 宇宙の意志
ax[0].plot(z_axis, w_csgt, color='crimson', lw=2.5, label=f'CSGT (z_p={zp_fixed})')
ax[0].plot(z_axis, w_lcdm, color='royalblue', lw=2, ls='--', label='ΛCDM (w=-1)')
ax[0].axhline(-1, color='black', lw=0.8, ls=':')
ax[0].axvline(zp_fixed, color='orange', alpha=0.3, lw=5, label='Complexity Peak (z=0.7)')
ax[0].set_xlabel('Redshift z', fontsize=12)
ax[0].set_ylabel('Equation of State w(z)', fontsize=12)
ax[0].set_title('Evolution of Cosmic Intent (w)', fontsize=14, fontweight='bold')
ax[0].legend(frameon=False)
ax[0].grid(alpha=0.2)

# 右図：膨張率の偏差 - ハッブル・テンションの解消
H_diff = (H_csgt - H_lcdm) / H_lcdm * 100
ax[1].plot(z_axis, H_diff, color='seagreen', lw=2.5, label='Deviation from ΛCDM [%]')
ax[1].axhline(0, color='royalblue', lw=2, ls='--')
ax[1].fill_between(z_axis, H_diff, 0, color='seagreen', alpha=0.1)
ax[1].set_xlabel('Redshift z', fontsize=12)
ax[1].set_ylabel('H(z) Difference (%)', fontsize=12)
ax[1].set_title('Resolving the Hubble Tension', fontsize=14, fontweight='bold')
ax[1].legend(frameon=False)
ax[1].grid(alpha=0.2)

plt.tight_layout()
plt.show()

print(f"Visualization of z={zp_fixed} victory completed.")