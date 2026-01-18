import matplotlib.pyplot as plt
import numpy as np

# ベストフィットパラメータ (ULTIMATE JOINT 結果)
A_final = 0.5570
sigma_final = 0.470 # 前回実行時の値
w_off_final = -0.990 # 固定付近
z_peak = 0.7

def w_z_ultimate(z):
    # CSGTの状態方程式: w(z) = w_off - A * exp(-(z-0.7)^2 / (2*sigma^2))
    # ※ dipとして機能させるため A の符号に注意
    return w_off_final - A_final * np.exp(-(z - z_peak)**2 / (2 * sigma_final**2))

z_plot = np.linspace(0, 2.5, 300)
w_plot = [w_z_ultimate(zg) for zg in z_plot]

plt.figure(figsize=(12, 7), facecolor='#fdfcfc')
plt.plot(z_plot, w_plot, color='#D40072', lw=4, label='CSGT: Ultimate Joint (SN+BAO)')
plt.axhline(-1, color='#333333', ls='--', alpha=0.6, label='ΛCDM (w=-1)')

# ファントム領域（w < -1）を強調
plt.fill_between(z_plot, -1, w_plot, where=(np.array(w_plot) < -1), 
                 color='#FF69B4', alpha=0.2, label='Phantom / Information Domain')

# ピークの注釈
min_w = min(w_plot)
plt.scatter([z_peak], [min_w], color='#D40072', s=100, zorder=5)
plt.annotate(f'The Love Dip\nw_min ≈ {min_w:.3f}', 
             xy=(z_peak, min_w), xytext=(z_peak+0.2, min_w-0.1),
             arrowprops=dict(arrowstyle='->', lw=2, color='#D40072'),
             fontsize=14, fontweight='bold', color='#D40072')

plt.title("Cosmological Evolution: The Information-Theoretic Pulse", fontsize=16, fontweight='bold')
plt.xlabel("Redshift (z)", fontsize=13)
plt.ylabel("Equation of State w(z)", fontsize=13)
plt.ylim(-1.8, -0.8)
plt.xlim(0, 2.3)
plt.legend(loc='lower right', frameon=True, fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

plt.show()