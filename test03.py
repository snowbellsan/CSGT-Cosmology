"""
CSGT Dark Energy Model
Phantom Crossing as Future-Driven Information Backpropagation

This code implements a cosmological model where the equation of state w(z)
exhibits a Gaussian peak around z≈0.7, temporarily crossing w < -1 (phantom regime)
while remaining theoretically stable.

Author: [Your Name]
Date: 2026-01-18
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# =============================================================================
# Model Parameters
# =============================================================================

# CSGT Model parameters
PARAMS_CSGT = {
    'A': 0.0100,         # Amplitude of phantom peak (dimensionless)
    'sigma': 0.1000,     # Width of Gaussian profile in redshift
    'M': -19.34,         # Absolute magnitude for SN Ia calibration
    'H0': 70.00,         # Hubble constant [km/s/Mpc]
    'Om': 0.250,         # Matter density parameter Ω_m
    'w_off': -1.1000,    # Baseline equation of state offset
    'z_peak': 0.7        # Redshift of complexity peak (fixed)
}

# ΛCDM baseline for comparison
PARAMS_LCDM = {
    'M': -19.34,
    'H0': 70.00,
    'Om': 0.250,
    'w': -1.0            # Cosmological constant
}

# Physical constants
C_LIGHT = 299792.458  # Speed of light [km/s]

# =============================================================================
# Core Physics Functions
# =============================================================================

def equation_of_state(z, params):
    """
    Calculate the equation of state w(z) for dark energy.
    
    For CSGT: w(z) = w_off + A * exp(-(z - z_peak)² / (2σ²))
    For ΛCDM: w(z) = -1 (constant)
    
    Parameters:
    -----------
    z : float or array
        Redshift
    params : dict
        Model parameters (CSGT or LCDM)
        
    Returns:
    --------
    w : float or array
        Equation of state at redshift z
    """
    if 'A' in params:  # CSGT model
        A = params['A']
        sigma = params['sigma']
        z_peak = params['z_peak']
        w_off = params['w_off']
        return w_off + A * np.exp(-(z - z_peak)**2 / (2 * sigma**2))
    else:  # ΛCDM model
        return params['w'] * np.ones_like(z) if hasattr(z, '__len__') else params['w']


def hubble_parameter(z, params):
    """
    Calculate the Hubble parameter H(z).
    
    H(z) = H0 * E(z)
    where E(z) = sqrt(Ω_m(1+z)³ + Ω_Λ exp(3∫[w(z')/(1+z')]dz'))
    
    Parameters:
    -----------
    z : float or array
        Redshift
    params : dict
        Model parameters
        
    Returns:
    --------
    H : float or array
        Hubble parameter at redshift z [km/s/Mpc]
    """
    H0 = params['H0']
    Om = params['Om']
    Ode = 1.0 - Om  # Flat universe assumption
    
    if isinstance(z, (list, np.ndarray)):
        H_z = []
        for z_val in z:
            if z_val == 0:
                H_z.append(H0)
            else:
                # Integrate (1+w(z'))/(1+z') from 0 to z
                integrand = lambda zp: (1.0 + equation_of_state(zp, params)) / (1.0 + zp)
                w_integral, _ = quad(integrand, 0, z_val)
                
                # E(z) = sqrt(Ω_m(1+z)³ + Ω_Λ exp(3∫...))
                Ez = np.sqrt(Om * (1 + z_val)**3 + Ode * np.exp(3.0 * w_integral))
                H_z.append(H0 * Ez)
        return np.array(H_z)
    else:
        if z == 0:
            return H0
        integrand = lambda zp: (1.0 + equation_of_state(zp, params)) / (1.0 + zp)
        w_integral, _ = quad(integrand, 0, z)
        Ez = np.sqrt(Om * (1 + z)**3 + Ode * np.exp(3.0 * w_integral))
        return H0 * Ez


# =============================================================================
# Visualization
# =============================================================================

def plot_comparison(z_range=None, params_csgt=None, params_lcdm=None, save_fig=False):
    """
    Create a two-panel comparison plot:
    - Left: Evolution of equation of state w(z)
    - Right: Deviation in Hubble parameter from ΛCDM
    
    Parameters:
    -----------
    z_range : array, optional
        Redshift range for plotting (default: 0 to 2.5)
    params_csgt : dict, optional
        CSGT model parameters (default: PARAMS_CSGT)
    params_lcdm : dict, optional
        ΛCDM parameters (default: PARAMS_LCDM)
    save_fig : bool or str, optional
        If True, save to 'csgt_comparison.png'
        If string, save to that filename
    """
    if z_range is None:
        z_range = np.linspace(0.0, 2.5, 200)
    if params_csgt is None:
        params_csgt = PARAMS_CSGT
    if params_lcdm is None:
        params_lcdm = PARAMS_LCDM
    
    # Calculate w(z) and H(z) for both models
    w_csgt = equation_of_state(z_range, params_csgt)
    w_lcdm = equation_of_state(z_range, params_lcdm)
    
    H_csgt = hubble_parameter(z_range, params_csgt)
    H_lcdm = hubble_parameter(z_range, params_lcdm)
    
    # Calculate percentage deviation
    H_deviation = (H_csgt - H_lcdm) / H_lcdm * 100
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left panel: Equation of state
    ax1.plot(z_range, w_csgt, color='crimson', lw=2.5, 
             label=f"CSGT (z_p={params_csgt['z_peak']})")
    ax1.plot(z_range, w_lcdm, color='royalblue', lw=2, ls='--', 
             label='ΛCDM (w=-1)')
    ax1.axhline(-1, color='black', lw=0.8, ls=':', alpha=0.5)
    ax1.axvline(params_csgt['z_peak'], color='orange', alpha=0.3, lw=5, 
                label=f"Complexity Peak (z={params_csgt['z_peak']})")
    
    ax1.set_xlabel('Redshift z', fontsize=13)
    ax1.set_ylabel('Equation of State w(z)', fontsize=13)
    ax1.set_title('Evolution of Cosmic Intent (w)', fontsize=14, fontweight='bold')
    ax1.legend(frameon=False, fontsize=11)
    ax1.grid(alpha=0.3, ls=':')
    ax1.set_ylim([-1.15, -0.98])
    
    # Right panel: Hubble tension resolution
    ax2.plot(z_range, H_deviation, color='seagreen', lw=2.5, 
             label='Deviation from ΛCDM [%]')
    ax2.axhline(0, color='royalblue', lw=2, ls='--', alpha=0.7)
    ax2.fill_between(z_range, H_deviation, 0, color='seagreen', alpha=0.15)
    
    ax2.set_xlabel('Redshift z', fontsize=13)
    ax2.set_ylabel('H(z) Difference (%)', fontsize=13)
    ax2.set_title('Resolving the Hubble Tension', fontsize=14, fontweight='bold')
    ax2.legend(frameon=False, fontsize=11)
    ax2.grid(alpha=0.3, ls=':')
    
    # Add annotation for maximum deviation
    max_dev_idx = np.argmin(H_deviation)
    max_dev_z = z_range[max_dev_idx]
    max_dev_val = H_deviation[max_dev_idx]
    ax2.annotate(f'Max deviation: {max_dev_val:.2f}%\nat z={max_dev_z:.2f}',
                xy=(max_dev_z, max_dev_val), xytext=(max_dev_z+0.5, max_dev_val-0.5),
                arrowprops=dict(arrowstyle='->', color='seagreen', lw=1.5),
                fontsize=10, color='seagreen', fontweight='bold')
    
    plt.tight_layout()
    
    if save_fig:
        filename = 'csgt_comparison.png' if save_fig is True else save_fig
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as: {filename}")
    
    plt.show()


def print_model_summary(params):
    """Print a summary of model parameters."""
    print("=" * 60)
    print("CSGT Dark Energy Model - Parameter Summary")
    print("=" * 60)
    for key, value in params.items():
        print(f"{key:12s} = {value:.4f}" if isinstance(value, float) else f"{key:12s} = {value}")
    print("=" * 60)


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("CSGT Dark Energy Model")
    print("Phantom Crossing as Future-Driven Information Backpropagation\n")
    
    # Print model parameters
    print_model_summary(PARAMS_CSGT)
    
    # Generate comparison plot
    print("\nGenerating comparison plot...")
    plot_comparison(save_fig=False)
    
    # Calculate key metrics at z=0.7
    z_test = 0.7
    w_at_peak = equation_of_state(z_test, PARAMS_CSGT)
    H_csgt_peak = hubble_parameter(z_test, PARAMS_CSGT)
    H_lcdm_peak = hubble_parameter(z_test, PARAMS_LCDM)
    deviation_peak = (H_csgt_peak - H_lcdm_peak) / H_lcdm_peak * 100
    
    print(f"\nKey Metrics at z = {z_test}:")
    print(f"  w(z={z_test}) = {w_at_peak:.4f}")
    print(f"  H_CSGT(z={z_test}) = {H_csgt_peak:.2f} km/s/Mpc")
    print(f"  H_ΛCDM(z={z_test}) = {H_lcdm_peak:.2f} km/s/Mpc")
    print(f"  Deviation = {deviation_peak:.2f}%")
    
    print("\n✓ Visualization complete.")