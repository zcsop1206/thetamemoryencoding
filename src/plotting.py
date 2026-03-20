import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter
import numpy as np

PURPLE = '#7F77DD'
BLUE   = '#378ADD'
GRAY   = '#B4B2A9'
RED    = '#D85A30'

def make_figure(results: dict, save_path: str):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(
        'Theta-rhythmic episodic memory encoding\n'
        'Dense sampling paradigm · Behavioral replication of Biba et al. (2026)',
        fontsize=12, fontweight='bold'
    )
    
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # Panel A - Raw hit rate vs SOA
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.text(0.02, 0.96, 'A', transform=ax_a.transAxes, fontsize=14, fontweight='bold', va='top')
    ax_a.spines[['top','right']].set_visible(False)
    
    soas = results.get('soas', np.linspace(50, 500, 90))
    hit_rate = results.get('hit_rate_raw', np.random.uniform(0.5, 0.7, 90))
    
    ax_a.scatter(soas, hit_rate, alpha=0.3, color=GRAY)
    
    if len(hit_rate) > 9:
        smoothed = savgol_filter(hit_rate, window_length=9, polyorder=3)
        ax_a.plot(soas, smoothed, color=PURPLE)
    else:
        smoothed = hit_rate
        ax_a.plot(soas, smoothed, color=PURPLE)
        
    sem = results.get('hit_rate_sem', None)
    if sem is not None:
        ax_a.fill_between(soas, smoothed - sem, smoothed + sem, color=PURPLE, alpha=0.2)
        
    ax_a.set_xlabel("Stimulus onset asynchrony (ms)")
    ax_a.set_ylabel("Hit rate")
    
    # Panel B - Detrended signal
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.text(0.02, 0.96, 'B', transform=ax_b.transAxes, fontsize=14, fontweight='bold', va='top')
    ax_b.spines[['top','right']].set_visible(False)
    
    detrended = results.get('detrended', np.zeros_like(soas))
    windowed = results.get('windowed', np.zeros_like(soas))
    
    ax_b.plot(soas, detrended, color=GRAY, alpha=0.6, label='Detrended')
    ax_b.plot(soas, windowed, color=PURPLE, linewidth=1.5, label='Windowed')
    
    ax_b.set_xlabel("SOA (ms)")
    ax_b.set_ylabel("Detrended hit rate")
    
    max_val = np.max(windowed)
    ax_b.annotate('', xy=(100, max_val*0.8), xytext=(267, max_val*0.8),
                  arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
                  
    # Panel C - Power spectrum
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.text(0.02, 0.96, 'C', transform=ax_c.transAxes, fontsize=14, fontweight='bold', va='top')
    ax_c.spines[['top','right']].set_visible(False)
    
    freqs = results.get('freqs', np.linspace(1, 30, 100))
    power = results.get('power', np.ones_like(freqs))
    ap_fit = results.get('ap_fit', np.ones_like(freqs))
    
    log_power = np.log10(power)
    ax_c.fill_between(freqs, log_power, color='lightgray', alpha=0.5)
    ax_c.plot(freqs, ap_fit, color='black', linestyle='--')
    
    # Fill periodic residual
    ax_c.fill_between(freqs, ap_fit, log_power, where=(log_power > ap_fit), color=PURPLE, alpha=0.5)
    
    # Shade theta band
    ax_c.axvspan(3, 10, color=BLUE, alpha=0.15)
    
    theta_cf = results.get('theta_peak_cf', None)
    if theta_cf is not None:
        ax_c.axvline(theta_cf, color=PURPLE, linestyle='--')
        ax_c.text(theta_cf + 0.5, np.max(log_power)*0.9, f"CF = {theta_cf:.1f} Hz", color=PURPLE)
        
    r2 = results.get('r_squared', 0.0)
    ax_c.text(0.95, 0.95, f"R² = {r2:.2f}", transform=ax_c.transAxes, ha='right', va='top')
    
    ax_c.set_xlim(0, 20)
    ax_c.set_xlabel("Frequency (Hz)")
    ax_c.set_ylabel("log₁₀ Power")
    
    # Panel D - Permutation test
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.text(0.02, 0.96, 'D', transform=ax_d.transAxes, fontsize=14, fontweight='bold', va='top')
    ax_d.spines[['top','right']].set_visible(False)
    
    null_dist = results.get('null_dist', np.random.randn(5000))
    obs = results.get('observed_power', 2.0)
    pval = results.get('p_value', 0.01)
    z_score = results.get('effect_size', 2.5)
    n_perms = len(null_dist)
    
    counts, bins, patches = ax_d.hist(null_dist, bins=50, color=GRAY)
    for c, b, p in zip(counts, bins[:-1], patches):
        if b >= obs:
            p.set_facecolor(RED)
            p.set_alpha(0.4)
            
    ax_d.axvline(obs, color=RED)
    ax_d.text(0.95, 0.95, f"p = {pval:.3f}\nz = {z_score:.2f}\nn = {n_perms:,} permutations", 
              transform=ax_d.transAxes, ha='right', va='top')
              
    ax_d.set_xlabel("Theta peak power (null)")
    ax_d.set_ylabel("Count")
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    plt.savefig(save_path, dpi=300)
    plt.close()
