import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
VNA Synthetic Data Generator for Milk Adulteration (Urea Detection)
------------------------------------------------------------------
Role: Senior RF Microwave Engineer & Data Scientist
Physics Context: UHF RFID (Alien 9640) + VNA Measurement
RF Principle: Dielectric Perturbation & Resonant Shift
"""

def simulate_s11_curve(f, f0, gamma, depth, baseline, noise_level=0.05):
    """
    Generates a Lorentzian Return Loss dip (S11) with realistic noise.
    
    Mathematically: S11(f) = Baseline - Depth * [ (gamma^2) / ((f - f0)^2 + gamma^2) ]
    f: Frequency array (MHz)
    f0: Resonant Frequency (MHz)
    gamma: Linewidth (related to Q-factor)
    depth: Depth of the dip (dB)
    baseline: Far-from-resonance magnitude (dB)
    """
    # 1. Physics-based Lorentzian dip
    s11 = baseline - depth * (gamma**2 / ((f - f0)**2 + gamma**2))
    
    # 2. Add realistic hardware noise (Gaussian thermal noise)
    noise = np.random.normal(0, noise_level, len(f))
    
    # 3. Baseline drift (Minor low-freq instability)
    drift = np.linspace(np.random.normal(0, 0.02), np.random.normal(0, 0.02), len(f))
    
    return s11 + noise + drift

def generate_vna_dataset(num_samples=1000):
    print(f"Generating {num_samples} synthetic VNA scans...")
    
    # Frequency range: 850 MHz to 880 MHz (UHF band)
    freqs = np.linspace(850, 880, 150)
    
    data = []
    
    for i in range(num_samples):
        # A. Randomize Urea Percentage (0% to 5%)
        urea_pct = np.random.uniform(0, 5)
        
        # B. Class Label: Pure if Urea < 0.1%, else Adulterated
        label = 0 if urea_pct < 0.1 else 1
        
        # C. Apply RF Physics Rules:
        # 1. Resonant Shift: urea increases permittivity -> f0 shifts DOWN
        #    Rule: 866 MHz baseline, shifting -0.8 MHz per 1% urea.
        f0_shift = -0.8 * urea_pct
        f0 = 866.0 + f0_shift + np.random.normal(0, 0.05) # Add slight jitter
        
        # 2. Dampening: urea increases conductivity -> losses increase -> dip depth decreases
        #    Rule: 15 dB depth baseline, losing 1.5 dB per 1% urea.
        depth = 15.0 - (1.5 * urea_pct) + np.random.normal(0, 0.1)
        
        # 3. Q-Factor change: linewidth (gamma) increases with loss
        gamma = 2.0 + (0.1 * urea_pct)
        
        # D. Run Physics engine
        s11_scan = simulate_s11_curve(freqs, f0, gamma, depth, baseline=-0.5)
        
        # E. Package data
        row = {
            'Sample_ID': f"S_{i:04d}",
            'Urea_Percentage': round(urea_pct, 4),
            'Class_Label': label
        }
        # Add frequency columns: f_850.0, f_850.2, ...
        for idx, val in enumerate(s11_scan):
            row[f'f_{freqs[idx]:.2f}'] = val
            
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv('vna_adulteration_dataset.csv', index=False)
    print("Dataset saved to vna_adulteration_dataset.csv")
    return df, freqs

def visualize_dielectric_shift(freqs):
    print("Generating visualization of simulated dielectric shift...")
    
    plt.figure(figsize=(12, 6))
    plt.style.use('dark_background') # Professional high-contrast theme
    
    percentages = [0, 1, 3, 5]
    colors = ['#00FF00', '#FFFF00', '#FF8000', '#FF0000']
    
    for pct, color in zip(percentages, colors):
        f0 = 866.0 - (0.8 * pct)
        depth = 15.0 - (1.5 * pct)
        gamma = 2.0 + (0.1 * pct)
        
        scan = simulate_s11_curve(freqs, f0, gamma, depth, baseline=-0.5, noise_level=0.02)
        plt.plot(freqs, scan, label=f'Urea {pct}%', color=color, linewidth=2)
    
    plt.axvline(x=866, color='gray', linestyle='--', alpha=0.5, label='Baseline Resonance (866 MHz)')
    plt.title('VNA Synthetic Data: Simulated S11 Return Loss Shift (850-880 MHz)', fontsize=14, pad=15)
    plt.xlabel('Frequency (MHz)', fontsize=12)
    plt.ylabel('Magnitude S11 (dB)', fontsize=12)
    plt.legend(loc='lower right', frameon=True, facecolor='black')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('vna_simulated_shift.png')
    print("Visualization saved to vna_simulated_shift.png")

if __name__ == "__main__":
    df, freqs = generate_vna_dataset(num_samples=1000)
    visualize_dielectric_shift(freqs)
