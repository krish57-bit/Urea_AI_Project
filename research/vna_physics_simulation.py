import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
VNA Physics Simulation for Urea Adulteration (Co-Founder Edition)
-----------------------------------------------------------------
System: UHF RFID Alien 9640 Squiggle + Keysight E5063A
Physics: Dielectric Perturbation in the Reactive Near-Field (3mm gap)
"""

def simulate_vna_scan(f, urea_pct, noise_lvl=0.08):
    """
    Simulates S11 Return Loss based on dielectric loading.
    - urea_pct: 0, 1, 3, or 5
    - Dielectric Loading: Urea increases permittivity (Freq Shift) and conductivity (Loss/Broadening)
    """
    
    # 1. Resonant Frequency (f0)
    # Baseline resonance is 866 MHz (Pure Milk). 
    # urea effectively increases the dielectric constant, lowering the resonance.
    f0_shift = -1.2 * urea_pct 
    f0 = 866.0 + f0_shift + np.random.normal(0, 0.1) # Jitter
    
    # 2. Return Loss Depth (Dampening)
    # Pure milk has a deep dip (~25dB). Urea adds ionic conductivity, dampening the resonance.
    depth_loss = 2.5 * urea_pct
    depth = 25.0 - depth_loss + np.random.normal(0, 0.2)
    
    # 3. Linewidth (Broadening / Q-Factor)
    # Urea increases the imaginary part of permittivity (loss tangent), broadening the dip.
    gamma = 2.0 + (0.3 * urea_pct)
    
    # 4. Generate Lorentzian Dip
    baseline = np.random.normal(-0.8, 0.05)
    s11 = baseline - depth * (gamma**2 / ((f - f0)**2 + gamma**2))
    
    # 5. Add Hardware Noise & Frequency Drift
    noise = np.random.normal(0, noise_lvl, len(f))
    jitter = np.sin(f / 10.0) * 0.05 # Minor spectral ripple
    
    return s11 + noise + jitter

def generate_co_founder_dataset(num_samples=1000):
    print("Generating High-Fidelity VNA Physics Dataset...")
    
    # Frequency Sweep: 800 - 950 MHz (180 points)
    freqs = np.linspace(800, 950, 180)
    
    data = []
    urea_classes = [0, 1, 3, 5]
    
    for i in range(num_samples):
        urea_pct = np.random.choice(urea_classes)
        
        # Simulate Scan
        scan = simulate_vna_scan(freqs, urea_pct)
        
        # Structure Row
        row = {
            'Sample_ID': f'T_{i:04d}',
            'Urea_Percentage': urea_pct,
            'Label': 0 if urea_pct == 0 else 1
        }
        
        for idx, val in enumerate(scan):
            row[f'f_{freqs[idx]:.2f}'] = round(val, 4)
            
        data.append(row)
        
    df = pd.DataFrame(data)
    df.to_csv('vna_physics_dataset.csv', index=False)
    print("Dataset saved to vna_physics_dataset.csv")
    return df, freqs

def visualize_co_founder_physics(freqs):
    print("Generating Publication-Quality Physics Visualization...")
    
    plt.figure(figsize=(12, 7))
    plt.style.use('bmh') # Clean engineering style
    
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#c0392b'] # Pure -> High Adulteration
    labels = ['Pure Milk (0%)', 'Low Urea (1%)', 'Med Urea (3%)', 'High Urea (5%)']
    
    for i, pct in enumerate([0, 1, 3, 5]):
        scan = simulate_vna_scan(freqs, pct, noise_lvl=0.03)
        plt.plot(freqs, scan, label=labels[i], color=colors[i], linewidth=2.5, alpha=0.9)
        
    plt.axvline(x=866, color='black', linestyle='--', alpha=0.4, label='Baseline Frequency (866 MHz)')
    plt.title('Simulated S11 Return Loss: Dielectric Shift & Dampening (Urea Adulteration)', fontsize=14)
    plt.xlabel('Frequency (MHz)', fontsize=12)
    plt.ylabel('Return Loss S11 (dB)', fontsize=12)
    plt.legend(loc='lower right', facecolor='white', frameon=True)
    plt.ylim(-30, 2)
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('vna_physics_shift.png', dpi=300)
    print("Visualization saved to vna_physics_shift.png")

if __name__ == "__main__":
    df, freqs = generate_co_founder_dataset(1000)
    visualize_co_founder_physics(freqs)
