# Urea AI Detection System

A TinyML-optimized sensing pipeline for detecting Milk Urea adulteration using UHF RFID (866 MHz) and 1D Convolutional Neural Networks (CNNs).

## Project Overview
This project leverages the dielectric perturbation theory of RF signals. Urea alters the complex permittivity of milk, causing a measurable shift in the resonant frequency and dampening of the $S_{11}$ Return Loss signal from a submerged RFID tag (Alien 9640 "Squiggle").

### Repository Structure
- **`data/`**: Core datasets for spectral analysis and composition mapping.
- **`firmware/`**: ESP32-S3 C++ code for Data Acquisition (DAQ) and Signal Pre-processing.
- **`hardware/`**: OpenSCAD monolithic 3D-printable sensing jig (3mm air gap).
- **`models/`**: Production-ready TFLite binaries for Edge AI deployment.
- **`research/`**: 1D-CNN training scripts, physics-driven VNA simulations, and multi-output regression models.
- **`scripts/`**: Production utilities, including the Serial Logger and Unified Analyst tool.

## Key Features
- **TinyML Optimized**: Urea detector model footprint of just **35.6 KB**.
- **Physics-Informed**: Simulation engine for VNA signatures based on Lorentzian oscillators.
- **Hardware-Hardened**: Firmware includes Watchdog oversight, Brownout protection, and Anomaly detection.
- **HIL Ready**: Integrated "Hardware-in-the-Loop" logging for production retraining.

## Quick Start
1. **Calibration**: Flash `firmware/milk_sensor_daq.ino` to an ESP32-S3.
2. **Data Collection**: Run `scripts/log_esp32_data.py` to capture real-world hardware scans.
3. **Training**: Use `research/retrain_production_model.py` to harden the CNN against your specific hardware noise.
4. **Analysis**: Run `scripts/milk_analyzer.py` for a full Milk Quality Audit Report.

---
**Technical Lead**: Krish (Technical Co-Founder)
**Role**: AI Engineering & RF Sensing
