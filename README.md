# Atomic Magnetometer for Electromagnetic Induction Imaging  

**Authors**:  
- Boris Baudel – École normale supérieure Rennes, Department of Mechatronics  
- Pr. Ferrucio Renzoni – University College London, Department of Physics and Astronomy  
- Dr. Han Yao – University College London, Department of Physics and Astronomy  

---

## 📖 Introduction  
Electromagnetic induction imaging (EMI) leverages ultra-sensitive **atomic magnetometers (AM)**, specifically **radio-frequency atomic magnetometers (RF-AM)**. These devices detect oscillating magnetic fields with high precision, making them ideal for EMI applications.  

This study explores experimental values for **high-resolution EMI systems** applied to different materials. Focus is placed on:  
- **Sensitivity and frequency range** of RF-AM sensors  
- **Image and signal processing methods**  

A **single-channel rubidium RF-AM** was developed:  
- Operating near room temperature  
- Sensitivity: **55 fT/Hz**  
- Linewidth: **36 Hz**  
- Effective across the **kHz–MHz band**  
- Small sensor volume ⇒ improved spatial resolution  

Results show successful high-resolution EMI on materials with conductivities ranging from **6 × 10⁷ S/m** to **500 S/m**, for samples of a few cm³ and imaging resolution of ~**1 mm**.  

Potential biomedical applications include **heart conductivity imaging** (future work at ~2 MHz).  

---

## ⚙️ Experimental Setup  

We use an **unshielded RF-AM** based on Savukov *et al.* (2005).  
- **Core**: Rubidium (Rb) alkali vapor cell  
- **Spin polarization**: Circularly polarized pump beam + parallel DC bias field (**BBIAS**)  
- **Frequency tuning**: Helmholtz coils (Zeeman effect)  
- **Calibration**: Known AC magnetic field  
- **Detection**: Probe beam polarization rotation → polarimeter → lock-in amplifier + spectrum analyzer  

### Key Components  
- **Polarimeter**: Polarizing beam splitter + balanced photodiode (Thorlabs PDB210A)  
- **LIA**: Ametek 7280 DSP  
- **SA**: Anritsu MS2718B  

---

## 🌀 Electromagnetic Induction Imaging (EMI)  

- **Primary field (B₁)** induces eddy currents in sample  
- **Secondary field (B₂)** carries info about material properties  
- **Phase-sensitive mapping** of total field reconstructs the image  

---

## 🔧 Working Conditions  

- **Probe/Pump modulation**: Single path, 300 MHz  
- **RF sensitivity**: 2.05 × 10⁻¹²  
- **AOM modulation**:  
  - Pump: **1.4 V**  
  - Probe: **1.6 V**  
- **Temperature**: ~100 °C  

---

## 🖼️ Image Processing Techniques  

1. **Gaussian filter** – noise reduction, structure preservation  
2. **Convolution** – filter application  
3. **Cubic interpolation** – filling missing values  
4. **Distance calculation** – detection of structural features (e.g., two holes in a sample)  

---

## 📐 Lorentzian Fitting  

Lorentzian curves are fitted to magnetic resonance signals:  

$$
\tilde{S}_x (\omega_{RF}) = \frac{S_0 B_{RF} \gamma \, \Gamma}{4[(\omega_{RF} - \Omega_L)^2 + \Gamma^2/4]}
$$

$$
\tilde{S}_y (\omega_{RF}) = \frac{S_0 B_{RF} \gamma (\Omega_L - \omega_{RF})}{2[(\omega_{RF} - \Omega_L)^2 + \Gamma^2/4]}
$$

- **S̃x**: Lorentzian absorption component  
- **S̃y**: Dispersive component  
- **Γ**: Linewidth (linked to sensitivity)  
- **ΩL**: Larmor frequency  

This model enables refined analysis of resonance signals, improving sensitivity for **medical imaging** and **geophysical exploration**.  

---

## 🧪 Imaging Results (Copper Coin Example)  

The **Lock-in Amplifier (LIA)** provides four key outputs:  
- **X**: In-phase (absorptive)  
- **Y**: Out-of-phase (dispersive)  
- **R = √(X² + Y²)**: Amplitude  
- **Φ = arctan(Y/X)**: Phase  

These outputs, combined with **spectrum analyzer traces**, are stored for detailed data analysis.  

Applications include:  
- Quantum computing  
- Magnetic field sensing  
- Optical signal analysis  

---

## 📊 Figures  

- **Fig. 1**: Probe & Pump lasers with RF coil in portable AM  
- **Fig. 2**: Sample with two holes  
- **Fig. 3**: Experimental setup of AM for EMI  
- **Fig. 4**: Principle of EMI (polarizations + RF modulation)  
- **Fig. 5**: Data processing (raw → Gaussian filtering → gradient removal)  
- **Fig. 6**: Two-hole detection & centering  
- **Fig. 7**: Lorentzian fitting results (circular copper)  
- **Fig. 8**: Probe & pump laser lightpath with AOM control  

---

## 🚀 Future Work  

- Biomedical imaging (heart conductivity at 2 MHz)  
- Enhanced multi-channel magnetometer arrays  
- Advanced image reconstruction algorithms  

---

## 📚 References  

- Savukov, I. M., et al. (2005). *Tunable Atomic Magnetometer for Detection of Radio-Frequency Magnetic Fields*.  

---

👉 This project demonstrates the feasibility of **atomic magnetometry for high-resolution EMI**, bridging physics, signal processing, and imaging.  
