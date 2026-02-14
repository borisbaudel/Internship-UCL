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
## Project Summary — Resonant Scan Processing & Eddy Current Mapping

This project processes frequency-scan measurements acquired on a 2D grid (e.g., over a conductive target)
to reconstruct spatial maps of resonant response parameters and derive **eddy-current distributions**
from a magnetic-field-related signal proxy.

Pipeline overview:

1. **Per-pixel spectral fitting** (Lorentzian + dispersive response)
2. **Scan re-ordering** (serpentine scan correction) + smoothing
3. **Optional feature metrology** from averaged cross-sections
4. **Eddy-current mapping** (first-order proportional model; diffusion-based physics in references)

---

## 1) Per-pixel Lorentzian / dispersive fitting (spectral feature extraction)

For each pixel of the scan grid `(x_dim × y_dim)`, the measured traces are:

- `data_x(pixel, :)` : in-phase / absorptive-like channel  
- `data_y(pixel, :)` : quadrature / dispersive-like channel  
- `data_w(pixel, :)` : frequency axis

Two models are fitted using nonlinear least squares (`lsqcurvefit`):

**Lorentzian amplitude model (X-channel)**
\[
X(\omega) = A \frac{\gamma^2}{\gamma^2 + (\omega-\omega_0)^2} + C
\]

**Dispersive / derivative-Lorentzian model (Y-channel)**
\[
Y(\omega) = A \gamma \frac{\omega-\omega_0}{\gamma^2 + (\omega-\omega_0)^2} + C
\]

This absorptive/dispersive decomposition is standard for resonance lineshapes and quadrature detection.

**References**
- S. M. Kay, *Fundamentals of Statistical Signal Processing: Estimation Theory*, Prentice Hall, 1993 (least-squares / parameter estimation background).
- A. Oppenheim, A. Willsky, S. Nawab, *Signals and Systems*, 2nd ed., Prentice Hall (quadrature signals / LTI systems background).
- For resonance lineshapes and dispersive quadratures: see standard spectroscopy/lock-in detection treatments (e.g., Stanford Research Systems lock-in amplifier application notes; general lock-in detection references).

---

## 2) Scan re-ordering → image reconstruction (serpentine scan correction)

The acquisition is performed in a **serpentine pattern** (alternating scan direction row by row).
`process_image_single2()` restores the correct spatial arrangement:

- even rows kept left→right  
- odd rows flipped right→left  
- global rotation to match the lab coordinate convention

A light Gaussian smoothing is applied to reduce pixel noise while preserving spatial structure.

**References**
- R. C. Gonzalez, R. E. Woods, *Digital Image Processing*, 4th ed., Pearson (image filtering, smoothing, interpolation concepts).
- MATLAB documentation: `imgaussfilt`, `fspecial('gaussian')`, `conv2`, `padarray` (implementation details).

---

## 3) Spatial feature metrology (optional)

`interpo22()` computes a 1D cross-section by averaging a band of rows, then:
- interpolates the profile (cubic interpolation),
- smooths it,
- extracts extrema on left/right segments,
- estimates a characteristic spacing (e.g., diameter proxy) from midpoints of max/min pairs.

**References**
- Gonzalez & Woods, *Digital Image Processing* (profile extraction, smoothing, peak detection basics).
- MATLAB Curve Fitting Toolbox documentation: `fit(...,'cubicinterp')`, smoothing utilities.

---

## 4) Eddy-current estimation and visualization

Assuming the reconstructed map is proportional to a magnetic-field amplitude \(B(x,y)\),
the code computes a **first-order eddy-current density magnitude proxy**:

\[
J(x,y) \approx \sigma \, B(x,y) \, \omega \, d
\]

where:
- \(\sigma\) is electrical conductivity (e.g., copper \(\sigma \approx 5.8\times10^7\,\mathrm{S/m}\))
- \(d\) is thickness (m)
- \(\omega = 2\pi f\) is angular frequency (rad/s)

Contours of `J(x,y)` are overlaid on the background map to visualize spatial current distribution.

**Important note (physics):**  
This proportional relation is intended for **qualitative mapping / visualization**.  
For quantitative eddy-current reconstruction, the correct physics is governed by Maxwell’s equations,
which reduce (in good conductors, harmonic regime) to a **diffusion equation** with skin depth:
\[
\delta = \sqrt{\frac{2}{\mu \sigma \omega}}
\]
and geometry/boundary conditions strongly affect the actual current distribution.



<img width="832" height="1166" alt="image" src="https://github.com/user-attachments/assets/b6f4361a-dc8b-47d9-aa5a-0c1e6ae04a2e" />



**References (eddy currents & EM diffusion)**
- D. J. Griffiths, *Introduction to Electrodynamics*, 4th ed., Pearson (conductors, skin depth, EM in matter).
- J. D. Jackson, *Classical Electrodynamics*, 3rd ed., Wiley (harmonic fields in conductors, diffusion/skin effect).
- C. V. Dodd, W. E. Deeds, “Analytical Solutions to Eddy-Current Probe-Coil Problems,” *Journal of Applied Physics* 39, 2829 (1968) — classic analytic eddy-current coil/half-space solutions.
- N. Bowler, “Eddy-current interaction with an ideal crack,” *Journal of Applied Physics* 75, 8128 (1994) — eddy currents and conductivity/defect interaction (useful for how geometry matters).
- For practical inversion / eddy-current imaging: D. Jiles, *Introduction to Magnetism and Magnetic Materials* (background), and NDT literature on eddy-current testing.

---

## Typical Usage

1) Fit per pixel:

```matlab
[xmidMatrix, ymidMatrix, rMatrix, phiMatrix] = lorentzian_fit(x_dim, y_dim, data_length_x, data_x, data_y, data_w, 0);
