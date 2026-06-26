# Hertz_Fits

**Automated Hertz contact mechanics fitting and stress relaxation analysis of AFM force-distance curves**

Cell stiffness (Young's modulus) is extracted from AFM force-distance curves by fitting the Hertz contact mechanics model to the approach segment. This pipeline reads raw JPK force files, fits the Hertz model to each approach curve, and optionally fits a stress relaxation model to time-dependent indentation data — producing per-cell quantitative mechanical readouts.

---

## Pipeline overview

```
.jpk-force files
        │
        ▼
 H_DataReat.py        ← Load raw AFM data, extract approach segments,
 (data ingestion)        convert to force-distance, apply contact point detection
        │
        ├──► H_HertzFit.py     ← Fit Hertz model → Young's modulus (E, kPa) per curve
        │
        └──► H_SRFits.py       ← Fit stress relaxation model → viscoelastic parameters
```

---

## Scripts

| Script | Description |
|---|---|
| `H_DataReat.py` | Reads `.jpk-force` files. Extracts height and vDeflection from the approach segment. Converts raw deflection to force (nN) and calculates tip-sample distance. Applies baseline correction and contact point detection. |
| `H_HertzFit.py` | Fits the Hertz contact mechanics model (spherical indenter) to the approach curve. Returns Young's modulus `E` in kPa per force curve. Handles both hard-surface calibration and cell indentation curves. |
| `H_SRFits.py` | Fits stress relaxation curves from constant-indentation segments. Extracts viscoelastic parameters (relaxation time, moduli) using a power law or standard linear solid (SLS) model. |

---

## Methods

**Hertz model:** For a spherical indenter of radius `R` on a flat elastic half-space, the indentation force is:

```
F = (4/3) * E* * sqrt(R) * δ^(3/2)
```

where `E* = E / (1 - ν²)` is the reduced modulus, `δ` is indentation depth, and `ν` is Poisson's ratio (typically assumed 0.5 for cells). `E` is extracted per curve by nonlinear least-squares fitting (`scipy.optimize.curve_fit`).

**Contact point detection:** The onset of indentation is identified from the approach curve as the point where deflection deviates from baseline, determined by threshold or derivative-based methods.

**Stress relaxation:** Fitted to the time-dependent force decay at constant indentation using either a power law `F(t) = A * t^(-α)` or a two-component SLS model, extracting instantaneous and equilibrium moduli.

---

## Setup

```bash
git clone https://github.com/GiuAlt/Hertz_Fits.git
cd Hertz_Fits
pip install jpkfile numpy pandas scipy matplotlib
```

> **Note:** Raw `.jpk-force` data files are not included. Set the data directory and cantilever spring constant `k` in `H_DataReat.py` to match your experiment. Indenter tip radius `R` should be set to match your probe geometry.

---

## Context

Developed during my PhD in Biophysics at ETH Zurich to characterise the mechanical phenotype of cancer cells, tumour spheroids, and organoids. Hertz fits were used to quantify cell stiffness (Young's modulus) across drug treatment conditions and cell lines, contributing to analyses published in *Nature Materials*. Tether force measurements from the same experiments are processed in [Tethers_Extraction](https://github.com/GiuAlt/Tethers_Extraction).

---

*Giulia Ammirati · [github.com/GiuAlt](https://github.com/GiuAlt) · ETH Zurich, 2024*
