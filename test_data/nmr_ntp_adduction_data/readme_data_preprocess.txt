NTP Adduction NMR Kinetic Data Processing

Data Collection and Peak Selection

For ATP, we manually selected all visible peaks using Mnova. These peaks represent both unmodified and modified nucleotides. At the first time point, the selected peak regions and their corresponding integrals are as follows:

Peak Range (ppm)	Integral Value
9.534–9.479	0.5545
8.554–8.516	229.305
8.519–8.453	242.121
8.444–8.300 (C8)	41173.5
8.249–8.224	215.439
8.214–8.157	337.572
8.153–7.970 (C2)	35034.6
7.932–7.873	228.227
8.292–8.248	-22.6069

The total signal is conserved across time points, indicating complete accounting of protons for these peaks.
	•	Peak 8 corresponds to: C8, modified-C8, C2, modified-C2
	•	Peak 6 corresponds to: C1’, modified C1’

_raw values refer to raw peak integrals without any normalization.

⸻

Fraction Decrease (Pre-normalization)

To estimate the extent of nucleotide modification over time, we initially calculate fractional decrease values (denoted f_dec_1 and f_dec_2) for the major peaks in the region.
	•	Example (first time point, total integral = 77438):
	•	C2: 35034.6 / 77438 ≈ 0.53
	•	C8: 41173.5 / 77438 ≈ 0.45

These ratios track the disappearance of the original species over time. However:

Important: These are not true percentages. The integral-to-proton ratio differs between unmodified and modified species. Therefore, while these values resemble percentages, they are only approximate pre-normalization indicators.

⸻

Normalization and Quantification

To obtain a more accurate estimate of the extent of modification, we:
	1.	Fit exponential decay curves to the raw integrals of the major unmodified peaks (e.g., C2, C8).
	2.	Extrapolate to estimate the maximum peak integral (at t = 0 s) using first-order kinetics.
	3.	Normalize each observed value against this extrapolated value to estimate the fraction of unmodified species remaining.

Example:
	•	At t = 493 s:
	•	Observed C2 fraction = 0.53
	•	Extrapolated max fraction = 0.5315 (from t = 0)
	•	→ Normalized: 0.53 / 0.5315 ≈ 0.997
	•	→ Interpreted as ~99.7% of C2 remaining at 493 s

⸻

Final Quantification and Modeling
	•	Apply the above normalization for both NTP adduction peaks and DMS degradation peaks.
	•	Calculate fraction of each species (e.g., frac_C2, frac_C8).
	•	Convert these fractions to absolute concentrations using an estimated starting concentration.
	•	Use these concentration values as inputs for simultaneous ODE fitting:
	•	One ODE for DMS degradation
	•	One ODE for NTP adduction

Note: For adduct formation, only one fraction (e.g., C2) is used to model adduct kinetics:
\text{[C2-CH}3] = 1 - \text{frac}\text{C2}