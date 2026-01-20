# CS2_35 Battery Insights & Hypothesis

## 1. Incremental Capacity Analysis (dQ/dV) Interpretation

### Why does the curve look like this?

The **dQ/dV (Differential Capacity)** curve corresponds to the derivative of the Charge/Discharge capacity with respect to Voltage. It transforms voltage plateaus (flat regions in V vs Q) into peaks.

- **The Shape**: The curve shows a prominent **negative peak around 3.6V**.
  - This is characteristic of **LiCoO2 (LCO)** cathode chemistry discharging against graphite.
  - The strong peak represents the primary phase transition where Lithium ions are inserted back into the Cathode lattice structure.
  - The "sharpness" of the peak indicates a well-defined voltage plateau, meaning the battery delivers the majority of its energy at this consistent voltage (~3.6V).

### Behavioral Analysis (Observed in Plots)

- **Stability**: The curves for Cycle 1, 10, 25, 40, and 50 are nearly identical and overlapping.
- **Hypothesis**: The battery is in its **early life "fresh" state**. There is minimal **Loss of Active Material (LAM)** (which would lower the peak) or **Loss of Lithium Inventory (LLI)** (which would shift the peak).
- **Polarization**: The lack of significant horizontal shift indicates that **Internal Resistance** has not increased substantially over these 50 cycles.

---

## 2. Field Behavior & Data Quality

### Observed vs Expected Behavior

| Field                      | Expected Behavior                         | Observed Behavior                                                           | Hypothesis / Conclusion                                                                                                                                        |
| -------------------------- | ----------------------------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **DC Internal Resistance** | Gradual increase over hundreds of cycles. | Stable around **0.09 Ohms** (90 mOhm) with an initial stabilization period. | **Normal**. The battery is healthy and resistance is stable.                                                                                                   |
| **AC Impedance**           | Complex spectrum (Real/Imaginary parts).  | **Flat 0.0**.                                                               | **Missing Data**. The test equipment likely did not record EIS (Electrochemical Impedance Spectroscopy) data for these cycles, or the column is a placeholder. |
| **ACI Phase Angle**        | Non-zero phase shift.                     | **Flat 0.0**.                                                               | **Missing Data**. Confirming no EIS data available.                                                                                                            |

---

## 3. Feature Importance & Interpretability Hypothesis

### Model Findings

Our Random Forest model identified **`Cycle_Index`** as the dominant predictor (Importance: ~0.98) for Capacity.

### Hypothesis: Why this feature importance?

1.  **Time-Dependent Degradation**: Even in early life, the most consistent predictable factor for capacity change is simply "time" (represented by Cycle Index). The degradation, while slow, is **monotonic**.
2.  **Lack of Stress Variation**: The test scenario is uniform (Constant 1C Discharge, Constant CCCV Charge). There are no varying stress factors (like different Temperatures or C-rates) that would make other features (like Voltage variance or Temperature) more important.
3.  **Voltage as State-of-Charge**: Voltage features (`Voltage_mean`, `Voltage_min`) are less important for predicting _total_ capacity because the test limits (4.2V to 2.7V) are fixed. They would be crucial for predicting _instantaneous_ SOC, but not the total health of the cell in this specific constant-cycling scenario.

### Conclusion for Future Feature Building

- **Constraint**: Use `Cycle_Index` carefully. It works for _this_ specific continuous test, but fails if the battery rests for months.
- **Recommendation**: Construct **"Throughput"** features (Cumulative Energy Throughput) instead of just Cycle Index to make the model robust to different cycling protocols.
