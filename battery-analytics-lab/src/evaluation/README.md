# Evaluation Strategy

## Overview

This document outlines the strategy for evaluating battery data, starting from standardization to advanced analysis.

## 1. Data Standardization (Evolution Phase)

The standardization process ("evolution phase") ensures that all raw data files, regardless of their source format (CSV, Excel) or column naming conventions, are transformed into a unified schema.

### Standardized Schema

The following fields are extracted and normalized:

- **Identifiers**: `Data_Point`, `Step_Index`, `Cycle_Index`
- **Time**: `Test_Time` (seconds), `Date_Time`, `Step_Time`
- **Electrical**: `Current` (A), `Voltage` (V)
- **Capacity/Energy**: `Charge_Capacity` (Ah), `Discharge_Capacity` (Ah), `Charge_Energy` (Wh), `Discharge_Energy` (Wh)
- **Derived**: `dV/dt` (V/s), `Internal_Resistance` (Ohm)
- **Impedance**: `AC_Impedance` (Ohm), `ACI_Phase_Angle` (Deg)
- **Metadata**: `Is_FC_Data`

### Ingestion Logic

- **Column Mapping**: Handles variations like `Test_Time(s)` vs `Test_Time`.
- **File Support**: Supports both `.csv` and `.xlsx` formats.
- **Output**: Standardized data is saved (e.g., as Parquet or standardized CSV) for efficient downstream processing.

## 2. Analysis Strategies

### Univariate Analysis

Focus on the distribution and behavior of individual variables:

- **Voltage**: Histogram of voltage levels, voltage vs. time to identify plateaus.
- **Current**: Load profile analysis, current distribution.
- **Capacity**: Capacity fade over cycles (Cycle Life).
- **Temperature**: Thermal characterization (if available).

### Multivariate Analysis

Examine relationships between variables:

- **Voltage vs. Capacity**: Differential Voltage Analysis (DVA) - dV/dQ.
- **Current vs. Voltage**: Internal resistance estimation, HPPC analysis.
- **Cycle Index vs. Capacity/Resistance**: Degradation modeling and SOH (State of Health) estimation.
- **Impedance Spectroscopy**: Nyquist and Bode plots for EIS data.

## 3. Project Standards & Standardization

- **Naming Convention**: Snake_Case or PascalCase as defined in `schema_definition.py`.
- **Units**: SI units (Seconds, Amperes, Volts, etc.) are enforced.
- **Validation**: Automated checks for schema compliance during ingestion.
