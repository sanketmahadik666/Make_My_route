# Analysis Report: CS2_35 Battery Data

## 1. Descriptive Analysis via EDA
The dataset was loaded successfully. Below are the descriptive statistics:

|       |   Data_Point |   Test_Time |       Step_Time |   Step_Index |   Cycle_Index |       Current |      Voltage |   Charge_Capacity |   Discharge_Capacity |   Charge_Energy |   Discharge_Energy |           dV/dt |   Internal_Resistance |   Is_FC_Data |   AC_Impedance |   ACI_Phase_Angle |   cycle_number |   original_row_count |   data_quality_score |
|:------|-------------:|------------:|----------------:|-------------:|--------------:|--------------:|-------------:|------------------:|---------------------:|----------------:|-------------------:|----------------:|----------------------:|-------------:|---------------:|------------------:|---------------:|---------------------:|---------------------:|
| count |     16961    |  16961      | 16961           |  16961       |    16961      | 16961         | 16961        |        16961      |           16961      |      16961      |         16961      | 16961           |         16961         |        16961 |          16961 |             16961 |          16961 |                16961 |      16961           |
| mean  |      8481    | 286990      |  2256.79        |      3.84164 |       25.6142 |    -0.0138591 |     3.863    |           25.2198 |              24.7544 |        100.464  |            90.5678 |    -0.000104946 |             0.088433  |            0 |              0 |                 0 |              1 |                16961 |          0.9         |
| std   |      4896.36 | 165745      |  1654.87        |      2.35508 |       14.4037 |     0.766336  |     0.237285 |           14.458  |              14.4663 |         57.5766 |            52.9861 |     0.00109921  |             0.010294  |            0 |              0 |                 0 |              0 |                    0 |          2.22051e-16 |
| min   |         1    |     30.0007 |     2.52631e-06 |      1       |        1      |    -1.10029   |     2.6993   |            0      |               0      |          0      |             0      |    -0.0147964   |             0         |            0 |              0 |                 0 |              1 |                16961 |          0.9         |
| 25%   |      4241    | 143551      |   840.415       |      2       |       13      |    -1.09957   |     3.72371  |           12.844  |              11.979  |         51.1966 |            43.7711 |    -9.71317e-05 |             0.0883364 |            0 |              0 |                 0 |              1 |                16961 |          0.9         |
| 50%   |      8481    | 286728      |  2071.04        |      2       |       26      |     0.549936  |     3.89838  |           25.1901 |              24.8307 |        100.348  |            90.7404 |     3.23772e-05 |             0.0891469 |            0 |              0 |                 0 |              1 |                16961 |          0.9         |
| 75%   |     12721    | 429619      |  3331.6         |      7       |       38      |     0.550117  |     4.0266   |           37.862  |              37.0293 |        150.789  |           135.534  |     6.47545e-05 |             0.0906039 |            0 |              0 |                 0 |              1 |                16961 |          0.9         |
| max   |     16961    | 574901      |  5930.44        |      9       |       50      |     1.01284   |     4.20014  |           50.104  |              50.1451 |        199.589  |           183.583  |     0.00168362  |             0.0939245 |            0 |              0 |                 0 |              1 |                16961 |          0.9         |

### Visualizations
- **Voltage Profile**: See `plots/voltage_vs_time.png`
- **Current Profile**: See `plots/current_vs_time.png`

## 2. Feature Engineering & Quality
We aggregated data at the cycle level to observe trends.
- **Capacity Fade**: `plots/capacity_fade.png` shows the degradation of discharge capacity over cycles.

## 3. Interpretability Analysis
We trained a Random Forest Regressor to predict `Discharge_Capacity` based on aggregated cycle features.

**Model Performance:**
- R2 Score: 0.9926
- MSE: 9.3973e-01

**Feature Importance:**
Top features driving the capacity prediction:
- **Cycle_Index**: 0.9873
- **Voltage_mean**: 0.0057
- **Current_mean**: 0.0049
- **Voltage_min**: 0.0021
- **Voltage_max**: 0.0000

See `plots/feature_importance.png` for visual representation.
