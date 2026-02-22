"""
Simple ML-based energy prediction model.
Inputs: distance (m), slope (grade), vehicle efficiency (Wh/km).
Output: energy consumption in Wh.
Model is isolated here for clarity and future replacement (e.g. more features, other libs).
"""
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class EnergyModel:
    """
    Predicts energy consumption in Wh from distance, slope, and efficiency.
    Uses scikit-learn LinearRegression; no custom algorithms.
    """

    def __init__(self, fit_intercept: bool = True):
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression(fit_intercept=fit_intercept)),
        ])
        self._fitted = False

    def fit(
        self,
        distance_m: np.ndarray,
        grade: np.ndarray,
        efficiency_wh_per_km: np.ndarray,
        energy_wh: np.ndarray,
    ) -> "EnergyModel":
        """Fit model: X = (distance_m, grade, 1/efficiency), y = energy_wh."""
        # Inverse efficiency so higher efficiency => less energy per km
        inv_eff = np.where(efficiency_wh_per_km > 0, 1.0 / efficiency_wh_per_km, 1.0 / 200.0)
        X = np.column_stack([distance_m, grade, inv_eff])
        self._pipeline.fit(X, energy_wh)
        self._fitted = True
        return self

    def predict_wh(
        self,
        distance_m: float | np.ndarray,
        grade: float | np.ndarray,
        efficiency_wh_per_km: float,
    ) -> float | np.ndarray:
        """
        Predict energy consumption in Wh.
        grade: rise/run (e.g. 0.05 for 5% slope).
        """
        scalar = np.isscalar(distance_m) or (isinstance(distance_m, np.ndarray) and distance_m.ndim == 0)
        if scalar:
            distance_m = np.array([float(distance_m)])
            grade = np.array([float(grade)])
        else:
            distance_m = np.asarray(distance_m, dtype=float)
            grade = np.asarray(grade, dtype=float)
        inv_eff = 1.0 / max(efficiency_wh_per_km, 1.0)
        X = np.column_stack([distance_m, grade, np.full_like(distance_m, inv_eff)])
        out = self._pipeline.predict(X)
        return float(out[0]) if scalar else out

    def save(self, path: str | Path) -> None:
        """Persist model (sklearn pipeline) for later load."""
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self._pipeline, "fitted": self._fitted}, path)

    @classmethod
    def load(cls, path: str | Path) -> "EnergyModel":
        """Load a previously saved model."""
        import joblib
        data = joblib.load(path)
        m = cls()
        m._pipeline = data["pipeline"]
        m._fitted = data.get("fitted", True)
        return m


def get_default_energy_model() -> EnergyModel:
    """
    Return a pre-fitted model using a simple physics-inspired baseline:
    energy_wh ≈ (distance_km * efficiency_wh_per_km) * (1 + slope_factor * |grade|).
    Fitted on synthetic data so the pipeline works out of the box.
    """
    np.random.seed(42)
    n = 500
    distance_m = np.random.uniform(100, 100_000, n)
    grade = np.random.uniform(-0.15, 0.15, n)
    efficiency_wh_per_km = np.random.uniform(120, 250, n)
    # Baseline: Wh = distance_km * eff * (1 + 2*|grade|) + noise
    distance_km = distance_m / 1000.0
    energy_wh = distance_km * efficiency_wh_per_km * (1.0 + 2.0 * np.abs(grade)) + np.random.normal(0, 500, n)
    energy_wh = np.maximum(energy_wh, 100)

    model = EnergyModel()
    model.fit(distance_m, grade, efficiency_wh_per_km, energy_wh)
    return model


def predict_energy_wh(
    distance_m: float,
    grade: float,
    efficiency_wh_per_km: float,
    model: Optional[EnergyModel] = None,
) -> float:
    """Convenience: predict energy in Wh using default or provided model."""
    if model is None:
        model = get_default_energy_model()
    return float(model.predict_wh(distance_m, grade, efficiency_wh_per_km))
