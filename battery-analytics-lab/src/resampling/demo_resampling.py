"""
Battery Analytics Lab - Resampling Demo
Phase 2: Data Resampling for Uniformity

This demo script shows how to use the voltage-based resampling functionality
to prepare battery cycling data for machine learning models.

Author: Battery Analytics Lab Team
Date: 2026-01-22
Version: 1.0
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.resampling import VoltageResampler, ResamplingValidator


def create_sample_cycle_data():
    """Create sample battery cycle data for demonstration."""
    # Create a voltage range typical for Li-ion discharge (4.2V to 3.0V)
    voltage = np.linspace(4.2, 3.0, 100)  # Decreasing voltage during discharge

    # Simulate capacity decrease (simplified model)
    # Capacity starts high and decreases as voltage drops
    capacity_base = 2.5  # Ah
    capacity_noise = np.random.normal(0, 0.02, len(voltage))
    capacity = capacity_base * (1 - 0.3 * (4.2 - voltage) / 1.2) + capacity_noise
    capacity = np.maximum(capacity, 0)  # Ensure non-negative

    # Simulate current (mostly constant with some noise)
    current = -2.0 + np.random.normal(0, 0.05, len(voltage))  # Negative for discharge

    # Create DataFrame
    df = pd.DataFrame({
        'voltage_v': voltage,
        'capacity_ah': capacity,
        'current_a': current,
        'temperature_c': 25.0 + np.random.normal(0, 2, len(voltage))
    })

    return df


def demo_voltage_resampling():
    """Demonstrate voltage-based resampling functionality."""
    print("Battery Analytics Lab - Voltage-Based Resampling Demo")
    print("=" * 60)

    # Initialize resampler
    print("\n1. Initializing VoltageResampler...")
    resampler = VoltageResampler()
    print("   ✓ VoltageResampler initialized successfully")
    print(f"   ✓ Grid: {resampler.num_grid_points} points from {resampler.voltage_min}V to {resampler.voltage_max}V")

    # Create sample data
    print("\n2. Creating sample battery cycle data...")
    original_data = create_sample_cycle_data()
    print(f"   ✓ Generated {len(original_data)} sample data points")
    print(f"   ✓ Voltage range: {original_data['voltage_v'].min():.3f}V to {original_data['voltage_v'].max():.3f}V")
    print(f"   ✓ Capacity range: {original_data['capacity_ah'].min():.3f}Ah to {original_data['capacity_ah'].max():.3f}Ah")

    # Perform resampling
    print("\n3. Performing voltage-based resampling...")
    resampled_data, metadata = resampler.resample_cycle_data(
        original_data, cycle_number=1, cell_id="DEMO_CELL_001"
    )

    if metadata['status'] == 'success':
        print("   ✓ Resampling completed successfully")
        print(f"   ✓ Resampled to {len(resampled_data)} uniform voltage points")
        print(f"   ✓ Voltage step: {resampler.voltage_step*1000:.0f} mV")
        print(f"   ✓ Interpolation method: {metadata['interpolation_method']}")

        # Show quality metrics
        if 'quality_metrics' in metadata:
            metrics = metadata['quality_metrics']
            print("\n4. Quality Metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(".4f")
                else:
                    print(f"   ✓ {key}: {value}")
    else:
        print(f"   ✗ Resampling failed: {metadata.get('error', 'Unknown error')}")
        return

    # Initialize validator
    print("\n5. Validating resampled data...")
    validator = ResamplingValidator()

    # Perform validation
    validation_result = validator.validate_resampled_data(
        resampled_data, original_data, metadata
    )

    print(f"   ✓ Validation status: {validation_result['overall_status']}")
    print(f"   ✓ Checks performed: {len(validation_result['checks_performed'])}")
    print(f"   ✓ Checks passed: {len(validation_result['passed_checks'])}")

    if validation_result['warnings']:
        print(f"   ⚠ Warnings: {len(validation_result['warnings'])}")

    # Show sample of resampled data
    print("\n6. Sample of resampled data:")
    print(resampled_data.head(10)[['voltage_v', 'capacity_ah', 'current_a']].to_string(index=False))

    # Summary
    print("\n7. Summary:")
    print(f"   • Original data points: {len(original_data)}")
    print(f"   • Resampled data points: {len(resampled_data)}")
    print(f"   • Voltage grid uniformity: {resampler.voltage_step*1000:.0f} mV steps")
    print("   • Ready for ML model input with fixed vector length")
    print("\n🎉 Voltage-based resampling demo completed successfully!")
    print("\nThis demonstrates how the resampling module ensures uniform input vectors")
    print("for machine learning models by interpolating battery data onto a fixed")
    print("voltage grid, aligning electrochemical features across different cycles.")


def main():
    """Main demo function."""
    try:
        demo_voltage_resampling()
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
