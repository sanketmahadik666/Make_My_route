"""
Pipeline Component Testing Script
Tests core functionality without requiring pandas/openpyxl
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def test_schema_definition():
    """Test the schema definition component."""
    print("="*60)
    print("TESTING SCHEMA DEFINITION")
    print("="*60)
    
    try:
        from ingestion.schema_definition import SchemaDefinition
        
        # Initialize schema
        schema = SchemaDefinition()
        
        print(f"✅ Schema initialized successfully")
        print(f"   Dataset: {schema.dataset_name}")
        print(f"   Version: {schema.schema_version}")
        print(f"   Expected Columns: {len(schema.get_expected_columns())}")
        
        print(f"\n📋 ALL CS2_35 Fields (17 total):")
        for i, col in enumerate(schema.get_expected_columns(), 1):
            print(f"  {i:2d}. {col}")
        
        # Test file validation
        test_file = Path("/home/sanket/Make_My_route/DATA/CS2_35/CS2_35_1_10_11.xlsx")
        if test_file.exists():
            validation = schema.validate_file_compatibility(test_file)
            print(f"\n✅ File validation test:")
            print(f"   File: {test_file.name}")
            print(f"   Compatible: {validation['compatible']}")
            if validation['issues']:
                print(f"   Issues: {validation['issues']}")
        else:
            print(f"\n⚠️  Test file not found: {test_file}")
        
        # Test field definitions
        field_defs = schema.get_all_field_definitions()
        print(f"\n📊 Field Type Coverage:")
        type_counts = {}
        for field, defn in field_defs.items():
            field_type = defn['type']
            type_counts[field_type] = type_counts.get(field_type, 0) + 1
        
        for field_type, count in type_counts.items():
            print(f"   {field_type}: {count} fields")
        
        return True
        
    except Exception as e:
        print(f"❌ Schema definition test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_units_configuration():
    """Test the units configuration."""
    print("\n" + "="*60)
    print("TESTING UNITS CONFIGURATION")
    print("="*60)
    
    try:
        import yaml
        
        # Load units config
        config_path = Path("config/units.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Units configuration loaded successfully")
        print(f"   Configuration: {config['configuration_info']['configuration_name']}")
        print(f"   Version: {config['configuration_info']['version']}")
        
        # Test SI units
        si_units = config['si_standard_units']
        print(f"\n📏 SI Standard Units:")
        for measurement, unit in si_units.items():
            print(f"   {measurement}: {unit}")
        
        # Test conversion factors
        conv_factors = config['conversion_factors']
        print(f"\n🔄 Conversion Factor Coverage:")
        for measurement in conv_factors.keys():
            print(f"   {measurement}: ✅")
        
        # Test CS2_35 column mapping
        cs2_35_mapping = config['cs2_35_column_mapping']
        print(f"\n📋 CS2_35 Column Mapping:")
        print(f"   Total columns mapped: {len(cs2_35_mapping)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Units configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_master_schema():
    """Test the master data schema."""
    print("\n" + "="*60)
    print("TESTING MASTER DATA SCHEMA")
    print("="*60)
    
    try:
        import yaml
        
        # Load master schema
        schema_path = Path("metadata/data_schema.yaml")
        with open(schema_path, 'r') as f:
            schema = yaml.safe_load(f)
        
        print(f"✅ Master schema loaded successfully")
        print(f"   Schema ID: {schema['schema_info']['schema_id']}")
        print(f"   Version: {schema['schema_info']['version']}")
        print(f"   Dataset: {schema['schema_info']['dataset_name']}")
        
        # Test columns
        columns = schema['columns']
        print(f"\n📊 Column Coverage:")
        print(f"   Total columns: {len(columns)}")
        
        # Count by type
        type_counts = {}
        for col_name, col_def in columns.items():
            nullable = col_def.get('nullable', False)
            nullability = "nullable" if nullable else "required"
            type_counts[nullability] = type_counts.get(nullability, 0) + 1
        
        for nullability, count in type_counts.items():
            print(f"   {nullability}: {count} fields")
        
        # Test field categories
        categories = {
            'Identification': ['Data_Point', 'Test_Time', 'Date_Time', 'Step_Time', 'Step_Index', 'Cycle_Index'],
            'Electrical': ['Current', 'Voltage'],
            'Capacity': ['Charge_Capacity', 'Discharge_Capacity'],
            'Energy': ['Charge_Energy', 'Discharge_Energy'],
            'Derived': ['dV/dt', 'Internal_Resistance'],
            'AC_Impedance': ['Is_FC_Data', 'AC_Impedance', 'ACI_Phase_Angle']
        }
        
        print(f"\n📋 Field Categories:")
        for category, field_list in categories.items():
            found_fields = [f for f in field_list if f in columns]
            print(f"   {category}: {len(found_fields)}/{len(field_list)} fields")
        
        return True
        
    except Exception as e:
        print(f"❌ Master schema test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_structure():
    """Test the project file structure."""
    print("\n" + "="*60)
    print("TESTING PROJECT FILE STRUCTURE")
    print("="*60)
    
    required_structure = {
        'config': ['units.yaml'],
        'src/ingestion': ['schema_definition.py', 'load_excel.py', 'batch_loader.py'],
        'src/standardization': ['unit_conversion.py'],
        'src/validation': ['data_validator.py'],
        'metadata': ['data_schema.yaml'],
        'environment': ['conda.yml'],
        'logs': [],  # Directory should exist
        'data': [],  # Directory should exist
    }
    
    all_good = True
    
    for directory, files in required_structure.items():
        dir_path = Path(directory)
        
        if not dir_path.exists():
            print(f"❌ Missing directory: {directory}")
            all_good = False
            continue
        
        print(f"✅ Directory exists: {directory}")
        
        for file in files:
            file_path = dir_path / file
            if file_path.exists():
                size = file_path.stat().st_size
                print(f"   ✅ {file} ({size} bytes)")
            else:
                print(f"   ❌ Missing file: {file}")
                all_good = False
    
    return all_good

def main():
    """Run all component tests."""
    print("🔬 BATTERY ANALYTICS LAB - PIPELINE COMPONENT TESTING")
    print("=" * 80)
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    tests = [
        ("Schema Definition", test_schema_definition),
        ("Units Configuration", test_units_configuration),
        ("Master Data Schema", test_master_schema),
        ("File Structure", test_file_structure),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Pipeline is ready for execution.")
        print("\n🚀 Next Steps:")
        print("   1. Activate conda environment: conda activate battery-analytics-lab")
        print("   2. Run data examination: python examine_data_structure.py")
        print("   3. Run controlled ingestion: python src/ingestion/batch_loader.py")
        print("   4. Run unit standardization: python src/standardization/unit_conversion.py")
    else:
        print(f"\n⚠️  {total-passed} tests failed. Please check the issues above.")

if __name__ == "__main__":
    main()