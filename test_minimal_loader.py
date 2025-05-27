from connectome_analysis.data.loaders import create_dataset_loader

try:
    loader = create_dataset_loader('abide')
    print("✓ Loader creation successful")
except Exception as e:
    print(f"✗ Loader error: {e}")
