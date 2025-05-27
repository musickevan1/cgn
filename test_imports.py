import sys
import traceback

modules_to_test = [
    'connectome_analysis',
    'connectome_analysis.data',
    'connectome_analysis.data.loaders',
    'connectome_analysis.data.preprocessing',
    'connectome_analysis.data.connectome'
]

for module in modules_to_test:
    try:
        __import__(module)
        print(f"✓ {module}")
    except Exception as e:
        print(f"✗ {module}: {e}")
        traceback.print_exc()
