"""
Comprehensive error detection for CGN Phase 4
"""
import os
import sys
import subprocess
import importlib.util
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


def check_syntax_errors():
    """Check all Python files for syntax errors"""
    errors = []
    src_path = Path("src")

    for py_file in src_path.rglob("*.py"):
        try:
            with open(py_file, 'r') as f:
                compile(f.read(), py_file, 'exec')
        except SyntaxError as e:
            errors.append({
                'file': str(py_file),
                'line': e.lineno,
                'error': str(e),
                'type': 'syntax'
            })

    return errors

def check_import_errors():
    """Check module imports"""
    modules_to_test = [
        'src.connectome_analysis.models.baseline',
        'src.connectome_analysis.models.gnn_models',
        'src.connectome_analysis.training.cross_validation',
        'src.connectome_analysis.training.experiment'
    ]

    errors = []
    for module in modules_to_test:
        try:
            spec = importlib.util.spec_from_file_location(
                module, module.replace('.', '/') + '.py'
            )
            if spec is not None and spec.loader is not None: # Add check for None
                imported_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(imported_module)
            else:
                 errors.append({
                    'module': module,
                    'error': f"Could not get module spec or loader for {module}",
                    'type': 'import'
                })
        except Exception as e:
            errors.append({
                'module': module,
                'error': str(e),
                'type': 'import'
            })

    return errors

def main():
    print("🔍 CGN Phase 4 Error Detection")
    print("=" * 50)

    # Check syntax errors
    syntax_errors = check_syntax_errors()
    if syntax_errors:
        print(f"❌ Found {len(syntax_errors)} syntax errors:")
        for error in syntax_errors:
            print(f"  📁 {error['file']}:{error['line']} - {error['error']}")
    else:
        print("✅ No syntax errors found")

    # Check import errors
    import_errors = check_import_errors()
    if import_errors:
        print(f"❌ Found {len(import_errors)} import errors:")
        for error in import_errors:
            print(f"  📦 {error['module']} - {error['error']}")
    else:
        print("✅ No import errors found")

    return len(syntax_errors) + len(import_errors)

if __name__ == "__main__":
    error_count = main()
    sys.exit(error_count)
