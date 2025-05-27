#!/usr/bin/env python3
"""
Install script for CGN project dependencies.
Handles optional neuroimaging packages gracefully.
"""

import subprocess
import sys
from pathlib import Path

def install_package(package_name, optional=False):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        if optional:
            print(f"⚠️  Optional package {package_name} failed to install: {e}")
            print("   This may be due to system dependencies. Project will work with limited functionality.")
            return False
        else:
            print(f"❌ Failed to install required package {package_name}: {e}")
            return False

def main():
    print("CGN Dependencies Installer")
    print("=" * 30)
    
    # Required packages
    required_packages = [
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "networkx>=3.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "jupyter>=1.0.0"
    ]
    
    # Optional neuroimaging packages
    optional_packages = [
        "nibabel>=5.0.0",
        "nilearn>=0.10.0",
        "torch-geometric>=2.3.0",
        "pytorch-lightning>=2.0.0",
        "transformers>=4.30.0"
    ]
    
    print("Installing required packages...")
    failed_required = []
    for package in required_packages:
        if not install_package(package):
            failed_required.append(package)
    
    print("\nInstalling optional packages...")
    failed_optional = []
    for package in optional_packages:
        if not install_package(package, optional=True):
            failed_optional.append(package)
    
    print("\n" + "=" * 30)
    print("INSTALLATION SUMMARY")
    print("=" * 30)
    
    if not failed_required:
        print("✅ All required packages installed successfully!")
    else:
        print(f"❌ Failed required packages: {failed_required}")
        return 1
    
    if not failed_optional:
        print("✅ All optional packages installed successfully!")
    else:
        print(f"⚠️  Some optional packages failed: {failed_optional}")
        print("   You can still use the project with limited functionality.")
    
    print("\nYou can now run the project notebooks and scripts!")
    return 0

if __name__ == "__main__":
    exit(main())
