#!/usr/bin/env python3
"""
Install dependencies for the Bloomberg API script
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install {package}")
        return False

def main():
    print("Installing dependencies for Bloomberg API script...")
    print("=" * 50)
    
    # Install basic dependencies
    packages = ["pandas", "openpyxl"]
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print("\n" + "=" * 50)
    print(f"Installed {success_count}/{len(packages)} packages successfully.")
    
    print("\n" + "=" * 50)
    print("BLOOMBERG API (blpapi) INSTALLATION INSTRUCTIONS:")
    print("=" * 50)
    print("1. You need to download the Bloomberg C++ SDK from:")
    print("   https://developer.bloomberg.com/")
    print("2. Extract the C++ SDK and note the path")
    print("3. Set environment variable BLPAPI_ROOT to the C++ SDK path")
    print("4. Install Microsoft C++ Build Tools from:")
    print("   https://visualstudio.microsoft.com/visual-cpp-build-tools/")
    print("5. Then run: pip install ./blpapi-3.25.3")
    print("\nAlternative: Look for pre-built wheel files (.whl) in the")
    print("Bloomberg Developer Portal that match your Python version.")
    
    print("\n" + "=" * 50)
    print("Your script will work once blpapi is installed!")
    print("=" * 50)

if __name__ == "__main__":
    main() 