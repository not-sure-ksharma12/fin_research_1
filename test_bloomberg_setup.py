#!/usr/bin/env python3
"""
Test Bloomberg API setup and provide installation guidance
"""

import os
import sys
import ctypes
from pathlib import Path

def test_blpapi_import():
    """Test if blpapi can be imported"""
    try:
        import blpapi
        print("✅ blpapi imported successfully!")
        print(f"   Version: {blpapi.__version__}")
        return True
    except ImportError as e:
        print(f"❌ blpapi import failed: {e}")
        return False

def test_cpp_library():
    """Test if the C++ library can be loaded"""
    try:
        # Try to load the Bloomberg C++ library
        cpp_lib_path = Path("blpapi_cpp_3.25.3.1/lib/blpapi3_64.dll")
        if cpp_lib_path.exists():
            print(f"✅ Found C++ library: {cpp_lib_path}")
            return True
        else:
            print(f"❌ C++ library not found at: {cpp_lib_path}")
            return False
    except Exception as e:
        print(f"❌ Error testing C++ library: {e}")
        return False

def check_environment():
    """Check environment setup"""
    print("🔍 Checking Bloomberg API environment...")
    print("=" * 50)
    
    # Check BLPAPI_ROOT
    blpapi_root = os.environ.get('BLPAPI_ROOT')
    if blpapi_root:
        print(f"✅ BLPAPI_ROOT set to: {blpapi_root}")
    else:
        print("❌ BLPAPI_ROOT not set")
    
    # Check if C++ SDK exists
    cpp_sdk_path = Path("blpapi_cpp_3.25.3.1")
    if cpp_sdk_path.exists():
        print(f"✅ C++ SDK found at: {cpp_sdk_path}")
    else:
        print("❌ C++ SDK not found")
    
    # Check for required files
    lib_path = cpp_sdk_path / "lib" / "blpapi3_64.dll"
    include_path = cpp_sdk_path / "include"
    
    if lib_path.exists():
        print(f"✅ C++ library found: {lib_path}")
    else:
        print(f"❌ C++ library not found: {lib_path}")
    
    if include_path.exists():
        print(f"✅ Include directory found: {include_path}")
    else:
        print(f"❌ Include directory not found: {include_path}")

def main():
    print("Bloomberg API Setup Test")
    print("=" * 50)
    
    # Check environment
    check_environment()
    print()
    
    # Test C++ library
    cpp_ok = test_cpp_library()
    print()
    
    # Test blpapi import
    blpapi_ok = test_blpapi_import()
    print()
    
    print("=" * 50)
    if blpapi_ok:
        print("🎉 Bloomberg API is ready to use!")
        print("You can now run: python nvda_options_fetcher.py")
    else:
        print("❌ Bloomberg API setup incomplete")
        print("\n📋 NEXT STEPS:")
        print("1. Install Microsoft C++ Build Tools:")
        print("   https://visualstudio.microsoft.com/visual-cpp-build-tools/")
        print("2. Download and install the 'C++ build tools' workload")
        print("3. Restart your terminal/command prompt")
        print("4. Run: cd blpapi-3.25.3 && pip install .")
        print("\nAlternative: Look for pre-built wheel files (.whl) on")
        print("the Bloomberg Developer Portal that match your Python version.")

if __name__ == "__main__":
    main() 