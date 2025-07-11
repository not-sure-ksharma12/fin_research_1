# Bloomberg API Setup Guide

## Current Status ✅
- ✅ Virtual environment created
- ✅ Basic dependencies installed (pandas, openpyxl)
- ❌ Bloomberg API (blpapi) - **NEEDS SETUP**

## Bloomberg API Installation Steps

### Option 1: Complete Setup (Recommended)

#### Step 1: Install Microsoft C++ Build Tools
1. Go to: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Download "Build Tools for Visual Studio"
3. Run installer and select "C++ build tools" workload
4. Install (this may take 10-15 minutes)

#### Step 2: Download Bloomberg C++ SDK
1. Go to: https://developer.bloomberg.com/
2. Sign in with your Bloomberg credentials
3. Download the **Bloomberg C++ SDK** (not the Python SDK)
4. Extract to a folder (e.g., `C:\bloomberg\cpp-sdk`)

#### Step 3: Set Environment Variables
```powershell
$env:BLPAPI_ROOT = "C:\bloomberg\cpp-sdk"
```

#### Step 4: Install blpapi
```powershell
cd blpapi-3.25.3
pip install .
```

### Option 2: Pre-built Wheel (Easier)

1. Go to: https://developer.bloomberg.com/
2. Look for **pre-built wheel files** (.whl) in the downloads section
3. Download the wheel that matches your Python version (3.12) and Windows 64-bit
4. Install with: `pip install path\to\blpapi-3.25.3-cp312-cp312-win_amd64.whl`

### Option 3: Alternative Bloomberg APIs

If blpapi setup is too complex, consider these alternatives:

#### Bloomberg Data License (DL) API
- More modern API
- Better documentation
- Easier setup

#### Bloomberg Desktop API (DAPI)
- Direct integration with Bloomberg Terminal
- No separate installation needed
- Limited to Bloomberg Terminal users

## Testing the Installation

Once blpapi is installed, test it:

```python
import blpapi
print("Bloomberg API installed successfully!")
```

## Running Your Script

After blpapi is installed:

```powershell
python nvda_options_fetcher.py
```

## Troubleshooting

### Common Issues:

1. **"Microsoft Visual C++ 14.0 or greater is required"**
   - Solution: Install Microsoft C++ Build Tools

2. **"BLPAPI_ROOT environment variable isn't defined"**
   - Solution: Set the BLPAPI_ROOT environment variable to your C++ SDK path

3. **"Could not find blpapi library"**
   - Solution: Ensure the C++ SDK contains the required .dll files

4. **Bloomberg Terminal Connection Issues**
   - Ensure Bloomberg Terminal is running
   - Check that you're connected to Bloomberg network
   - Verify your Bloomberg credentials

## Next Steps

1. Choose one of the installation options above
2. Follow the steps for your chosen method
3. Test the installation
4. Run your NVDA options script

## Support

- Bloomberg Developer Portal: https://developer.bloomberg.com/
- Bloomberg API Documentation: Available on the developer portal
- Microsoft C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/ 