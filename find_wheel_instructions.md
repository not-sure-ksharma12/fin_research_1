# Alternative: Find Pre-built Wheel Files

If you don't want to install Microsoft C++ Build Tools, you can look for pre-built wheel files.

## Steps:

1. **Go to Bloomberg Developer Portal:**
   https://developer.bloomberg.com/

2. **Look for "Python SDK" or "Wheel Files" in the downloads section**

3. **Download the wheel file that matches:**
   - Python version: 3.12
   - Platform: Windows 64-bit
   - Architecture: amd64
   - Example filename: `blpapi-3.25.3-cp312-cp312-win_amd64.whl`

4. **Install the wheel file:**
   ```powershell
   .\venv\Scripts\Activate.ps1
   pip install path\to\blpapi-3.25.3-cp312-cp312-win_amd64.whl
   ```

## What to Look For:

- Files ending in `.whl`
- Filename should contain:
  - `blpapi-3.25.3` (version)
  - `cp312` (Python 3.12)
  - `win_amd64` (Windows 64-bit)

## If No Wheel Files Available:

You'll need to install Microsoft C++ Build Tools as described in the main instructions.

## Quick Test After Installation:

```powershell
python -c "import blpapi; print('Bloomberg API installed successfully!')"
``` 