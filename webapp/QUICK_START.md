# 🚀 CRCL Dashboard - Quick Start Guide

## ⚡ Get Started in 30 Seconds

### Option 1: Use the startup script (Recommended)
```bash
cd webapp
./start.sh
```

### Option 2: Manual startup
```bash
cd webapp
source venv/bin/activate
python app.py
```

## 🌐 Access the Dashboard
Open your browser and go to: **http://localhost:5001**

## 📊 What You'll See

1. **Control Panel** - Select dates and data type
2. **Summary Stats** - Total trades, P&L, returns
3. **Data Visualization** - Charts or trade tables based on your selection

## 🎯 Features Ready to Use

✅ **Company Selection**: CRCL (expandable for others)  
✅ **Date Range Picker**: Select start/end dates  
✅ **Data Type Selection**: Hourly graphs or trade logs  
✅ **Real-time Data**: Reads from your Excel and log files  
✅ **Interactive Charts**: Chart.js powered visualizations  
✅ **Responsive Design**: Works on desktop and mobile  

## 📁 Data Sources
- **Hourly Data**: `CRCL_hourly_data.xlsx`
- **Trades Log**: `CRCL_trades.log`

## 🔧 If Something Goes Wrong

1. **Check terminal logs** for error messages
2. **Verify data files exist** in the specified paths
3. **Restart the application** with `./start.sh`

## 🎨 Customization
- Modify `templates/index.html` for UI changes
- Update `app.py` for backend logic
- Add new companies by updating the ticker dropdown

---
**Need help?** Check the full `README.md` for detailed documentation.
