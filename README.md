# 🧠📈 Forex Trading Bot with Deep Learning and Automated Execution

This project is an intelligent Forex trading bot that uses deep learning to make trade decisions and execute orders automatically. It supports two execution modes based on broker compatibility — via the MetaTrader 5 API or UI automation for brokers that restrict automated access.

---

## 🚀 Features

- **LSTM-based deep learning model** for time series prediction:
  - Multiple stacked LSTM layers to capture temporal patterns
  - Dropout layers to reduce overfitting
  - Dense output layer with softmax for multi-class prediction (Buy / Sell / Hold)
- Automatically places and modifies orders based on model predictions.
- Supports both **MetaTrader 5 API** and **screen automation** (for brokers that block MT5 API).
- Configurable stop-loss and take-profit settings.
- Designed for high-frequency, real-time trading using technical indicators.
- Modular and extensible for additional trading logic or broker support.

---

## 🛠️ Technologies Used

### 📊 Machine Learning & Data
- **TensorFlow / Keras** – LSTM-based deep learning model  
- **NumPy / Pandas** – Data processing and manipulation  
- **TA-Lib** – Generating technical indicators  
- **Scikit-learn** – Data scaling and evaluation metrics  

### 🖥️ Automation & Execution
- **MetaTrader5** – API to place and manage trades  
- **PyAutoGUI** – Screen control (clicks, typing, screenshots)  
- **Pywinauto** – Automating Windows UI interactions  
- **Pytesseract** – OCR to extract price data from screenshots  

### 🧰 System Requirements
- Python 3.9+
- Windows OS (required for MetaTrader + screen automation)
- Tesseract-OCR (for screen-based price reading)

---

## 🔁 Workflow Overview

1. **Data Preparation**  
   Generate technical indicators using TA-Lib and prepare labeled time series data.

2. **Model Training**  
   Train a multi-class classification LSTM model:
   ```bash
   python train_model.py
   ```

3. **Broker Type Selection**
   - ✅ If your broker supports MetaTrader 5: use `mt5_trade.py`
   - ❌ If not, use screen automation with `ui_automation_trade.py`

4. **Live Execution**
   - Load trained LSTM model
   - Fetch market data
   - Predict signal (BUY/SELL/HOLD)
   - Execute order and set SL/TP

---

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/forex-trading-bot.git
   cd forex-trading-bot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract) and add the path to system environment variables.

---

## 🧠 Training the LSTM Model

Make sure `historical_data.csv` is in the appropriate directory.

```bash
python train_model.py
```

The model includes:
- `LSTM` layers for temporal pattern learning  
- `Dropout` layers to prevent overfitting  
- `Dense` output layer for multi-class classification  
- Compiled with `categorical_crossentropy` loss and `Adam` optimizer

Model is saved as `trained_model.h5`.

---

## ▶️ Running the Bot

### ✅ MetaTrader 5 API Method

Use this if your broker allows direct MT5 access:

```bash
python mt5_trade.py
```

### ❌ Screen Automation Method

Use this for brokers that don’t support automation:

```bash
python ui_automation_trade.py
```

You can customize trade direction, pip size, and region values in the script.

---

## 📸 Screenshot Automation Notes

- The bot captures a region of the screen and uses OCR to read the entry price.
- PyAutoGUI and Pywinauto simulate human actions to:
  - Place a trade
  - Open the modify trade window
  - Set Stop Loss and Take Profit prices relative to prediction and current price

---

## ⚠️ Disclaimer

> This project is intended for educational and experimental purposes only.  
> Trading financial markets involves risk. Use at your own discretion and risk.

---

## 🤝 Contributions

Want to contribute? PRs and ideas are welcome!

1. Fork the repo  
2. Create a feature branch  
3. Submit a pull request  

---

## 📬 Contact

Feel free to reach out for collaboration or questions!
