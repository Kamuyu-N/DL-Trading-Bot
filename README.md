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

---

## 🔁 Workflow Overview

1. **Data Preparation**  
   - `FeatureCreation.py` handles feature engineering with technical indicators.
   - Data file `eurusd-15m.csv` contains historical price data used for model training.

2. **Model Training**  
   - `model.py` builds and trains the LSTM model.
   - Outputs a trained file like `trained_model.h5`.

3. **Broker Type Selection**
   - ✅ If your broker supports MetaTrader 5: use `MetaTrader.py`
   - ❌ If not, use screen automation via `OrderAutomation.py`

4. **Trade Execution**
   - `AssistanceFunctions.py` contains helper logic for modifying orders and setting SL/TP.
   - Choose execution mode and run real-time predictions and trades.

---

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Kamuyu-N/DL-Trading-Bot.git
   cd DL-Trading-Bot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract) and add the path to system environment variables.

---

## 🧠 Training the LSTM Model

Make sure `eurusd-15m.csv` is present.

```bash
python model.py
```

The model includes:
- `LSTM` layers for sequence learning  
- `Dropout` layers to prevent overfitting  
- `Dense` output for class prediction (Buy, Sell, Hold)  
- Compiled with `categorical_crossentropy` and `Adam` optimizer

---

## ▶️ Running the Bot

### ✅ MetaTrader 5 API Method

Use this if your broker allows direct MT5 access:

```bash
python MetaTrader.py
```

### ❌ Screen Automation Method

Use this for brokers that don’t support automation:

```bash
python OrderAutomation.py
```

You can customize trade direction, pip size, and screen region inside the script.

---

## 📸 Screenshot Automation Notes

- `OrderAutomation.py` captures a region of the screen and uses OCR to extract prices.
- `PyAutoGUI` and `Pywinauto` simulate user input to:
  - Place a trade
  - Open trade modification
  - Set Stop Loss and Take Profit

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
