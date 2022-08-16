import yfinance as yf
import matplotlib.pyplot as plt

# Lade Microsoft Aktiendaten
msft = yf.Ticker("MSFT")

# Get stock info
msft.info

# Get historical market data
hist = msft.history(period="max")
hist['Close'].plot(figsize=(16, 9))

# Get data
dates = hist['Close'].axes[0]
stock_values = hist['Close'].values

# Plotte Daten
plt.plot(dates, stock_values)
plt.show()

# --- Projektarbeit ---
