import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import scipy.optimize as opt
import time
import yaml




def load_price_data(filepath, datetime_col=None, price_cols=None):
    """
    Load electricity price data from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    datetime_col : str, optional
        Name of the column containing datetime information (will be parsed).
    price_cols : list of str, optional
        List of column names to keep (e.g. ['SBP', 'SSP']).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with datetime index (if provided).
    """
    # Read CSV
    df = pd.read_csv(filepath)

    # If a datetime column is provided, parse it
    if datetime_col:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df.set_index(datetime_col, inplace=True)

    # Keep only relevant price columns if specified
    if price_cols:
        missing = [col for col in price_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")
        df = df[price_cols]

    # Handle missing values (optional: forward fill then drop remaining)
    df = df.ffill().dropna()

    return df

def replay_feed(data, delay=0.0):
    """
    Simulate a live feed from historical data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Time-indexed DataFrame with electricity prices.
    delay : float
        Seconds to wait between timesteps (simulated real-time).
    
    Yields
    ------
    (timestamp, row) : tuple
        Each row of data with its timestamp.
    """
    for timestamp, row in data.iterrows():
        yield timestamp, row
        time.sleep(delay)  # wait before sending the next tick



df = pd.read_csv("SystemSellAndBuyPrices-2025-06-29T12_30_00.000Z-2025-07-06T12_30_00.000Z.csv", parse_dates=["SettlementDate"])
df.set_index("SettlementDate", inplace=True)

def load_config(filepath):
    with open(filepath, "r") as f:
        config = yaml.safe_load(f)
    return config

# Load configuration files
battery_cfg = load_config('BatteryConfig.yaml')
trading_cfg = load_config('TradingConfig.yaml')
# Extract battery parameters        
battery_capacity = battery_cfg['battery']['capacity_kwh']
max_charge_kw = battery_cfg['battery']['max_charge_kw']
max_discharge_kw = battery_cfg['battery']['max_discharge_kw']
efficiency = battery_cfg['battery']['efficiency']
soc_init = battery_cfg['battery']['soc_init']
degradation_rate = battery_cfg['battery']['degradation_rate']
# Extract trading parameters
starting_balance = trading_cfg['trading']['starting_balance']
transaction_fee = trading_cfg['trading']['transaction_fee']
max_trade_volume = trading_cfg['trading']['max_trade_volume']


df["ssp_lag1"] = df["SystemSellPrice"].shift(1)
df["ssp_lag2"] = df["SystemSellPrice"].shift(2)
df["ssp_ma6"]  = df["SystemSellPrice"].rolling(6).mean()
df["sbp_ma6"]  = df["SystemBuyPrice"].rolling(6).mean()
df["spread"]   = df["SystemBuyPrice"] - df["SystemSellPrice"]
df["MidPrice"] = (df["SystemSellPrice"] + df["SystemBuyPrice"]) / 2
df["Return"] = np.log(df["MidPrice"] / df["MidPrice"].shift(1)) # Volatility is computed from log returns
df["RollingVol"] = df["Return"].rolling(window=48).std() * np.sqrt(48) # Annualize the volatility


x_vals = []
y_vals = [] 
y1_vals = []
# y2_vals = []


# plt.plot(x_vals, y_vals, label='SystemSellPrice', color='blue')
# plt.plot(x_vals, y1_vals, label='6-hr MA SSP', color='orange', marker='o')
# plt.plot(x_vals, y2_vals, label='Spread', color='green', marker='x')


index = count()

def animate(i):
    x_vals.append(next(index))
    y_vals.append(df["SystemSellPrice"].iloc[i % len(df)])
    y1_vals.append(df["ssp_ma6"].iloc[i % len(df)])
    # plt.legend()
    
    plt.cla()
    plt.plot(x_vals, y_vals, label='Dynamic Line', color='blue')
    plt.plot(x_vals, y1_vals, label='6-hr MA SSP', color='orange', marker='o')
    # plt.xlabel('Time Step')
    # plt.ylabel('Price')
    # plt.title('Dynamic Graph of System Sell Price and 6-hr MA SSP')
    # plt.legend()
    # plt.grid(True)

ani = FuncAnimation(plt.gcf(), animate, interval=10)

plt.tight_layout()
plt.show()
