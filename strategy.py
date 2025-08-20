import yaml
import numpy as np
import pandas as pd

from train_data_assembly import train_data_assembly

df = train_data_assembly()

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


# Track battery state
charge = battery_capacity * soc_init    # start half full
charge_list = []
profit_list = []
charge_sale_vol = 100
Capacity_remaining = battery_capacity - charge

# Buy/Sell 
Strong = 0.2
Mid = 0.15
Weak = 0.1

### Defining Buy/Sell vol
StrongBuyVol =  Capacity_remaining * Strong
MidBuyVol = Capacity_remaining * Mid
WeakBuyVol = Capacity_remaining * Weak

StrongSellVol = charge * Strong
MidSellVol = charge * Mid
WeakSellVol = charge * Weak

#### Features

df["ssp_lag1"] = df["SystemSellPrice"].shift(1)
df["ssp_lag2"] = df["SystemSellPrice"].shift(2)
df["ssp_ma6"]  = df["SystemSellPrice"].rolling(6).mean()
df["sbp_ma6"]  = df["SystemBuyPrice"].rolling(6).mean()
df["ssp_rolling_mean_48"] = df["SystemSellPrice"].rolling(48).mean()
df["spread"]   = df["SystemBuyPrice"] - df["SystemSellPrice"]
df['ssp_rolling_std_3']  = df['SystemSellPrice'].rolling(3).std()

##### Demand Features
df["Demand_lag1"] = df["Demand"].shift(1)
df["Demand_lag2"] = df["Demand"].shift(2)
df["Demand_ma6"]  = df["Demand"].rolling(6).mean()
df["Demand_rolling_std_3"] = df["Demand"].rolling(3).std()

##### Time Features
# df['hour'] = df['StartTime'].dt.hour
# df['day_of_week'] = df['StartTime'].dt.dayofweek  # 0=Monday
# df['month'] = df['StartTime'].dt.month
# df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int) 
df['hour'] = df.index.hour
df['day_of_week'] =  df.index.dayofweek  # 0=Monday
df['month'] =  df.index.month
df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int) 

##### Battery Features
df["Battery_Level"] = soc_init  # Initial state of charge

##### Market Dynamics
df['is_price_spike'] = (df['SystemSellPrice'] > df['ssp_rolling_mean_48']*1.5).astype(int)

##### Volatility 
epsilon = 1e-6
df["MidPrice"] = (df["SystemSellPrice"] + df["SystemBuyPrice"]) / 2
df["Return"] = np.log(df["MidPrice"] / df["MidPrice"].shift(1) + epsilon) # Volatility is computed from log returns
df["RollingVol"] = df["Return"].rolling(window=48).std() * np.sqrt(48) # Annualize the volatility
df["RollingVol"]
df.dropna()

def StrongWeakMechanism(df=df):
    threshold = df["SystemSellPrice"].rolling(window=48).std().mean() * 0.3
    df['Action'] = np.where(df["SystemSellPrice"] < df["ssp_ma6"] - threshold*2, 'Strong Buy',
                            np.where(df["SystemSellPrice"] < df["ssp_ma6"] - threshold, 'Mid Buy',
                                np.where(df["SystemSellPrice"] < df["ssp_ma6"] - threshold * 0.5, 'Weak Buy',
                                    np.where(df["SystemSellPrice"] > df["ssp_ma6"] + threshold * 2, 'Strong Sell',
                                        np.where(df["SystemSellPrice"] > df["ssp_ma6"] + threshold, 'Mid Sell',
                                                 np.where(df["SystemSellPrice"] > df["ssp_ma6"] + threshold * 0.5, "Weak Sell", 'HOLD'))))))
                            
    return df['Action']


df['Action'] = StrongWeakMechanism(df)

X = df[['SystemSellPrice', 'ssp_lag1', 'ssp_lag2', 'ssp_ma6', 'hour', 'day_of_week', 'month',
         'is_weekend', 'Battery_Level', 'Demand_lag1', 'Demand_lag2', 'Demand_ma6', 'Demand_rolling_std_3',
           'ssp_rolling_mean_48', 'ssp_rolling_std_3']]
y = df['Action']
