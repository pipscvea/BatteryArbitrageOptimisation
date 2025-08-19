import pandas as pd
import glob
import os


def merge_price_csvs(file_list, datetime_col='StartTime', price_col='SystemSellPrice'):
  """
  Merge multiple price CSV files into a single continuous DataFrame.
  
  Parameters:
  - file_list: list of CSV file paths
  - datetime_col: name of the datetime column in each CSV
  - price_col: name of the price column in each CSV
  
  Returns:
  - merged_df: pandas DataFrame with datetime index and price column, sorted and deduplicated
  """
  dfs = []
  
  for file in file_list:
      df = pd.read_csv(file, parse_dates=[datetime_col])
      dfs.append(df)
  
  merged_df = pd.concat(dfs, ignore_index=True)
  merged_df = merged_df.drop_duplicates(subset=datetime_col)
  merged_df = merged_df.sort_values(by=datetime_col)
  merged_df.set_index(datetime_col, inplace=True)
  
  return merged_df

def merge_with_demand(price_df, demand_csv, datetime_col='StartTime', how='inner'):
  """
  Merge a price DataFrame with a demand CSV file on datetime.
  
  Parameters:
  - price_df: pandas DataFrame with datetime index
  - demand_csv: path to demand CSV file
  - datetime_col: name of the datetime column in demand CSV
  - how: type of merge ('inner', 'left', 'outer', 'right')
  
  Returns:
  - merged_df: DataFrame with both price and demand columns
  """
  
  # Load demand CSV
  demand_df = pd.read_csv(demand_csv, parse_dates=[datetime_col])
  demand_df.set_index(datetime_col, inplace=True)
  whole_merged_df = price_df.merge(demand_df, left_index=True, right_index=True, how=how)
  whole_merged_df = whole_merged_df.sort_values(by=datetime_col)
  
  return whole_merged_df


def train_data_assembly():
  price_folder_path = "SystemSellAndBuyPrices"
  price_file_pattern = os.path.join(price_folder_path, "SystemSellAndBuyPrices-2017*.csv")
  price_file_list = glob.glob(price_file_pattern)

  merged_prices_df = merge_price_csvs(price_file_list)
  merge = merge_with_demand(merged_prices_df, 'RollingSystemDemand\\RollingSystemDemand-2017-01-01T00_00_00.000Z-2017-03-05T17_00_00.000Z.csv')
  return merge

