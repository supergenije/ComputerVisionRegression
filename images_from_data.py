import timeseries_to_gaf as ttg
from pandas.tseries.holiday import USFederalHolidayCalendar as Calendar
from multiprocessing import Pool
import pandas as pd
import os
import datetime as dt
import ray
import numpy as np
import talib

from influxdb_dao.engine import InfluxDAO

from typing import *


PATH = os.path.dirname(__file__)
IMAGES_PATH = os.path.join(PATH, 'GramianAngularFields/TRAIN')
TEST_PATH = os.path.join(PATH, 'GramianAngularFields/TEST')
DATA_PATH = os.path.join(PATH, 'TimeSeries')

HOST="http://192.168.2.40:8086"
ORG="MarketData"
BUCKET="market_data"
TOKEN ="-hzZ7-09Y_NWKbjPFIVRVxeyOoK0P857Ype1guP3sk5lisZDT-erGJZ9F2oxaXD554H31glI0dJSQRbESICdvg=="


def data_to_image_preprocess() -> None:
    """
    :return: None
    """
    print('PROCESSING DATA')
    ive_data = 'IBM_adjusted.txt'
    col_name = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = pd.read_csv(os.path.join(DATA_PATH, ive_data), names=col_name, header=None)
    # Drop unnecessary data_slice
    df = df.drop(['High', 'Low', 'Volume', 'Open'], axis=1)
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], infer_datetime_format=True)
    df = df.groupby(pd.Grouper(key='DateTime', freq='1h')).mean().reset_index()     # '1min'
    df['Close'] = df['Close'].replace(to_replace=0, method='ffill')
    # Remove non trading days and times
    clean_df = clean_non_trading_times(df)
    # Send to slicing
    set_gaf_data(clean_df)

def load_data_form_idb(symbol: str, measurement: str, start_date: dt.datetime, end_date: dt.datetime) -> pd.DataFrame:
    enigine = InfluxDAO(host=HOST, token=TOKEN, org=ORG)
    df = enigine.get_bar_data(instrument=symbol, bucket=BUCKET, measurement=measurement, start_date=start_date, end_date=end_date)
    #df = df.drop(['high', 'low', 'volume', 'open'], axis=1)
    data_df = df.loc[symbol].reset_index()
    data_df['DateTime'] = data_df['_time']
    data_df = data_df.drop(['_time', '_measurement'],axis=1)

    return data_df
    #clean_df = clean_non_trading_times(data_df)
    

def clean_non_trading_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    :param df: Data with weekends and holidays
    :return trading_data:
    """
    # Weekends go out
    df = df[df['DateTime'].dt.weekday < 5].reset_index(drop=True)
    df = df.set_index('DateTime')
    # Remove non trading hours
    #df = df.between_time('9:00', '16:00')
    df.reset_index(inplace=True)
    # Holiday days we want to delete from data_slice
    holidays = Calendar().holidays(start='2000-01-01', end='2021-12-31')
    m = df['DateTime'].isin(holidays)
    clean_df = df[~m].copy()
    return clean_df.fillna(method='ffill')

#@ray.remote
def set_gaf_data(df: pd.DataFrame) -> None:
    """
    :param df: DataFrame data_slice
    :return: None
    """
    DATA_CHUNK = 250
    dates = df['DateTime'].dt.date
    dates = dates.drop_duplicates()
    list_dates = dates.apply(str).tolist()
    index = DATA_CHUNK
    # Container to store data_slice for the creation of GAF
    decision_map = {key: [] for key in ['LONG', 'SHORT']}
    while index < len(list_dates) - 1:
        # Select appropriate timeframe
        data_slice = df.loc[(df['DateTime'] > list_dates[index - DATA_CHUNK]) & (df['DateTime'] < list_dates[index])]
        
                # Group data_slice by time frequency
        # for freq in ['1d']:#,'5d','25d','125d']:
        #     #for freq in ['1h', '2h', '4h', '1d','1M','1Y']:
        #     group_dt = data_slice.groupby(pd.Grouper(key='DateTime', freq=freq)).mean().reset_index()
        #     group_dt = group_dt.dropna()
        # 
        #     gafs.append(group_dt['close'].tail(DATA_CHUNK))
        #    gafs.append(ta.RSI())

        gafs = [data_slice['close'].dropna(),
                talib.RSI(data_slice['close'], timeperiod=14).dropna(),
                talib.ADX(high=data_slice['high'], low=data_slice['low'],close=data_slice['close'], timeperiod=14).dropna()
                ]
        
        # gafs.append(data_slice['close'])
        # gafs.append(ta.RSI(data_slice['close'], timeperiod=14))
        
        # Decide what trading position we should take on that day
        future_value = df[df['DateTime'].dt.date.astype(str) == list_dates[index]]['close'].iloc[-1]
        current_value = data_slice['close'].iloc[-1]
        decision = trading_action(future_close=future_value, current_close=current_value)
        decision_map[decision].append([list_dates[index - 1], gafs])
        index += 1
    print('GENERATING IMAGES')
    # Generate the images from processed data_slice

    generate_gaf(decision_map)
    # Log stuff
    dt_points = dates.shape[0]
    total_short = len(decision_map['SHORT'])
    total_long = len(decision_map['LONG'])
    images_created = total_short + total_long
    print("========PREPROCESS REPORT========:\nTotal Data Points: {0}\nTotal Images Created: {1}"
          "\nTotal LONG positions: {2}\nTotal SHORT positions: {3}".format(dt_points,
                                                                           images_created,
                                                                           total_short,
                                                                           total_long))


def trading_action(future_close: int, current_close: int) -> str:
    """
    :param future_close: Integer
    :param current_close: Integer
    :return: Folder destination as String
    """
    current_close = current_close
    future_close = future_close
    return 'LONG' if current_close < future_close else 'SHORT'

@ray.remote
def plot_image(decision: str, image_data: pd.DataFrame):
    to_plot = [ttg.create_gaf(x)['gadf'] for x in image_data[1]]
    ttg.create_images(X_plots=to_plot,
                    image_name='{0}'.format(image_data[0].replace('-', '_')),
                    destination=decision)

def generate_gaf(images_data: Dict[str, pd.DataFrame]) -> None:
    """
    :param images_data:
    :return:
    """
    tasks = []
    for decision, data in images_data.items():
        for image_data in data:
            task = plot_image.remote(decision, image_data)
            tasks.append(task)
            #plot_image(decision, image_data)
            # to_plot = [ttg.create_gaf(x)['gadf'] for x in image_data[1]]
            # ttg.create_images(X_plots=to_plot,
            #                   image_name='{0}'.format(image_data[0].replace('-', '_')),
            #                   destination=decision)
    ray.get(tasks)

if __name__ == "__main__":
    ray.init()
    #pool = Pool(os.cpu_count())
    print(dt.datetime.now())
    print('CONVERTING TIME-SERIES TO IMAGES')
    data_df = load_data_form_idb(symbol='AGG-STK-SMART-USD', measurement='1 day', start_date=dt.datetime(2010,1,1), end_date=dt.datetime(2022,1,1))
    set_gaf_data(data_df)
    # task = set_gaf_data.remote(data_df)
    # assert ray.get(task) is None
    #pool.apply(set_gaf_data(data_df))
    #data_to_image_preprocess
    print('DONE!')
    print(dt.datetime.now())
