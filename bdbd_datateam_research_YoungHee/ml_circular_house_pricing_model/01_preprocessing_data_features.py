# -*- coding: utf-8 -*-

import sys
sys.path.append('/Users/yoingheechoi/workspace/boodong/bdbd_datateam_project01')
for _p in sys.path:
    print(_p)

import os

print('*** current_working_dir:{}'.format(os.getcwd()))

import numpy as np
import pandas as pd
import datetime
import multiprocessing as mulproc
import chardet
import glob

from apis import feature_api as fa
from apis import data_preprocessing_api as dpa
import utils.functions as F
from utils.log import Log
from configs import script_config

log = Log()

def detect_encoding(path):
    with open(path, 'rb') as rawdata:
        result = chardet.detect(rawdata.read(10000))
        print(result)

def transform_csv(path):
    df = pd.read_csv(path, encoding='euc-kr')
    df.to_csv(path, encoding='utf-8-sig')

def save_csv(df, path):
    df.to_csv(path, encoding='utf-8-sig')
    log.info('succeeded to save df into {}'.format(path))

#############################################################################################
# 1 사분면 데이터
#############################################################################################

# 주택보급률 데이터 전처리 하는 함수 (1 사분면 : 주택재고 (S))
def preprocessing_house_supply_rate(dest_path):
    df = pd.read_csv('./data/house_supply_rate.csv')
    df = df.drop(['Unnamed: 0'], axis=1)
    df.loc[:, '시점'] = pd.to_datetime(df['시점'].astype(str), format='%Y')
    df = df.replace('-', np.nan)

    cols = df.columns.values
    cols = cols[~np.isin(cols, ['구분(1)', '시점'])]
    df.loc[:, cols] = df[cols].astype(float)
    
    df = df.groupby(by='구분(1)', as_index=False, group_keys=False).apply(lambda _df: _df.set_index('시점').resample('MS').first())

    df = df.reset_index()
    df.loc[:, '구분(1)'] = df['구분(1)'].ffill()
    df.loc[:, cols] = df[cols].interpolate(method='linear', limit_direction='both', axis=0)

    save_csv(df, f'{dest_path}/house_supply_rate2.csv')

# 연도별 인구 데이터 전처리 하는 함수 (1 사분면 : 인구 (p))
def preprocessing_population_years(path):
    df = pd.read_csv(path)
    for _col in df.columns:
        if 'Unnamed' in _col:
            df = df.drop([_col], axis=1)

    cols = df.columns
    cols = cols[~np.isin(cols, ['행정구역'])]

    area_list = df['행정구역'].unique()

    df_total_result = pd.DataFrame()

    for _area in area_list:

        df_result = pd.DataFrame()
        df_result.loc[:, 'date'] = pd.to_datetime(cols.str.split('_').str[0].unique(), format='%Y년%m월')
        df_result.loc[:, 'area'] = _area
        
        cat_cols = cols.str.split('_').str[1].unique()

        for _cat_col in cat_cols:
            _cat_col_group = []

            for _col in cols:
                if _cat_col in _col:
                    _cat_col_group.append(_col)

            df_result.loc[:, _cat_col] = df[df['행정구역'] == _area][_cat_col_group].T.squeeze().astype(str).str.replace(',', '').astype(float).values

        if df_total_result.empty is True:
            df_total_result = df_result
        else:
            df_total_result = df_total_result.append(df_result, ignore_index=True)

    return df_total_result

# 전체 인구 데이터 전처리 하는 함수 (1 사분면 : 인구 (p))
def preprocessing_population_total(dest_path):
    df_result = pd.DataFrame()
    file_list = glob.glob('./data/populations/*.csv')
    for _file in file_list:
        log.info('_file:{}'.format(_file))
        df = preprocessing_population_years(_file)

        if df_result.empty is True:
            df_result = df
        else:
            df_result = df_result.append(df, ignore_index=True)

    
    df_result = df_result.sort_values(by=['area', 'date'], axis=0)
    save_csv(df_result, f'{dest_path}/population2.csv')

# 주택담보대출금리 (신규취급액 기준) (r) 전처리 하는 함수 (1 사분면 : 금리(r))
    # 주택담보대출 금리 신규취급액 기준
    # 2001.09-2021.11
    # 연이율 %
def preprocessing_house_loan_rate_new(dest_path):
    df = pd.read_csv('./data/금리_주택담보대출_신규취급액기준.csv')
    df = df.iloc[3:-3]
    df = df.drop(['Unnamed: 0'], axis=1)
    df = df.rename(columns={'통계표':'date', '4.2.2.1 신규취급액 기준':'rate'})
    df.loc[:, 'date'] = pd.to_datetime(df['date'])
    df.loc[:, 'rate'] = df['rate'].astype(float)
    df = df.dropna()

    save_csv(df, f'{dest_path}/house_loan_rate_new2.csv')

# 주택담보대출금리 (잔액 기준) (r) 전처리 하는 함수 (1 사분면 : 금리(r))
    # 주택담보대출 금리 잔액 기준
    # 2009.09-2021.11
    # 연이율 %
def preprocessing_house_loan_rate_balance(dest_path):
    df = pd.read_csv('./data/금리_주택담보대출_잔액기준.csv')
    df = df.iloc[3:-3]
    df = df.drop(['Unnamed: 0'], axis=1)
    df = df.rename(columns={'통계표':'date', '4.2.2.2 잔액 기준':'rate'})
    df.loc[:, 'date'] = pd.to_datetime(df['date'])
    df.loc[:, 'rate'] = df['rate'].astype(float)
    df = df.dropna()

    save_csv(df, f'{dest_path}/house_loan_rate_balance2.csv')

#############################################################################################
# 2 사분면 데이터
#############################################################################################

def main():

    dest_path = './data/preprocessed'

    F.make_dir(dest_path)

    # 1 사분면 데이터
    preprocessing_house_supply_rate(dest_path)
    preprocessing_population_total(dest_path)
    preprocessing_house_loan_rate_new(dest_path)
    preprocessing_house_loan_rate_balance(dest_path)
    
    
    return

if __name__ == '__main__':
    main()
    print('finished')
