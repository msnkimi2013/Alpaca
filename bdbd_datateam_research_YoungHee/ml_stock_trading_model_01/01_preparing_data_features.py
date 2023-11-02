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

from apis import feature_api as fa
from apis import data_preprocessing_api as dpa
import utils.functions as F
from utils.log import Log
from configs import script_config

log = Log()

# 상승률 계산
def get_ratio(begin, end):
    diff = end - begin
    ratio = diff / begin
    return ratio

# 실거래가격 리턴 (지역, 단지)
def get_transaction_by_type(si, gu, dong, danji_name, danji_id, trade_type, last_date):
    if danji_name == '':
        return fa.get_transaction_area_mean(si, gu, dong, year_last_date=last_date, _type=trade_type).drop(['date'], axis=1)
    else:
        return fa.get_transaction_danji(si, gu, dong, danji_name, danji_id, year_last_date=last_date, _type=trade_type).drop(['date'], axis=1)

# 평당 매매가, 전세가, 전세가율 리턴
def get_sise_features(si, gu, dong, danji_name, danji_id):
    _today = datetime.datetime.today()
    
    df_tran_trade = get_transaction_by_type(si, gu, dong, danji_name, danji_id, 'trading', _today)
    if df_tran_trade.empty is True:
        log.warn('df_tran_trade is empty. {} {} {}'.format(si, gu, dong))
        raise Exception('error1')

    df_tran_rent = get_transaction_by_type(si, gu, dong, danji_name, danji_id, 'rent', _today)
    if df_tran_rent.empty is True:
        log.warn('df_tran_rent is empty. {} {} {}'.format(si, gu, dong))
        raise Exception('error2') 

    df_merged = pd.merge(df_tran_trade, df_tran_rent, how='left', left_index=True, right_index=True, suffixes=('_x', '_y'))
    df_merged = df_merged.dropna()

    # 전세가율 계산
    df_merged.loc[:, 'rent_ratio'] = df_merged['unit_price_x'] / df_merged['unit_price_y']

    months = 12

    # 매매가 과거 1년간 상승률
    df_merged.loc[:, 'sell_inc_ratio_1'] = get_ratio(df_merged['unit_price_x'].shift(months), df_merged['unit_price_x'])

    # 전세가 과거 1년간 상승률
    df_merged.loc[:, 'rent_inc_ratio_1'] = get_ratio(df_merged['unit_price_y'].shift(months), df_merged['unit_price_y'])

    # 전세가율 과거 1년간 상승률
    df_merged.loc[:, 'rent_ratio_inc_ratio_1'] = get_ratio(df_merged['rent_ratio'].shift(months), df_merged['rent_ratio'])

    df_merged = df_merged.dropna()
    df_merged = df_merged.rename(columns={"unit_price_x":"sell_unit_price", "unit_price_y":"rent_unit_price"})

    return df_merged

# 단지 속성값 리턴
def get_danji_features(si, gu, dong, danji_name, danji_id):

    _today = datetime.datetime.today()

    ##########################################################
    # 아파트 속성 값들
    ##########################################################
    df_attr = fa.load_danji_attributes(si, gu, dong, danji_name, danji_id)
    if df_attr.empty is True:
        log.warn('df_attr is empty. {} {} {} {} {}'.format(si, gu, dong, danji_name, danji_id))
        raise Exception('error1') 

    # 세대수
    total_households = df_attr['total_households'].values[0]

    # 준공월
    completion_month = df_attr['completion_month'].values[0]

    try:
        completion_month = datetime.datetime.strptime(completion_month, '%Y.%m')
    except Exception as e:
        try:
            completion_month = datetime.datetime.strptime(completion_month, '%Y.%m.%d')
        except Exception as e:
            raise Exception('error2')

    # 준공월 이후 일수
    delta_days = (_today - completion_month).days

    # 용적률
    floor_area_ratio = df_attr['floor_area_ratio'].values[0]

    # 건설사
    constructor = df_attr['constructor'].values[0]
    

    return total_households, delta_days, floor_area_ratio, constructor

# 지역 평균과 아파트 단지 매매가, 전세가, 전세가율 차이 계산하여 리턴
def get_gap(df_si, df_gu, df_dong, df_danji):

    # 지역 단위별로 컬럼명 변경
    columns = df_si.columns
    for _col in columns:
        df_si = df_si.rename(columns={_col:f"{_col}_si"})
        df_gu = df_gu.rename(columns={_col:f"{_col}_gu"})
        df_dong = df_dong.rename(columns={_col:f"{_col}_dong"})
        df_danji = df_danji.rename(columns={_col:f"{_col}_danji"})

    # 지역 평균과 단지의 df를 모두 병합
    df_merged = pd.merge(df_danji, df_dong, how='left', left_index=True, right_index=True)
    df_merged = pd.merge(df_merged, df_gu, how='left', left_index=True, right_index=True)
    df_merged = pd.merge(df_merged, df_si, how='left', left_index=True, right_index=True)

    df_merged = df_merged.dropna()

    # 지역 평균간 차이를 계산
    for _col in columns:
        df_merged.loc[:, f'{_col}_sigu_gap'] = df_merged[f'{_col}_si'] - df_merged[f'{_col}_gu']
        df_merged.loc[:, f'{_col}_sidong_gap'] = df_merged[f'{_col}_si'] - df_merged[f'{_col}_dong']
        df_merged.loc[:, f'{_col}_sidanji_gap'] = df_merged[f'{_col}_si'] - df_merged[f'{_col}_danji']

        df_merged.loc[:, f'{_col}_gudong_gap'] = df_merged[f'{_col}_gu'] - df_merged[f'{_col}_dong']
        df_merged.loc[:, f'{_col}_gudanji_gap'] = df_merged[f'{_col}_gu'] - df_merged[f'{_col}_danji']

        df_merged.loc[:, f'{_col}_dongdanji_gap'] = df_merged[f'{_col}_dong'] - df_merged[f'{_col}_danji']

    return df_merged

# 데이터 피쳐 만드는 함수
def build_data_features(si, gu_list):
    
    df_danji_features = pd.DataFrame()

    ##########################################################
    # 시 평균 평당 매매가, 전세가, 전세가율, 매매가 상승률, 전세가 상승률
    ##########################################################
    df_sise_feature_si = get_sise_features(si, '', '', '', '')
    
    for gu in gu_list:

        try:
            ##########################################################
            # 구 평균 평당 매매가, 전세가, 전세가율, 매매가 상승률, 전세가 상승률
            ##########################################################
            df_sise_feature_gu = get_sise_features(si, gu, '', '', '')
        except Exception as e:
            continue

        dong_list = fa.get_dong_list(si, gu)
        for dong in dong_list:

            try:
                ##########################################################
                # 동 평균 평당 매매가, 전세가, 전세가율, 매매가 상승률, 전세가 상승률
                ##########################################################
                df_sise_feature_dong = get_sise_features(si, gu, dong, '', '')
            except Exception as e:
                continue

            danji_list = fa.get_danji_list(si, gu, dong)
            for _danji in danji_list:

                danji_name = _danji[0]
                danji_id = _danji[1]

                try:
                    ##########################################################
                    # 단지 평균 평당 매매가, 전세가, 전세가율, 매매가 상승률, 전세가 상승률
                    ##########################################################
                    df_sise_feature_danji = get_sise_features(si, gu, dong, danji_name, danji_id)
                    df_sise_feature_danji = df_sise_feature_danji[df_sise_feature_danji.index >= datetime.datetime(2012,1,1)]
                    if df_sise_feature_danji.index[0] > datetime.datetime(2012,1,1):
                        log.warn('{} {} {} {} {}. time-series is not long enough. {} > 2021-01-01. shape:{}'.format(si, gu, dong, danji_name, danji_id, df_sise_feature_danji.index[0], df_sise_feature_danji.shape))
                        continue

                    ##########################################################
                    # 지역평균간 갭, 지역평균과 아파트 갭 (매매가, 전세가, 전세가율)
                    ##########################################################
                    df_sise_gap = get_gap(df_sise_feature_si, df_sise_feature_gu, df_sise_feature_dong, df_sise_feature_danji)

                    ##########################################################
                    # 단지 속성값
                    ##########################################################
                    total_households, delta_days, floor_area_ratio, constructor = get_danji_features(si, gu, dong, danji_name, danji_id)
                except Exception as e:
                    continue

                area_key1 = f'{si}_{gu}_{dong}'
                area_key2 = f'{si}_{gu}'
                area_key3 = f'{si}_{gu}_{dong}_{danji_name}_{danji_id}'

                df_sise_gap.loc[:, 'date'] = df_sise_gap.index
                df_sise_gap.loc[:, 'year'] = df_sise_gap.index.year
                df_sise_gap.loc[:, 'month'] = df_sise_gap.index.month
                df_sise_gap.loc[:, 'si'] = si
                df_sise_gap.loc[:, 'gu'] = gu
                df_sise_gap.loc[:, 'dong'] = dong
                df_sise_gap.loc[:, 'danji_name'] = danji_name
                df_sise_gap.loc[:, 'danji_id'] = danji_id
                df_sise_gap.loc[:, 'area_key1'] = area_key1
                df_sise_gap.loc[:, 'area_key2'] = area_key2
                df_sise_gap.loc[:, 'area_key3'] = area_key3
                df_sise_gap.loc[:, 'total_households'] = total_households
                df_sise_gap.loc[:, 'delta_days'] = delta_days
                df_sise_gap.loc[:, 'floor_area_ratio'] = floor_area_ratio
                df_sise_gap.loc[:, 'constructor'] = constructor

                if df_danji_features.empty is True:
                    df_danji_features = df_sise_gap
                else:
                    df_danji_features = df_danji_features.append(df_sise_gap)

    return df_danji_features

def build_additional_featrues(path):

    df_danji_features = pd.read_csv(f'{path}/df_danji_features.csv')
    df_danji_features = df_danji_features.drop(['date.1'], axis=1)

    categorical_columns = ['si', 'gu', 'dong', 'area_key1', 'area_key2', 'constructor', 'danji_name', 'danji_id', 'year', 'month']

    key_columns = ['area_key3', 'date']

    df_danji_features = df_danji_features.set_index(key_columns)

    column_list = df_danji_features.columns

    # indexing multiindex df examples ##############################################################
    # df_danji_features.loc[pd.IndexSlice[df_danji_features.index[0][0], :], 'total_households']
    # df_danji_features.loc[pd.IndexSlice[df_danji_features.index[0][0], :], :].index
    # df_danji_features.T.loc[_col, :].loc[:, '2012-02-01']
    # #df_si.apply(lambda _df: _df.set_index['area_key3', 'date'].T.loc['total_households', :])
    # pd.qcut(df_danji_features.T.loc['total_households', :], 100, labels=False, duplicates='drop')
    ################################################################################################
    
    _q = 100

    for _col in column_list:
        if _col in categorical_columns:
            continue
        if _col in key_columns:
            continue

        log.info('build qcut column for [{}]'.format(_col))

        df_danji_features.loc[:, f'q_all_{_col}'] = pd.qcut(df_danji_features.T.loc[_col, :], _q, labels=False, duplicates='drop')
        df_danji_features.loc[:, f'q_si_{_col}'] = df_danji_features.groupby(by=['si'], as_index=False, group_keys=False).apply(lambda _df: pd.qcut(_df.T.loc[_col, :], _q, labels=False, duplicates='drop'))
        df_danji_features.loc[:, f'q_gu_{_col}'] = df_danji_features.groupby(by=['si', 'gu'], as_index=False, group_keys=False).apply(lambda _df: pd.qcut(_df.T.loc[_col, :], _q, labels=False, duplicates='drop'))
        df_danji_features.loc[:, f'q_dong_{_col}'] = df_danji_features.groupby(by=['si', 'gu', 'dong'], as_index=False, group_keys=False).apply(lambda _df: pd.qcut(_df.T.loc[_col, :], _q, labels=False, duplicates='drop'))

        # df_danji_features.loc[:, f'q_all_{_col}'] = pd.qcut(df_danji_features.set_index(['area_key3', 'date']).T.loc[_col, :], 100, labels=False, duplicates='drop')
        # df_danji_features.loc[:, f'q_si_{_col}'] = df_danji_features.groupby(by=['si']).apply(lambda _df: pd.qcut(_df.set_index(['area_key3', 'date']).T.loc[_col, :], _q, labels=False, duplicates='drop'))
        # df_danji_features.loc[:, f'q_gu_{_col}'] = df_danji_features.groupby(by=['si', 'gu']).apply(lambda _df: pd.qcut(_df.set_index(['area_key3', 'date']).T.loc[_col, :], _q, labels=False, duplicates='drop'))
        # df_danji_features.loc[:, f'q_dong_{_col}'] = df_danji_features.groupby(by=['si', 'gu', 'dong']).apply(lambda _df: pd.qcut(_df.set_index(['area_key3', 'date']).T.loc[_col, :], _q, labels=False, duplicates='drop'))

    log.info('intermediate check. shape:{}'.format(df_danji_features.shape))
    df_danji_features.to_csv(f'{path}/df_danji_features_qcut_intermediate.csv', encoding='utf-8-sig')

    df_danji_features = df_danji_features.reset_index()

    # 지역 코드 (시, 구, 동)
    for _col in categorical_columns:
        df_danji_features.loc[:, _col] = pd.factorize(df_danji_features[_col])[0]

    df_danji_features = df_danji_features.dropna()
    log.info('succeeded to add qcut columns. shape:{}'.format(df_danji_features.shape))
    df_danji_features.to_csv(f'{path}/df_danji_features_qcut.csv', encoding='utf-8-sig')

    return df_danji_features

def parallel_proc(args):
    si = args[0]
    gu_list = args[1]

    return build_data_features(si, gu_list)

def parallel_main_si(si):
    gu_list = fa.get_gu_list(si)

    ######### single-processing ##########
    #df_danji_features = build_data_features(si, gu_list)
    ######################################

    ######### muilti-processing #########
    splited_data = dpa.split_data_for_parallel_proc(gu_list)

    args = []
    for d in splited_data:
        args.append([si]+[d])

    with mulproc.Pool(dpa.__cpu_count) as p:
    #with mulproc.Pool(1) as p:
       df_danji_features = pd.concat(p.map(parallel_proc, args))
    ######################################

    return df_danji_features

def build_all_data_features(si_list, path):
    results = []
    for si in si_list:
        log.info('si:{}'.format(si))
        results.append(parallel_main_si(si))

    df_danji_features = pd.concat(results)

    danji_count = df_danji_features.set_index(['area_key3', 'date']).groupby(level='area_key3').size().shape[0]
    date_count = df_danji_features.set_index(['area_key3', 'date']).groupby(level='area_key3').size()[0]

    df_danji_features.to_csv(f'{path}/df_danji_features.csv', encoding='utf-8-sig')
    log.info('succeeded to build df. shape:{} danji_count:{} date_count:{}'.format(df_danji_features.shape, danji_count, date_count))

def main():
    si_list = script_config._config_si_list

    path = './data/features'
    F.make_dir(path)
    
    #build_all_data_features(si_list, path)
    build_additional_featrues(path)
    return

def test():
    df = pd.read_csv('./data/features/df_danji_features_서울특별시_2020_added.csv')
    return

if __name__ == '__main__':
    main()
    #test()
    print('finished')
