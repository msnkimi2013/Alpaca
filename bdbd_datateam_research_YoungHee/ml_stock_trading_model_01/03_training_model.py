# -*- coding: utf-8 -*-

import sys
import os

import numpy as np
import pandas as pd
import datetime
import multiprocessing as mulproc

import sys

print('*** current_working_dir:{}'.format(os.getcwd()))

from log import Log


log = Log()

# 피쳐 데이터 불러오는 함수
def read_data_features():
    df = pd.read_csv('./data/features/df_danji_features_qcut.csv')
    df = df.drop(['area_key1', 'area_key2', 'danji_name', 'danji_id'], axis=1)                          # delete 4 columns
    danji_list = df['area_key3'].unique()
    danji_count = len(danji_list)
    min_date = df['date'].min()
    max_date = df['date'].max()

    log.info('df.shape:{}'.format(df.shape))
    log.info('danji_count:{} time-period:{}-{}'.format(danji_count, min_date, max_date))
    return df, danji_list

# 단지별로 시계열 길이를 나눠주는 함수
def split_df_by_len(df, level_name, _len):
    df1 = df.groupby(level=level_name, as_index=False, group_keys=False).apply(lambda _df: _df.iloc[:_len])
    df2 = df.groupby(level=level_name, as_index=False, group_keys=False).apply(lambda _df: _df.iloc[_len:])
    return df1, df2

# df의 index 정보 (단지 개수, 시계열 최소, 최대) 프린트 해주는 함수
def print_index_info(df, _name):
    _danji_count = len(df.groupby(level='area_key3'))
    _min_date = df.loc[df.index[0][0], :].index[0]
    _max_date = df.loc[df.index[0][0], :].index[-1]

    log.info('[{}] shape:{}'.format(_name, df.shape))
    log.info('[{}] danji_count:{} time-period:({}) - ({})'.format(_name, _danji_count, _min_date, _max_date))
    
# features와 targets를 분리해 주는 함수
def split_features_and_targets(df, target_months):
    target_col = 'sell_unit_price_sidanji_gap'                                                            # Y
    df = df.set_index(['area_key3', 'date'])
    df.loc[:, 'target_value'] = df.groupby(level='area_key3')[target_col].shift(-target_months)           # shifting in apt group
    df = df.dropna()                                                                                      # drop blank cell
    df_features = df.drop(['target_value'], axis=1)                                                  #
    df_targets = df[['target_value']]

    print_index_info(df_features, 'df_features')
    print_index_info(df_targets, 'df_targets')

    return df_features, df_targets

# 훈련 데이터와 테스트 데이터를 분리해 주는 함수
def split_train_and_test(df_features, df_targets):
    total_len = len(df_features.loc[df_features.index[0][0], :].index)
    train_len = int(total_len * 0.8)

    df_features_train, df_features_test = split_df_by_len(df_features, 'area_key3', train_len)
    df_targets_train, df_targets_test = split_df_by_len(df_targets, 'area_key3', train_len)

    print_index_info(df_features_train, 'df_features_train')
    print_index_info(df_targets_train, 'df_targets_train')

    print_index_info(df_features_test, 'df_features_test')
    print_index_info(df_targets_test, 'df_targets_test')

    df_features_train = df_features_train.reset_index()
    df_targets_train = df_targets_train.reset_index()

    df_features_test = df_features_test.reset_index()
    df_targets_test = df_targets_test.reset_index()

    return df_features_train, df_targets_train, df_features_test, df_targets_test


def main():
    # 예측 개월 수
    target_months = 36

    # 피쳐 데이터 불러오기
    df, danji_list = read_data_features()

    # 피쳐 데이터와 타겟 데이터 나누기
    df_features, df_targets = split_features_and_targets(df, target_months)

    # 훈련 데이터와 테스트 데이터 나누기
    df_features_train, df_targets_train, df_features_test, df_targets_test = split_train_and_test(df_features, df_targets)

    log.info('df_features_train.shape:{}'.format(df_features_train.shape))
    log.info('df_targets_train.shape:{}'.format(df_targets_train.shape))
    log.info('df_features_test.shape:{}'.format(df_features_test.shape))
    log.info('df_targets_test.shape:{}'.format(df_targets_test.shape))

    # Do something
    # ...

    return

if __name__ == '__main__':
    main()
    print('finished')
