# -*- coding: utf-8 -*-

import sys
import os

print('*** current_working_dir:{}'.format(os.getcwd()))

import numpy as np
import pandas as pd
import datetime
import multiprocessing as mulproc
import chardet
import glob
from log import Log

log = Log()

#############################################################################################
# 데이터 불러오는 함수들
#############################################################################################

# 전처리된 주택보급률 데이터 불러오는 함수 
    # desc : 1 사분면 : 주택재고 (S)
    # columns : 구분(1), 시점, 가구수, 주택수, 보급률, 가구수(등록센서스), 주택수(등록센서스), 보급률(등록센서스)
        # 구분(1) : 지역명
    # time-period :
        # 가구수, 주택수, 보급률은 2014년까지 밖에 없음
        # 가구수(등록센서스), 주택수(등록센서스), 보급률(등록센서스)는 2010년 부터 2020년까지 있음
    # unit : 천호, 천가구, %, 월단위
def read_house_supply_rate():
    df = pd.read_csv('./data/preprocessed/house_supply_rate2.csv')
    df = df.drop(['Unnamed: 0'], axis=1)
    df.loc[:, '시점'] = pd.to_datetime(df['시점'])
    df = df.set_index(['구분(1)', '시점'])
    return df

# 전처리된 인구 데이터 리턴 하는 함수
    # desc : 1 사분면 : 인구 (p)
    # columns : area, date, 총인구수, 세대수, 세대당 인구, 남자 인구수, 여자 인구수, 남여 비율
    # unit : 명, %, 월단위
    # time-period : 2008-2021
def read_population():
    df = pd.read_csv('./data/preprocessed/population2.csv')
    df = df.drop(['Unnamed: 0'], axis=1)
    df.loc[:, 'date'] = pd.to_datetime(df['date'])
    df = df.set_index(['area', 'date'])
    return df

# 전처리된 주택담보대출금리 불러오는 함수 (신규취급액 기준) 
    # desc : 1 사분면 : 금리(r)
    # columns : rate
    # time-period : 2001.09-2021.11
    # unit : 연이율 %, 월단위
def read_house_loan_rate_new():
    df = pd.read_csv('./data/preprocessed/house_loan_rate_new2.csv')
    df.loc[:, 'date'] = pd.to_datetime(df['date'])
    df = df.set_index(['date'])
    return df

# 전처리된 주택담보대출금리 불러오는 함수 (잔액 기준)
    # desc : 1 사분면 : 금리(r)
    # columns : rate
    # time-period : 2009.09-2021.11
    # unit : 연이율 %, 월단위
def read_house_loan_rate_balance():
    df = pd.read_csv('./data/preprocessed/house_loan_rate_balance2.csv')
    df.loc[:, 'date'] = pd.to_datetime(df['date'])
    df = df.set_index(['date'])
    return df


def main():
    # do something
    # ...
    return

if __name__ == '__main__':
    main()
    print('finished')