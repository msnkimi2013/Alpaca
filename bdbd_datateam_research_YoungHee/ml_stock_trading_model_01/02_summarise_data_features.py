# -*- coding: utf-8 -*-

import sys
sys.path.append('/Users/yoingheechoi/workspace/boodong/bdbd_datateam_project01')
for _p in sys.path:
    print(_p)

import traceback
import os
import glob

print('*** current_working_dir:{}'.format(os.getcwd()))

import pandas as pd
import datetime
import multiprocessing as mulproc

from apis import feature_api as fa
from apis import data_preprocessing_api as dpa
import utils.functions as F
from utils.log import Log
from configs import script_config

log = Log()

def load_data_features(file_path):
    try:
        df = pd.read_csv(file_path)
        for _col in df.columns:
            if 'Unnamed:' in _col:
                df = df.drop(_col, axis=1)
        return df
    except FileNotFoundError as e:
        log.error('file not found. file_path:{}'.format(file_path))
        raise e
    except Exception as e:
        log.error('exception occured. file_path:{} e:{}'.format(file_path, e))
        traceback.print_exc(file=sys.stdout)
        raise e



def summarise_features():

    path = './data/features'

    file_list = glob.glob(f'{path}/*_added.csv')



    for _file in file_list:
        df = load_data_features(_file)

        print('\n\n---------- start summary ---------- file_path:{}'.format(_file))
        print('df.shape:{}'.format(df.shape))

        for _col in df.columns:
            unique_values = df[_col].unique()

            print('\n')
            print('col:{}'.format(_col)) 
            print('unique_values:')
            print('{}'.format(unique_values))
            print('describe:')
            print('{}'.format(df[_col].describe()))
            print('\n')
    
    return

if __name__ == '__main__':
    #main()
    summarise_features()
    print('finished')
