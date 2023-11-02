from os import pread
import pandas as pd
import numpy as np
import time
import pickle
import argparse

from itertools import combinations
from tqdm import tqdm

from apis import search_public_path, get_Gwanju_subway_time

# TODO jupyter notebook 에 여기저기 나눠져 있는 accessibility 계산하는데 필요한  table들 계산하는거 정리해두기

def create_Gwnaju_subway_table(base_table, target_path):
    # 해당 지역 지하철 역 숫자만큼 빈 테이블 만들기
    table_length = len(base_table[base_table['지역명'] == '광주'])
    stations = base_table[base_table['지역명'] == '광주']['지하철코드']
    target_sf = pd.DataFrame(np.zeros((table_length, table_length), dtype=int), columns=stations, index=stations)
    # 광주 도시철도 공사 api 가지고 테이블 작성함.
    sf = get_Gwanju_subway_time()
    for _, row in sf.iterrows():
        target_sf.iloc[row.start_station_id-1, row.end_station_id-1] = row.station_time
    target_sf.to_pickle(target_path)


def station_to_station_tablemaker(base_table, region, call_limit, last_txt=None, last_file=None):
    """ 지하철 길찾기 api 이용해서 전체 역간 이동시간 테이블 만드는 함수 - 돈들어가 서 현재 문제임. """

    # 해당 지역 지하철 역 숫자만큼 빈 테이블 만들기
    table_length = len(base_table[base_table['지역명'] == region])
    stations = base_table[base_table['지역명'] == region]['지하철코드']
    if last_txt is None:
        sf = pd.DataFrame(np.zeros((table_length, table_length), dtype=int), columns=stations, index=stations)
    else:
        sf = pd.read_csv(last_file, index_col=0)

    # 0.5초 간격으로 api call 던짐
    error_set = []
    flag = 0
    all_comb = list(combinations(stations.tolist(), 2))
    if last_txt is not None:
        with open(last_txt, 'r') as f:
            last_start, last_end = f.readline().split(',')
            last_idx = all_comb.index((last_start, last_end))

    print("This call start from last end of index-{}, {}, {}".format(last_idx, last_start, last_end))
    print("This region has total {} station. It will start call API for all {} connection...".format(table_length, len(all_comb)))
    for i, (start, end) in enumerate(tqdm(all_comb, desc="Station to station api calls on progress", total=len(all_comb))):
        start_coord = (base_table.loc[base_table['지하철코드'] == start, 'X좌표'].iloc[0], base_table.loc[base_table['지하철코드'] == start, 'Y좌표'].iloc[0])
        end_coord = (base_table.loc[base_table['지하철코드'] == end, 'X좌표'].iloc[0], base_table.loc[base_table['지하철코드'] == end, 'Y좌표'].iloc[0])
        if i < last_idx: continue
        #print(start, end, start_coord, end_coord)
        response = search_public_path(start_coord, end_coord, search_type=1)
        if response[0] is None:
            error_set.append([start, end, response[1]])
            if response[1]['code'] == '-98':
                sf.loc[start, end] = 3
                sf.loc[start, start] = 0
                sf.loc[end, start] = 3
            continue
        total_time = response[1][0]['info']['totalTime']
        sf.loc[start, end] = total_time
        sf.loc[start, start] = 0
        sf.loc[end, start] = total_time
        flag += 1
        if flag >= call_limit:
            last = f"{start},{end}"
            break
        time.sleep(0.5)

    sf.to_csv('station_to_station_Busan_v1.0.csv')

    with open('error_case_Busan.pkl', 'wb') as f:
        pickle.dump(error_set, f)
    with open('numbers_on_count.txt', 'w') as f:
        f.write(last)
    print(error_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SubToSub Table Maker')
    parser.add_argument('--api_limit', '-l', type=int, 
                    help='How much api you want to you')
    args = parser.parse_args()

    base_table = pd.read_csv('~/bdbd_datateam_research/data/kms_tables/지하철코드정보.csv', sep="|")

    """
    station_to_station_tablemaker(base_table, '부산', args.api_limit, 
                                  last_txt="/Users/toyaji/bdbd_datateam_research/numbers_on_count.txt", 
                                  last_file="/Users/toyaji/bdbd_datateam_research/station_to_station_Busan_v1.0.csv")

    """
    create_Gwnaju_subway_table(base_table, 'station_to_station_Gwangju_v1.2.pkl')


