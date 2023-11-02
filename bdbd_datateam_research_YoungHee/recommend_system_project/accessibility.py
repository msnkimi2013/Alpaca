import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error
from pathlib import Path
from haversine import haversine_vector
from itertools import combinations
from easydict import EasyDict
from apis import search_public_path

# 테이블 관련 경로
AtoS = "/Users/toyaji/bdbd_datateam_research/data/apt_to_subway_v1.0.pkl" # 아파트 to 지하철역 거리 테이블
StoS = "/Users/toyaji/bdbd_datateam_research/data/station_to_station_Gwangju_v1.2.pkl" # 지하철역 to 지하철역 이동시간 테이블
Aattr = "/Users/toyaji/bdbd_datateam_research/data/recommend_table_v1.0.pkl" # 아파트 속성 정리 테이블
Sinfo = "/Users/toyaji/bdbd_datateam_research/data/subway_info_v1.0.pkl" # 지하철 역 정보 테이블

class AptRecommenderByAccess(object):
    def __init__(self, apt_attr_table_path, apt_to_subway_table_path, subway_to_subway_path, subway_info_path,
                radius_for_points_to_subway=500, radius_for_apts_to_subway=600, k_for_clusters=2, weights_for_scoring=None):
        super().__init__()

        self.load_tables(apt_attr_table_path, apt_to_subway_table_path, subway_to_subway_path, subway_info_path)
        # 지하철 코드 정보 미리 list로 가지고 있기
        self.sub_coords = list(zip(self.sub_info.X, self.sub_info.Y))

        # set parameters for calc
        self._radius_for_points_to_subway = radius_for_points_to_subway
        self._radius_for_apts_to_subway = radius_for_apts_to_subway
        self._k_for_clusters = k_for_clusters
        self._weights_for_scoring = weights_for_scoring

    def load_tables(self, apt_attr_table_path, apt_to_subway_table_path, subway_to_subway_path, subway_info_path, format='pickle'):
        # pathlib 써야 운영체제 따라 경로 문제 막을 수 있어
        apt_attr_table_path = Path(apt_attr_table_path)
        apt_to_subway_table_path = Path(apt_to_subway_table_path)
        subway_to_subway_path = Path(subway_to_subway_path)
        subway_info_path = Path(subway_info_path)

        assert apt_attr_table_path.exists()
        assert apt_to_subway_table_path.exists()
        assert subway_to_subway_path.exists()
        assert subway_info_path.exists()

        if format == 'pickle':
            self.apt_attrs = pd.read_pickle(apt_attr_table_path)
            self.apt_to_sub = pd.read_pickle(apt_to_subway_table_path)
            self.sub_to_sub = pd.read_pickle(subway_to_subway_path)
            self.sub_info = pd.read_pickle(subway_info_path)
        elif format == 'csv':
            self.apt_attrs = pd.read_csv(apt_attr_table_path)
            self.apt_to_sub = pd.read_csv(apt_to_subway_table_path)
            self.sub_to_sub = pd.read_csv(subway_to_subway_path)
            self.sub_info = pd.read_csv(subway_info_path)
        else:
            raise FileNotFoundError("테이블 파일 확장자 확인 필요!")

        # TODO index column 확인하는 작업 필요함.


    def __call__(self, *points) -> dict:
        self.receive_points(*points, radius=self._radius_for_points_to_subway)
        self.calc_weighted_avg_time_for_all()
        self.get_candidates_of_k_best_cluster(self._k_for_clusters, self._radius_for_apts_to_subway)
        return self.candidates_apt

    def receive_points(self, *points, radius=500):
        # 출퇴근지점 등 나타내는 포인트 여러개 받아서 반경(m 단위) 안에 지하철 역 찾아서 dictionary로 mapping 반환함.
        self.points = list(points)
        self.points_to_sub_map = {}
        # 주어진 입력 포인트들에서 전체 지하철역하고 거리 계산해서 주어진 반경 안에 있는 지하철 역만 골라내서 지하철 정보 매핑하기
        point_to_sub_dist = haversine_vector(self.points, self.sub_coords, comb=True, unit='m')
        idxs, cols = np.where(point_to_sub_dist <= radius)
        for idx, col in zip(idxs, cols):
            self.points_to_sub_map[col] = EasyDict(self.sub_info.iloc[idx])

    def calc_weighted_avg_time_for_all(self, weights=None):
        # 주어진 point 연결 지하철 정보들 가지고 전체 지하철역들 이동시간 뽑아서 가중치로 더해
        sub_codes_idxs = [info.code for _, info in self.points_to_sub_map.items()]
        # 가중치 따로 안 주어지면 그냥 1/n 으로 mean해서 역 서치함.
        if weights is None:
            no_points = len(sub_codes_idxs)
            weights = np.ones(no_points) / no_points
        # exponental average 로 점수줘서 어느쪽에 가중치 주느냐 따라서 점수가 그쪽 가까울수록 잘 나오게 만듦.
        time_for_subs = self.sub_to_sub[sub_codes_idxs]
        self.score_for_clusters = time_for_subs.apply(lambda x: mean_squared_error(weights, np.divide(x, np.sum(x))), axis=1)

    def get_candidates_of_k_best_cluster(self, k=1, radius=600):
        # score 기준으로 상위 k개 cluster 선택하고 주변 아파트 정보 가져와 
        self.candidates_apt = {}
        for _ in range(k):
            best = self.score_for_clusters.idxmin()
            self.score_for_clusters.pop(best)

            # 뽑힌 cluster 기준역에서 반경(radius) 안에 아파트 검색해서 후보군으로 인덱스 가져와
            candidates_apt_idx = self.apt_to_sub[self.apt_to_sub[best] < radius].index
            candidates_apt_attr = self.apt_attrs[self.apt_attrs.index.isin(candidates_apt_idx)]
            self.candidates_apt[best] = candidates_apt_attr

    @property
    def cluster_subs(self):
        return self.sub_info.set_index('code').loc[list(self.candidates_apt.keys())]

    @property
    def apts_of_cluster(self):
        return self.candidates_apt
    
        


        
    
if __name__ == '__main__':
    # 테스트 용으로 광주 아파트 이용
    pointA = (126.93222940515193, 35.13275605066436)
    pointB = (126.80040831732504, 35.1426913510177)

    recommender = AptRecommenderByAccess(Aattr, AtoS, StoS, Sinfo)
    recommender(pointA, pointB)
