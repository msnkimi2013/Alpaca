import requests
import json
import math
import time
import pandas as pd


from haversine import haversine

KAKAO = "KakaoAK b78407deb13ac4862a36927925102116"
TMAP = 'l7xx43134f13f1dc45f6b35a865f66b6e93f'
ODSAY = 'VqvM1e5gy3b74/mnKKG1SMnPUt/0849zjIxGcatpQNA'   # 1000 제한
GyeonggiDataDream ='a84973971b6d4ba2bcb19dc0da2e544a'


################# 카카오 apis ##################

def get_keyword_search(keyword, coordinates, radius):
    """ 검색어로 반경안에 주변 정보 가져올 수 있음 """
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    header = {"Authorization": KAKAO}
    params = {
        "query": f"{keyword}",
        #"category_group_code": "MT1",
        "x": coordinates[0],
        "y": coordinates[1],
        "radius": radius,
    }
    r = requests.get(url, headers=header, params=params)
    t = r.json()
    return t['documents']

def get_group_search(group, coordinates, radius):
    """ group 카테고리 코드로 주변 정보 가져올 수 있음 """
    url = "https://dapi.kakao.com/v2/local/search/category.json"
    header = {"Authorization": KAKAO}
    params = {
        "category_group_code": f"{group}",
        "x": coordinates[0],
        "y": coordinates[1],
        "radius": radius,
    }
    r = requests.get(url, headers=header, params=params)
    t = r.json()
    return t['documents']

def get_coordinates(address, exact=True):
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    header = {"Authorization": KAKAO}
    params = {
        "query": f"{address}",
        "analyze_type": "exact" if exact else "similar",
    }
    r = requests.get(url, headers=header, params=params)
    t = r.json()
    t = t['documents'][0]
    return (float(t['x']), float(t['y']))

################# ODsay apis ##################

def search_public_path(start_coordinates, end_coordinates, search_type):
    """ 
    좌표로 대중교통 경로 검색하여 전체 소요시간, 거리 등 정보와 경로 list 제공 
    API 문서 : https://lab.odsay.com/guide/releaseReference#searchPubTransPath
    """
    url = "https://api.odsay.com/v1/api/searchPubTransPathT"
    SX = start_coordinates[0]
    SY = start_coordinates[1]
    EX = end_coordinates[0]
    EY = end_coordinates[1]
    search_type = search_type if search_type in [0, 1, 2] else 0

    r = requests.get(f'{url}?apiKey={ODSAY}&SX={SX}&SY={SY}&EX={EX}&EY={EY}&SearchPathType={search_type}')
    info = json.loads(r.text)
    if 'error' in info.keys():
        print(info)
        return None, info['error']
    else:
        path = info['result'].pop('path')
        return info['result'], path


################# Tmap apis ##################

def search_car_path(start_coordinates, end_coordinates, departure_time):
    """ 
    Tmap api 타임머신 자동차 길 안내 api 
    출발지, 도착지, 출발시간 요렇게가 기본 3개라고 볼 수 있음. - 도착시간 기준으로도 검색 가능
    """
    url = "https://apis.openapi.sk.com/tmap/routes/prediction"

    header = {
        "appkey": TMAP,
        "version": "1",
        "callback": ""
    }
    payload =  {
        "routesInfo": {
            "departure" : {
    			"name" : "departur_point",
    			"lon" : start_coordinates[0],
    			"lat" : start_coordinates[1]
    			},
    		"destination" : {
    			"name" : "arrival_point",
    			"lon" : end_coordinates[0],
    			"lat" : end_coordinates[1]
    		    },
    		"predictionType" : "departure",
    		"predictionTime" : departure_time,
        }
    }

    r = requests.post(url, json=payload, headers=header)
    info = json.loads(r.text)
    maininfo = info['features'][0]['properties']
    needed_keys = ['totalDistance', 'totalTime', 'totalFare', 'taxiFare', 'departureTime', 'arrivalTime']
    return {key: maininfo[key] for key in needed_keys}


############ 지하철 table 관련 api ###########

def get_Gwanju_subway_time():
    """ 광주 지하철 전체 역에서 다른 역으로 이동시간하고 실제 거리 table 가져오는 api """
    time_table = []
    for i in range(20):
        r = requests.get(f"http://www.grtc.co.kr/subway/openapi/StationTimeInformation.do?rbsIdx=317&doctype=json&station_id={i+1}")
        times = r.json()

        for t in times:
            time_table.append([t['start_station_id'], t['end_station_id'], t['start_station_name'], 
                               t['end_station_name'], t['station_time'], t['station_distance']])
        time.sleep(0.3)

    df = pd.DataFrame(time_table, columns=['start_station_id', 'end_station_id', 'start_station_name',
                                           'end_station_name', 'station_time', 'station_distance'])
    return df

############ 거리 관련 함수 ##############

def get_dist_by_address(adr1, adr2, unit='km') -> float:
    """ 두 주소간 좌표로 haver dist 구해줌 """
    x1, y1 = get_coordinates(adr1)
    x2, y2 = get_coordinates(adr2)
    return haversine((y1, x1), (y2, x2), unit)

def check_if_in_radius(apt_coords, point, radius, unit='km') -> bool:
    """ 반경안에 해당 좌표값 들어가는지 check """
    if haversine(apt_coords, point, unit) < radius:
        return True
    else:
        return False

def check_if_intersect(apt_coords, *point_rad_set, unit='km') -> bool:
    """ 주어진 좌표에 반경 넣었을 때 overlab영역 안에 있는지 check """
    bool_list = []
    for point, radius in point_rad_set:
        bool_list.append(check_if_in_radius(apt_coords, point, radius, unit))

    if all(bool_list):
        return True
    else:
        return False
