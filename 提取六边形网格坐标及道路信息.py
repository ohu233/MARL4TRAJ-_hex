# -*- coding: utf-8 -*-
import geopandas as gpd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

path = "江苏安徽浙江上海——三省一市地理数据\基础数据"

# 六边形网格文件
hex_shp = f"{path}\\三省一市区划图_six_200.shp"

# 道路名称:县道、普铁、省道、高速收费站、高速、国道、高铁、火车站
roads = ["县道", "2024普铁", "省道", "2024高速收费站", "2024高速", "国道", "2024高铁", "2024火车站"]
road_shp_list = [f"{path}\\{road}.shp" for road in roads]
road_col_suffix = {
    "县道": "xd",
    "2024普铁": "pt",
    "省道": "sd",
    "2024高速收费站": "gs_sfz",
    "2024高速": "gs",
    "国道": "gd",
    "2024高铁": "gt",
    "2024火车站": "hcz",
}

# 输出
output_shp = f"{path}\\xyz_all_road_flag.shp"
output_csv = f"{path}\\xyz_all_road_flag.csv"
side_len = 200
# =======================================================

# 读取六边形 + 中心点
print("读取六边形网格...")
hex = gpd.read_file(hex_shp)
hex["cx"] = hex.geometry.centroid.x
hex["cy"] = hex.geometry.centroid.y

# 计算每个六边形中心的经纬度（转换为 WGS84）
hex_centroids = gpd.GeoDataFrame(geometry=gpd.points_from_xy(hex["cx"], hex["cy"]), crs=hex.crs)
hex_centroids = hex_centroids.to_crs("EPSG:4326")
hex["lon"] = hex_centroids.geometry.x
hex["lat"] = hex_centroids.geometry.y

# 研究区的左下角作为 xyz 原点 (0,0,0)
min_x = hex["cx"].min()
min_y = hex["cy"].min()

cx_norm = hex["cx"].values - min_x
cy_norm = hex["cy"].values - min_y

# Flat-top 尖朝左右 立方体坐标
q = (cx_norm * 2.0 / 3.0) / side_len
r = (-cx_norm / 3.0 + np.sqrt(3)/3.0 * cy_norm) / side_len

hex["x"] = np.round(q).astype(int)
hex["z"] = np.round(r).astype(int)
hex["y"] = -hex["x"] - hex["z"]
# =================================================================

# 道路判断
def calc_road_flag(hex_gdf, line_gdf):
    flag = np.zeros(len(hex_gdf), dtype=np.int8)
    sidx = line_gdf.sindex
    for idx, poly in enumerate(hex_gdf.geometry):
        candidates = list(sidx.intersection(poly.bounds))
        if not candidates:
            continue
        for cid in candidates:
            if poly.intersects(line_gdf.geometry.iloc[cid]):
                flag[idx] = 1
                break
    return flag

# 循环处理所有道路
for i, road_name in enumerate(roads):
    print(f"正在处理：{road_name}")
    road_gdf = gpd.read_file(road_shp_list[i]).to_crs(hex.crs)
    hex[f"has_{road_col_suffix[road_name]}"] = calc_road_flag(hex, road_gdf)

# 输出字段
keep_cols = ["x","y","z","lon","lat","geometry"] + [f"has_{road_col_suffix[r]}" for r in roads]
hex_out = hex[keep_cols].copy()

# 保存
hex_out.to_file(output_shp, encoding="utf-8")
hex_out.drop(columns=["geometry"]).to_csv(output_csv, index=False, encoding="utf-8-sig")
