"""
网格坐标 ↔ Beijing 1954 GK CM 111E ↔ WGS84 ↔ Web Mercator 转换。

坐标系说明:
  网格 (grid_x, grid_y): 0-563 列, 0-528 行, 每格 1000m
  Beijing 1954 3 Degree GK CM 111E (EPSG:2434): 投影米制坐标
  WGS84 (EPSG:4326): 经纬度
  Web Mercator (EPSG:3857): contextily 使用的坐标系
"""

import os
import sys

# 修复 PROJ 数据库版本冲突：pip 安装的 rasterio 自带的 PROJ DLL
# 需要匹配版本的 proj.db。必须先于 rasterio 导入前设置 PROJ_DATA。
# 查找 rasterio 自带的 proj_data 目录。
def _find_rasterio_proj_data():
    """Locate the proj_data directory bundled with rasterio."""
    for p in sys.path:
        candidate = os.path.join(p, "rasterio", "proj_data")
        if os.path.isdir(candidate):
            return candidate
    return None

_proj_data = _find_rasterio_proj_data()
if _proj_data:
    os.environ.setdefault("PROJ_DATA", _proj_data)

import numpy as np
import pyproj

# ============================================================
# Beijing 1954 GK CM 111E 网格参数 (来自 wgs84tobj1954.py)
# ============================================================
X_MIN = 988144.781509014
X_MAX = 1552144.781509014
Y_MIN = 3417504.294282121
Y_MAX = 3946504.294282121
GRID_SIZE = 1000  # 每格 1000 米

GRID_COLS = int((X_MAX - X_MIN) // GRID_SIZE)  # 564
GRID_ROWS = int((Y_MAX - Y_MIN) // GRID_SIZE)  # 529

# ============================================================
# WKT 定义 (来自 wgs84tobj1954.py)
# ============================================================
WGS84_WKT = (
    'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],'
    'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433],AUTHORITY["EPSG",4326]]'
)

BEIJING1954_WKT = (
    'PROJCS["Beijing_1954_3_Degree_GK_CM_111E",'
    'GEOGCS["GCS_Beijing_1954",DATUM["D_Beijing_1954",SPHEROID["Krasovsky_1940",6378245.0,298.3]],'
    'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],'
    'PROJECTION["Gauss_Kruger"],PARAMETER["False_Easting",500000.0],'
    'PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",111.0],'
    'PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],'
    'UNIT["Meter",1.0],AUTHORITY["EPSG",2434]]'
)

# ============================================================
# pyproj 转换器 (缓存，避免重复创建)
# ============================================================
_wgs84_crs = pyproj.CRS.from_wkt(WGS84_WKT)
_beijing1954_crs = pyproj.CRS.from_wkt(BEIJING1954_WKT)
_web_mercator_crs = pyproj.CRS.from_epsg(3857)

_bj_to_wgs84 = pyproj.Transformer.from_crs(_beijing1954_crs, _wgs84_crs, always_xy=True)
_wgs84_to_bj = pyproj.Transformer.from_crs(_wgs84_crs, _beijing1954_crs, always_xy=True)
_wgs84_to_merc = pyproj.Transformer.from_crs(_wgs84_crs, _web_mercator_crs, always_xy=True)
_merc_to_wgs84 = pyproj.Transformer.from_crs(_web_mercator_crs, _wgs84_crs, always_xy=True)


def grid_to_beijing(grid_x, grid_y):
    """网格坐标 → Beijing 1954 米制坐标 (cell 中心)"""
    bx = X_MIN + (np.asarray(grid_x) + 0.5) * GRID_SIZE
    by = Y_MIN + (np.asarray(grid_y) + 0.5) * GRID_SIZE
    return bx, by


def beijing_to_wgs84(bx, by):
    """Beijing 1954 米制 → WGS84 (lon, lat)"""
    lon, lat = _bj_to_wgs84.transform(np.asarray(bx), np.asarray(by))
    return lon, lat


def grid_to_wgs84(grid_x, grid_y):
    """网格坐标 → WGS84 (lon, lat)"""
    bx, by = grid_to_beijing(grid_x, grid_y)
    return beijing_to_wgs84(bx, by)


def grid_to_mercator(grid_x, grid_y):
    """网格坐标 → Web Mercator (EPSG:3857)"""
    lon, lat = grid_to_wgs84(grid_x, grid_y)
    mx, my = _wgs84_to_merc.transform(np.asarray(lon), np.asarray(lat))
    return mx, my


def mercator_to_grid(mx, my):
    """Web Mercator → 网格坐标"""
    lon, lat = _merc_to_wgs84.transform(np.asarray(mx), np.asarray(my))
    bx, by = _wgs84_to_bj.transform(np.asarray(lon), np.asarray(lat))
    grid_x = (np.asarray(bx) - X_MIN) / GRID_SIZE
    grid_y = (np.asarray(by) - Y_MIN) / GRID_SIZE
    return grid_x, grid_y


def grid_bounds_to_mercator(x_min, x_max, y_min, y_max):
    """网格范围 → Web Mercator 范围"""
    corners_x = np.array([x_min, x_max, x_min, x_max])
    corners_y = np.array([y_min, y_min, y_max, y_max])
    mx, my = grid_to_mercator(corners_x, corners_y)
    return mx.min(), mx.max(), my.min(), my.max()


def full_grid_bounds_mercator():
    """全图网格范围 → Web Mercator 范围"""
    return grid_bounds_to_mercator(0, GRID_COLS, 0, GRID_ROWS)


# ============================================================
# 六边形坐标转换
# ============================================================
def hex_to_mercator(hex_mapdata_raw, q_arr, r_arr, s_arr):
    """
    六边形 cube 坐标 → Web Mercator (EPSG:3857)。
    通过 hex_grid.pkl 的原始 dict 查找 lon/lat。
    """
    q_arr = np.asarray(q_arr).ravel()
    r_arr = np.asarray(r_arr).ravel()
    s_arr = np.asarray(s_arr).ravel()

    lons = np.zeros(len(q_arr))
    lats = np.zeros(len(q_arr))

    for i in range(len(q_arr)):
        key = (int(q_arr[i]), int(r_arr[i]), int(s_arr[i]))
        entry = hex_mapdata_raw.get(key)
        if entry is not None:
            lons[i] = entry['lon']
            lats[i] = entry['lat']

    mx, my = _wgs84_to_merc.transform(lons, lats)
    return mx, my


def hex_bounds_mercator(hex_mapdata_raw, q_min=None, q_max=None, r_min=None, r_max=None):
    """
    六边形区域 → Web Mercator 范围。
    如果未指定范围，使用全部数据的边界。
    """
    if q_min is not None:
        # 用 corner 估算
        corners_q = np.array([q_min, q_max, q_min, q_max])
        corners_r = np.array([r_min, r_min, r_max, r_max])
        corners_s = -corners_q - corners_r
        mx, my = hex_to_mercator(hex_mapdata_raw, corners_q, corners_r, corners_s)
        return mx.min(), mx.max(), my.min(), my.max()

    # 全量数据边界
    lons = np.array([v['lon'] for v in hex_mapdata_raw.values()])
    lats = np.array([v['lat'] for v in hex_mapdata_raw.values()])
    corners_lon = np.array([lons.min(), lons.max(), lons.min(), lons.max()])
    corners_lat = np.array([lats.min(), lats.min(), lats.max(), lats.max()])
    mx, my = _wgs84_to_merc.transform(corners_lon, corners_lat)
    return mx.min(), mx.max(), my.min(), my.max()
