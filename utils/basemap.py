"""
统一底图渲染：OSM 道路瓦片或旧 JPEG 底图。
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import contextily as ctx
import xyzservices

from utils.geo_utils import full_grid_bounds_mercator

# 设为 False 可回退到旧 JPEG 底图
USE_OSM_BASEMAP = True

OSM_PROVIDER = xyzservices.providers.Gaode.Normal


def add_osm_basemap(ax, alpha=1.0, zoom=None):
    """在给定 Axes 上叠加 OSM 道路瓦片底图。Axes 需已在 EPSG:3857 空间。"""
    try:
        ctx.add_basemap(
            ax,
            crs="EPSG:3857",
            source=OSM_PROVIDER,
            zoom=zoom or "auto",
            alpha=alpha,
            reset_extent=False,
        )
    except Exception as e:
        print(f"[basemap] OSM 瓦片下载失败，回退到 JPEG 底图: {e}")
        _add_jpeg_fallback(ax)


def set_ax_extent(ax, xmin, xmax, ymin, ymax):
    """设置 Axes 范围并添加底图。坐标需为 Web Mercator (EPSG:3857)。"""
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if USE_OSM_BASEMAP:
        add_osm_basemap(ax)
    else:
        _add_jpeg_fallback(ax)
    ax.set_aspect("equal")


def set_full_extent(ax):
    """设置全图范围并添加底图。"""
    xmin, xmax, ymin, ymax = full_grid_bounds_mercator()
    set_ax_extent(ax, xmin, xmax, ymin, ymax)


def _add_jpeg_fallback(ax):
    """旧 JPEG 底图回退方案（矩形 grid，仅向后兼容）。"""
    bg_img = mpimg.imread(r"figur\jiangsu\js.jpg")
    ax.imshow(bg_img, extent=[0, 564, 0, 529], aspect="equal", alpha=1)
