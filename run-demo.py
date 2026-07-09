#Viewshed analysis (binary): visible / not-visible areas around a heritage site
#================================================================================
import os
import sys
import json
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import ListedColormap
from pyproj import Transformer
import shapely
from shapely.geometry import Point, box
from shapely.strtree import STRtree

import pandas as pd
import geopandas as gpd
import pathlib as pb

plt.rcParams["font.family"] = "TH Sarabun New"
plt.rcParams["font.size"] = 16

dir_app = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

UTM_EPSG = "EPSG:32647"       # UTM zone 47N - correct for Bangkok (~lon 100-102E)


class ViewshedAnalysis:
    def __init__(self, heritage_x: float, heritage_y: float, heritage_h: float):
        # Default Value ข้อมูลที่ตั้งแหล่งศิลปกรรม
        to_utm = Transformer.from_crs("EPSG:4326", UTM_EPSG, always_xy=True)
        self.heritage_x, self.heritage_y = to_utm.transform(heritage_x, heritage_y)
        self.heritage_h = heritage_h        # ความสูงแหล่งศิลปกรรม 
        # Default Value ข้อมูลการมองเห็น
        self.obs_eye_height = 1.7            # ความสูงระดับการมองเห็นmeter
        self.ground_z = 0.0
        self.radius_m = 250.0               # รัศมีการมองเห็น observation distance
        self.cell_size = 10.0               # ขนาดกริดที่ได้จากการวิเคราะห์
        self.list_bld = []
        self.list_buff_bld = []
        self.BUILDING_SEARCH_MARGIN = 0.0
        self.grid_x = None
        self.grid_y = None
        self.visible_rect = None
        self.hx = None
        self.hy = None
        self.hex_r = None
        self.visible_hex = None

    def load_building(self, dir_osm_bld):
        # column: geom(32647), height
        # ใช้ bkk_footprints_utm_fixed.geojson ที่ reproject เป็น UTM 32647 และ
        # ซ่อม geometry invalid ไว้ล่วงหน้าแล้ว (สร้างด้วย prepare_utm_fixed_geojson.py)
        path_object = pb.Path(dir_osm_bld)
        with open(path_object, encoding="utf-8") as f:
            text = f.read()
        geoms = shapely.get_parts(shapely.from_geojson(text))   # one geometry per feature
        heights = np.array([feat["properties"]["AGL"] for feat in json.loads(text)["features"]], dtype=float)
        self.list_bld = [{"geom": g, "height": float(h)} for g, h in zip(geoms, heights)]
        return self.list_bld

    def build_rect_grid(self):
        n = int(np.ceil(self.radius_m / self.cell_size))
        offsets = np.arange(-n, n + 1) * self.cell_size
        grid_x, grid_y = np.meshgrid(self.heritage_x + offsets, self.heritage_y + offsets)
        dist_from_site = np.sqrt((grid_x - self.heritage_x) ** 2 + (grid_y - self.heritage_y) ** 2)
        return grid_x, grid_y, dist_from_site
    
    def build_hex_grid(self):
        hex_r = self.cell_size / np.sqrt(3)      # circumradius
        dx = 1.5 * hex_r                    # column spacing
        dy = self.cell_size                      # row spacing (= flat-to-flat width)

        ncols = int(np.ceil(self.radius_m / dx)) + 1
        nrows = int(np.ceil(self.radius_m / dy)) + 1
        cols, rows = np.meshgrid(np.arange(-ncols, ncols + 1), np.arange(-nrows, nrows + 1), indexing="ij")

        hx = self.heritage_x + cols * dx
        hy = self.heritage_y + rows * dy + np.where(cols % 2 != 0, dy / 2, 0.0)   # odd columns offset

        hx, hy = hx.ravel(), hy.ravel()
        dist = np.hypot(hx - self.heritage_x, hy - self.heritage_y)
        keep = dist <= self.radius_m
        return hx[keep], hy[keep], dist[keep], hex_r

    def compute_viewshed_rect(self):
        grid_x, grid_y, dist = self.build_rect_grid()
        in_radius = dist <= self.radius_m
        visible = np.full(grid_x.shape, -1, dtype=np.int8)   # -1 = outside AOI
        visible[in_radius] = self.compute_visibility(grid_x[in_radius], grid_y[in_radius], dist[in_radius])
        return grid_x, grid_y, visible

    def compute_viewshed_hex(self):
        hx, hy, hd, hex_r = self.build_hex_grid()
        visible = self.compute_visibility(hx, hy, hd)
        return hx, hy, hex_r, visible

    def compute_visibility(self, ox, oy, od):
        geoms = np.array([b["geom"] for b in self.list_buff_bld], dtype=object)
        heights = np.array([b["height"] for b in self.list_buff_bld])
        tree = STRtree(geoms)
        t0 = time.time()
        # One sightline per point: observer -> site
        coords = np.empty((ox.size, 2, 2))
        coords[:, 0, 0], coords[:, 0, 1] = ox, oy
        coords[:, 1, 0], coords[:, 1, 1] = self.heritage_x, self.heritage_y
        lines = shapely.linestrings(coords)
        obs_pts = shapely.points(ox, oy)

        line_idx, geom_idx = tree.query(lines, predicate="intersects")

        visible = np.ones(ox.size, dtype=np.int8)

        end_obs, end_site = self.ground_z + self.obs_eye_height, self.heritage_h
        h_lo, h_hi = min(end_obs, end_site), max(end_obs, end_site)

        pair_h = heights[geom_idx]
        visible[line_idx[pair_h > h_hi]] = 0
        check = (pair_h > h_lo) & (pair_h <= h_hi)
        li, gi = line_idx[check], geom_idx[check]

        inter = shapely.intersection(lines[li], geoms[gi])
        d_enter = shapely.distance(obs_pts[li], inter)
        d_exit = d_enter + shapely.length(inter)    # exact for a single crossing segment

        od_safe = np.where(od > 1e-6, od, 1.0)      # avoid 0/0 at the site's own cell
        t_enter = d_enter / od_safe[li]
        t_exit = d_exit / od_safe[li]
        h_enter = end_obs * (1 - t_enter) + end_site * t_enter
        h_exit = end_obs * (1 - t_exit) + end_site * t_exit
        los_height_min = np.minimum(h_enter, h_exit)

        visible[li[heights[gi] > los_height_min]] = 0
        visible[od < 1e-6] = 1     # the site's own cell is always visible
        logger.info(f"Computed {ox.size} cells in {time.time() - t0:.1f}s")

        return visible
    

    def export_geojson(self, output_path):
        to_wgs = Transformer.from_crs(UTM_EPSG, "EPSG:4326", always_xy=True)
        outputDict = {
            "type": "FeatureCollection",
            "name": "Viewshed_Grid_Result",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::4326"}},
            "features": []
        }
        featureList = []

        # มุมของแต่ละ cell คำนวณใน UTM เมตร แล้วค่อยแปลงเป็น WGS84
        # เรียงทวนเข็มนาฬิกาตาม right-hand rule ของ RFC 7946
        if self.visible_rect is not None:
            # Rectangle grid: สี่เหลี่ยม 4 มุม (LL -> LR -> UR -> UL)
            mask = self.visible_rect != -1   # ตัด cell นอกรัศมีศึกษาออก
            grid_ids = np.arange(self.visible_rect.size).reshape(self.visible_rect.shape)[mask]
            vis_vals = self.visible_rect[mask]
            half = self.cell_size / 2
            cx, cy = self.grid_x[mask], self.grid_y[mask]
            corner_x = np.stack([cx - half, cx + half, cx + half, cx - half], axis=1)   # (N, 4)
            corner_y = np.stack([cy - half, cy - half, cy + half, cy + half], axis=1)
        elif self.visible_hex is not None:
            # Hexagonal grid: หกเหลี่ยม flat-top 6 มุม (vertex ที่มุม 0, 60, ..., 300 องศา
            # รอบ center = ทวนเข็มนาฬิกา และ orientation ตรงกับ build_hex_grid)
            grid_ids = np.arange(self.visible_hex.size)   # hex กรองในรัศมีไว้แล้ว ไม่มีค่า -1
            vis_vals = self.visible_hex
            cx, cy = self.hx, self.hy
            angles = np.deg2rad(np.arange(0, 360, 60))
            corner_x = cx[:, None] + self.hex_r * np.cos(angles)[None, :]   # (N, 6)
            corner_y = cy[:, None] + self.hex_r * np.sin(angles)[None, :]
        else:
            logger.error("export_geojson: no viewshed result to export — run main() first")
            return

        # weight สำหรับ color ramp ระดับการมองเห็น: 0 = มองไม่เห็น,
        # 1-5 = มองเห็น โดยแบ่งรัศมีศึกษาเป็น 5 ช่วงเท่า ๆ กันตามระยะจากแหล่งศิลปกรรม
        # (5 = ช่วงใกล้สุด เห็นชัดที่สุด, 1 = ช่วงไกลสุดแต่ยังมองเห็น)
        dist = np.hypot(cx - self.heritage_x, cy - self.heritage_y)
        weight_cls = np.clip(5 - (dist / (self.radius_m / 5)).astype(int), 1, 5)
        weights = np.where(vis_vals == 1, weight_cls, 0)

        n_corners = corner_x.shape[1]
        lons, lats = to_wgs.transform(corner_x.ravel(), corner_y.ravel())   # แปลงทุกมุมในคำสั่งเดียว
        lons, lats = lons.reshape(-1, n_corners), lats.reshape(-1, n_corners)

        for gid, v, w, lon4, lat4 in zip(grid_ids, vis_vals, weights, lons, lats):
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": []
                },
                "properties": {}
            }

            propertiesDict = {}
            propertiesDict['grid_id'] = int(gid)
            propertiesDict['visible'] = int(v)
            propertiesDict['weight'] = int(w)

            feature['properties'] = propertiesDict
            ring = [[float('%.6f' % x), float('%.6f' % y)] for x, y in zip(lon4, lat4)]
            ring.append(ring[0])   # ปิด ring: จุดสุดท้ายต้องซ้ำจุดแรก
            feature['geometry']['coordinates'] = [ring]
            featureList.append(feature)

        outputDict['features'] = featureList
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(outputDict, f, ensure_ascii=False)

        # Summary Report
        n_vis = sum(1 for feat in featureList if feat['properties']['visible'] == 1)
        n_tot = len(featureList)
        pct = f"{100 * n_vis / n_tot:.1f}%" if n_tot else "n/a"
        logger.info(f"Saved Complete. ({n_vis}/{n_tot} cells visible, {pct})")

    def main(self, GRID_TYPE):
        site_pt = Point(self.heritage_x, self.heritage_y)
        self.list_buff_bld = [b for b in self.list_bld if b["geom"].distance(site_pt) <= self.radius_m + self.BUILDING_SEARCH_MARGIN]
        logger.info(f"Loaded {len(self.list_bld)} buildings, {len(self.list_buff_bld)} within analysis range")

        if GRID_TYPE == "Rectangle":
            self.grid_x, self.grid_y, self.visible_rect = self.compute_viewshed_rect()
        elif GRID_TYPE == "Hexagonal":
            self.hx, self.hy, self.hex_r, self.visible_hex = self.compute_viewshed_hex()
        else:
            raise ValueError(f"unknown GRID_TYPE: {GRID_TYPE}")
        # plot show result
        #self.plot_result_rect()







if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    # ข้อมูลแหล่งศิลปกรรม
    HERITAGE_SITE = (100.5175699, 13.7185468)
    HERITAGE_HEIGHT = 40.0
    # ข้อมูลขอบเขตอาคาร (reproject เป็น UTM 32647 และซ่อม geometry invalid ไว้ล่วงหน้าแล้ว)
    file_osm_bld = "bkk_footprints_utm_fixed.geojson"
    dir_osm_bld = os.path.join(dir_app, file_osm_bld)
    # รูปแบบการแสดงผล
    grids_type = "Hexagonal" # Rectangle or Hexagonal

    analysis = ViewshedAnalysis(HERITAGE_SITE[0], HERITAGE_SITE[1], HERITAGE_HEIGHT)
    analysis.load_building(dir_osm_bld)
    analysis.main(grids_type)


    # ผลการวิเคราะห์
    file_viewshed = "viewshed_grid_result.geojson"
    dir_output_viewshed = os.path.join(dir_app, file_viewshed)
    analysis.export_geojson(dir_output_viewshed)

    # ระดับความใกล้ที่ยังมองเห็น