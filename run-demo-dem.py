#Viewshed analysis (binary) + DEM: visible / not-visible areas around a heritage site
#================================================================================
# เหมือน run-demo.py แต่วิเคราะห์ร่วมกับ DEM (พื้นดินต่างระดับจริง):
#   - ตา observer   = ground(DEM ที่ cell) + obs_eye_height      (ต่างกันทุก cell)
#   - ยอด heritage  = ground(DEM ที่ site) + heritage_h
#   - ยอดตึก        = ground(DEM ที่ตึก, sample ตอนโหลด) + AGL
# ทุกความสูงเทียบกันบนฐาน absolute elevation (m MSL) แทน flat-terrain assumption
#================================================================================
import os
import sys
import json
import time
import math
import logging
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import ListedColormap
from pyproj import Transformer
import shapely
from shapely.geometry import Point, box
from shapely.strtree import STRtree

# เครื่องนี้มี PROJ_LIB ของระบบ (PostGIS/OSGeo4W) ที่เวอร์ชันไม่ตรงกับ PROJ ที่
# rasterio bundle มาเอง ทำให้เปิดไฟล์ raster ที่มี CRS แล้ว error "Cannot find
# proj.db" — ต้องชี้ PROJ_LIB/PROJ_DATA ไปที่ของ rasterio ก่อน import rasterio
_rio_spec = importlib.util.find_spec("rasterio")
if _rio_spec and _rio_spec.origin:
    _rio_proj_data = os.path.join(os.path.dirname(_rio_spec.origin), "proj_data")
    if os.path.isdir(_rio_proj_data):
        os.environ["PROJ_LIB"] = _rio_proj_data
        os.environ["PROJ_DATA"] = _rio_proj_data

import rasterio

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
        self.heritage_h = heritage_h        # ความสูงแหล่งศิลปกรรม (เหนือพื้นดินที่ site)
        # Default Value ข้อมูลการมองเห็น
        self.obs_eye_height = 1.7            # ความสูงระดับการมองเห็น meter (เหนือพื้นดิน)
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
        self.hex_ids = None
        self.visible_hex = None
        # ข้อมูลระดับพื้นดินจาก DEM (ตั้งค่าโดย load_dem) — แทน flat-terrain assumption
        self.dem_data = None
        self.dem_inv_transform = None
        self.heritage_ground = 0.0          # ground elevation (m MSL) ที่ตำแหน่งแหล่งศิลปกรรม

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

    def load_dem(self, dir_dem):
        """โหลด DEM (GeoTIFF, ต้องเป็น EPSG:32647 หน่วยเมตร) เฉพาะ window ที่คลุม
        พื้นที่ศึกษา + ตึกทั้งหมด (ไม่อ่านทั้งไฟล์ จึงรองรับ DEM ระดับประเทศขนาดใหญ่ได้)
        แล้ว: 1) เก็บ ground elevation ที่ตำแหน่งแหล่งศิลปกรรม
              2) sample msl ให้ตึกทุกหลังที่ representative point (จุดในตึกเสมอ)
        ต้องเรียกหลัง load_building()"""
        assert self.list_bld, "load_dem ต้องเรียกหลัง load_building()"

        # bbox ที่ต้องใช้ = ตึกทั้งหมด + วงรัศมีศึกษา (เผื่อขอบ 2 พิกเซลด้วย floor/ceil)
        bld_xy = shapely.get_coordinates(np.array([b["geom"] for b in self.list_bld], dtype=object))
        r = self.radius_m + self.BUILDING_SEARCH_MARGIN
        minx = min(bld_xy[:, 0].min(), self.heritage_x - r)
        maxx = max(bld_xy[:, 0].max(), self.heritage_x + r)
        miny = min(bld_xy[:, 1].min(), self.heritage_y - r)
        maxy = max(bld_xy[:, 1].max(), self.heritage_y + r)

        with rasterio.open(pb.Path(dir_dem)) as src:
            assert src.crs is not None and str(src.crs).upper() == UTM_EPSG.upper(), \
                f"DEM ต้องเป็น {UTM_EPSG} (ได้ {src.crs}) — reproject ก่อนใช้"
            inv = ~src.transform
            col_min, row_min = inv * (minx, maxy)
            col_max, row_max = inv * (maxx, miny)
            col_off, row_off = math.floor(col_min), math.floor(row_min)
            window = rasterio.windows.Window(
                col_off, row_off,
                math.ceil(col_max) - col_off, math.ceil(row_max) - row_off)
            nodata = src.nodata if src.nodata is not None else -9999.0
            self.dem_data = src.read(1, window=window, boundless=True, fill_value=nodata)
            self.dem_inv_transform = ~src.window_transform(window)

        self.heritage_ground = float(
            self.ground_at(np.array([self.heritage_x]), np.array([self.heritage_y]))[0])

        # spatial join: ground elevation ของตึกแต่ละหลัง (จุดเดียวต่อตึกเพียงพอ
        # เพราะ footprint เล็กกว่าพิกเซล DEM มาก) — ยอดตึก absolute = msl + AGL
        pts = shapely.get_coordinates(
            shapely.point_on_surface(np.array([b["geom"] for b in self.list_bld], dtype=object)))
        msl = self.ground_at(pts[:, 0], pts[:, 1])
        for b, z in zip(self.list_bld, msl):
            b["msl"] = float(z)

        logger.info(f"DEM loaded {self.dem_data.shape}, heritage ground = {self.heritage_ground:.1f} m MSL, "
                    f"building msl range {msl.min():.1f}..{msl.max():.1f} m")

    def ground_at(self, x, y):
        """Ground elevation (m MSL) ที่พิกัด UTM (x, y) — bilinear interpolation
        ระหว่าง 4 พิกเซลรอบจุด (ค่าพิกเซลถือเป็นความสูงที่ center ของพิกเซล)
        ให้ค่าพื้นไล่ระดับเนียนแม้ cell_size เล็กกว่าพิกเซล DEM
        พิกัดนอกขอบ DEM ถูก clamp เข้าขอบ"""
        cols, rows = self.dem_inv_transform * (x, y)
        h, w = self.dem_data.shape
        # -0.5: จาก edge-based pixel coords เป็นตำแหน่งเทียบ center ของพิกเซล
        fc = np.clip(np.asarray(cols) - 0.5, 0, w - 1)
        fr = np.clip(np.asarray(rows) - 0.5, 0, h - 1)
        c0 = np.minimum(fc.astype(int), w - 2)
        r0 = np.minimum(fr.astype(int), h - 2)
        tc, tr = fc - c0, fr - r0

        d = self.dem_data
        return (d[r0, c0] * (1 - tr) * (1 - tc) + d[r0, c0 + 1] * (1 - tr) * tc
                + d[r0 + 1, c0] * tr * (1 - tc) + d[r0 + 1, c0 + 1] * tr * tc).astype(float)

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

    def observer_open_mask(self, ox, oy):
        """True = จุด observer อยู่นอกตัวอาคาร (คนยืนได้จริง)
        cell ที่ center ตกในตัวตึกไม่ใช่จุดยืนมองจริง — ตัดออกจากการวิเคราะห์
        (ต้องเรียกหลังตั้ง list_buff_bld แล้ว)"""
        geoms = np.array([b["geom"] for b in self.list_buff_bld], dtype=object)
        tree = STRtree(geoms)
        pi, _ = tree.query(shapely.points(ox, oy), predicate="within")
        open_mask = np.ones(ox.size, dtype=bool)
        open_mask[np.unique(pi)] = False
        logger.info(f"Excluded {int((~open_mask).sum())} / {ox.size} cells inside building "
                    f"footprints ({open_mask.sum()} observer cells remain)")
        return open_mask

    def compute_viewshed_rect(self):
        grid_x, grid_y, dist = self.build_rect_grid()
        in_radius = dist <= self.radius_m
        # -1 = นอกรัศมีศึกษา หรือ center อยู่ในตัวตึก (ไม่วิเคราะห์/ไม่ export)
        visible = np.full(grid_x.shape, -1, dtype=np.int8)
        ox, oy, od = grid_x[in_radius], grid_y[in_radius], dist[in_radius]
        open_mask = self.observer_open_mask(ox, oy)
        vis_sub = np.full(ox.size, -1, dtype=np.int8)
        vis_sub[open_mask] = self.compute_visibility(ox[open_mask], oy[open_mask], od[open_mask])
        visible[in_radius] = vis_sub
        return grid_x, grid_y, visible

    def compute_viewshed_hex(self):
        hx, hy, hd, hex_r = self.build_hex_grid()
        # เก็บ index เดิมไว้เป็น grid_id ก่อนตัด cell ในตัวตึกออก
        # เพื่อให้ grid_id ของ cell ที่เหลือคงที่ เทียบกับผลเวอร์ชันก่อน ๆ ได้
        ids = np.arange(hx.size)
        open_mask = self.observer_open_mask(hx, hy)
        hx, hy, hd = hx[open_mask], hy[open_mask], hd[open_mask]
        self.hex_ids = ids[open_mask]
        visible = self.compute_visibility(hx, hy, hd)
        return hx, hy, hex_r, visible

    def compute_visibility(self, ox, oy, od):
        """LOS test ต่อจุด observer โดยทุกความสูงเป็น absolute elevation (m MSL)
        จาก DEM — รองรับพื้นต่างระดับจริง (ต้องเรียก load_dem ก่อน)"""
        assert self.dem_data is not None, "compute_visibility ต้องเรียกหลัง load_dem()"

        geoms = np.array([b["geom"] for b in self.list_buff_bld], dtype=object)
        bld_top = np.array([b["msl"] + b["height"] for b in self.list_buff_bld])
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

        # ปลายเส้นสายตาสองข้าง: ฝั่ง observer ต่างกันทุก cell ตามพื้นดินใต้เท้า
        end_obs = self.ground_at(ox, oy) + self.obs_eye_height   # array ต่อ cell
        end_site = self.heritage_ground + self.heritage_h        # scalar

        # ความสูงเส้นสายตาแต่ละคู่ (เส้น, ตึก) แกว่งอยู่ระหว่างปลายสองข้างเสมอ:
        # ตึกที่ยอดสูงกว่า max(ปลาย) บังแน่ถ้าตัดเส้น / ต่ำกว่า min(ปลาย) ไม่มีวันบัง
        pair_obs = end_obs[line_idx]
        pair_top = bld_top[geom_idx]
        h_hi = np.maximum(pair_obs, end_site)
        h_lo = np.minimum(pair_obs, end_site)

        visible[line_idx[pair_top > h_hi]] = 0
        check = (pair_top > h_lo) & (pair_top <= h_hi)
        li, gi = line_idx[check], geom_idx[check]
        obs_e = pair_obs[check]

        inter = shapely.intersection(lines[li], geoms[gi])
        d_enter = shapely.distance(obs_pts[li], inter)
        d_exit = d_enter + shapely.length(inter)    # exact for a single crossing segment

        od_safe = np.where(od > 1e-6, od, 1.0)      # avoid 0/0 at the site's own cell
        t_enter = d_enter / od_safe[li]
        t_exit = d_exit / od_safe[li]
        h_enter = obs_e * (1 - t_enter) + end_site * t_enter
        h_exit = obs_e * (1 - t_exit) + end_site * t_exit
        los_height_min = np.minimum(h_enter, h_exit)

        visible[li[pair_top[check] > los_height_min]] = 0
        visible[od < 1e-6] = 1     # the site's own cell is always visible
        logger.info(f"Computed {ox.size} cells in {time.time() - t0:.1f}s")

        return visible


    def export_geojson(self, output_path):
        """ส่งออกผล viewshed เป็น GeoJSON polygon grid ตาม RFC 7946:
        - พิกัดเป็น WGS84 lon/lat เสมอ จึงไม่มี member "crs" (spec ปัจจุบันตัดออกแล้ว)
        - exterior ring เรียงทวนเข็มนาฬิกา (right-hand rule) และปิด ring
        - grid_id ใส่เป็น Feature "id" ตาม spec และคงไว้ใน properties เพื่อให้เห็นในตาราง GIS
        - พิกัดปัดทศนิยม 6 ตำแหน่ง (~0.1 ม.) ตามที่ spec แนะนำ
        ("name" ระดับ FeatureCollection เป็น foreign member ที่ spec อนุญาต — QGIS ใช้เป็นชื่อ layer)"""
        to_wgs = Transformer.from_crs(UTM_EPSG, "EPSG:4326", always_xy=True)

        if self.visible_rect is not None:
            # Rectangle grid: สี่เหลี่ยม 4 มุม (LL -> LR -> UR -> UL = ทวนเข็ม)
            mask = self.visible_rect != -1   # ตัด cell นอกรัศมีศึกษาออก
            grid_ids = np.arange(self.visible_rect.size).reshape(self.visible_rect.shape)[mask]
            vis_vals = self.visible_rect[mask]
            half = self.cell_size / 2
            cx, cy = self.grid_x[mask], self.grid_y[mask]
            corner_x = np.stack([cx - half, cx + half, cx + half, cx - half], axis=1)   # (N, 4)
            corner_y = np.stack([cy - half, cy - half, cy + half, cy + half], axis=1)
        elif self.visible_hex is not None:
            # Hexagonal grid: หกเหลี่ยม flat-top 6 มุม (vertex ที่มุม 0, 60, ..., 300 องศา
            # รอบ center = ทวนเข็ม และ orientation ตรงกับ build_hex_grid)
            grid_ids = self.hex_ids   # index เดิมก่อนตัด cell ในตัวตึก = PK คงที่
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
        lons = lons.round(6).reshape(-1, n_corners)
        lats = lats.round(6).reshape(-1, n_corners)

        features = []
        for gid, v, w, lon_ring, lat_ring in zip(grid_ids, vis_vals, weights, lons, lats):
            ring = [[float(x), float(y)] for x, y in zip(lon_ring, lat_ring)]
            ring.append(ring[0])   # ปิด ring: จุดสุดท้ายต้องซ้ำจุดแรก
            features.append({
                "type": "Feature",
                "id": int(gid),
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {"grid_id": int(gid), "visible": int(v), "weight": int(w)},
            })

        fc = {"type": "FeatureCollection", "name": "Viewshed_Grid_Result", "features": features}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(fc, f, ensure_ascii=False)

        # Summary Report
        n_vis = sum(1 for feat in features if feat['properties']['visible'] == 1)
        n_tot = len(features)
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


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    # ข้อมูลแหล่งศิลปกรรม
    HERITAGE_SITE = (100.5175699, 13.7185468)
    HERITAGE_HEIGHT = 40.0
    # ข้อมูลขอบเขตอาคาร (reproject เป็น UTM 32647 และซ่อม geometry invalid ไว้ล่วงหน้าแล้ว)
    file_osm_bld = "bkk_footprints_utm_fixed.geojson"
    dir_osm_bld = os.path.join(dir_app, file_osm_bld)
    # ข้อมูล DEM (GeoTIFF, EPSG:32647) — โหลดเฉพาะ window รอบพื้นที่ศึกษา
    file_dem = "dem.tif"
    dir_dem = os.path.join(dir_app, file_dem)
    # รูปแบบการแสดงผล
    grids_type = "Hexagonal" # Rectangle or Hexagonal

    analysis = ViewshedAnalysis(HERITAGE_SITE[0], HERITAGE_SITE[1], HERITAGE_HEIGHT)
    analysis.load_building(dir_osm_bld)
    analysis.load_dem(dir_dem)
    analysis.main(grids_type)

    # ผลการวิเคราะห์
    file_viewshed = "viewshed_grid_result_dem.geojson"
    dir_output_viewshed = os.path.join(dir_app, file_viewshed)
    analysis.export_geojson(dir_output_viewshed)
