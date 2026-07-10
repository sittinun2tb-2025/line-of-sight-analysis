#Sensitivity analysis: ความน่าจะเป็นที่มองเห็นแหล่งศิลปกรรม เมื่อความสูงตึก (AGL) ไม่แน่นอน
#================================================================================
# แนวคิด (Monte Carlo):
#   ความสูงตึกจากข้อมูลจริงมีความคลาดเคลื่อน — cell ที่แพ้/ชนะกันหลักเศษเมตร
#   อาจพลิกผลได้ถ้า AGL คลาดไปเพียง 1-2 ม. สคริปต์นี้จึงรันวิเคราะห์ซ้ำ N รอบ
#   โดยสุ่มรบกวน AGL ของทุกตึก (uniform ±AGL_ERROR_M) แล้วนับสัดส่วนรอบที่
#   แต่ละ cell มองเห็น -> "p_visible" (0.0-1.0)
#
# เคล็ดความเร็ว: การสุ่มเปลี่ยนเฉพาะ "ความสูง" ไม่ได้เปลี่ยนรูปทรง/ตำแหน่งตึก
#   ดังนั้นงานเรขาคณิตราคาแพง (STRtree query + intersection) ทำครั้งเดียว
#   เก็บความสูงต่ำสุดของเส้นสายตาต่อคู่ (sightline, building) ไว้ แต่ละรอบ
#   Monte Carlo เหลือแค่เทียบ array -> รันหลายร้อยรอบได้ในเวลาไม่กี่วินาที
#
# ใช้ตรรกะ LOS เดียวกับ run-demo-dem.py (โหลด class ผ่าน importlib) และมี
# ตัวตรวจ baseline: ผลรอบ "ไม่รบกวน" ต้องตรงกับ compute_visibility ของ
# โมเดลหลักทุก cell ไม่เช่นนั้นหยุดทันที (กันโค้ดสอง path เพี้ยนจากกัน)
#================================================================================
import os
import sys
import json
import time
import logging
import importlib.util

import numpy as np
import shapely
from shapely.geometry import Point
from shapely.strtree import STRtree
from pyproj import Transformer

dir_app = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# โหลด ViewshedAnalysis จาก run-demo-dem.py (ชื่อไฟล์มี "-" จึง import ปกติไม่ได้)
_spec = importlib.util.spec_from_file_location("run_demo_dem", os.path.join(dir_app, "run-demo-dem.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ViewshedAnalysis = _mod.ViewshedAnalysis
UTM_EPSG = _mod.UTM_EPSG


class SensitivityAnalysis:
    def __init__(self, base: "ViewshedAnalysis", n_runs=200, agl_error_m=2.0, seed=42,
                 stable_p=0.95):
        """base = ViewshedAnalysis ที่ผ่าน load_building() และ load_dem() มาแล้ว"""
        self.a = base
        self.n_runs = n_runs
        self.agl_error_m = agl_error_m
        self.seed = seed
        self.stable_p = stable_p          # p >= ค่านี้ = มองเห็นแน่, p <= 1-ค่านี้ = ถูกบังแน่
        self.grid_type = None
        # ผลลัพธ์ (ตั้งค่าโดย prepare/run)
        self.grid_ids = None
        self.p_visible = None
        self.visible_base = None

    # ------------------------------------------------------------------
    def prepare(self, GRID_TYPE):
        """สร้าง grid + งานเรขาคณิตทั้งหมด (ทำครั้งเดียว ใช้ซ้ำทุกรอบ Monte Carlo)"""
        a = self.a
        self.grid_type = GRID_TYPE

        site_pt = Point(a.heritage_x, a.heritage_y)
        a.list_buff_bld = [b for b in a.list_bld
                           if b["geom"].distance(site_pt) <= a.radius_m + a.BUILDING_SEARCH_MARGIN]
        logger.info(f"Loaded {len(a.list_bld)} buildings, {len(a.list_buff_bld)} within analysis range")

        if GRID_TYPE == "Rectangle":
            grid_x, grid_y, dist = a.build_rect_grid()
            in_radius = dist <= a.radius_m
            self.ox, self.oy, self.od = grid_x[in_radius], grid_y[in_radius], dist[in_radius]
            # grid_id = flat row-major index ของ grid เต็ม (PK เดียวกับ export ปกติ)
            self.grid_ids = np.arange(dist.size).reshape(dist.shape)[in_radius]
        elif GRID_TYPE == "Hexagonal":
            self.ox, self.oy, self.od, self.hex_r = a.build_hex_grid()
            self.grid_ids = np.arange(self.ox.size)
        else:
            raise ValueError(f"unknown GRID_TYPE: {GRID_TYPE}")

        # ตัด cell ที่ center อยู่ในตัวตึกออก (คนยืนมองตรงนั้นไม่ได้จริง)
        # grid_ids ของ cell ที่เหลือคงเดิม จึงยัง join กับผลเวอร์ชันก่อน ๆ ได้
        open_mask = a.observer_open_mask(self.ox, self.oy)
        self.ox, self.oy, self.od = self.ox[open_mask], self.oy[open_mask], self.od[open_mask]
        self.grid_ids = self.grid_ids[open_mask]

        # ---- งานเรขาคณิต ทำครั้งเดียว (mirror ส่วนต้นของ compute_visibility) ----
        t0 = time.time()
        geoms = np.array([b["geom"] for b in a.list_buff_bld], dtype=object)
        self.msl = np.array([b["msl"] for b in a.list_buff_bld])
        self.h0 = np.array([b["height"] for b in a.list_buff_bld])
        tree = STRtree(geoms)

        coords = np.empty((self.ox.size, 2, 2))
        coords[:, 0, 0], coords[:, 0, 1] = self.ox, self.oy
        coords[:, 1, 0], coords[:, 1, 1] = a.heritage_x, a.heritage_y
        lines = shapely.linestrings(coords)
        obs_pts = shapely.points(self.ox, self.oy)

        # ทุกคู่ (sightline, building) ที่ตัดกันจริง — ไม่กรองด้วยความสูง เพราะ
        # ความสูงจะเปลี่ยนทุกรอบ Monte Carlo
        self.li, self.gi = tree.query(lines, predicate="intersects")

        end_obs = a.ground_at(self.ox, self.oy) + a.obs_eye_height   # ต่อ cell
        end_site = a.heritage_ground + a.heritage_h                   # scalar

        inter = shapely.intersection(lines[self.li], geoms[self.gi])
        d_enter = shapely.distance(obs_pts[self.li], inter)
        d_exit = d_enter + shapely.length(inter)

        od_safe = np.where(self.od > 1e-6, self.od, 1.0)
        t_enter = d_enter / od_safe[self.li]
        t_exit = d_exit / od_safe[self.li]
        pair_obs = end_obs[self.li]
        h_enter = pair_obs * (1 - t_enter) + end_site * t_enter
        h_exit = pair_obs * (1 - t_exit) + end_site * t_exit
        # ความสูงเส้นสายตา ณ จุดต่ำสุดของช่วงที่ผ่านตึกนี้ — ตึกบังเมื่อยอด > ค่านี้
        self.los_min = np.minimum(h_enter, h_exit)

        logger.info(f"Precomputed {len(self.li)} sightline-building pairs "
                    f"for {self.ox.size} cells in {time.time() - t0:.1f}s")

        self._check_baseline()

    # ------------------------------------------------------------------
    def _visibility_from_heights(self, heights):
        """คำนวณ visible (0/1) ของทุก cell จากความสูงตึกชุดที่กำหนด — numpy ล้วน"""
        top = self.msl + heights                       # ยอดตึก absolute (m MSL)
        blocked = top[self.gi] > self.los_min          # ต่อคู่
        visible = np.ones(self.ox.size, dtype=np.int8)
        visible[self.li[blocked]] = 0
        visible[self.od < 1e-6] = 1                    # cell ของตัว site เห็นเสมอ
        return visible

    def _check_baseline(self):
        """ผลแบบไม่รบกวนต้องตรงกับ compute_visibility ของโมเดลหลักทุก cell"""
        base_fast = self._visibility_from_heights(self.h0)
        base_ref = self.a.compute_visibility(self.ox, self.oy, self.od)
        if not np.array_equal(base_fast, base_ref):
            n_diff = int((base_fast != base_ref).sum())
            raise RuntimeError(f"baseline mismatch กับ compute_visibility ({n_diff} cells) — "
                               "ตรรกะสอง path ไม่ตรงกัน ห้ามใช้ผลต่อ")
        self.visible_base = base_ref
        logger.info(f"Baseline check OK ({int(base_ref.sum())}/{base_ref.size} cells visible)")

    # ------------------------------------------------------------------
    def run(self):
        """Monte Carlo: สุ่ม AGL ±agl_error_m (uniform) n_runs รอบ -> p_visible ต่อ cell
        (เปลี่ยนเป็น normal ได้โดยใช้ rng.normal(0, sigma) แทน rng.uniform)"""
        rng = np.random.default_rng(self.seed)
        t0 = time.time()
        count = np.zeros(self.ox.size, dtype=np.int32)
        for _ in range(self.n_runs):
            noise = rng.uniform(-self.agl_error_m, self.agl_error_m, size=self.h0.size)
            heights = np.maximum(self.h0 + noise, 0.0)   # ความสูงติดลบไม่มีจริง
            count += self._visibility_from_heights(heights)
        self.p_visible = count / self.n_runs

        stable_vis = int((self.p_visible >= self.stable_p).sum())
        stable_blk = int((self.p_visible <= 1 - self.stable_p).sum())
        uncertain = self.p_visible.size - stable_vis - stable_blk
        logger.info(f"Monte Carlo {self.n_runs} runs (AGL ±{self.agl_error_m} m) "
                    f"in {time.time() - t0:.1f}s")
        logger.info(f"stable visible: {stable_vis}, stable blocked: {stable_blk}, "
                    f"uncertain: {uncertain} ({100 * uncertain / self.p_visible.size:.1f}%)")
        return self.p_visible

    # ------------------------------------------------------------------
    def export_geojson(self, output_path):
        """ส่งออก polygon grid ตาม RFC 7946 (แบบเดียวกับ export_geojson หลัก)
        properties: grid_id, visible (baseline), p_visible, stability"""
        a = self.a
        to_wgs = Transformer.from_crs(UTM_EPSG, "EPSG:4326", always_xy=True)

        if self.grid_type == "Rectangle":
            half = a.cell_size / 2
            cx, cy = self.ox, self.oy
            corner_x = np.stack([cx - half, cx + half, cx + half, cx - half], axis=1)
            corner_y = np.stack([cy - half, cy - half, cy + half, cy + half], axis=1)
        else:   # Hexagonal (flat-top, ทวนเข็ม — orientation ตรงกับ build_hex_grid)
            angles = np.deg2rad(np.arange(0, 360, 60))
            corner_x = self.ox[:, None] + self.hex_r * np.cos(angles)[None, :]
            corner_y = self.oy[:, None] + self.hex_r * np.sin(angles)[None, :]

        n_corners = corner_x.shape[1]
        lons, lats = to_wgs.transform(corner_x.ravel(), corner_y.ravel())
        lons = lons.round(6).reshape(-1, n_corners)
        lats = lats.round(6).reshape(-1, n_corners)

        lo, hi = 1 - self.stable_p, self.stable_p
        features = []
        for gid, vb, p, lon_ring, lat_ring in zip(self.grid_ids, self.visible_base,
                                                   self.p_visible, lons, lats):
            ring = [[float(x), float(y)] for x, y in zip(lon_ring, lat_ring)]
            ring.append(ring[0])
            stability = "visible" if p >= hi else ("blocked" if p <= lo else "uncertain")
            features.append({
                "type": "Feature",
                "id": int(gid),
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {"grid_id": int(gid), "visible": int(vb),
                               "p_visible": round(float(p), 3), "stability": stability},
            })

        fc = {"type": "FeatureCollection", "name": "Viewshed_Sensitivity_Result",
              "features": features}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(fc, f, ensure_ascii=False)
        logger.info(f"Saved {output_path} ({len(features)} cells, "
                    f"n_runs={self.n_runs}, AGL ±{self.agl_error_m} m)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    # ข้อมูลแหล่งศิลปกรรม
    HERITAGE_SITE = (100.5175699, 13.7185468)
    HERITAGE_HEIGHT = 40.0
    # ข้อมูลขอบเขตอาคาร + DEM + โครงข่ายทางสาธารณะ (เตรียมไว้ล่วงหน้าแล้ว)
    dir_osm_bld = os.path.join(dir_app, "bkk_footprints_utm_fixed.geojson")
    dir_dem = os.path.join(dir_app, "dem.tif")
    dir_osm_roads = os.path.join(dir_app, "bkk_osm_roads_utm.geojson")
    # รูปแบบ grid และพารามิเตอร์ Monte Carlo
    grids_type = "Hexagonal"      # Rectangle or Hexagonal
    N_RUNS = 200                  # จำนวนรอบสุ่ม (ความละเอียดของ p = 1/N_RUNS)
    AGL_ERROR_M = 2.0             # +-ความคลาดเคลื่อนความสูงตึกที่สมมติ (uniform ±)
    SEED = 42                     # ล็อกไว้เพื่อให้ผลทำซ้ำได้
    # ผลการวิเคราะห์
    dir_output = os.path.join(dir_app, "viewshed_sensitivity_result.geojson")

    analysis = ViewshedAnalysis(HERITAGE_SITE[0], HERITAGE_SITE[1], HERITAGE_HEIGHT)
    analysis.load_building(dir_osm_bld)
    analysis.load_dem(dir_dem)
    analysis.load_osm_roads(dir_osm_roads)

    sens = SensitivityAnalysis(analysis, n_runs=N_RUNS, agl_error_m=AGL_ERROR_M, seed=SEED)
    sens.prepare(grids_type)
    sens.run()
    sens.export_geojson(dir_output)
