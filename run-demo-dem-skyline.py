#Skyline analysis: เส้นขอบฟ้าที่ผู้สังเกตมองเห็นจริง เมื่อเงยหน้ามองไปยังแหล่งศิลปกรรม
#================================================================================
# ต่อยอดจาก run-demo-dem.py (โมเดล DEM): กำหนด "จุดที่คนยืน" (observer) แล้ว
#   1) คำนวณ silhouette ของอาคารรอบตัวในมุมกวาด ±FOV รอบทิศเล็งไปยังแหล่ง
#      โดยแต่ละอาคารให้ (ช่วงมุมกวาด, มุมเงยของยอด) จากความสูงจริง (msl+AGL),
#      ระยะจริง และระดับตาผู้สังเกต (ground DEM + eye height)
#   2) เทียบมุมเงยของยอดแหล่งศิลปกรรมกับเส้นขอบฟ้า ณ ทิศเล็ง (มุม 0°)
#      -> ยอดโผล่พ้น = มองเห็น / จมใต้เส้นขอบฟ้า = ถูกบัง
#   3) plot กราฟเส้นขอบฟ้า (skyline) พร้อมตำแหน่งยอดแหล่ง -> skyline_from_observer_3d.png
#
# ใช้ ViewshedAnalysis (อาคาร + DEM) ชุดเดียวกับ pipeline หลัก จึงได้ระดับพื้นดิน
# และความสูงอาคารแบบ absolute (m MSL) สอดคล้องกับการวิเคราะห์ viewshed
#================================================================================
import os
import sys
import math
import logging
import importlib.util

import numpy as np
import shapely
from shapely.geometry import Point
from pyproj import Transformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse

dir_app = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

plt.rcParams["font.family"] = "TH Sarabun New"

# โหลด ViewshedAnalysis จาก run-demo-dem.py (ชื่อไฟล์มี "-" จึง import ปกติไม่ได้)
_spec = importlib.util.spec_from_file_location("run_demo_dem", os.path.join(dir_app, "run-demo-dem.py"))
_rdd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rdd)
ViewshedAnalysis = _rdd.ViewshedAnalysis
UTM_EPSG = _rdd.UTM_EPSG

# สีชุดเดียวกับผลวิเคราะห์อื่น
C_VIS = "#1D9E75"; C_BLK = "#D85A30"; C_EDGE = "#8A948E"
C_SIL = "#AEB8B0"; C_SIL_E = "#4A4A4A"; C_INK = "#1B2A26"   # C_SIL_E = ขอบตึกสีเทาเข้ม


class SkylineAnalysis:
    def __init__(self, base: "ViewshedAnalysis", fov_deg=60.0,
                 bld_search_m=430.0, elev_cap_deg=55.0, near_skip_m=18.0):
        """base = ViewshedAnalysis ที่ผ่าน load_building() และ load_dem() มาแล้ว"""
        self.a = base
        self.fov = fov_deg
        self.bld_search_m = bld_search_m       # รัศมีดึงอาคารมาทำ silhouette
        self.elev_cap = elev_cap_deg           # จำกัดมุมเงยสูงสุด (กันอาคารชิดตัวพุ่งเกิน)
        self.near_skip = near_skip_m           # ข้ามอาคารที่ชิดผู้สังเกตเกินไป (ฉากหน้า)
        self.to_utm = Transformer.from_crs("EPSG:4326", UTM_EPSG, always_xy=True)

    # ------------------------------------------------------------------
    def set_observer(self, lon=None, lat=None, eye_height_m=1.70,
                     bearing_deg=None, dist_m=None):
        """กำหนดจุดที่คนยืน: ระบุ (lon, lat) โดยตรง หรือระบุ (bearing_deg, dist_m)
        = ทิศ/ระยะจากแหล่งศิลปกรรม (bearing 0°=เหนือ, 90°=ตะวันออก)
        eye_height_m = ความสูงระดับตาคน (เมตร) เหนือพื้นดิน ณ จุดยืน (ค่าเริ่มต้น 1.70 ม.)"""
        a = self.a
        if lon is not None and lat is not None:
            self.ox, self.oy = self.to_utm.transform(lon, lat)
        elif bearing_deg is not None and dist_m is not None:
            b = math.radians(bearing_deg)
            self.ox = a.heritage_x + dist_m * math.sin(b)
            self.oy = a.heritage_y + dist_m * math.cos(b)
        else:
            raise ValueError("ต้องระบุ (lon, lat) หรือ (bearing_deg, dist_m)")

        self.eye_height = eye_height_m
        self.o_ground = float(a.ground_at(np.array([self.ox]), np.array([self.oy]))[0])
        self.eye_abs = self.o_ground + self.eye_height
        self.dist_oh = math.hypot(a.heritage_x - self.ox, a.heritage_y - self.oy)
        # bearing observer -> heritage (x=east, y=north)
        self.az_h = math.degrees(math.atan2(a.heritage_x - self.ox, a.heritage_y - self.oy))
        return self

    def _relaz(self, px, py):
        """มุมกวาดของจุด (px,py) เทียบทิศเล็งไปยังแหล่ง (อยู่ในช่วง -180..180)"""
        d = math.degrees(math.atan2(px - self.ox, py - self.oy)) - self.az_h
        return (d + 180) % 360 - 180

    # ------------------------------------------------------------------
    def compute(self):
        """คำนวณ silhouette (bars) + มุมเงยยอดแหล่ง + สถานะมองเห็น/ถูกบัง"""
        a = self.a
        obs_pt = Point(self.ox, self.oy)
        top_h = a.heritage_ground + a.heritage_h
        self.elev_h = math.degrees(math.atan2(top_h - self.eye_abs, self.dist_oh))

        bars = []   # (ra_min, ra_max, elev, dist)
        for b in a.list_bld:
            g = b["geom"]
            if g.distance(obs_pt) > self.bld_search_m:
                continue
            c = shapely.get_coordinates(g)
            ras = [self._relaz(x, y) for x, y in c]
            ra_min, ra_max = min(ras), max(ras)
            if ra_max < -self.fov or ra_min > self.fov:
                continue
            if ra_max - ra_min > 170:            # กัน artifact เมื่ออาคารคร่อมด้านหลัง
                continue
            dn = min(math.hypot(x - self.ox, y - self.oy) for x, y in c)
            if dn < self.near_skip:
                continue
            top = b["msl"] + b["height"]
            elev = min(math.degrees(math.atan2(top - self.eye_abs, dn)), self.elev_cap)
            bars.append((max(ra_min, -self.fov), min(ra_max, self.fov), elev, dn))
        self.bars = bars

        # เส้นขอบฟ้า ณ ทิศเล็ง (มุม 0°): มุมเงยสูงสุดของอาคารที่คร่อม 0° และอยู่ใกล้กว่าแหล่ง
        self.sil0 = max([e for (r0, r1, e, d) in bars if r0 <= 0 <= r1 and d < self.dist_oh - 2],
                        default=0.0)
        self.visible = self.elev_h > self.sil0 + 0.1
        logger.info(f"Observer ground {self.o_ground:.1f} m, eye {self.eye_abs:.1f} m MSL, "
                    f"dist to site {self.dist_oh:.0f} m")
        logger.info(f"Heritage elev {self.elev_h:.1f}deg vs skyline {self.sil0:.1f}deg "
                    f"-> {'VISIBLE' if self.visible else 'BLOCKED'}  ({len(bars)} buildings in FOV)")
        return self

    # ------------------------------------------------------------------
    def _envelope(self):
        """ขอบบนสุดของเงาอาคาร (skyline upper envelope) เป็นเส้นขั้นบันได -> (xs, ys)"""
        edges = sorted({-self.fov, self.fov}
                       | {r for (r0, r1, e, d) in self.bars for r in (r0, r1)})
        xs, ys = [], []
        for xL, xR in zip(edges[:-1], edges[1:]):
            mid = 0.5 * (xL + xR)
            h = max([e for (r0, r1, e, d) in self.bars if r0 <= mid <= r1], default=0.0)
            xs += [xL, xR]; ys += [h, h]
        return xs, ys

    # ------------------------------------------------------------------
    def plot(self, out_path):
        """วาดกราฟเส้นขอบฟ้า (skyline) ที่ผู้สังเกตมองเห็นจริง ลง out_path"""
        fig, ax = plt.subplots(figsize=(13, 5.2))

        # พื้นหลังท้องฟ้า: ไล่เฉดสีฟ้า (สว่างที่ขอบฟ้า -> เข้มด้านบน) + เมฆนุ่ม ๆ
        ymax = max(46, self.elev_h * 1.5)
        grad = np.linspace(0.0, 1.0, 256)[:, None]
        c_hor = np.array([0.87, 0.94, 0.99])     # ใกล้ขอบฟ้า (สว่าง)
        c_top = np.array([0.27, 0.51, 0.83])     # ด้านบน (ฟ้าเข้ม)
        sky = np.repeat((c_hor * (1 - grad) + c_top * grad)[:, None, :], 2, axis=1)
        ax.imshow(sky, extent=[-self.fov, self.fov, 0, ymax], origin="lower",
                  aspect="auto", zorder=0, interpolation="bilinear")
        crng = np.random.default_rng(7)
        for cxp, cyp, cw in [(-44, 39, 22), (-10, 42, 28), (22, 35, 24), (47, 40, 18)]:
            for _ in range(7):
                ex = cxp + crng.uniform(-cw * 0.4, cw * 0.4)
                ey = cyp + crng.uniform(-2.0, 2.0)
                ew = cw * crng.uniform(0.4, 0.7)
                ax.add_patch(Ellipse((ex, ey), ew, ew * 0.32, fc="white",
                                     ec="none", alpha=0.16, zorder=1))

        # silhouette อาคาร (วาดไกลก่อน)
        for (r0, r1, e, d) in sorted(self.bars, key=lambda bar: -bar[3]):
            ax.add_patch(Rectangle((r0, 0), r1 - r0, e, fc=C_SIL, ec=C_SIL_E, lw=0.6))
        ax.axhline(0, color=C_EDGE, lw=1.0)

        # เส้นขอบฟ้า (skyline) = ขอบบนสุดของเงาอาคาร วาดเป็นเส้นแดงซ้อนทับ
        sx, sy = self._envelope()
        ax.plot(sx, sy, color="#E11", lw=2.0, zorder=8, solid_joinstyle="miter")

        # ยอดแหล่งศิลปกรรม ณ ทิศเล็ง (มุม 0°)
        mc = C_VIS if self.visible else C_BLK
        ax.plot([0], [self.elev_h], "*", color=mc, ms=14, mec="white", mew=1.4, zorder=10)
        ax.plot([0, 0], [0, self.elev_h], color=mc, lw=1.8,
                ls=("-" if self.visible else (0, (3, 2))), zorder=9)
        status = "มองเห็น (โผล่พ้นเส้นขอบฟ้า)" if self.visible else "ถูกบัง (ต่ำกว่าเส้นขอบฟ้า)"

        ax.set_xlim(-self.fov, self.fov)
        ax.set_ylim(0, ymax)
        step = 30 if self.fov >= 60 else 10
        ax.set_xticks(np.arange(-self.fov, self.fov + 1, step))
        ax.set_xlabel(f"มุมกวาดซ้าย–ขวา รอบทิศเล็งไปยังแหล่ง (องศา, ±{self.fov:.0f}°)", fontsize=13)
        ax.set_ylabel("มุมเงย (องศา)", fontsize=13)
        ax.set_title(f"กราฟเส้นขอบฟ้า (Skyline) ที่ผู้สังเกตมองเห็นจริง — เงยหน้ามองไปทางแหล่งศิลปกรรม\n"
                     f"ระยะ {self.dist_oh:.0f} ม. · ยอดแหล่งมุมเงย {self.elev_h:.1f}° "
                     f"เทียบเส้นขอบฟ้า {self.sil0:.1f}° จึง{status}",
                     fontsize=15, fontweight="bold", color=C_INK, loc="left", pad=10)
        ax.tick_params(labelsize=11)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"Saved {out_path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    # ข้อมูลแหล่งศิลปกรรม (lon, lat)
    HERITAGE_SITE = (100.5175699, 13.7185468)
    HERITAGE_HEIGHT = 40.0
    # จุดที่คนยืน (lon, lat) รูปแบบเดียวกับ HERITAGE_SITE
    OBSERVER_SITE = (100.5175467, 13.7149312)
    # ข้อมูลที่เตรียมไว้ล่วงหน้า
    dir_osm_bld = os.path.join(dir_app, "bkk_footprints_utm_fixed.geojson")
    dir_dem = os.path.join(dir_app, "dem.tif")
    OUTPUT_PNG = os.path.join(dir_app, "skyline_from_observer_3d.png")

    analysis = ViewshedAnalysis(HERITAGE_SITE[0], HERITAGE_SITE[1], HERITAGE_HEIGHT)
    analysis.load_building(dir_osm_bld)
    analysis.load_dem(dir_dem)

    sky = SkylineAnalysis(analysis)                      # FOV_DEG = 60° (ค่า default ใน __init__)
    sky.set_observer(lon=OBSERVER_SITE[0], lat=OBSERVER_SITE[1])   # eye height = 1.70 ม. (ค่า default)
    sky.compute()
    sky.plot(OUTPUT_PNG)
