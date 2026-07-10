#Route-based viewshed: การมองเห็นแหล่งศิลปกรรมตามแนวถนน/เส้นทางเข้าถึง (serial vision)
#================================================================================
# เฟส A — ทั้งโครงข่าย: วางจุด observer ทุก ROUTE_SPACING_M เมตร บนทุกเส้นของ
#   โครงข่ายทางเดิน (bkk_osm_roads_utm.geojson) คำนวณ visible + p_visible ต่อจุด
#   -> viewshed_route_result.geojson (Point) + สรุปสัดส่วนการมองเห็นรายถนน (CSV)
#
# เฟส B — เส้นทางเข้าถึงที่กำหนด: หา shortest path (networkx) จากจุดต้นทาง
#   แต่ละจุดใน ROUTE_ORIGINS ไปยังแหล่งศิลปกรรม แล้ววิเคราะห์ตามลำดับระยะทาง
#   (chainage) -> viewshed_route_paths.geojson + กราฟ serial vision profile
#   (viewshed_route_profiles.png): เดินเข้ามาเมตรที่เท่าไรเริ่มเห็น / หายช่วงไหน
#
# ใช้เครื่องคำนวณเดียวกับ pipeline หลักทั้งหมด: ViewshedAnalysis (run-demo-dem.py,
# โมเดล DEM) + SensitivityAnalysis.prepare_points (run-sensitivity.py, Monte Carlo)
# ข้อจำกัดที่ประกาศ: จุด observer อยู่บน centerline ถนน (คนจริงเดินริมทาง ต่างกัน 2-5 ม.)
#================================================================================
import os
import sys
import json
import logging
import importlib.util

import numpy as np
import shapely
from shapely.geometry import Point
from pyproj import Transformer
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

dir_app = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load(fname, modname):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(dir_app, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_dem_mod = _load("run-demo-dem.py", "run_demo_dem")
_sens_mod = _load("run-sensitivity.py", "run_sensitivity")
ViewshedAnalysis = _dem_mod.ViewshedAnalysis
SensitivityAnalysis = _sens_mod.SensitivityAnalysis
UTM_EPSG = _dem_mod.UTM_EPSG

plt.rcParams["font.family"] = "TH Sarabun New"
plt.rcParams["font.size"] = 14


class RouteAnalysis:
    def __init__(self, base: "ViewshedAnalysis", roads_path, spacing_m=10.0,
                 n_runs=200, agl_error_m=2.0, seed=42):
        self.a = base
        self.spacing = spacing_m
        self.n_runs = n_runs
        self.agl_error_m = agl_error_m
        self.seed = seed
        self.to_wgs = Transformer.from_crs(UTM_EPSG, "EPSG:4326", always_xy=True)

        with open(roads_path, encoding="utf-8") as f:
            text = f.read()
        self.edge_geoms = shapely.get_parts(shapely.from_geojson(text))
        self.edge_props = [f["properties"] for f in json.loads(text)["features"]]
        logger.info(f"Road network loaded: {len(self.edge_geoms)} edges")

    # ------------------------------------------------------------------
    def _set_blockers(self, od_max):
        """ตึกที่เป็นตัวบังได้ = ตึกที่อยู่ห่างแหล่งไม่เกินระยะจุด observer ไกลสุด
        (เส้นสายตาทุกเส้นอยู่ในวงกลมรัศมีนั้นรอบแหล่งเสมอ — convexity)"""
        a = self.a
        site_pt = Point(a.heritage_x, a.heritage_y)
        a.list_buff_bld = [b for b in a.list_bld if b["geom"].distance(site_pt) <= od_max]
        logger.info(f"Blocker set: {len(a.list_buff_bld)} buildings within {od_max:.0f} m of site")

    def _analyze_points(self, ox, oy):
        """คำนวณ visible (baseline) + p_visible ของจุด observer ชุดหนึ่ง
        คืน (open_mask ที่ใช้กรอง, visible, p_visible) — จุดใน mask=False ถูกตัดทิ้ง"""
        a = self.a
        od = np.hypot(ox - a.heritage_x, oy - a.heritage_y)
        self._set_blockers(od.max() + 1.0)

        # ตัดจุดที่ตกในตัวตึก (ทางลอด/ความคลาดเคลื่อน OSM กับ footprint)
        open_mask = a.observer_open_mask(ox, oy)
        ox, oy, od = ox[open_mask], oy[open_mask], od[open_mask]

        sens = SensitivityAnalysis(a, n_runs=self.n_runs, agl_error_m=self.agl_error_m,
                                   seed=self.seed)
        sens.prepare_points(ox, oy, od, np.arange(ox.size))
        sens.run()
        return open_mask, sens.visible_base, sens.p_visible

    # ------------------------------------------------------------------
    # เฟส A: ทั้งโครงข่าย
    # ------------------------------------------------------------------
    def analyze_network(self, out_geojson, out_csv):
        # sample จุดทุก spacing เมตร บนทุกเส้น (รวมจุดปลายเส้น)
        xs, ys, eids, chain = [], [], [], []
        for i, g in enumerate(self.edge_geoms):
            ds = np.arange(0, g.length + self.spacing / 2, self.spacing)
            ds[-1] = min(ds[-1], g.length)
            pts = shapely.line_interpolate_point(g, ds)
            c = shapely.get_coordinates(pts)
            xs.append(c[:, 0]); ys.append(c[:, 1])
            eids.append(np.full(len(ds), i)); chain.append(ds)
        ox = np.concatenate(xs); oy = np.concatenate(ys)
        eids = np.concatenate(eids); chain = np.concatenate(chain)
        logger.info(f"Phase A: {ox.size} route points sampled every {self.spacing:.0f} m")

        open_mask, visible, p = self._analyze_points(ox, oy)
        ox, oy = ox[open_mask], oy[open_mask]
        eids, chain = eids[open_mask], chain[open_mask]

        # export จุด (WGS84, RFC 7946)
        lons, lats = self.to_wgs.transform(ox, oy)
        features = []
        for k in range(ox.size):
            pr = self.edge_props[eids[k]]
            features.append({
                "type": "Feature", "id": int(k),
                "geometry": {"type": "Point",
                             "coordinates": [round(float(lons[k]), 6), round(float(lats[k]), 6)]},
                "properties": {"edge_id": int(eids[k]), "name": pr["name"],
                               "highway": pr["highway"], "chainage_m": round(float(chain[k]), 1),
                               "visible": int(visible[k]), "p_visible": round(float(p[k]), 3)},
            })
        fc = {"type": "FeatureCollection", "name": "Viewshed_Route_Result", "features": features}
        with open(out_geojson, "w", encoding="utf-8") as f:
            json.dump(fc, f, ensure_ascii=False)
        logger.info(f"Saved {out_geojson} ({len(features)} points, "
                    f"{int(visible.sum())} visible = {100 * visible.mean():.1f}%)")

        # สรุปรายถนน (รวมตามชื่อ+ประเภท; ไม่มีชื่อ = unnamed)
        agg = {}
        for k in range(ox.size):
            pr = self.edge_props[eids[k]]
            key = (pr["name"] or "(unnamed)", pr["highway"])
            n, v = agg.get(key, (0, 0))
            agg[key] = (n + 1, v + int(visible[k]))
        rows = sorted(agg.items(), key=lambda kv: -kv[1][0])
        with open(out_csv, "w", encoding="utf-8-sig") as f:
            f.write("name,highway,n_points,n_visible,pct_visible\n")
            for (name, hw), (n, v) in rows:
                f.write(f'"{name}",{hw},{n},{v},{100 * v / n:.1f}\n')
        logger.info(f"Saved {out_csv} ({len(rows)} roads)")

    # ------------------------------------------------------------------
    # เฟส B: เส้นทางเข้าถึงที่กำหนด (serial vision)
    # ------------------------------------------------------------------
    def _build_graph(self):
        """สร้างกราฟจาก edge geometries: node = ปลายเส้น (ปัดพิกัด 0.1 ม.)"""
        G = nx.Graph()
        for i, g in enumerate(self.edge_geoms):
            c = shapely.get_coordinates(g)
            u = (round(c[0, 0], 1), round(c[0, 1], 1))
            v = (round(c[-1, 0], 1), round(c[-1, 1], 1))
            w = g.length
            # เส้นคู่ขนานระหว่าง node เดียวกัน: เก็บเส้นสั้นสุด
            if not G.has_edge(u, v) or G[u][v]["weight"] > w:
                G.add_edge(u, v, weight=w, idx=i)
        return G

    @staticmethod
    def _nearest_node(G, x, y, max_snap=150.0):
        nodes = np.array(list(G.nodes))
        d = np.hypot(nodes[:, 0] - x, nodes[:, 1] - y)
        j = int(d.argmin())
        if d[j] > max_snap:
            raise ValueError(f"จุดอยู่ห่างโครงข่ายเกิน {max_snap:.0f} m (ใกล้สุด {d[j]:.0f} m)")
        return tuple(nodes[j]), float(d[j])

    def _route_line(self, origin_xy):
        """shortest path จาก origin ไปยัง node ที่ใกล้แหล่งที่สุด -> LineString เดียว"""
        a = self.a
        G = self._build_graph()
        src, d_src = self._nearest_node(G, *origin_xy)
        dst, d_dst = self._nearest_node(G, a.heritage_x, a.heritage_y)
        logger.info(f"snap origin {d_src:.0f} m, snap site {d_dst:.0f} m")
        path = nx.shortest_path(G, src, dst, weight="weight")

        coords = []
        for u, v in zip(path, path[1:]):
            g = self.edge_geoms[G[u][v]["idx"]]
            c = shapely.get_coordinates(g).tolist()
            if round(c[0][0], 1) != u[0] or round(c[0][1], 1) != u[1]:
                c = c[::-1]                       # กลับทิศ edge ให้เดินจาก u -> v
            coords.extend(c if not coords else c[1:])
        return shapely.LineString(coords)

    def analyze_route(self, name, origin_lonlat, ax):
        """วิเคราะห์ 1 เส้นทาง วาดกราฟลง ax และคืน features (จุดตามเส้นทาง)"""
        a = self.a
        to_utm = Transformer.from_crs("EPSG:4326", UTM_EPSG, always_xy=True)
        origin = to_utm.transform(*origin_lonlat)

        line = self._route_line(origin)
        ds = np.arange(0, line.length + self.spacing / 2, self.spacing)
        ds[-1] = min(ds[-1], line.length)
        c = shapely.get_coordinates(shapely.line_interpolate_point(line, ds))
        ox, oy = c[:, 0], c[:, 1]

        open_mask, visible, p = self._analyze_points(ox, oy)
        ox, oy, ds = ox[open_mask], oy[open_mask], ds[open_mask]

        # ตัวชี้วัด serial vision
        vis_idx = np.flatnonzero(visible == 1)
        first_glimpse = float(ds[vis_idx[0]]) if vis_idx.size else None
        pct = 100 * visible.mean()
        run_best = run_cur = 0
        for v in visible:                          # ช่วงเห็นต่อเนื่องยาวสุด (จำนวนจุด)
            run_cur = run_cur + 1 if v == 1 else 0
            run_best = max(run_best, run_cur)
        logger.info(f"Route '{name}': length {line.length:.0f} m, visible {pct:.1f}% of points, "
                    f"first glimpse @ {first_glimpse if first_glimpse is not None else 'never'} m, "
                    f"longest visible stretch ~{run_best * self.spacing:.0f} m")

        # กราฟ profile: x = ระยะเดินจากต้นทาง, y = p_visible (แถบเขียว = visible baseline)
        ax.fill_between(ds, 0, p, step="mid", color="#9FE1CB", alpha=0.9,
                        label="p_visible (AGL ±2 ม.)")
        ax.step(ds, p, where="mid", color="#0F6E56", lw=1.2)
        ax.scatter(ds[visible == 1], np.full(vis_idx.size, 1.06), s=8, color="#1D9E75",
                   marker="s", label="visible (baseline)")
        if first_glimpse is not None:
            ax.axvline(first_glimpse, color="#D85A30", ls="--", lw=1.2)
            ax.annotate(f"เห็นครั้งแรก\n{first_glimpse:.0f} ม.", (first_glimpse, 0.55),
                        textcoords="offset points", xytext=(8, 0), color="#993C1D", fontsize=11)
        ax.set_ylim(-0.05, 1.15)
        ax.set_xlim(0, ds.max() * 1.02)
        ax.set_ylabel("p_visible")
        ax.set_title(f"{name} — เดินเข้าหาแหล่ง {line.length:.0f} ม. "
                     f"(เห็น {pct:.0f}% ของทาง)", loc="left", fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)

        lons, lats = self.to_wgs.transform(ox, oy)
        return [{
            "type": "Feature", "id": int(k),
            "geometry": {"type": "Point",
                         "coordinates": [round(float(lons[k]), 6), round(float(lats[k]), 6)]},
            "properties": {"route": name, "chainage_m": round(float(ds[k]), 1),
                           "visible": int(visible[k]), "p_visible": round(float(p[k]), 3)},
        } for k in range(ox.size)]

    def analyze_routes(self, origins, out_geojson, out_png):
        fig, axes = plt.subplots(len(origins), 1, figsize=(11, 3.4 * len(origins)),
                                 squeeze=False)
        features = []
        for (name, lon, lat), ax in zip(origins, axes[:, 0]):
            features.extend(self.analyze_route(name, (lon, lat), ax))
        axes[-1, 0].set_xlabel("ระยะเดินจากจุดต้นทาง (ม.)")
        fig.suptitle("Serial vision profile — การมองเห็นแหล่งศิลปกรรมระหว่างเดินเข้าถึง",
                     fontsize=16, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {out_png}")

        fc = {"type": "FeatureCollection", "name": "Viewshed_Route_Paths", "features": features}
        with open(out_geojson, "w", encoding="utf-8") as f:
            json.dump(fc, f, ensure_ascii=False)
        logger.info(f"Saved {out_geojson} ({len(features)} points)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    # ข้อมูลแหล่งศิลปกรรม
    HERITAGE_SITE = (100.5175699, 13.7185468)
    HERITAGE_HEIGHT = 40.0
    # ข้อมูลที่เตรียมไว้ล่วงหน้า
    dir_osm_bld = os.path.join(dir_app, "bkk_footprints_utm_fixed.geojson")
    dir_dem = os.path.join(dir_app, "dem.tif")
    dir_osm_roads = os.path.join(dir_app, "bkk_osm_roads_utm.geojson")
    # พารามิเตอร์ route analysis
    ROUTE_SPACING_M = 10.0
    # จุดต้นทางเฟส B (ชื่อ, lon, lat) — ตัวอย่าง: ปลายถนนหลักที่ไกลจากแหล่งที่สุด
    # แก้/เพิ่มเป็นป้ายรถเมล์ ท่าเรือ ลานจอดรถจริงได้ตามต้องการ
    ROUTE_ORIGINS = [
        ("จากถนนเจริญกรุง (เหนือ)", 100.5163271, 13.7253312),
        ("จากถนนสาทรใต้ (ตะวันออก)", 100.5244179, 13.7200189),
    ]

    analysis = ViewshedAnalysis(HERITAGE_SITE[0], HERITAGE_SITE[1], HERITAGE_HEIGHT)
    analysis.load_building(dir_osm_bld)
    analysis.load_dem(dir_dem)

    route = RouteAnalysis(analysis, dir_osm_roads, spacing_m=ROUTE_SPACING_M)

    # เฟส A: ทั้งโครงข่าย
    route.analyze_network(os.path.join(dir_app, "viewshed_route_result.geojson"),
                          os.path.join(dir_app, "viewshed_route_summary.csv"))

    # เฟส B: เส้นทางเข้าถึงที่กำหนด
    route.analyze_routes(ROUTE_ORIGINS,
                         os.path.join(dir_app, "viewshed_route_paths.geojson"),
                         os.path.join(dir_app, "viewshed_route_profiles.png"))
