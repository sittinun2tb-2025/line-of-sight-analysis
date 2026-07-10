#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import time

import numpy as np
import shapely
from pyproj import Transformer

dir_app = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(dir_app, "bkk_footprints.geojson")            # WGS84 lon/lat
OUT_PATH = os.path.join(dir_app, "bkk_footprints_utm_fixed.geojson")  # EPSG:32647, valid
OSM_OUT_PATH = os.path.join(dir_app, "bkk_osm_roads_utm.geojson")         # EPSG:32647, walk network
UTM_EPSG = "EPSG:32647"
OSM_BBOX_MARGIN_DEG = 0.001    # ขยายกรอบดึง OSM เผื่อขอบ (~110 ม.)


def prepare_footprints_utm():
    """เตรียมข้อมูลอาคาร: reproject + ซ่อม geometry
    คืนค่า bbox (minlon, minlat, maxlon, maxlat) ของข้อมูลต้นทาง (WGS84)
    สำหรับใช้เป็นกรอบดึงข้อมูล OSM ต่อ"""
    with open(SRC_PATH, encoding="utf-8") as f:
        text = f.read()
    data = json.loads(text)

    geoms = shapely.get_parts(shapely.from_geojson(text))
    assert len(geoms) == len(data["features"]), "geometry/property count mismatch"

    lonlat = shapely.get_coordinates(geoms)
    bbox_wgs = (lonlat[:, 0].min(), lonlat[:, 1].min(),
                lonlat[:, 0].max(), lonlat[:, 1].max())

    to_utm = Transformer.from_crs("EPSG:4326", UTM_EPSG, always_xy=True)
    e, n = to_utm.transform(lonlat[:, 0], lonlat[:, 1])
    geoms = shapely.set_coordinates(geoms, np.column_stack([e, n]))

    bad = ~shapely.is_valid(geoms)
    print(f"Fixing {bad.sum()} / {len(geoms)} invalid geometries...")
    geoms[bad] = shapely.buffer(geoms[bad], 0)
    still_bad = int((~shapely.is_valid(geoms)).sum())
    if still_bad:
        print(f"Warning: {still_bad} geometries still invalid after buffer(0)")

    for feat, geom in zip(data["features"], geoms):
        feat["geometry"] = json.loads(shapely.to_geojson(geom))
    data["crs"] = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:{UTM_EPSG.replace(':', '::')}"}}

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"Wrote {OUT_PATH}")

    return bbox_wgs


def _tag_str(v):
    """OSM tag อาจเป็น list (ทางที่ถูก merge ตอน simplify) — แปลงเป็น string เดียว"""
    if isinstance(v, (list, tuple)):
        return "|".join(str(x) for x in v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return str(v)


def prepare_osm_roads(bbox_wgs, out_path=OSM_OUT_PATH):
    """ดาวน์โหลดโครงข่ายถนน/ทางเดิน (walk) จาก OSM ในกรอบ bbox (WGS84)
    reproject เป็น EPSG:32647 แล้วเขียน GeoJSON (LineString ต่อ edge)
    properties: osmid, highway, name, width (ถ้ามี), length_m"""
    import osmnx as ox   # import ในฟังก์ชัน: ไม่บังคับติดตั้งถ้าไม่ใช้ขั้นนี้

    minlon, minlat, maxlon, maxlat = bbox_wgs
    m = OSM_BBOX_MARGIN_DEG
    bbox = (minlon - m, minlat - m, maxlon + m, maxlat + m)   # (left, bottom, right, top)

    print(f"Downloading OSM walk network for bbox {tuple(round(v, 5) for v in bbox)} ...")
    G = ox.graph_from_bbox(bbox, network_type="walk", retain_all=True)
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True).to_crs(UTM_EPSG)

    features = []
    for i, (_, row) in enumerate(edges.iterrows()):
        features.append({
            "type": "Feature",
            "id": i,
            "geometry": json.loads(shapely.to_geojson(row.geometry)),
            "properties": {
                "osmid": _tag_str(row.get("osmid")),
                "highway": _tag_str(row.get("highway")),
                "name": _tag_str(row.get("name")),
                "width": _tag_str(row.get("width")),
                "length_m": round(float(row["length"]), 1),
            },
        })

    fc = {"type": "FeatureCollection",
          "name": "OSM_Roads_Walk",
          "crs": {"type": "name", "properties": {"name": f"urn:ogc:def:crs:{UTM_EPSG.replace(':', '::')}"}},
          "features": features}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)

    import collections
    hw = collections.Counter(f["properties"]["highway"] for f in features)
    print(f"Wrote {out_path}  ({len(features)} edges)")
    print("  highway types:", dict(hw.most_common(8)))


def main():
    t0 = time.perf_counter()

    bbox_wgs = prepare_footprints_utm()

    try:
        prepare_osm_roads(bbox_wgs)
    except Exception as e:
        print(f"Warning: OSM road prep skipped — {type(e).__name__}: {e}")
        print("(ต้องต่ออินเทอร์เน็ตถึง Overpass API; ข้อมูลอาคารเตรียมเสร็จตามปกติแล้ว)")

    print(f"Done in {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
