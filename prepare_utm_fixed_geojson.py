#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""One-time prep: reproject bkk_footprints.geojson to EPSG:32647 and fix
invalid (self-intersecting) building geometries, so run-demo.py's
load_building() doesn't have to pay that cost on every run.

Run again whenever bkk_footprints.geojson changes.
"""

import json
import os
import time

import numpy as np
import shapely
from pyproj import Transformer

dir_app = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(dir_app, "bkk_footprints.geojson")            # WGS84 lon/lat
OUT_PATH = os.path.join(dir_app, "bkk_footprints_utm_fixed.geojson")  # EPSG:32647, valid
UTM_EPSG = "EPSG:32647"


def main():
    t0 = time.perf_counter()

    with open(SRC_PATH, encoding="utf-8") as f:
        text = f.read()
    data = json.loads(text)

    geoms = shapely.get_parts(shapely.from_geojson(text))
    assert len(geoms) == len(data["features"]), "geometry/property count mismatch"

    to_utm = Transformer.from_crs("EPSG:4326", UTM_EPSG, always_xy=True)
    lonlat = shapely.get_coordinates(geoms)
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

    print(f"Wrote {OUT_PATH} in {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
