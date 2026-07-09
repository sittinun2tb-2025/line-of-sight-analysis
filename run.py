#Viewshed analysis (binary): visible / not-visible areas around a heritage site
#================================================================================

import json
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import ListedColormap
from pyproj import Transformer
import shapely
from shapely.geometry import Point, box
from shapely.strtree import STRtree


GEOJSON_PATH = "bkk_footprints.geojson"     # building footprints (WGS84 lon/lat)
HEIGHT_FIELD = "AGL"                        # Above Ground Level ระยะความสูงจากพื้นดิน
UTM_EPSG = "EPSG:32647"       

# ข้อมูลแหล่งศิปลกรรม
SITE_LON, SITE_LAT = 100.5175699, 13.7185468   # heritage site location
SITE_HEIGHT = 40.0            # metres, วัดพระแก้วมีความสูงจากฐานถึงยอด 40 เมตร

# ข้อมูลการมองเห็น
OBSERVER_EYE_HEIGHT = 1.7     # metres, standard eye height for a standing pedestrian
GROUND_Z = 0.0                # flat-terrain assumption (buildings-only DSM, no DTM)
RADIUS = 250.0                # metres, radius of the study area around the site
CELL_SIZE = 10.0              # metres, grid resolution
BUILDING_SEARCH_MARGIN = 100.0

MIN_LABEL_AREA = 40.0         # sq. metres, skip AGL height labels on footprints smaller than this
GRID_EDGE_COLOR = "#D9D9D9"   # light grey border between grid cells

OUTPUT_PNG_RECT = f"viewshed_binary_rectangle_{SITE_HEIGHT:.0f}m.png"
OUTPUT_PNG_HEX = f"viewshed_binary_hexagonal_{SITE_HEIGHT:.0f}m.png"
OUTPUT_PNG_LINE = f"viewshed_heritage_line_{SITE_HEIGHT:.0f}m.png"
OUTPUT_GEOJSON = "viewshed_grid_result.geojson"

plt.rcParams["font.family"] = "TH Sarabun New"
plt.rcParams["font.size"] = 16

# ---------------------------------------------------------------------------
# 1. Load and reproject building footprints
# ---------------------------------------------------------------------------

def load_buildings(geojson_path, to_utm, height_field=HEIGHT_FIELD):
    """Load a GeoJSON of building footprints and reproject to the UTM CRS.

    Returns a list of dicts: {"geom": shapely Polygon/MultiPolygon, "height": float}
    """
    with open(geojson_path, encoding="utf-8") as f:
        text = f.read()

    # Parse all geometries in one GEOS call — much faster than shape() per feature
    geoms = shapely.get_parts(shapely.from_geojson(text))
    print (geoms)
    
    features = json.loads(text)["features"]
    assert len(geoms) == len(features), "geometry/property count mismatch (null geometry?)"

    heights = np.full(len(geoms), np.nan)
    for i, feat in enumerate(features):
        try:
            heights[i] = float(feat["properties"][height_field])
        except (KeyError, TypeError, ValueError):
            pass  # stays NaN -> dropped below

    valid = ~np.isnan(heights)
    if not valid.all():
        print(f"Skipped {np.count_nonzero(~valid)} feature(s) "
              f"with missing/invalid '{height_field}' value")
    geoms, heights = geoms[valid], heights[valid]

    # Reproject every vertex of every footprint in a single pyproj call
    lonlat = shapely.get_coordinates(geoms)
    e, n = to_utm.transform(lonlat[:, 0], lonlat[:, 1])
    geoms = shapely.set_coordinates(geoms, np.column_stack([e, n]))

    # Fix minor self-intersections, only where actually needed
    bad = ~shapely.is_valid(geoms)
    geoms[bad] = shapely.buffer(geoms[bad], 0)

    keep = ~shapely.is_empty(geoms)
    return [{"geom": g, "height": h} for g, h in zip(geoms[keep], heights[keep])]


# ---------------------------------------------------------------------------
# 2. Build the analysis grid (square cells, or flat-top hexagons)
# ---------------------------------------------------------------------------

def build_grid(site_x, site_y, radius, cell_size):
    n = int(np.ceil(radius / cell_size))
    offsets = np.arange(-n, n + 1) * cell_size
    grid_x, grid_y = np.meshgrid(site_x + offsets, site_y + offsets)
    dist_from_site = np.sqrt((grid_x - site_x) ** 2 + (grid_y - site_y) ** 2)
    return grid_x, grid_y, dist_from_site


def build_hex_grid(site_x, site_y, radius, cell_size):
    """Generate flat-top hexagon centres covering a circle of `radius` m
    around the site, limited to the circle.

    `cell_size` is the hexagon's width across flats (the same "resolution"
    meaning as a square grid's cell size). Returns the centre coordinates,
    their distance from the site, and the hexagon's circumradius
    (centre-to-vertex distance, needed to actually draw the hexes).
    """
    hex_r = cell_size / np.sqrt(3)      # circumradius
    dx = 1.5 * hex_r                    # column spacing
    dy = cell_size                      # row spacing (= flat-to-flat width)

    ncols = int(np.ceil(radius / dx)) + 1
    nrows = int(np.ceil(radius / dy)) + 1
    cols, rows = np.meshgrid(np.arange(-ncols, ncols + 1),
                              np.arange(-nrows, nrows + 1), indexing="ij")

    hx = site_x + cols * dx
    hy = site_y + rows * dy + np.where(cols % 2 != 0, dy / 2, 0.0)   # odd columns offset

    hx, hy = hx.ravel(), hy.ravel()
    dist = np.hypot(hx - site_x, hy - site_y)
    keep = dist <= radius
    return hx[keep], hy[keep], dist[keep], hex_r


# ---------------------------------------------------------------------------
# 3. Line-of-sight viewshed (vectorised, works for any set of observer points)
# ---------------------------------------------------------------------------

def compute_visibility(site_x, site_y, site_height, buildings, ox, oy, od,
                        eye_h=OBSERVER_EYE_HEIGHT, ground_z=GROUND_Z):
    """Test line-of-sight to the site from each observer point (ox, oy), at
    distance od from the site, all at once.

    A point sees the site unless some building blocks the sightline. The
    ray's height is a straight-line interpolation between the observer's eye
    level and the site top, so — regardless of which end is higher — its
    lowest point over any sub-segment of the line is at whichever endpoint
    (entry or exit from a given building's footprint) has the lower height.
    Comparing the building's height against that minimum works whether the
    site is above, below, or level with the observer's eyes.

    Returns an int8 array (0/1) the same length as ox/oy.
    """
    geoms = np.array([b["geom"] for b in buildings], dtype=object)
    heights = np.array([b["height"] for b in buildings])
    tree = STRtree(geoms)

    t0 = time.time()

    # One sightline per point: observer -> site
    coords = np.empty((ox.size, 2, 2))
    coords[:, 0, 0], coords[:, 0, 1] = ox, oy
    coords[:, 1, 0], coords[:, 1, 1] = site_x, site_y
    lines = shapely.linestrings(coords)
    obs_pts = shapely.points(ox, oy)

    # All (sightline, building) pairs that actually intersect, in one bulk query
    line_idx, geom_idx = tree.query(lines, predicate="intersects")

    visible = np.ones(ox.size, dtype=np.int8)

    # The ray's height only ever ranges between these two endpoint values
    # (whichever order they're in). A building above the higher one always
    # blocks; one at or below the lower one never does; only heights in
    # between need the exact entry/exit test.
    end_obs, end_site = ground_z + eye_h, site_height
    h_lo, h_hi = min(end_obs, end_site), max(end_obs, end_site)

    pair_h = heights[geom_idx]
    visible[line_idx[pair_h > h_hi]] = 0
    check = (pair_h > h_lo) & (pair_h <= h_hi)
    li, gi = line_idx[check], geom_idx[check]

    # Where the sightline enters and exits this building's footprint —
    # both are colinear with the (straight) sightline, so the nearest and
    # farthest intersection points are found via distance from the observer.
    inter = shapely.intersection(lines[li], geoms[gi])
    d_enter = shapely.distance(obs_pts[li], inter)
    d_exit = d_enter + shapely.length(inter)   # exact for a single crossing segment

    od_safe = np.where(od > 1e-6, od, 1.0)   # avoid 0/0 at the site's own cell
    t_enter = d_enter / od_safe[li]
    t_exit = d_exit / od_safe[li]
    h_enter = end_obs * (1 - t_enter) + end_site * t_enter
    h_exit = end_obs * (1 - t_exit) + end_site * t_exit
    los_height_min = np.minimum(h_enter, h_exit)

    visible[li[heights[gi] > los_height_min]] = 0
    visible[od < 1e-6] = 1           # the site's own cell is always visible

    print(f"Computed {ox.size} cells in {time.time() - t0:.1f}s")
    return visible


def compute_viewshed_rect(site_x, site_y, site_height, buildings, radius, cell_size):
    grid_x, grid_y, dist = build_grid(site_x, site_y, radius, cell_size)
    in_radius = dist <= radius
    visible = np.full(grid_x.shape, -1, dtype=np.int8)   # -1 = outside AOI
    visible[in_radius] = compute_visibility(
        site_x, site_y, site_height, buildings,
        grid_x[in_radius], grid_y[in_radius], dist[in_radius])
    return grid_x, grid_y, visible


def compute_viewshed_hex(site_x, site_y, site_height, buildings, radius, cell_size):
    hx, hy, hd, hex_r = build_hex_grid(site_x, site_y, radius, cell_size)
    visible = compute_visibility(site_x, site_y, site_height, buildings, hx, hy, hd)
    return hx, hy, hex_r, visible


# ---------------------------------------------------------------------------
# 4. Plot the result
# ---------------------------------------------------------------------------

def _plot_common(ax, site_x, site_y, site_height, radius, buildings, title, legend_elems=None):
    """Buildings, heritage-site marker, legend, scalebar, north arrow, axes —
    everything shared between the rectangle- and hexagon-grid renders."""
    aoi_box = box(site_x - radius - 20, site_y - radius - 20,
                  site_x + radius + 20, site_y + radius + 20)
    for b in buildings:
        if not b["geom"].intersects(aoi_box):
            continue
        shade = min(0.85, 0.15 + b["height"] / 250 * 0.7)
        color = (1 - shade * 0.7,) * 3
        polys = [b["geom"]] if b["geom"].geom_type == "Polygon" else list(b["geom"].geoms)
        for p in polys:
            xs, ys = p.exterior.xy
            ax.fill(xs, ys, facecolor=color, edgecolor="#5f5e5a", linewidth=0.3, zorder=2)
            if p.area >= MIN_LABEL_AREA:
                label_pt = p.representative_point()
                ax.text(label_pt.x, label_pt.y, f"{b['height']:.0f}",
                        ha="center", va="center", fontsize=4, color="#26215C", zorder=4)

    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(site_x + radius * np.cos(theta), site_y + radius * np.sin(theta),
            linestyle="--", linewidth=1, color="#444441", zorder=3)

    ax.plot(site_x, site_y, marker="o", markersize=10, color="#E24B4A",
            markeredgecolor="white", markeredgewidth=1, zorder=5)
    ax.annotate(f"Heritage site\n({site_height:.0f} m)", (site_x, site_y),
                textcoords="offset points", xytext=(12, 12), fontsize=10,
                color="#26215C", fontweight="bold", zorder=6)

    ax.set_xlim(site_x - radius - 20, site_x + radius + 20)
    ax.set_ylim(site_y - radius - 20, site_y + radius + 20)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(title, fontsize=12, fontweight="bold")

    if legend_elems is None:
        legend_elems = [
            mpatches.Patch(facecolor="#5DCAA5", alpha=0.85, label="Visible"),
            mpatches.Patch(facecolor="#F09595", alpha=0.85, label="Not visible (blocked)"),
            mpatches.Patch(facecolor="#888780", label="Surrounding buildings (darker = taller)"),
            plt.Line2D([0], [0], linestyle="--", color="#444441", label=f"Study radius ({radius:.0f} m)"),
        ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=9, framealpha=0.9)

    scalebar_len = 50
    sb_x0, sb_y0 = site_x - radius - 10, site_y - radius - 5
    ax.plot([sb_x0, sb_x0 + scalebar_len], [sb_y0, sb_y0], color="black", linewidth=3, zorder=10)
    ax.text(sb_x0 + scalebar_len / 2, sb_y0 + 8, "50 m", ha="center", fontsize=9, zorder=10)

    na_x, na_y = site_x + radius - 10, site_y + radius - 15
    ax.annotate("N", xy=(na_x, na_y + 25), xytext=(na_x, na_y),
                arrowprops=dict(facecolor="black", width=3, headwidth=10, headlength=10),
                ha="center", fontsize=11, fontweight="bold", zorder=10)


def _save_plot(fig, output_png):
    plt.tight_layout()
    #plt.show()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_png}")


def _draw_rect_grid(ax, grid_x, grid_y, visible, cell_size):
    half = cell_size / 2
    x_edges = np.append(grid_x[0, :] - half, grid_x[0, -1] + half)
    y_edges = np.append(grid_y[:, 0] - half, grid_y[-1, 0] + half)

    cmap = ListedColormap(["#F1EFE8", "#F09595", "#5DCAA5"])  # outside / blocked / visible
    ax.pcolormesh(x_edges, y_edges, visible + 1, cmap=cmap, vmin=0, vmax=2,
                  shading="flat", alpha=0.85, zorder=1,
                  edgecolors=GRID_EDGE_COLOR, linewidth=0.3)


def plot_result_rect(grid_x, grid_y, visible, site_x, site_y, site_height, radius,
                      cell_size, buildings, output_png):
    fig, ax = plt.subplots(figsize=(9, 9))

    _draw_rect_grid(ax, grid_x, grid_y, visible, cell_size)

    _plot_common(ax, site_x, site_y, site_height, radius, buildings,
                 f"Viewshed analysis (radius {radius:.0f} m, grid {cell_size:.0f} m)")
    _save_plot(fig, output_png)


def plot_result_hex(hx, hy, hex_r, visible, site_x, site_y, site_height, radius,
                     cell_size, buildings, output_png):
    fig, ax = plt.subplots(figsize=(9, 9))

    colors = np.where(visible == 1, "#5DCAA5", "#F09595")  # visible / blocked
    hexagons = [mpatches.RegularPolygon((x, y), numVertices=6, radius=hex_r,
                                         orientation=-np.pi / 2)  # flat-top, matches build_hex_grid
                for x, y in zip(hx, hy)]
    ax.add_collection(PatchCollection(hexagons, facecolor=colors, edgecolor=GRID_EDGE_COLOR,
                                       linewidth=0.3, alpha=0.85, zorder=1))

    _plot_common(ax, site_x, site_y, site_height, radius, buildings,
                 f"Viewshed analysis (radius {radius:.0f} m, hex grid {cell_size:.0f} m)")
    _save_plot(fig, output_png)


def plot_sightlines(grid_x, grid_y, visible, site_x, site_y, site_height, radius,
                     cell_size, buildings, output_png):
    """Draw the rectangle grid plus a line-of-sight segment (P1 = heritage
    site -> grid cell centre) for every cell where the site is actually
    visible, i.e. no building blocks that sightline."""
    fig, ax = plt.subplots(figsize=(9, 9))

    _draw_rect_grid(ax, grid_x, grid_y, visible, cell_size)

    vis_mask = visible == 1
    ox, oy = grid_x[vis_mask], grid_y[vis_mask]
    # same row-major flat index used as the PK in export_geojson
    grid_id = np.arange(visible.size).reshape(visible.shape)[vis_mask]

    p1 = np.column_stack([np.full(ox.size, site_x), np.full(ox.size, site_y)])
    p2 = np.column_stack([ox, oy])
    segments = np.stack([p1, p2], axis=1)   # shape (N, 2, 2): N segments of 2 points each

    ax.add_collection(LineCollection(segments, colors="#3C3489", linewidths=0.6,
                                      alpha=0.7, zorder=1.5))

    for gid, x, y in zip(grid_id, ox, oy):
        ax.text(x, y, str(gid), ha="center", va="center", fontsize=6,
                color="#1B1B1B", fontweight="bold", zorder=6)

    legend_elems = [
        plt.Line2D([0], [0], color="#3C3489", linewidth=2, label="Visible sightline"),
        mpatches.Patch(facecolor="#5DCAA5", alpha=0.85, label="Visible cell"),
        mpatches.Patch(facecolor="#F09595", alpha=0.85, label="Not visible cell (blocked)"),
        mpatches.Patch(facecolor="#888780", label="Surrounding buildings (darker = taller)"),
        plt.Line2D([0], [0], linestyle="--", color="#444441", label=f"Study radius ({radius:.0f} m)"),
    ]
    _plot_common(ax, site_x, site_y, site_height, radius, buildings,
                 f"Heritage sightlines - visible only (radius {radius:.0f} m, "
                 f"grid {cell_size:.0f} m)", legend_elems=legend_elems)
    _save_plot(fig, output_png)


# ---------------------------------------------------------------------------
# 5. Export the grid result as GeoJSON (WGS84 points, for use in any GIS)
# ---------------------------------------------------------------------------

def export_geojson(grid_x, grid_y, visible, cell_size, to_wgs, output_path):
    mask = visible != -1
    # grid_id = flat row-major index into the full grid -> stable per-cell PK,
    # unaffected by how many cells later get filtered in/out of the AOI
    grid_id = np.arange(visible.size).reshape(visible.shape)[mask]
    lons, lats = to_wgs.transform(grid_x[mask], grid_y[mask])   # all points in one call
    features = [{
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [round(lon, 7), round(lat, 7)]},
        "properties": {"grid_id": int(gid), "visible": int(v), "cell_size_m": cell_size},
    } for gid, lon, lat, v in zip(grid_id, lons, lats, visible[mask])]

    fc = {"type": "FeatureCollection",
          "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
          "features": features}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)

    n_vis = sum(1 for feat in features if feat["properties"]["visible"] == 1)
    n_tot = len(features)
    pct = f"{100 * n_vis / n_tot:.1f}%" if n_tot else "n/a"
    print(f"Saved {output_path}  ({n_vis}/{n_tot} cells visible, {pct})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    to_utm = Transformer.from_crs("EPSG:4326", UTM_EPSG, always_xy=True)
    to_wgs = Transformer.from_crs(UTM_EPSG, "EPSG:4326", always_xy=True)

    site_x, site_y = to_utm.transform(SITE_LON, SITE_LAT)

    all_buildings = load_buildings(GEOJSON_PATH, to_utm)
    site_pt = Point(site_x, site_y)
    buildings = [b for b in all_buildings
                 if b["geom"].distance(site_pt) <= RADIUS + BUILDING_SEARCH_MARGIN]
    print(f"Loaded {len(all_buildings)} buildings, {len(buildings)} within analysis range")

    grid_x, grid_y, visible_rect = compute_viewshed_rect(
        site_x, site_y, SITE_HEIGHT, buildings, RADIUS, CELL_SIZE)
    plot_result_rect(grid_x, grid_y, visible_rect, site_x, site_y, SITE_HEIGHT, RADIUS,
                      CELL_SIZE, buildings, OUTPUT_PNG_RECT)

    hx, hy, hex_r, visible_hex = compute_viewshed_hex(
        site_x, site_y, SITE_HEIGHT, buildings, RADIUS, CELL_SIZE)
    plot_result_hex(hx, hy, hex_r, visible_hex, site_x, site_y, SITE_HEIGHT, RADIUS,
                     CELL_SIZE, buildings, OUTPUT_PNG_HEX)

    plot_sightlines(grid_x, grid_y, visible_rect, site_x, site_y, SITE_HEIGHT, RADIUS,
                     CELL_SIZE, buildings, OUTPUT_PNG_LINE)

    export_geojson(grid_x, grid_y, visible_rect, CELL_SIZE, to_wgs, OUTPUT_GEOJSON)


if __name__ == "__main__":
    main()
