#!/usr/bin/env python3
"""
NAPOLEONIC - rapid terrain & infrastructure mapping from Sentinel-2

Takes the newest low-cloud Sentinel-2 L2A scene over an area of interest and
extracts: tree lines / woodland, built-up areas, and candidate roads. It can
fuse OpenStreetMap roads and buildings (with height tags) and writes a map,
a GeoJSON layer pack and a plain-language report.

    python napoleonic.py --demo                                   # synthetic test scene
    python napoleonic.py --lat 48.74 --lon 37.58 --radius-km 3    # live Sentinel-2
    python napoleonic.py --lat 48.74 --lon 37.58 --osm            # + OSM fusion

Dependencies: numpy scipy scikit-image matplotlib rasterio pyproj pystac-client requests

IMPORTANT LIMITS (also written into every report)
 * Sentinel-2 is 10 m/pixel. It cannot measure building height, see vehicles
   or people, or resolve narrow tracks. Heights come from OSM tags only.
 * Imagery is 2-5 days old at best and cloud can hide the ground.
 * All thresholds are heuristics and need local calibration and validation.
 * This is a decision-support aid, not a substitute for verified ground truth.
"""
import argparse
import datetime as dt
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import filters, measure, morphology
from rasterio import features as rfeatures
from rasterio.transform import Affine, from_origin
from rasterio.warp import transform_geom

PIX = 10.0  # metres per pixel (Sentinel-2 10 m bands)


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    ndvi_veg: float = 0.45          # vegetation threshold
    canopy_swir: float = 0.18       # tree canopy is dark in SWIR (crops are brighter)
    adaptive_canopy: bool = True    # pick the SWIR split automatically (Otsu) within vegetation
    canopy_swir_lo: float = 0.10    # bounds for the automatic split
    canopy_swir_hi: float = 0.22
    canopy_red: float = 0.035       # very dark red -> dense canopy
    min_canopy_px: int = 4
    line_max_width_px: float = 5.0  # mean width <= 50 m -> "linear" canopy
    line_min_len_px: int = 6        # skeleton >= 60 m
    ndbi_built: float = 0.0
    built_tex: float = 0.02         # visible-band local std (towns are heterogeneous)
    built_density: float = 0.30
    road_hi_pct: float = 99.0       # hysteresis: strong ridge seeds (percentile)
    road_lo_pct: float = 96.5       # hysteresis: weak ridge pixels connected to seeds
    road_min_len_px: int = 10       # >= 100 m
    tall_m: float = 12.0            # OSM height treated as "multi-storey"


@dataclass
class Scene:
    bands: dict                      # B02,B03,B04,B08,B11 reflectance 0-1, same grid
    scl: np.ndarray                  # scene classification layer
    transform: Affine
    crs: str
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Acquisition
# --------------------------------------------------------------------------- #
def bbox_from_point(lat, lon, radius_km):
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * math.cos(math.radians(lat)))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def _catalog():
    from pystac_client import Client
    return Client.open("https://earth-search.aws.element84.com/v1")


def search_scenes(bbox, max_cloud=30, lookback_days=30):
    """List available Sentinel-2 L2A scenes (newest first) as plain dicts."""
    end = dt.date.today()
    start = end - dt.timedelta(days=lookback_days)
    items = _catalog().search(
        collections=["sentinel-2-l2a"], bbox=bbox, datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}}, max_items=60).items()
    out = [{"id": i.id, "datetime": i.datetime.isoformat(), "date": i.datetime.strftime("%Y-%m-%d"),
            "cloud": float(i.properties.get("eo:cloud_cover", 0.0))} for i in items]
    return sorted(out, key=lambda d: d["datetime"], reverse=True)


def fetch_scene(bbox, max_cloud=30, lookback_days=30, item_id=None):
    """Sentinel-2 L2A scene (newest low-cloud one, or a chosen id) from public AWS COGs."""
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    from rasterio.enums import Resampling

    cat = _catalog()
    if item_id:
        found = list(cat.search(collections=["sentinel-2-l2a"], ids=[item_id]).items())
        if not found:
            raise SystemExit(f"Scene {item_id} not found.")
        item = found[0]
    else:
        end = dt.date.today()
        start = end - dt.timedelta(days=lookback_days)
        items = list(cat.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=f"{start}/{end}",
                                query={"eo:cloud_cover": {"lt": max_cloud}}, max_items=50).items())
        if not items:
            raise SystemExit("No Sentinel-2 scene under the cloud limit. Raise the cloud limit or search further back.")
        item = max(items, key=lambda i: i.datetime)
    keys = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir", "B11": "swir16", "SCL": "scl"}

    def read(asset, shape=None):
        with rasterio.open(item.assets[asset].href) as src:
            b = transform_bounds("EPSG:4326", src.crs, *bbox)
            win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
            if shape is None:
                return src.read(1, window=win), src.window_transform(win), str(src.crs)
            rs = Resampling.nearest if asset == "scl" else Resampling.bilinear
            return src.read(1, window=win, out_shape=shape, resampling=rs), None, None

    red, transform, crs = read(keys["B04"])
    shape = red.shape
    raw = {"B04": red}
    for band in ("B02", "B03", "B08", "B11"):
        raw[band] = read(keys[band], shape)[0]
    scl = read(keys["SCL"], shape)[0]

    offset = 0.0 if item.properties.get("earthsearch:boa_offset_applied") else 1000.0
    bands = {k: np.clip((v.astype("float32") - offset) / 10000.0, 0, 1) for k, v in raw.items()}
    meta = {
        "source": "Sentinel-2 L2A (earth-search)",
        "scene_id": item.id,
        "acquired": item.datetime.isoformat(),
        "cloud_cover_pct": item.properties.get("eo:cloud_cover"),
        "bbox": bbox,
    }
    return Scene(bands, scl, transform, crs, meta)


def synthetic_scene(size=600, seed=7):
    """A made-up Eastern-European-style town scene so the pipeline can be tested offline."""
    rng = np.random.default_rng(seed)
    H = W = size
    yy, xx = np.mgrid[0:H, 0:W]
    # ground truth masks used to paint the scene
    refl = {k: np.zeros((H, W), "float32") for k in ("B02", "B03", "B04", "B08", "B11")}

    def paint(mask, b02, b03, b04, b08, b11, noise=0.004):
        for k, v in zip(("B02", "B03", "B04", "B08", "B11"), (b02, b03, b04, b08, b11)):
            refl[k][mask] = v + rng.normal(0, noise, mask.sum())

    # field mosaic (bare + cropped)
    blocks = np.zeros((H, W), int)
    for _ in range(45):
        x0, y0 = rng.integers(0, W - 80), rng.integers(0, H - 80)
        blocks[y0:y0 + rng.integers(60, 160), x0:x0 + rng.integers(60, 160)] = rng.integers(1, 4)
    paint(np.ones((H, W), bool), 0.07, 0.10, 0.13, 0.24, 0.24)  # default: bare soil
    paint(blocks == 1, 0.04, 0.08, 0.05, 0.40, 0.22)              # smooth crop
    paint(blocks == 2, 0.06, 0.09, 0.09, 0.30, 0.25)              # sparse crop

    # woodland block
    wood = (xx - 470) ** 2 + (yy - 120) ** 2 < 70 ** 2
    paint(wood, 0.025, 0.045, 0.025, 0.30, 0.15, noise=0.03)

    # tree lines (shelterbelts) along field edges
    treelines = np.zeros((H, W), bool)
    treelines[250:254, 60:330] = True
    treelines[60:330, 330:333] = True
    treelines[470:473, 120:380] = True
    paint(treelines, 0.025, 0.045, 0.025, 0.32, 0.15, noise=0.025)

    # river
    river = np.abs(yy - (400 + 40 * np.sin(xx / 70.0))) < 5
    paint(river, 0.06, 0.07, 0.04, 0.03, 0.01)

    # town: heterogeneous roofs, gardens, asphalt grid
    town = (xx > 200) & (xx < 360) & (yy > 280) & (yy < 440)
    paint(town, 0.05, 0.08, 0.10, 0.28, 0.22)
    for by in range(285, 435, 14):
        for bx in range(205, 355, 14):
            if rng.random() < 0.72:
                h, w = rng.integers(4, 9), rng.integers(4, 9)
                paint((yy >= by) & (yy < by + h) & (xx >= bx) & (xx < bx + w),
                      0.16, 0.17, 0.19 + rng.random() * 0.1, 0.27, 0.36 + rng.random() * 0.1, 0.01)
    roads = np.zeros((H, W), bool)
    for ry in range(280, 441, 28):
        roads[ry:ry + 2, 200:360] = True
    for rx in range(200, 361, 28):
        roads[280:440, rx:rx + 2] = True
    roads[350:352, 0:200] = True      # road in from the west
    roads[350:352, 360:600] = True    # road out to the east
    roads &= ~river
    paint(roads, 0.05, 0.055, 0.055, 0.10, 0.09, noise=0.004)

    scl = np.full((H, W), 4, "uint8")
    scl[river] = 6
    cloud = (xx - 80) ** 2 + (yy - 520) ** 2 < 35 ** 2
    for k in refl:
        refl[k][cloud] = 0.75
    scl[cloud] = 9
    bands = {k: np.clip(ndi.gaussian_filter(v, 0.5), 0, 1) for k, v in refl.items()}
    transform = from_origin(500000, 406000, PIX, PIX)  # neutral spot, deliberately not a real place
    meta = {
        "source": "SYNTHETIC test scene",
        "scene_id": "demo",
        "acquired": (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=3)).isoformat(),
        "cloud_cover_pct": round(100 * cloud.mean(), 1),
        "bbox": None,
    }
    return Scene(bands, scl, transform, "EPSG:32629", meta)


# --------------------------------------------------------------------------- #
# OpenStreetMap fusion (optional) - roads and buildings with height tags
# --------------------------------------------------------------------------- #
def fetch_osm(bbox):
    import requests
    w, s, e, n = bbox
    q = (f"[out:json][timeout:60];(way['highway']({s},{w},{n},{e});"
         f"way['building']({s},{w},{n},{e}););out geom tags;")
    r = requests.post("https://overpass-api.de/api/interpreter", data={"data": q}, timeout=120)
    r.raise_for_status()
    roads, buildings = [], []
    for el in r.json().get("elements", []):
        pts = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
        tags = el.get("tags", {})
        if len(pts) < 2:
            continue
        if "highway" in tags:
            roads.append({"type": "LineString", "coordinates": pts})
        elif "building" in tags and len(pts) >= 4:
            h = None
            try:
                if "height" in tags:
                    h = float(str(tags["height"]).replace("m", "").strip())
                elif "building:levels" in tags:
                    h = float(tags["building:levels"]) * 3.0
            except ValueError:
                pass
            buildings.append(({"type": "Polygon", "coordinates": [pts]}, h))
    return roads, buildings


def rasterize_geoms(geoms, scene, shape):
    if not geoms:
        return np.zeros(shape, bool)
    proj = [transform_geom("EPSG:4326", scene.crs, g) for g in geoms]
    return rfeatures.rasterize(proj, out_shape=shape, transform=scene.transform,
                               fill=0, default_value=1, all_touched=True).astype(bool)


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def nd(a, b):
    return (a - b) / (a + b + 1e-6)


def drop_small(mask, n):
    """Remove connected components smaller than n pixels."""
    lab, cnt = ndi.label(mask, structure=np.ones((3, 3)))
    if cnt == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= n
    keep[0] = False
    return keep[lab]


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def local_std(a, k):
    m = ndi.uniform_filter(a, k)
    return np.sqrt(np.maximum(ndi.uniform_filter(a * a, k) - m * m, 0))


def analyse(sc: Scene, p: Params, osm=None):
    B = sc.bands
    shape = B["B04"].shape
    valid = ~np.isin(sc.scl, [0, 1, 3, 8, 9, 10])  # no-data, saturated, shadow, cloud, cirrus
    ndvi = nd(B["B08"], B["B04"])
    ndbi = nd(B["B11"], B["B08"])
    mndwi = nd(B["B03"], B["B11"])
    water = ((mndwi > 0.1) | (sc.scl == 6)) & valid

    # --- canopy -> tree lines vs woodland --------------------------------- #
    veg = (ndvi > p.ndvi_veg) & ~water & valid
    near_water = ndi.binary_dilation(water, iterations=2)
    swir_thr = p.canopy_swir
    if p.adaptive_canopy and veg.sum() > 2000:
        try:  # data-driven split between dark canopy and brighter crops
            swir_thr = float(np.clip(filters.threshold_otsu(B["B11"][veg]), p.canopy_swir_lo, p.canopy_swir_hi))
        except Exception:
            swir_thr = p.canopy_swir
    canopy = veg & ~near_water & ((B["B11"] < swir_thr) | (B["B04"] < p.canopy_red))
    canopy = ndi.binary_closing(canopy, structure=disk(1))
    canopy = drop_small(canopy, p.min_canopy_px)
    lab = measure.label(canopy, connectivity=2)
    lines = np.zeros(shape, bool)
    n_lines = 0
    if lab.max():
        ids = np.arange(1, lab.max() + 1)
        skel = morphology.skeletonize(canopy)
        area = ndi.sum(canopy, lab, ids)
        skl = ndi.sum(skel, lab, ids)
        linear = (area / np.maximum(skl, 1) <= p.line_max_width_px) & (skl >= p.line_min_len_px)
        lines = np.isin(lab, ids[linear])
        n_lines = int(linear.sum())
        line_len_m = float(skl[linear].sum() * PIX * 1.1)
    else:
        line_len_m = 0.0
    woodland = canopy & ~lines

    # --- built-up --------------------------------------------------------- #
    vis = (B["B02"] + B["B03"] + B["B04"]) / 3
    vtex = local_std(vis, 5)
    built_raw = (ndbi > p.ndbi_built) & (ndvi < 0.25) & (vtex > p.built_tex) & ~water & valid
    dens = ndi.uniform_filter(built_raw.astype("float32"), 7)
    core = drop_small(built_raw & (dens > p.built_density), 6)
    footprint = ndi.uniform_filter(built_raw.astype("float32"), 21) > 0.06   # ~210 m window
    built = ndi.binary_fill_holes(footprint) & ~water & valid
    built = drop_small(built, 200)
    urban_zone = built

    # --- candidate roads (ridge filter on visible brightness) ------------- #
    v = ndi.gaussian_filter(0.5 * vis + 0.5 * B["B11"], 0.7)
    ridge = np.maximum(filters.sato(v, sigmas=[1, 1.5], black_ridges=False),
                       filters.sato(v, sigmas=[1, 1.5], black_ridges=True))
    ridge[ndi.binary_dilation(~valid | water, iterations=5) | ndi.binary_dilation(canopy, iterations=2)
          | ndi.binary_dilation(core, iterations=3)] = 0
    r = ridge[valid]
    lo, hi = np.percentile(r, [p.road_lo_pct, p.road_hi_pct])
    rskel = morphology.skeletonize(filters.apply_hysteresis_threshold(ridge, lo, hi))
    rl = measure.label(rskel, connectivity=2)
    sizes = np.bincount(rl.ravel())
    keep = sizes >= p.road_min_len_px
    keep[0] = False
    roads = keep[rl]
    # Streets inside dense settlements are not resolvable at 10 m (the filter just traces building gaps),
    # so imagery-derived roads are only reported outside built-up areas. OSM covers urban streets.
    roads = drop_small(roads & ~ndi.binary_erosion(built, iterations=2), p.road_min_len_px)

    out = {"valid": valid, "water": water, "canopy": canopy, "lines": lines,
           "woodland": woodland, "built": built, "roads": roads, "urban_zone": urban_zone,
           "osm_roads": None, "osm_buildings": None, "osm_tall": None, "ndvi": ndvi}

    # --- OSM fusion -------------------------------------------------------- #
    if osm:
        o_roads, o_b = osm
        out["osm_roads"] = rasterize_geoms(o_roads, sc, shape)
        out["osm_buildings"] = rasterize_geoms([g for g, _ in o_b], sc, shape)
        out["osm_tall"] = rasterize_geoms([g for g, h in o_b if h and h >= p.tall_m], sc, shape)
        out["osm_counts"] = {"roads": len(o_roads), "buildings": len(o_b),
                             "tall": sum(1 for _, h in o_b if h and h >= p.tall_m),
                             "with_height": sum(1 for _, h in o_b if h)}

    px_km2 = (PIX / 1000) ** 2
    stats = {
        "aoi_km2": round(shape[0] * shape[1] * px_km2, 2),
        "canopy_swir_used": round(swir_thr, 3),
        "valid_pct": round(100 * valid.mean(), 1),
        "water_km2": round(water.sum() * px_km2, 2),
        "treelines_count": n_lines,
        "treelines_length_km": round(line_len_m / 1000, 2),
        "woodland_km2": round(woodland.sum() * px_km2, 2),
        "builtup_km2": round(built.sum() * px_km2, 2),
        "candidate_road_km": round(roads.sum() * PIX * 1.1 / 1000, 2),
    }
    if osm:
        stats["osm_road_km"] = round(out["osm_roads"].sum() * PIX * 0.9 / 1000, 2)
        stats.update({f"osm_{k}": v for k, v in out["osm_counts"].items()})
    return out, stats


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def rgb(sc):
    img = np.dstack([sc.bands["B04"], sc.bands["B03"], sc.bands["B02"]])
    img = np.clip(img / 0.28, 0, 1) ** 0.8
    return img


def layer_to_geojson(mask, sc, name):
    feats = []
    for geom, val in rfeatures.shapes(mask.astype("uint8"), mask=mask, transform=sc.transform, connectivity=8):
        feats.append({"type": "Feature", "properties": {"layer": name},
                      "geometry": transform_geom(sc.crs, "EPSG:4326", geom)})
    return feats


def render_map(sc, res, path, max_zoom=4):
    """High-quality static map: upsampled + sharpened true colour, outlined layers, scale bar, north arrow."""
    import matplotlib.patheffects as pe
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch

    H, W = sc.bands["B04"].shape
    z = int(np.clip(1800 // max(H, W), 1, max_zoom))
    img = rgb(sc)
    big = img
    if z > 1:
        big = np.clip(np.stack([ndi.zoom(img[..., i], z, order=3) for i in range(3)], -1), 0, 1)
        blur = np.stack([ndi.gaussian_filter(big[..., i], 0.45 * z) for i in range(3)], -1)
        big = np.clip(big + 0.8 * (big - blur), 0, 1)   # unsharp mask
    ext = (0, W * PIX / 1000, H * PIX / 1000, 0)

    def up(mask):
        m = mask.astype("float32")
        return ndi.zoom(m, z, order=1) if z > 1 else m

    canvas = big.astype("float32").copy()   # all layers are blended into ONE image (fast to draw)
    outlines = []

    def blend(cov, color, alpha):
        a = (cov * alpha)[..., None]
        canvas[:] = canvas * (1 - a) + np.array(to_rgb(color), "float32") * a

    def area(mask, color, alpha):
        if mask is None or not mask.any():
            return
        a = up(mask)
        blend((a > 0.5).astype("float32"), color, alpha)
        outlines.append((a, color))

    def line(mask, color, casing=False, alpha=0.95):
        if mask is None or not mask.any():
            return
        if casing:  # dark halo keeps thin lines visible on any background
            blend(np.clip((up(ndi.binary_dilation(mask, iterations=1)) - 0.25) / 0.35, 0, 1), "#000000", 0.6)
        blend(np.clip((up(mask) - 0.25) / 0.35, 0, 1), color, alpha)

    area(res["water"], "#38bdf8", 0.35)
    area(res["woodland"], "#22c55e", 0.30)
    area(res["built"], "#ef4444", 0.28)
    line(res["lines"], "#facc15")
    line(res["roads"], "#ffffff", casing=True)
    line(res["osm_roads"], "#22d3ee", casing=True, alpha=0.9)
    area(res["osm_tall"], "#e879f9", 0.9)
    area(~res["valid"], "#111827", 0.55)

    fig, ax = plt.subplots(figsize=(13, 12))
    ax.imshow(np.clip(canvas, 0, 1), extent=ext, interpolation="lanczos")
    for a, color in outlines:
        if a.max() > 0.5:
            ax.contour(a, levels=[0.5], colors=[color], linewidths=1.1, extent=ext, origin="upper")

    # scale bar + north arrow
    width_m = W * PIX
    bar_m = max(b for b in (100, 200, 500, 1000, 2000, 5000) if b <= width_m * 0.28) if width_m >= 400 else 100
    x1 = ext[1] - 0.04 * ext[1]
    y1 = ext[2] - 0.05 * ext[2]
    halo = [pe.withStroke(linewidth=3, foreground="black")]
    ax.plot([x1 - bar_m / 1000, x1], [y1, y1], color="white", lw=4, solid_capstyle="butt", path_effects=halo)
    ax.text(x1 - bar_m / 2000, y1 - 0.012 * ext[2], f"{bar_m} m" if bar_m < 1000 else f"{bar_m // 1000} km",
            color="white", ha="center", va="bottom", fontsize=11, fontweight="bold", path_effects=halo)
    ax.annotate("N", xy=(0.955, 0.965), xytext=(0.955, 0.885), xycoords="axes fraction", color="white",
                ha="center", fontsize=15, fontweight="bold", path_effects=halo,
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5, path_effects=halo))

    legend = [Patch(color="#facc15", label="Tree lines"), Patch(color="#22c55e", label="Woodland"),
              Patch(color="#ef4444", label="Built-up area"), Patch(color="#ffffff", label="Candidate roads (imagery)"),
              Patch(color="#38bdf8", label="Water"), Patch(color="#111827", label="Cloud / no data")]
    if res["osm_roads"] is not None:
        legend += [Patch(color="#22d3ee", label="OSM roads"), Patch(color="#e879f9", label="Multi-storey (OSM tags)")]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=10, frameon=True, title="Layers")
    ax.set_xlabel("Distance east (km)")
    ax.set_ylabel("Distance south (km)")
    cc = sc.meta.get("cloud_cover_pct")
    ax.set_title(f"NAPOLEONIC  |  {sc.meta['source']}  |  acquired {sc.meta['acquired'][:10]}"
                 + (f"  |  scene cloud {cc:.0f}%" if cc is not None else ""), fontsize=12)
    fig.savefig(path, dpi=170, bbox_inches="tight", pil_kwargs={"compress_level": 3})
    plt.close(fig)


def write_report(sc, stats, path):
    acq = dt.datetime.fromisoformat(sc.meta["acquired"].replace("Z", "+00:00"))
    age = (dt.datetime.now(dt.timezone.utc) - acq).days
    lines = [
        "# NAPOLEONIC - terrain & infrastructure report", "",
        f"- **Source:** {sc.meta['source']} (`{sc.meta['scene_id']}`)",
        f"- **Acquired:** {sc.meta['acquired'][:16]} UTC  (**{age} days old**)",
        f"- **Scene cloud cover:** {sc.meta['cloud_cover_pct']}%   |   **Usable area:** {stats['valid_pct']}%",
        f"- **Area of interest:** {stats['aoi_km2']} km2 at 10 m/pixel", "",
        "## Extracted features",
        f"- Tree lines: **{stats['treelines_count']}** segments, about **{stats['treelines_length_km']} km** total",
        f"- Woodland: **{stats['woodland_km2']} km2**",
        f"- Built-up area: **{stats['builtup_km2']} km2**",
        f"- Candidate roads from imagery: about **{stats['candidate_road_km']} km**",
        f"- Water: {stats['water_km2']} km2",
    ]
    if "osm_roads" in stats:
        lines += ["", "## OpenStreetMap fusion",
                  f"- OSM roads: {stats['osm_roads']} ways, about {stats['osm_road_km']} km",
                  f"- OSM buildings: {stats['osm_buildings']} ({stats['osm_with_height']} with height data)",
                  f"- Multi-storey (>= 12 m by tag): **{stats['osm_tall']}**"]
    lines += ["", "## Confidence and limits",
              "- 10 m pixels: roads narrower than about 10 m, single buildings and any height are **not** measurable from Sentinel-2.",
              "- Tree lines depend on season: deciduous cover in winter is leaf-off, so NDVI under-detects it.",
              "- Bare soil can be mistaken for built-up; candidate roads include field boundaries.",
              "- Cloud and shadow areas are masked and hold no information.",
              "- Heuristic thresholds are uncalibrated. Validate against ground truth before relying on any number.",
              "- OSM data in conflict areas may be outdated or incomplete."]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Napoleonic: Sentinel-2 terrain & infrastructure mapper")
    ap.add_argument("--demo", action="store_true", help="run on a synthetic scene")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--radius-km", type=float, default=3.0)
    ap.add_argument("--max-cloud", type=float, default=30)
    ap.add_argument("--lookback", type=int, default=30, help="days to search back")
    ap.add_argument("--osm", action="store_true", help="fuse OpenStreetMap roads/buildings")
    ap.add_argument("--out", default="napoleonic_out")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = Params()

    if a.demo:
        sc = synthetic_scene()
        osm = None
    else:
        if a.lat is None or a.lon is None:
            ap.error("give --lat and --lon, or use --demo")
        if a.radius_km > 12:
            ap.error("radius above 12 km would download a lot of data; use <= 12")
        bbox = bbox_from_point(a.lat, a.lon, a.radius_km)
        print("Searching for the newest Sentinel-2 scene...")
        sc = fetch_scene(bbox, a.max_cloud, a.lookback)
        osm = None
        if a.osm:
            try:
                print("Fetching OpenStreetMap roads and buildings...")
                osm = fetch_osm(bbox)
            except Exception as e:  # network or API failure should not kill the run
                print(f"OSM fetch failed ({e}); continuing with imagery only.")

    print(f"Scene {sc.meta['scene_id']} acquired {sc.meta['acquired'][:10]}; analysing...")
    res, stats = analyse(sc, p, osm)

    render_map(sc, res, out / "map.png")
    write_report(sc, stats, out / "report.md")
    gj = []
    for name in ("lines", "woodland", "built", "roads", "osm_roads", "osm_tall"):
        if res.get(name) is not None and res[name].any():
            gj += layer_to_geojson(res[name], sc, name)
    (out / "features.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": gj}))
    (out / "summary.json").write_text(json.dumps({"meta": sc.meta, "stats": stats}, indent=2, default=str))
    try:
        import visuals
        prep = visuals.prepare_layers(sc, res)
        (out / "interactive_map.html").write_text(
            visuals.build_result_map(prep, demo=a.demo).get_root().render(), encoding="utf-8")
        extra = ", interactive_map.html"
    except Exception as e:  # folium/Pillow missing should not break the CLI
        extra = ""
        print(f"(interactive map skipped: {e})")
    print(json.dumps(stats, indent=2))
    print(f"Wrote map.png, report.md, features.geojson, summary.json{extra} to {out}/")


if __name__ == "__main__":
    main()
