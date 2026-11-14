"""
NAPOLEONIC - interactive result map.

Every detected layer is reprojected from the satellite's UTM grid into WGS84 (so it lines up exactly
with web basemaps), upsampled, smoothed and turned into a transparent image overlay. The overlays
are added to a Leaflet map on top of high-resolution basemap imagery, with layer toggles,
a measuring tool, fullscreen and a legend.
"""
import base64
import io
from concurrent.futures import ThreadPoolExecutor

import folium
import numpy as np
from folium import plugins
from PIL import Image, features as pil_features
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy import ndimage as ndi

import napoleonic as nap

WEBP_OK = pil_features.check("webp")


# --------------------------------------------------------------------------- #
# Reprojection to WGS84
# --------------------------------------------------------------------------- #
class Grid:
    """Destination WGS84 grid covering the scene, optionally upsampled for smoother overlays."""

    def __init__(self, sc, shape, max_px=1400):
        h, w = shape
        left, bottom, right, top = array_bounds(h, w, sc.transform)
        self.up = int(np.clip(max_px // max(h, w), 1, 4))
        self.transform, self.w, self.h = calculate_default_transform(
            sc.crs, "EPSG:4326", w, h, left, bottom, right, top,
            dst_width=w * self.up, dst_height=h * self.up)
        self.src_transform, self.src_crs = sc.transform, sc.crs
        west, north = self.transform.c, self.transform.f
        east = west + self.transform.a * self.w
        south = north + self.transform.e * self.h
        self.bounds = [[south, west], [north, east]]
        self.centre = ((south + north) / 2, (west + east) / 2)

    def warp(self, arr, resampling=Resampling.bilinear):
        dst = np.zeros((self.h, self.w), "float32")
        reproject(source=np.ascontiguousarray(arr, dtype="float32"), destination=dst,
                  src_transform=self.src_transform, src_crs=self.src_crs,
                  dst_transform=self.transform, dst_crs="EPSG:4326", resampling=resampling)
        return dst


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #
def _hex_rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _uri(rgba, lossy=False):
    im = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    if lossy and WEBP_OK:
        im.save(buf, "WEBP", quality=88, method=1)
        mime = "image/webp"
    else:
        im.save(buf, "PNG", compress_level=6)
        mime = "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def _soft(grid, mask):
    """Warp a boolean mask and return a smooth 0-1 coverage (anti-aliased edges)."""
    return np.clip((grid.warp(mask.astype("float32")) - 0.30) / 0.30, 0, 1)


def mask_layer(grid, mask, color, alpha, casing=False):
    core = _soft(grid, mask) * alpha
    rgb = np.zeros((grid.h, grid.w, 3), "float32")
    rgb[:] = _hex_rgb(color)
    if casing:  # dark halo around thin lines so they stay readable on any background
        halo = _soft(grid, ndi.binary_dilation(mask, iterations=1)) * 0.6 * alpha
        out_a = core + halo * (1 - core)
        rgb = rgb * (core / np.maximum(out_a, 1e-6))[..., None]
        core = out_a
    rgba = np.zeros((grid.h, grid.w, 4), "uint8")
    rgba[..., :3] = rgb.astype("uint8")
    rgba[..., 3] = (np.clip(core, 0, 1) * 255).astype("uint8")
    return rgba, False


def composite_layer(grid, bands, scale=0.28, gamma=0.8, sharpen=True):
    """3-band composite -> RGBA overlay (cubic upsample + mild unsharp mask)."""
    chans = [grid.warp(b, Resampling.cubic) for b in bands]
    cover = grid.warp(np.ones_like(bands[0]), Resampling.bilinear)
    img = np.clip(np.stack(chans, -1) / scale, 0, 1) ** gamma
    if sharpen and grid.up > 1:
        blur = np.stack([ndi.gaussian_filter(img[..., i], 0.45 * grid.up) for i in range(3)], -1)
        img = np.clip(img + 0.7 * (img - blur), 0, 1)
    rgba = np.zeros((grid.h, grid.w, 4), "uint8")
    rgba[..., :3] = (img * 255).astype("uint8")
    rgba[..., 3] = (np.clip((cover - 0.4) / 0.2, 0, 1) * 255).astype("uint8")
    return rgba, True


def ndvi_layer(grid, ndvi, valid):
    import matplotlib
    cmap = matplotlib.colormaps["RdYlGn"]
    n = grid.warp(np.clip((ndvi + 0.2) / 1.0, 0, 1))
    ok = grid.warp(valid.astype("float32")) > 0.5
    rgba = (cmap(n) * 255).astype("uint8")
    rgba[..., 3] = np.where(ok & (grid.warp(np.ones_like(ndvi)) > 0.5), 235, 0).astype("uint8")
    return rgba, True


# --------------------------------------------------------------------------- #
# Layer preparation (expensive: do once per analysis)
# --------------------------------------------------------------------------- #
OVERLAYS = [  # key, label, colour, alpha, dark casing for thin lines, shown by default
    ("water", "🟦 Water", "#38bdf8", 0.55, False, True),
    ("woodland", "🟩 Woodland", "#22c55e", 0.50, False, True),
    ("built", "🟥 Built-up area", "#ef4444", 0.42, False, True),
    ("lines", "🟨 Tree lines", "#facc15", 0.95, False, True),
    ("roads", "⬜ Candidate roads (imagery)", "#ffffff", 0.95, True, True),
    ("osm_roads", "🟦 OSM roads", "#22d3ee", 0.90, True, True),
    ("osm_tall", "🟪 Multi-storey buildings (OSM tags)", "#e879f9", 0.95, False, True),
    ("invalid", "⬛ Cloud / no data", "#111827", 0.60, False, True),
]


def prepare_layers(sc, res, max_px=1800):
    B = sc.bands
    grid = Grid(sc, B["B04"].shape, max_px)
    masks = dict(res)
    masks["invalid"] = ~res["valid"]

    jobs = [  # (group, label, z-index or default-on flag, callable returning (rgba, lossy))
        ("base", "🛰️ Sentinel-2 true colour", 3, lambda: composite_layer(grid, [B["B04"], B["B03"], B["B02"]])),
        ("base", "🌿 False colour (SWIR: vegetation green, built-up pink)", 2,
         lambda: composite_layer(grid, [B["B11"], B["B08"], B["B04"]], scale=0.45)),
        ("base", "🌱 Vegetation health (NDVI)", 1, lambda: ndvi_layer(grid, res["ndvi"], res["valid"])),
    ]
    for key, label, color, alpha, casing, on in OVERLAYS:
        m = masks.get(key)
        if m is not None and m.any():
            jobs.append(("overlay", label, on, (lambda m=m, c=color, a=alpha, k=casing: mask_layer(grid, m, c, a, k))))
    # GDAL warping runs strictly one at a time (not guaranteed thread-safe); only PNG/WebP encoding is parallel
    images = [j[3]() for j in jobs]
    with ThreadPoolExecutor(max_workers=4) as ex:
        uris = list(ex.map(lambda im: _uri(im[0], lossy=im[1]), images))
    layers = {"base": [], "overlay": []}
    for (group, label, extra, _), uri in zip(jobs, uris):
        layers[group].append((label, uri, extra))
    return {"bounds": grid.bounds, "centre": grid.centre, "layers": layers, "meta": sc.meta,
            "has_osm": res["osm_roads"] is not None}


# --------------------------------------------------------------------------- #
# Map construction (cheap: rebuilt when opacity changes)
# --------------------------------------------------------------------------- #
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"


def build_result_map(prep, opacity=0.85, demo=False, controls=True):
    s, w = prep["bounds"][0]
    n, e = prep["bounds"][1]
    m = folium.Map(location=prep["centre"], zoom_start=14, tiles=None, control_scale=True, max_zoom=20)

    if demo:  # synthetic scene: no real place, so no basemap
        m.get_root().header.add_child(folium.Element(
            "<style>.leaflet-container{background:#d1d5db !important}</style>"))
    else:
        folium.TileLayer(ESRI.format("World_Imagery"), attr="Esri, Maxar, Earthstar Geographics",
                         name="Satellite basemap (Esri, high-res)", max_zoom=20, max_native_zoom=19).add_to(m)
        folium.TileLayer("OpenStreetMap", name="Streets (OSM)", show=False, max_zoom=20).add_to(m)
        folium.TileLayer(ESRI.format("World_Topo_Map"), attr="Esri", name="Topographic",
                         show=False, max_zoom=20, max_native_zoom=19).add_to(m)
        folium.TileLayer(ESRI.format("Reference/World_Boundaries_and_Places"), attr="Esri",
                         name="Place names", overlay=True, show=True, max_zoom=20, max_native_zoom=19).add_to(m)

    for label, uri, z in prep["layers"]["base"]:
        on = demo and z == 3  # in demo mode the scene itself is the only background
        folium.raster_layers.ImageOverlay(image=uri, bounds=prep["bounds"], name=label, opacity=1.0,
                                          overlay=True, control=True, show=on, interactive=False,
                                          zindex=z).add_to(m)
    for i, (label, uri, on) in enumerate(prep["layers"]["overlay"]):
        folium.raster_layers.ImageOverlay(image=uri, bounds=prep["bounds"], name=label, opacity=opacity,
                                          overlay=True, control=True, show=on, interactive=False,
                                          zindex=10 + i).add_to(m)

    folium.Rectangle(prep["bounds"], color="#facc15", weight=2, dash_array="6 6", fill=False,
                     name="Scan area outline").add_to(m)
    if controls:
        plugins.Fullscreen(position="topleft").add_to(m)
        plugins.MeasureControl(position="topleft", primary_length_unit="meters",
                               primary_area_unit="sqkilometers").add_to(m)
        plugins.MousePosition(position="bottomright", num_digits=5).add_to(m)
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    m.fit_bounds([[s, w], [n, e]])

    meta = prep["meta"]
    cc = meta.get("cloud_cover_pct")
    note = ("Synthetic test scene, not a real place." if demo else
            "Basemap imagery may be older than the scan. Detected layers come from the Sentinel-2 scene.")
    legend = f"""
    <div style="position:absolute;bottom:62px;left:10px;z-index:9999;background:rgba(17,24,39,.9);color:#f9fafb;
                padding:10px 12px;border-radius:8px;font:12px/1.5 system-ui,sans-serif;max-width:270px;
                box-shadow:0 2px 10px rgba(0,0,0,.4)">
      <b>NAPOLEONIC scan</b><br>
      Acquired: {str(meta.get('acquired', ''))[:10]}{f' · cloud {cc:.0f}%' if cc is not None else ''}<br>
      <span style="opacity:.8">Scene: {meta.get('scene_id', '')}</span><br>
      <span style="opacity:.75;font-size:11px">10 m resolution. Line widths are exaggerated for visibility. {note}</span>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    return m
