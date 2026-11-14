"""
NAPOLEONIC - web interface.   Run with:   streamlit run app.py
Needs napoleonic.py and visuals.py in the same folder.
"""
import datetime as dt
import io
import json
import tempfile
import zipfile
from pathlib import Path

import folium
import streamlit as st
from streamlit_folium import st_folium

import napoleonic as nap
import visuals as vz

st.set_page_config(page_title="Napoleonic", page_icon="🛰️", layout="wide")
st.markdown("<style>.block-container{padding-top:1.6rem}</style>", unsafe_allow_html=True)
st.title("🛰️ Napoleonic")
st.caption("Rapid terrain and infrastructure mapping from Sentinel-2 satellite imagery")

ss = st.session_state
ss.setdefault("lat", 48.74)
ss.setdefault("lon", 37.58)
ss.setdefault("view_center", (48.74, 37.58))   # map view: changed only by deliberate actions
ss.setdefault("view_zoom", 11)
if "pending" in ss:  # a map click or search result, applied before the widgets are created
    ss["lat"], ss["lon"] = ss.pop("pending")

ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"


# ------------------------------------------------------------------ cached helpers
@st.cache_data(ttl=1800, max_entries=3, show_spinner=False)
def cached_scene(bbox, max_cloud, lookback, item_id):
    return nap.fetch_scene(bbox, max_cloud, lookback, item_id)


@st.cache_data(ttl=900, show_spinner=False)
def cached_search(bbox, max_cloud, lookback):
    return nap.search_scenes(bbox, max_cloud, lookback)


@st.cache_data(ttl=3600, max_entries=3, show_spinner=False)
def cached_osm(bbox):
    return nap.fetch_osm(bbox)


@st.cache_resource
def picker_base_map():
    """The picker's base map is built ONCE and never changes, so the browser never reloads it.
    The selection box is added dynamically (feature group) and the view is only moved on purpose."""
    m = folium.Map(location=(48.74, 37.58), zoom_start=11, tiles=None, control_scale=True, max_zoom=19)
    folium.TileLayer(ESRI.format("World_Imagery"), attr="Esri, Maxar, Earthstar Geographics",
                     name="Satellite", max_zoom=19).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Streets", show=False, max_zoom=19).add_to(m)
    folium.TileLayer(ESRI.format("Reference/World_Boundaries_and_Places"), attr="Esri",
                     name="Place names", overlay=True, show=True, max_zoom=19).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    return m


def geocode(q):
    """Place-name search via OpenStreetMap Nominatim (free, rate-limited)."""
    import requests
    r = requests.get("https://nominatim.openstreetmap.org/search",
                     params={"q": q, "format": "json", "limit": 1},
                     headers={"User-Agent": "napoleonic-hackathon-prototype"}, timeout=15)
    r.raise_for_status()
    hit = r.json()
    return (float(hit[0]["lat"]), float(hit[0]["lon"])) if hit else None


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Area of interest")
    mode = st.radio("Data source", ["Demo (synthetic scene)", "Live Sentinel-2"])
    demo = mode.startswith("Demo")
    lat = st.number_input("Latitude", key="lat", format="%.4f", disabled=demo)
    lon = st.number_input("Longitude", key="lon", format="%.4f", disabled=demo)
    radius = st.slider("Radius (km)", 1.0, 12.0, 3.0, 0.5, disabled=demo)
    max_cloud = st.slider("Max cloud cover (%)", 5, 80, 30, disabled=demo)
    lookback = st.slider("Search back (days)", 5, 90, 30, disabled=demo)
    use_osm = st.checkbox("Fuse OpenStreetMap roads/buildings", value=False, disabled=demo)

    bbox = tuple(round(v, 5) for v in nap.bbox_from_point(lat, lon, radius))
    item_id = None
    manual = st.checkbox("Choose a specific scene date", disabled=demo)
    if manual and not demo:
        try:
            scenes = cached_search(bbox, max_cloud, lookback)
            if scenes:
                labels = [f"{x['date']}  ·  {x['cloud']:.0f}% cloud" for x in scenes]
                item_id = scenes[labels.index(st.selectbox("Available scenes (newest first)", labels))]["id"]
            else:
                st.warning("No scenes under the cloud limit. Raise it or search further back.")
        except Exception as e:
            st.warning(f"Scene search failed: {e}")

    with st.expander("Detection settings (advanced)"):
        p = nap.Params()
        p.adaptive_canopy = st.checkbox("Auto-tune tree detection to the scene", value=p.adaptive_canopy)
        p.ndvi_veg = st.slider("Vegetation NDVI", 0.2, 0.8, p.ndvi_veg, 0.01)
        p.canopy_swir = st.slider("Canopy SWIR ceiling (manual)", 0.08, 0.30, p.canopy_swir, 0.01,
                                  disabled=p.adaptive_canopy)
        p.line_max_width_px = st.slider("Tree-line max width (px, 10 m each)", 2.0, 10.0,
                                        p.line_max_width_px, 0.5)
        p.line_min_len_px = st.slider("Tree-line min length (px)", 3, 30, p.line_min_len_px)
        p.road_hi_pct = st.slider("Road strictness (seed percentile)", 97.0, 99.9, p.road_hi_pct, 0.1)
        p.tall_m = st.slider("Multi-storey height (m, OSM)", 6.0, 30.0, p.tall_m, 1.0)

    run = st.button("Run analysis", type="primary", width="stretch")

# typed coordinates -> move the map there (clicks/searches record their own position so they don't re-centre)
if (round(lat, 5), round(lon, 5)) != ss.get("_seen_ll"):
    ss["view_center"] = (lat, lon)
    ss["_seen_ll"] = (round(lat, 5), round(lon, 5))


# ------------------------------------------------------------------ map picker
@st.fragment
def location_picker(radius_km):
    """Runs as a fragment: searching/zooming never re-runs the whole app. Only a click does."""
    q = st.text_input("Search a place (optional)", placeholder="e.g. Kramatorsk, Ukraine", key="place_q")
    if q and q != ss.get("last_query"):
        ss["last_query"] = q
        try:
            found = geocode(q)
        except Exception as e:
            st.warning(f"Search failed: {e}")
            found = None
        if found:
            ss.update(pending=found, view_center=found, view_zoom=12,
                      _seen_ll=(round(found[0], 5), round(found[1], 5)))
            st.rerun()
        elif found is None and q:
            st.warning("Place not found.")

    st.caption("Click the map to choose the centre. Drag and zoom freely: the view stays where you leave it. "
               "The yellow box is the area that will be scanned.")
    la, lo = ss["lat"], ss["lon"]
    w, s, e, n = nap.bbox_from_point(la, lo, radius_km)
    fg = folium.FeatureGroup(name="Scan area")
    fg.add_child(folium.Rectangle([[s, w], [n, e]], color="#facc15", weight=3, fill=True, fill_opacity=0.10))
    fg.add_child(folium.CircleMarker([la, lo], radius=6, color="#111827", weight=2, fill=True,
                                     fill_color="#facc15", fill_opacity=1))
    ret = st_folium(picker_base_map(), center=ss["view_center"], zoom=ss["view_zoom"],
                    feature_group_to_add=fg, height=520, use_container_width=True,
                    returned_objects=["last_clicked"], key="picker")
    click = (ret or {}).get("last_clicked")
    if click:
        key = (round(click["lat"], 5), round(click["lng"], 5))
        if key != ss.get("last_click"):          # a genuinely new click
            ss["last_click"] = key
            ss["pending"] = (click["lat"], click["lng"])
            ss["_seen_ll"] = key                  # keeps the view where the user left it
            st.rerun()
    st.write(f"**Selected:** {la:.4f}, {lo:.4f}   |   scan area about {2 * radius_km:.0f} × {2 * radius_km:.0f} km")


if not demo:
    with st.expander("📍 Pick the area on the map", expanded="out" not in ss):
        location_picker(radius)

# ------------------------------------------------------------------ run
if run:
    try:
        with st.status("Working...", expanded=True) as status:
            osm = None
            if demo:
                st.write("Generating synthetic scene")
                sc = nap.synthetic_scene()
            else:
                st.write("Retrieving Sentinel-2 imagery (first run for an area can take a minute)")
                sc = cached_scene(bbox, max_cloud, lookback, item_id)
                if use_osm:
                    st.write("Fetching OpenStreetMap data")
                    try:
                        osm = cached_osm(bbox)
                    except Exception as e:
                        st.warning(f"OSM fetch failed ({e}). Continuing with imagery only.")
            st.write("Analysing terrain")
            res, stats = nap.analyse(sc, p, osm)
            st.write("Rendering maps")
            tmp = Path(tempfile.mkdtemp())
            nap.render_map(sc, res, tmp / "map.png")
            nap.write_report(sc, stats, tmp / "report.md")
            gj = []
            for name in ("lines", "woodland", "built", "roads", "osm_roads", "osm_tall"):
                if res.get(name) is not None and res[name].any():
                    gj += nap.layer_to_geojson(res[name], sc, name)
            prep = vz.prepare_layers(sc, res)
            html = vz.build_result_map(prep, 0.85, demo=demo).get_root().render()
            out = {
                "png": (tmp / "map.png").read_bytes(),
                "report": (tmp / "report.md").read_text(encoding="utf-8"),
                "geojson": json.dumps({"type": "FeatureCollection", "features": gj}),
                "summary": json.dumps({"meta": sc.meta, "stats": stats}, indent=2, default=str),
                "html": html, "prep": prep, "stats": stats, "meta": sc.meta, "demo": demo,
            }
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("map.png", out["png"])
                z.writestr("report.md", out["report"])
                z.writestr("features.geojson", out["geojson"])
                z.writestr("summary.json", out["summary"])
                z.writestr("interactive_map.html", out["html"])
            out["zip"] = buf.getvalue()
            ss["out"] = out
            ss.pop("map_cache", None)
            status.update(label="Done", state="complete", expanded=False)
    except SystemExit as e:  # raised by the scene search when nothing matches
        st.error(str(e))
    except Exception as e:
        st.error(f"Analysis failed: {e}")

# ------------------------------------------------------------------ results
out = ss.get("out")
if not out:
    st.info("Choose an area and press **Run analysis**. Start with the demo to see how it works.")
else:
    s_, meta = out["stats"], out["meta"]
    try:
        age = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(meta["acquired"].replace("Z", "+00:00"))).days
        if not out["demo"] and age > 7:
            st.warning(f"This imagery is {age} days old. Treat it as history, not as the current situation.")
    except Exception:
        pass

    c = st.columns(6)
    c[0].metric("Tree lines", f"{s_['treelines_count']}", f"{s_['treelines_length_km']} km")
    c[1].metric("Woodland", f"{s_['woodland_km2']} km²")
    c[2].metric("Built-up", f"{s_['builtup_km2']} km²")
    c[3].metric("Candidate roads", f"{s_['candidate_road_km']} km")
    c[4].metric("Usable area", f"{s_['valid_pct']}%")
    c[5].metric("Scene date", str(meta["acquired"])[:10])

    t_map, t_img, t_rep, t_data = st.tabs(["🗺️ Interactive map", "🖼️ Scan image", "📄 Report", "🧾 Data"])
    with t_map:
        opacity = st.slider("Overlay strength", 0.2, 1.0, 0.85, 0.05, key="opacity")
        cache = ss.get("map_cache")
        if not cache or cache["opacity"] != opacity:
            cache = {"opacity": opacity, "map": vz.build_result_map(out["prep"], opacity, demo=out["demo"])}
            ss["map_cache"] = cache
        st_folium(cache["map"], height=700, use_container_width=True, returned_objects=[], key="result_map")
        st.caption("Use the layer panel (top right) to switch the satellite basemap, show the raw Sentinel-2 image, "
                   "false colour or vegetation health, and toggle each detected layer. The ruler tool measures "
                   "distances. Basemap imagery can be older than the scan.")
    with t_img:
        st.image(out["png"], width="stretch")
    with t_rep:
        st.markdown(out["report"])
    with t_data:
        st.json({"meta": meta, "stats": s_}, expanded=False)

    d = st.columns(5)
    d[0].download_button("Map (PNG)", out["png"], "map.png", "image/png", on_click="ignore")
    d[1].download_button("Interactive map (HTML)", out["html"], "interactive_map.html", "text/html", on_click="ignore")
    d[2].download_button("Report (MD)", out["report"], "report.md", "text/markdown", on_click="ignore")
    d[3].download_button("Layers (GeoJSON)", out["geojson"], "features.geojson", "application/geo+json", on_click="ignore")
    d[4].download_button("Everything (ZIP)", out["zip"], "napoleonic_results.zip", "application/zip", on_click="ignore")
    st.caption("Heuristic output at 10 m resolution. Validate against ground truth before relying on it.")
