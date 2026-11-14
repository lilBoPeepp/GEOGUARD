"""
NAPOLEONIC - web interface.   Run with:   streamlit run app.py
Needs napoleonic.py in the same folder.
"""
import json
import tempfile
from dataclasses import fields
from pathlib import Path

import pandas as pd
import streamlit as st

import napoleonic as nap

st.set_page_config(page_title="Napoleonic", page_icon="🛰️", layout="wide")
st.title("🛰️ Napoleonic")
st.caption("Rapid terrain and infrastructure mapping from Sentinel-2 satellite imagery")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Area of interest")
    mode = st.radio("Data source", ["Demo (synthetic scene)", "Live Sentinel-2"])
    lat = st.number_input("Latitude", value=48.74, format="%.4f", disabled=mode.startswith("Demo"))
    lon = st.number_input("Longitude", value=37.58, format="%.4f", disabled=mode.startswith("Demo"))
    radius = st.slider("Radius (km)", 1.0, 12.0, 3.0, 0.5, disabled=mode.startswith("Demo"))
    max_cloud = st.slider("Max cloud cover (%)", 5, 80, 30, disabled=mode.startswith("Demo"))
    lookback = st.slider("Search back (days)", 5, 90, 30, disabled=mode.startswith("Demo"))
    use_osm = st.checkbox("Fuse OpenStreetMap roads/buildings", value=False,
                          disabled=mode.startswith("Demo"))

    with st.expander("Detection thresholds (advanced)"):
        p = nap.Params()
        p.ndvi_veg = st.slider("Vegetation NDVI", 0.2, 0.8, p.ndvi_veg, 0.01)
        p.canopy_swir = st.slider("Canopy SWIR ceiling", 0.08, 0.30, p.canopy_swir, 0.01)
        p.line_max_width_px = st.slider("Tree-line max width (px, 10 m each)", 2.0, 10.0,
                                        p.line_max_width_px, 0.5)
        p.line_min_len_px = st.slider("Tree-line min length (px)", 3, 30, p.line_min_len_px)
        p.road_hi_pct = st.slider("Road strictness (seed percentile)", 97.0, 99.9, p.road_hi_pct, 0.1)
        p.tall_m = st.slider("Multi-storey height (m, OSM)", 6.0, 30.0, p.tall_m, 1.0)

    run = st.button("Run analysis", type="primary", width="stretch")

# ------------------------------------------------------------------ run
if run:
    try:
        with st.status("Working...", expanded=True) as status:
            osm = None
            if mode.startswith("Demo"):
                st.write("Generating synthetic scene")
                sc = nap.synthetic_scene()
            else:
                bbox = nap.bbox_from_point(lat, lon, radius)
                st.write("Searching for the newest Sentinel-2 scene")
                sc = nap.fetch_scene(bbox, max_cloud, lookback)
                if use_osm:
                    st.write("Fetching OpenStreetMap data")
                    try:
                        osm = nap.fetch_osm(bbox)
                    except Exception as e:
                        st.warning(f"OSM fetch failed ({e}). Continuing with imagery only.")
            st.write("Analysing terrain")
            res, stats = nap.analyse(sc, p, osm)
            tmp = Path(tempfile.mkdtemp())
            nap.render_map(sc, res, tmp / "map.png")
            nap.write_report(sc, stats, tmp / "report.md")
            gj = []
            for name in ("lines", "woodland", "built", "roads", "osm_roads", "osm_tall"):
                if res.get(name) is not None and res[name].any():
                    gj += nap.layer_to_geojson(res[name], sc, name)
            st.session_state["out"] = {
                "map": (tmp / "map.png").read_bytes(),
                "report": (tmp / "report.md").read_text(encoding="utf-8"),
                "geojson": json.dumps({"type": "FeatureCollection", "features": gj}),
                "stats": stats, "meta": sc.meta, "demo": mode.startswith("Demo"),
                "centre": (lat, lon),
            }
            status.update(label="Done", state="complete", expanded=False)
    except SystemExit as e:  # raised by fetch_scene when nothing matches
        st.error(str(e))
    except Exception as e:
        st.error(f"Analysis failed: {e}")

# ------------------------------------------------------------------ results
out = st.session_state.get("out")
if not out:
    st.info("Choose an area in the sidebar and press **Run analysis**. "
            "Start with the demo to see how it works.")
else:
    s = out["stats"]
    c = st.columns(5)
    c[0].metric("Tree lines", f"{s['treelines_count']}", f"{s['treelines_length_km']} km")
    c[1].metric("Woodland", f"{s['woodland_km2']} km²")
    c[2].metric("Built-up", f"{s['builtup_km2']} km²")
    c[3].metric("Candidate roads", f"{s['candidate_road_km']} km")
    c[4].metric("Usable area", f"{s['valid_pct']}%")

    tab_map, tab_report, tab_data = st.tabs(["Map", "Report", "Data"])
    with tab_map:
        st.image(out["map"], width="stretch")
        if not out["demo"]:
            st.caption("Location")
            st.map(pd.DataFrame({"lat": [out["centre"][0]], "lon": [out["centre"][1]]}), zoom=11)
    with tab_report:
        st.markdown(out["report"])
    with tab_data:
        st.json({"meta": out["meta"], "stats": s}, expanded=False)

    d = st.columns(3)
    d[0].download_button("Download map (PNG)", out["map"], "map.png", "image/png")
    d[1].download_button("Download report (MD)", out["report"], "report.md", "text/markdown")
    d[2].download_button("Download layers (GeoJSON)", out["geojson"], "features.geojson",
                         "application/geo+json")
    st.caption("Heuristic output at 10 m resolution. Validate against ground truth before relying on it.")
