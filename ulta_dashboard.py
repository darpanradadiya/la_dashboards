# ulta_dashboard.py  ────────────────────────────────────────────────────────
# One-week LA “Weekly Patterns” → brands that appear the SAME DAY as Ulta

import streamlit as st, pandas as pd, numpy as np
import plotly.express as px
from pathlib import Path
from itertools import chain

st.set_page_config("Ulta Beauty • Same-day Co-Visits (LA)", layout="wide")

# ── 0 ▸ CSV absolute path ---------------------------------------------------
CSV_PATH = Path(
    "/Users/darpanradadiya/Downloads/ULTA_BEAUTY_DASHBOARD/Data/la_foot_traffic.csv"
)

if not CSV_PATH.exists():
    st.error(f"CSV not found at:\n{CSV_PATH}\n"
             "→ check the path or edit CSV_PATH in ulta_dashboard.py")
    st.stop()

# ── 1 ▸ load file + compute partner stats ----------------------------------
@st.cache_data(show_spinner=False)
def load_ulta_partners(csv: Path):
    df = pd.read_csv(csv)

    # safe parser for dict-string → list
    def parse(raw):
        if not isinstance(raw, str) or raw.strip() in ("", "{}", "NaN"):
            return []
        try:
            d = eval(raw.replace("'", '"').replace('""', '"'))
            return list(d) if isinstance(d, dict) else []
        except Exception:
            return []

    df["brand_list"] = df["related_same_day_brand"].apply(parse)

    # find “Ulta” token
    ulta_tok = next(
        (b for b in set(chain.from_iterable(df["brand_list"])) if "ulta" in b.lower()),
        "Ulta Beauty"
    )

    ulta_mask = df["brand_list"].apply(lambda L: ulta_tok in L)
    ulta_days = int(ulta_mask.sum())

    # counts
    joint = (
        df[ulta_mask]["brand_list"].explode()
          .loc[lambda s: s != ulta_tok]
          .value_counts()
          .rename("joint_visits")
    )
    overall = (
        df["brand_list"].explode()
          .value_counts()
          .rename("brand_visits")
    )

    stats = pd.concat([joint, overall], axis=1)
    stats["support/Confidence"]    = stats["joint_visits"] / ulta_days
    stats["lift"]       = stats["support/Confidence"] / (stats["brand_visits"] / len(df))

    # ensure first column is called “partner” no matter what
    stats = stats.reset_index()
    stats.rename(columns={stats.columns[0]: "partner"}, inplace=True)

    stats = stats.sort_values("lift", ascending=False)
    return ulta_tok, ulta_days, stats, df

ULTA_TOKEN, ULTA_DAYS, PARTNERS, RAW_DF = load_ulta_partners(CSV_PATH)

# ── 2 ▸ sidebar controls ----------------------------------------------------
with st.sidebar:
    st.header("Filters")
    min_joint = st.slider("Min joint visits", 1,
                          int(PARTNERS["joint_visits"].max()), 5, 1)
    sort_col = st.selectbox("Sort table by",
                            ["lift", "joint_visits", "support"])
    highlight = st.multiselect(
        "Highlight partners", PARTNERS["partner"].head(15).tolist()
    )
    st.divider(); st.header("Map")
    map_partner = st.selectbox("Partner to map",
                               ["—"] + PARTNERS["partner"].tolist())

flt = (
    PARTNERS.query("joint_visits >= @min_joint")
            .sort_values(sort_col, ascending=False)
)

# ── 3 ▸ KPI tiles -----------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Ulta rows",       f"{ULTA_DAYS:,}")
c2.metric("Co-visit brands", f"{len(PARTNERS):,}")
c3.metric("Top lift",        f"{PARTNERS['lift'].max():.0f}×")

# ── 4 ▸ scatter chart -------------------------------------------------------
st.markdown("### 📊 Lift vs. joint-visit count")
fig = px.scatter(
    flt, x="joint_visits", y="lift", text="partner",
    labels={"joint_visits":"Joint visits with Ulta (week)",
            "lift":"Co-visit lift"},
    height=420
)
fig.update_traces(
    textposition="top center",
    marker=dict(color=np.where(
        flt["partner"].isin(highlight), "crimson", "steelblue"
    ))
)
st.plotly_chart(fig, use_container_width=True)

# ── 5 ▸ table + download ----------------------------------------------------
st.markdown("### 🏆 Partner table")
st.dataframe(flt.reset_index(drop=True), use_container_width=True, height=360)
st.download_button("⬇ Download CSV", flt.to_csv(index=False).encode(),
                   "ulta_covisit_partners.csv")

# ── 6 ▸ map (Ulta + one partner) -------------------------------------------
if map_partner != "—":
    st.markdown(f"### 🗺️ Ulta & **{map_partner}** store locations")

    # Ulta POIs
    ulta_geo = RAW_DF[
        RAW_DF["location_name"].str.contains("ulta", case=False, na=False)
    ][["location_name", "latitude", "longitude"]].dropna().drop_duplicates()

    # partner POIs (simple substring match)
    part_geo = RAW_DF[
        RAW_DF["location_name"].str.contains(map_partner, case=False, na=False)
    ][["location_name", "latitude", "longitude"]].dropna().drop_duplicates()

    if ulta_geo.empty or part_geo.empty:
        st.info("No coordinates found for one of the brands.")
    else:
        mdf = (
            pd.concat([
                ulta_geo.assign(category="Ulta"),
                part_geo.assign(category=map_partner)
            ]).reset_index(drop=True)
        )
        mfig = px.scatter_mapbox(
            mdf, lat="latitude", lon="longitude",
            color="category", hover_name="location_name",
            zoom=8, height=500,
            mapbox_style="open-street-map"
        )
        st.plotly_chart(mfig, use_container_width=True)

# ── 7 ▸ notes --------------------------------------------------------------
with st.expander("🔬 Method"):
    st.markdown(fr"""
* Anchor token detected → **{ULTA_TOKEN}**  
* One-week LA slice of SafeGraph Weekly Patterns.  
* **Lift** = P(partner&nbsp;|&nbsp;Ulta) ÷ P(partner overall).  
* Map uses simple `location_name` substring to find stores.
""")
