"""Single-page Chicago community-area dashboard. Exploratory; not clinical guidance."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st

st.set_page_config(page_title="Chicago Gun Violence & Healthcare", page_icon="🏙️", layout="wide")
DATA = Path(__file__).resolve().parent / "data"

@st.cache_data
def load_data():
    ny = pd.read_csv(DATA / "neighborhood_by_year.csv")
    profiles = pd.read_csv(DATA / "neighborhood_profiles.csv")
    hospitals = pd.read_csv(DATA / "trauma_hospitals.csv")
    geo = json.loads((DATA / "dashboard_map.geojson").read_text())
    for feature in geo["features"]:
        feature["properties"]["ca_num"] = str(feature["properties"]["ca_num"])
    return ny, profiles, hospitals, geo

@st.cache_resource
def fit_model():
    """Pooled negative binomial GLM; exploratory fixed dispersion, not causal."""
    ny = pd.read_csv(DATA / "neighborhood_by_year.csv")
    fields = ["shot_total", "population", "pct_poverty", "pct_unemployed", "year", "ca_name"]
    d = ny[fields].copy()
    for c in ["shot_total", "population", "pct_poverty", "pct_unemployed", "year"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=fields)
    d = d[(d.population > 0) & (d.shot_total >= 0)].copy()
    # Year fixed effects avoid treating different calendar years as equivalent.
    x = pd.get_dummies(d[["pct_poverty", "pct_unemployed", "year"]].assign(year=d.year.astype(str)),
                       columns=["year"], drop_first=True, dtype=float)
    x = sm.add_constant(x, has_constant="add")
    model = sm.GLM(d.shot_total.astype(float), x, family=sm.families.NegativeBinomial(alpha=1.0),
                   offset=np.log(d.population.astype(float))).fit()
    d["model_estimate"] = model.predict(x, offset=np.log(d.population.astype(float)))
    return model, list(x.columns), d

ny, profiles, hospitals, geo = load_data()

st.title("Myuzima: Chicago Gun Violence & Healthcare Dashboard")
st.caption(
    "Exploring gun violence, socioeconomic conditions, "
    "and healthcare access across Chicago's 77 community areas."
)

tab1, tab2, tab3 = st.tabs([
    "Neighborhoods & Gun Violence",
    "Socioeconomic Analysis",
    "Healthcare Impact"
])
areas = sorted(ny.ca_name.dropna().unique())
years = sorted(ny.year.dropna().astype(int).unique(), reverse=True)
a, b = st.columns([2, 1])
with a:
    area = st.selectbox("Community area", ["All Chicago"] + areas)
with b:
    year = st.selectbox("Shooting reporting year", years)
current = ny.loc[ny.year == year].copy()
focus = current if area == "All Chicago" else current[current.ca_name == area]
profile = profiles.loc[profiles.ca_name == area].iloc[0] if area != "All Chicago" else None

def display_number(v, decimals=0):
    return "N/A" if pd.isna(v) else f"{v:,.{decimals}f}"

def weighted_rate(df):
    population = df.population.sum()
    return df.shot_total.sum() / population * 10000 if population > 0 else np.nan
with tab1:
    st.subheader("Neighborhood overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Shooting victims", display_number(focus.shot_total.sum()))
    c2.metric("Fatal victims", display_number(focus.fatal.sum()))
    c3.metric("Victims per 10,000", display_number(weighted_rate(focus), 1))
    c4.metric("Population", display_number(focus.population.sum()))

p1, p2, p3 = st.columns(3)
if area == "All Chicago":
    p1.metric("Poverty (%)", "Select a neighborhood")
    p2.metric("Unemployment (%)", "Select a neighborhood")
    p3.metric("Estimated trauma drive", "Select a neighborhood")
else:
    row = focus.iloc[0]
    p1.metric("Poverty (%)", display_number(row.pct_poverty, 1))
    p2.metric("Unemployment (%)", display_number(row.pct_unemployed, 1))
    p3.metric("Estimated trauma drive", display_number(profile.trauma_drive_min, 1) + " min" if pd.notna(profile.trauma_drive_min) else "N/A")
st.caption("Poverty/unemployment definitions and source years require verification. Trauma drive times are estimates, NOT measured EMS response times.")

st.subheader("Map: shooting burden and hospital locations")
map_col, side_col = st.columns([1.65, 1], gap="large")
with map_col:
    measure = st.radio("Shade by", ["Victims per 10,000", "Total victims"], horizontal=True)
    value = "shot_rate_per_10k" if measure.startswith("Victims") else "shot_total"
    m = current[["ca_num", "ca_name", value]].copy()
    m["ca_num"] = m.ca_num.astype(str)
    fig = px.choropleth_mapbox(m, geojson=geo, locations="ca_num", featureidkey="properties.ca_num",
        color=value, hover_name="ca_name", color_continuous_scale="YlOrRd", mapbox_style="carto-positron",
        center={"lat": 41.84, "lon": -87.68}, zoom=9.5, opacity=0.75, height=520)
    valid = hospitals.dropna(subset=["lat", "lon"]).copy()
    # Different hospitals may have identical rounded coordinates; show all names on hover.
    valid["location_key"] = valid.lat.astype(str) + "," + valid.lon.astype(str)
    grouped = valid.groupby(["lat", "lon"], as_index=False).agg(hospital=("hospital", lambda s: "<br>".join(s)))
    fig.add_trace(go.Scattermapbox(lat=grouped.lat, lon=grouped.lon, mode="markers",
        marker={"size": 14, "color": "#123d66"}, text=grouped.hospital,
        hovertemplate="<b>Hospital(s)</b><br>%{text}<extra></extra>", name="Listed hospitals"))
    if area != "All Chicago":
        fig.update_layout(title={"text": f"Selected community area: {area}", "x": 0.03})
    fig.update_layout(margin={"l": 0, "r": 0, "t": 40, "b": 0})
   map_event = st.plotly_chart(
    fig,
    use_container_width=True,
    on_select="rerun",
    selection_mode="points",
    key="chicago_map"
)
selected_points = map_event.selection.points

if selected_points:
    point = selected_points[0]

    if "location" in point:
        selected_ca = str(point["location"])

        selected_row = current[
            current["ca_num"].astype(str) == selected_ca
        ]

        if not selected_row.empty:
            selected_name = selected_row.iloc[0]["ca_name"]

            st.success(f"Selected community area: {selected_name}")

            selected_data = selected_row.iloc[0]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Shooting victims",
                display_number(selected_data["shot_total"])
            )

            col2.metric(
                "Fatal victims",
                display_number(selected_data["fatal"])
            )

            col3.metric(
                "Victims per 10,000",
                display_number(selected_data["shot_rate_per_10k"], 1)
            )
                        st.markdown("#### Socioeconomic & Healthcare Profile")

            s1, s2, s3 = st.columns(3)

            s1.metric(
                "Poverty rate",
                f"{selected_data['pct_poverty']:.1f}%"
            )

            s2.metric(
                "Unemployment rate",
                f"{selected_data['pct_unemployed']:.1f}%"
            )

            selected_profile = profiles[
                profiles["ca_name"] == selected_name
            ]

            if not selected_profile.empty:
                drive_time = selected_profile.iloc[0]["trauma_drive_min"]

                s3.metric(
                    "Estimated trauma-center drive",
                    f"{drive_time:.1f} min" if pd.notna(drive_time) else "N/A"
                )
            else:
                s3.metric("Estimated trauma-center drive", "N/A")

            st.caption(
                "Poverty and unemployment are socioeconomic indicators. "
                "Trauma-center drive time is an estimate, not an actual "
                "ambulance response time."
            )
    st.caption("Markers sharing rounded coordinates list all hospitals on hover, including Stroger (Cook County). Verify coordinates before publication.")
with side_col:
    st.markdown("**Highest shooting-victim rates**")
    ranked = current.nlargest(10, "shot_rate_per_10k")[["ca_name", "shot_rate_per_10k"]]
    st.dataframe(ranked.rename(columns={"ca_name": "Community", "shot_rate_per_10k": "Victims / 10k"}),
                 hide_index=True, use_container_width=True)
    if profile is not None:
        st.markdown("**Healthcare access for selected community**")
        st.write("Nearest listed trauma center:", profile.nearest_center)
        st.write("Estimated clinics:", display_number(profile.clinics))

st.subheader("Shooting trends")
trend = ny if area == "All Chicago" else ny[ny.ca_name == area]
t = trend.groupby("year", as_index=False)[["fatal", "nonfatal"]].sum()
fig = px.bar(t.melt(id_vars="year", var_name="Outcome", value_name="Victims"), x="year", y="Victims",
             color="Outcome", barmode="stack", height=320)
st.plotly_chart(fig, use_container_width=True)
with tab2:
    st.subheader("Regression analysis — exploratory")
    st.write("A pooled negative-binomial model examines associations of poverty and unemployment with shooting-victim counts, adjusting for population and calendar year. This model does not prove cause and effect.")
    try:
        model, model_columns, modeled = fit_model()
        observed = modeled[modeled.year == year].copy()
        if area != "All Chicago":
            observed = observed[observed.ca_name == area]
        if not observed.empty:
            actual = observed.shot_total.sum()
            fitted = observed.model_estimate.sum()
            r1, r2 = st.columns(2)
            r1.metric("Observed victims", display_number(actual))
            r2.metric("Model-estimated victims", display_number(fitted, 1))
        fig = px.scatter(modeled[modeled.year == year], x="model_estimate", y="shot_total", hover_name="ca_name",
            labels={"model_estimate": "Model-estimated victims", "shot_total": "Observed victims"}, height=380)
        maxval = max(modeled.loc[modeled.year == year, ["model_estimate", "shot_total"]].max())
        fig.add_shape(type="line", x0=0, y0=0, x1=maxval, y1=maxval, line={"dash": "dash", "color": "gray"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Model uses fixed negative-binomial dispersion alpha=1 as a starting assumption; it is NOT a validated forecasting model. Predictors may share underlying source periods. Review model diagnostics, source definitions, and temporal dependence before using for decisions.")
        with st.expander("Model coefficients (exploratory)"):
            st.dataframe(pd.DataFrame({"Coefficient": model.params.index, "Estimate": model.params.values,
                                       "P-value": model.pvalues.values}), hide_index=True)

    st.subheader("What-if explorer — model-based illustration")
    st.caption("Adjusting a variable changes the fitted association only. It does not estimate what an intervention would cause.")
    chosen = area if area != "All Chicago" else st.selectbox("Choose a neighborhood for scenario", areas)
    baseline = modeled[(modeled.ca_name == chosen) & (modeled.year == year)]
    if len(baseline):
        rec = baseline.iloc[0]
        s1, s2 = st.columns(2)
        poverty = s1.slider("Hypothetical poverty (%)", 0.0, 100.0, float(rec.pct_poverty), 0.5)
        unemployment = s2.slider("Hypothetical unemployment (%)", 0.0, 100.0, float(rec.pct_unemployed), 0.5)
        x = pd.DataFrame(0.0, index=[0], columns=model_columns)
        x.loc[0, "const"] = 1.0
        x.loc[0, "pct_poverty"] = poverty
        x.loc[0, "pct_unemployed"] = unemployment
        year_col = "year_" + str(year)
        if year_col in x:
            x.loc[0, year_col] = 1.0
        prediction = float(model.predict(x, offset=np.log([rec.population])).iloc[0])
        st.metric("Illustrative model-estimated victims", display_number(prediction, 1),
                  delta=f"{prediction - rec.model_estimate:+.1f} vs baseline estimate")
except Exception as exc:
    st.error(f"Regression could not be fitted: {exc}")
with tab3:
    st.subheader("Hospital finances — separate 2024 snapshot")
    st.caption("Hospital-wide financial figures, NOT costs attributable to gunshot victims. Not all hospitals are listed.")
    finance_cols = ["hospital", "total_costs", "charity_care_cost", "uncompensated_care_cost"]
    st.dataframe(hospitals[finance_cols].rename(columns={"hospital": "Hospital", "total_costs": "Total costs ($)",
        "charity_care_cost": "Charity care ($)", "uncompensated_care_cost": "Uncompensated care ($)"}),
        hide_index=True, use_container_width=True)

with st.expander("Data notes and limitations — read before client presentation"):
    st.write("Shooting counts are victims, not distinct incidents. Rates use supplied population denominators. Socioeconomic variable definitions, ACS/source years, and hospital classifications require verification against primary sources. Estimated travel time is not actual EMS response time. Hospital finances are hospital-wide 2024 figures and are not shooting-specific. Regression is observational, exploratory, and not a causal or validated forecasting model. The dashboard covers the available years, not 2026 shooting data.")
    st.download_button("Download selected shooting data", focus.to_csv(index=False), file_name=f"chicago_{year}_selected.csv", mime="text/csv")
