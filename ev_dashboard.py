import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="EV Market 2026 — Price Prediction",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        border: 1px solid #e9ecef;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #1a1a2e;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("ev_market_2026.csv")
    df.columns = df.columns.str.strip()
    return df

@st.cache_resource
def train_models(df):
    feature_cols = [
        "battery_capacity_kwh", "range_miles", "charging_speed_kw",
        "acceleration_0_60_mph", "top_speed_mph", "horsepower", "torque_nm",
        "seating_capacity", "cargo_volume_cubic_ft", "weight_kg",
        "safety_rating", "autopilot_level", "warranty_years", "year",
        "drive_type", "body_type", "market_segment", "country_of_origin"
    ]
    df_model = df[feature_cols + ["price_usd"]].dropna()

    le_dict = {}
    cat_cols = ["drive_type", "body_type", "market_segment", "country_of_origin"]
    for col in cat_cols:
        le = LabelEncoder()
        df_model = df_model.copy()
        df_model[col] = le.fit_transform(df_model[col])
        le_dict[col] = le

    X = df_model[feature_cols]
    y = df_model["price_usd"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42),
        "Linear Regression": LinearRegression()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv = cross_val_score(model, X, y, cv=5, scoring="r2")
        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_test": y_test,
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
            "cv_r2_mean": cv.mean(),
            "cv_r2_std": cv.std(),
        }

    return results, le_dict, feature_cols, X_test, y_test

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ EV Dashboard")
    st.markdown("---")
    selected_brands = st.multiselect(
        "Filter by brand", sorted(df["brand"].unique()),
        default=sorted(df["brand"].unique())
    )
    selected_segments = st.multiselect(
        "Filter by segment", sorted(df["market_segment"].unique()),
        default=sorted(df["market_segment"].unique())
    )
    year_range = st.slider("Model year range", int(df["year"].min()), int(df["year"].max()),
                           (int(df["year"].min()), int(df["year"].max())))
    st.markdown("---")
    st.caption(f"Dataset: {len(df):,} records")

df_filtered = df[
    df["brand"].isin(selected_brands) &
    df["market_segment"].isin(selected_segments) &
    df["year"].between(year_range[0], year_range[1])
]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Exploratory Analysis", "🤖 ML Model", "🔮 Price Predictor"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## EV Market 2026 — Overview")
    st.markdown(f"Showing **{len(df_filtered):,}** records after filters.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Price", f"${df_filtered['price_usd'].mean():,.0f}")
    c2.metric("Avg Range", f"{df_filtered['range_miles'].mean():.0f} mi")
    c3.metric("Avg Horsepower", f"{df_filtered['horsepower'].mean():.0f} hp")
    c4.metric("Total Sales (sum)", f"{df_filtered['annual_sales_units'].sum():,.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        sales_by_brand = (
            df_filtered.groupby("brand")["annual_sales_units"].sum()
            .sort_values(ascending=True).tail(15)
            .reset_index()
        )
        sales_by_brand.columns = ["Brand", "Total Units"]
        fig = px.bar(
            sales_by_brand, x="Total Units", y="Brand",
            orientation="h", title="Annual Sales by Brand (top 15)",
            color="Total Units",
            color_continuous_scale="Blues"
        )
        fig.update_layout(coloraxis_showscale=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        seg_sales = df_filtered.groupby("market_segment")["annual_sales_units"].sum().reset_index()
        fig2 = px.pie(
            seg_sales, values="annual_sales_units", names="market_segment",
            title="Sales Share by Market Segment",
            color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4
        )
        fig2.update_traces(textinfo="label+percent")
        fig2.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        avg_price_country = (
            df_filtered.groupby("country_of_origin")["price_usd"].mean()
            .sort_values(ascending=False).reset_index()
        )
        avg_price_country.columns = ["Country", "Avg Price (USD)"]
        fig3 = px.bar(
            avg_price_country, x="Country", y="Avg Price (USD)",
            title="Avg Price by Country of Origin",
            color="Avg Price (USD)",
            color_continuous_scale="Oranges"
        )
        fig3.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        avg_range_seg = (
            df_filtered.groupby("market_segment")["range_miles"].mean()
            .sort_values(ascending=False).reset_index()
        )
        avg_range_seg.columns = ["Segment", "Avg Range (miles)"]
        fig4 = px.bar(
            avg_range_seg, x="Segment", y="Avg Range (miles)",
            title="Avg Range by Market Segment",
            color="Avg Range (miles)",
            color_continuous_scale="Greens"
        )
        fig4.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — Exploratory Analysis
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            df_filtered, x="price_usd", nbins=50,
            title="Price Distribution",
            labels={"price_usd": "Price (USD)"},
            color_discrete_sequence=["#4361ee"]
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            df_filtered, x="range_miles", nbins=40,
            title="Range Distribution",
            labels={"range_miles": "Range (miles)"},
            color_discrete_sequence=["#3a0ca3"]
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Price vs Key Features")
    x_axis = st.selectbox("X axis", [
        "range_miles", "battery_capacity_kwh", "horsepower",
        "charging_speed_kw", "acceleration_0_60_mph", "top_speed_mph",
        "annual_sales_units", "customer_rating"
    ])
    color_by = st.selectbox("Color by", ["market_segment", "country_of_origin", "drive_type", "body_type"])

    fig = px.scatter(
        df_filtered, x=x_axis, y="price_usd",
        color=color_by, hover_data=["brand", "model", "year"],
        title=f"Price vs {x_axis.replace('_', ' ').title()}",
        opacity=0.7,
        trendline="ols"
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Correlation Heatmap")
    num_cols = ["price_usd", "battery_capacity_kwh", "range_miles", "charging_speed_kw",
                "acceleration_0_60_mph", "top_speed_mph", "horsepower", "torque_nm",
                "cargo_volume_cubic_ft", "weight_kg", "safety_rating", "autopilot_level",
                "annual_sales_units", "customer_rating", "warranty_years"]
    corr = df_filtered[num_cols].corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Feature Correlation Matrix"
    )
    fig_corr.update_layout(height=550)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("### Box Plot — Price by Category")
    cat_col = st.selectbox("Group by", ["market_segment", "brand", "country_of_origin", "drive_type", "body_type"])
    fig_box = px.box(
        df_filtered, x=cat_col, y="price_usd",
        color=cat_col, title=f"Price Distribution by {cat_col.replace('_',' ').title()}",
        points="outliers"
    )
    fig_box.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_box, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — ML Model
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Machine Learning — Price Prediction Models")
    st.info("Training on the **full dataset** (filters don't apply here) for the best model quality.")

    with st.spinner("Training models... this takes a few seconds."):
        results, le_dict, feature_cols, X_test, y_test = train_models(df)

    # Model comparison table
    st.markdown("### Model Comparison")
    comp_df = pd.DataFrame([
        {
            "Model": name,
            "R² Score": f"{r['r2']:.4f}",
            "MAE": f"${r['mae']:,.0f}",
            "RMSE": f"${r['rmse']:,.0f}",
            "CV R² (mean ± std)": f"{r['cv_r2_mean']:.3f} ± {r['cv_r2_std']:.3f}"
        }
        for name, r in results.items()
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    best_model_name = max(results, key=lambda k: results[k]["r2"])
    st.success(f"🏆 Best model: **{best_model_name}** — R² = {results[best_model_name]['r2']:.4f}")

    # Model selector
    selected_model = st.selectbox("Select model to inspect", list(results.keys()), index=list(results.keys()).index(best_model_name))
    res = results[selected_model]

    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{res['r2']:.4f}")
    col2.metric("MAE", f"${res['mae']:,.0f}")
    col3.metric("RMSE", f"${res['rmse']:,.0f}")

    col_a, col_b = st.columns(2)

    with col_a:
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=res["y_test"], y=res["y_pred"],
            mode="markers", opacity=0.5,
            marker=dict(color="#4361ee", size=5),
            name="Predictions"
        ))
        mn = min(res["y_test"].min(), res["y_pred"].min())
        mx = max(res["y_test"].max(), res["y_pred"].max())
        fig_pred.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx],
            mode="lines", line=dict(color="red", dash="dash", width=2),
            name="Perfect fit"
        ))
        fig_pred.update_layout(
            title="Actual vs Predicted Price",
            xaxis_title="Actual Price (USD)",
            yaxis_title="Predicted Price (USD)",
            height=400
        )
        st.plotly_chart(fig_pred, use_container_width=True)

    with col_b:
        residuals = np.array(res["y_test"]) - np.array(res["y_pred"])
        fig_res = px.histogram(
            x=residuals, nbins=50,
            title="Residuals Distribution",
            labels={"x": "Residual (Actual − Predicted)"},
            color_discrete_sequence=["#f72585"]
        )
        fig_res.add_vline(x=0, line_dash="dash", line_color="black")
        fig_res.update_layout(height=400)
        st.plotly_chart(fig_res, use_container_width=True)

    # Feature importance (RF & GB only)
    if selected_model in ["Random Forest", "Gradient Boosting"]:
        st.markdown("### Feature Importance")
        model_obj = res["model"]
        importances = (
            pd.Series(model_obj.feature_importances_, index=feature_cols)
            .sort_values(ascending=True).reset_index()
        )
        importances.columns = ["Feature", "Importance"]
        fig_imp = px.bar(
            importances, x="Importance", y="Feature",
            orientation="h", title=f"Feature Importance — {selected_model}",
            color="Importance",
            color_continuous_scale="Purples"
        )
        fig_imp.update_layout(coloraxis_showscale=False, height=500)
        st.plotly_chart(fig_imp, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 — Price Predictor
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 🔮 Predict EV Price")
    st.markdown("Fill in the specs below and the best model will estimate the price.")

    with st.spinner("Loading model..."):
        results_p, le_dict_p, feature_cols_p, _, _ = train_models(df)

    best = results_p[max(results_p, key=lambda k: results_p[k]["r2"])]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Performance specs**")
        battery = st.slider("Battery capacity (kWh)", 20.0, 130.0, 75.0, 0.5)
        range_mi = st.slider("Range (miles)", 100, 500, 300)
        charging = st.slider("Charging speed (kW)", 20.0, 360.0, 150.0, 5.0)
        accel = st.slider("0–60 mph (seconds)", 2.0, 10.0, 5.0, 0.1)
        top_speed = st.slider("Top speed (mph)", 90, 200, 150)
        hp = st.slider("Horsepower", 100, 1050, 500)
        torque = st.slider("Torque (Nm)", 150, 1000, 500)

    with col2:
        st.markdown("**Body & features**")
        drive = st.selectbox("Drive type", sorted(df["drive_type"].unique()))
        body = st.selectbox("Body type", sorted(df["body_type"].unique()))
        seats = st.selectbox("Seating capacity", sorted(df["seating_capacity"].unique()))
        cargo = st.slider("Cargo volume (cu ft)", 10.0, 85.0, 40.0, 0.5)
        weight = st.slider("Weight (kg)", 1400, 3000, 2000)
        safety = st.selectbox("Safety rating", [3, 4, 5])
        autopilot = st.selectbox("Autopilot level", [0, 1, 2, 3])

    with col3:
        st.markdown("**Market info**")
        segment = st.selectbox("Market segment", sorted(df["market_segment"].unique()))
        country = st.selectbox("Country of origin", sorted(df["country_of_origin"].unique()))
        year = st.selectbox("Model year", sorted(df["year"].unique(), reverse=True))
        warranty = st.selectbox("Warranty (years)", sorted(df["warranty_years"].unique()))

    st.markdown("---")

    if st.button("⚡ Predict Price", type="primary", use_container_width=False):
        input_dict = {
            "battery_capacity_kwh": battery, "range_miles": range_mi,
            "charging_speed_kw": charging, "acceleration_0_60_mph": accel,
            "top_speed_mph": top_speed, "horsepower": hp, "torque_nm": torque,
            "seating_capacity": seats, "cargo_volume_cubic_ft": cargo,
            "weight_kg": weight, "safety_rating": safety,
            "autopilot_level": autopilot, "warranty_years": warranty,
            "year": year, "drive_type": drive, "body_type": body,
            "market_segment": segment, "country_of_origin": country
        }

        # Encode categoricals
        for col_name, le in le_dict_p.items():
            val = input_dict[col_name]
            if val in le.classes_:
                input_dict[col_name] = le.transform([val])[0]
            else:
                input_dict[col_name] = 0

        X_input = pd.DataFrame([input_dict])[feature_cols_p]
        predicted = best["model"].predict(X_input)[0]
        r2 = best["r2"]

        margin = predicted * (1 - r2) * 0.5

        st.markdown("### Predicted Price")
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated Price", f"${predicted:,.0f}")
        c2.metric("Lower bound", f"${max(0, predicted - margin):,.0f}")
        c3.metric("Upper bound", f"${predicted + margin:,.0f}")

        st.markdown(f"*Confidence interval based on model R² = {r2:.4f}. Model: **{max(results_p, key=lambda k: results_p[k]['r2'])}***")

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=predicted,
            number={"prefix": "$", "valueformat": ",.0f"},
            gauge={
                "axis": {"range": [0, 250000]},
                "bar": {"color": "#4361ee"},
                "steps": [
                    {"range": [0, 40000], "color": "#d4edda"},
                    {"range": [40000, 80000], "color": "#fff3cd"},
                    {"range": [80000, 130000], "color": "#fde2e4"},
                    {"range": [130000, 250000], "color": "#f8d7da"},
                ],
                "threshold": {"line": {"color": "red", "width": 3}, "thickness": 0.75, "value": predicted}
            },
            title={"text": "Predicted Price (USD)"}
        ))
        fig_gauge.update_layout(height=320)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Compare to similar vehicles
        st.markdown("### Similar vehicles in the dataset")
        similar = df[
            (df["market_segment"] == segment) &
            (df["range_miles"].between(range_mi - 40, range_mi + 40))
        ][["brand", "model", "year", "variant", "price_usd", "range_miles", "horsepower", "market_segment"]].copy()
        similar["price_usd"] = similar["price_usd"].map("${:,.0f}".format)
        st.dataframe(similar.head(10), use_container_width=True, hide_index=True)
