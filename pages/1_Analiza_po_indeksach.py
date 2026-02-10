from turtle import title, width
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import sys
import os

from model import TrendPredictor
from utils import load_data


# Konfiguracja strony
st.set_page_config(page_title="Analiza Trendów Sprzedaży", layout="wide")

try:
    df = load_data()
    predictor = TrendPredictor(df)
except Exception as e:
    st.error(f"Błąd wczytywania danych: {e}")
    st.stop()

st.title("Analiza sprzedaży")

st.sidebar.header("Filtry")
unique_segments = sorted(df["Segment"].unique())
selected_segment = st.sidebar.selectbox("1. Wybierz segment", ["Wszystkie"] + unique_segments)
if selected_segment == "Wszystkie":
    cats_in_seg = sorted(df["Grupa L2"].unique())
else:
    cats_in_seg = sorted(df[df["Segment"] == selected_segment]["Grupa L2"].unique())

selected_category = st.sidebar.selectbox("2. Kategoria (do indeksu)", ["Wszystkie"] + cats_in_seg)

mask = pd.Series([True] * len(df), index = df.index)
if selected_segment != "Wszystkie":
    mask &= (df["Segment"] == selected_segment)
if selected_category != "Wszystkie":
    mask &= (df["Grupa L2"] == selected_category)

inds_in_cat = sorted(df[mask]["index_id"].unique())
selected_index = st.sidebar.selectbox("3. Indeks", inds_in_cat)

# Zarządzanie pamięcia podręczną
if "ranking_df" not in st.session_state:
    st.session_state.ranking_df = None

tab1, tab2 = st.tabs(["Top wzrosty", "Analiza szczegółowa"])

with tab1:
    st.header(f"Najszybciej rosnące produkty: {selected_segment}")
    if st.button("Generuj / Odśwież ranking"):
        with st.spinner("Analizuję trendy wszystkich produktów..."):
            raw = predictor.generate_batch_ranking_fast(windows_week=12, min_sales=10)
            feats = df[["index_id", "Segment", "Producent"]].drop_duplicates().set_index("index_id")
            enriched = raw.set_index("index_id").join(feats).reset_index()
            st.session_state.ranking_df = enriched

    if st.session_state.ranking_df is not None:
        view_df = st.session_state.ranking_df.copy()

        if selected_segment != "Wszystkie":
            view_df = view_df[view_df["Segment"] == selected_segment]
        
        st.dataframe(
            view_df,
            column_config={
                "index_id": "Indeks",
                "total_sales": "Sprzedaż (12 tyg.)",
                "velocity_proxy": st.column_config.NumberColumn(
                    "Dynamika %", format="percent"
                ),
                "slope": st.column_config.NumberColumn(
                    "Trend (szt/tydz)", format="%.2f"
                ),
                "trend_status": st.column_config.TextColumn("Momentum"),
            },
            width="stretch",
            hide_index=True
        )

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Znaleziono {len(view_df)} aktywnych produktów.")

            # Wykres rozkładu momentum
            try:
                status_counts = view_df["trend_status"].value_counts()

                fig2 = go.Figure()

                fig2.add_trace(go.Pie(labels = status_counts.index,
                                    values=status_counts.values,
                                    hole=0.4,
                                    textinfo="label+percent"))
                
                colors_map = {
                    "Przyspiesza 🚀": "#2ecc71",
                    "Zwalnia 🔻": "#e74c3c",
                    "Stabilny ➡️": "#95a5a6"
                }
                fig2.update_traces(marker=dict(colors=[colors_map.get(x, "#333")
                                                    for x in status_counts.index]))
                
                fig2.update_layout(
                    title="Rozkład momentum",
                    margin=dict(t=30, b=0, l=0, r=0),
                    height=300
                )

                st.plotly_chart(fig2, width="stretch")
            except IndexError as e:
                st.error(f'Brak danych dla segmentu {view_df["Segment"]}')
                st.stop()
        with col2:
            top_grower = view_df.iloc[0]
            st.success(f"Lider wzrostów: {top_grower['index_id']} (+{top_grower['velocity_proxy']:.1%})")

    else:
        st.warning("Brak parametrów spełniajacych kryteria")

with tab2:
    st.sidebar.markdown("---")
    forecast_steps = st.sidebar.slider("Horyzont prognozy (tygodnie)", 2,24,8)

    # Główna przestrzeń
    st.title(f"Analiza: {selected_index}")

    # Przygotowanie danych z klasy
    series = predictor.prepare_series(selected_index)

    if series is not None:
        metrics = predictor.get_trend_metrics(series)
        current_state = predictor.analyze_current_state(series)
        momentum = predictor.calculate_slope(series)
        forecast_df = predictor.predict_future(series, steps=forecast_steps)

        # KPI scorecards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Aktualna Dynamika",
                    f"{current_state['velocity']:.1%}",
                    delta = current_state["status"], delta_color="normal", delta_arrow="auto")
        with col2:
            st.metric("Status Trendu",
                    momentum,
                    help="Czy trend przyśpiesza czy zwalnia?")
        with col3:
            st.metric("Nachylenie",
                    f"{metrics['slope']:.2f} szt./tydz.")
        with col4:
            st.metric("Wiarygodność (R2)",
                    f"{metrics['r2']:.2f}")
            
        #wykres
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x = series.index,
            y = series.values,
            mode = "lines+markers",
            line = dict(color="royalblue", width=2)
        ))

        # Prognoza
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df["predicted_sales"],
            mode="lines",
            name="Prognoza",
            line=dict(color="green", dash="dot", width=3)
        ))

        X_nums = np.arange(len(series))
        y_trend = metrics["slope"] * X_nums + metrics["intercept"]

        fig.add_trace(go.Scatter(
            x=series.index,
            y=y_trend,
            mode="lines",
            name="Linia trendu",
            line=dict(color="red", width=1, dash="dash"),
        ))

        fig.update_layout(
            title="Sprzedaż tygodniowa + prognoza",
            xaxis_title="Data",
            yaxis_title="Sztuki",
            hovermode="x unified"
        )

        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("Nie udało się pobrać danyych dla tego indeksu")