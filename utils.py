import streamlit as st
import pandas as pd

# wczytywanie danych
@st.cache_data
def load_data():
    df = pd.read_csv("data/prognoza_sprzedazy_indexy.csv")

    df.rename(columns={"Day [Date] PE-D01":"date",
                       "Units Sold ST-010":"sales",
                       "Product [Index] PR-P02":"index_id",
                       "Segment [Name] PR-S02":"Segment",
                       "APG L2 [Name] PR-AG6": "Grupa L2",
                       "Manufacturer [Local Name] PR-M09": "Producent"},
                       inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    return df