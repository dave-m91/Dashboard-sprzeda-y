from calendar import week
import stat
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np


class TrendPredictor:
    def __init__(self, data: pd.DataFrame):
        """Inicjalizacja danych"""
        self.data = data
        self.model = LinearRegression()

    def prepare_series(self, index_id: str):
        """Przygotowanie danych"""
        mask = self.data["index_id"] == index_id
        temp_df = self.data[mask].copy()

        if temp_df.empty:
            print(f"Brak danych dla indeksu: {index_id}")
            return None
        
        temp_df = temp_df.set_index("date")
        #Agregacja tygodni
        weekly_series = temp_df["sales"].resample("W").sum()
        # Uzupełnienie luk w przypadku braku sprzedaży
        weekly_series = weekly_series.asfreq("W", fill_value=0)
        return weekly_series

    def get_trend_metrics(self, series: pd.Series):
        """Regresja liniowa i R2"""
        y = series.values
        X = np.arange(len(y)).reshape(-1,1)
        self.model.fit(X, y)

        slope = self.model.coef_[0] # nachylenie z funkcji liniowej
        r2 = self.model.score(X, y)

        return {
            "slope": slope,
            "r2": r2,
            "intercept": self.model.intercept_
        }

    def predict_future(self, series: pd.Series, steps: int=4):
        """Prognoza na ilośc kroków w przód"""
        y = series.values
        X = np.arange(len(y)).reshape(-1,1)

        self.model.fit(X, y)
        last_x= X[-1,0]
        feature_X = np.arange(last_x + 1, last_x + 1 + steps).reshape(-1,1)

        forecast_values = self.model.predict(feature_X)
        forecast_values = np.maximum(forecast_values, 0)

        last_date = series.index[-1]
        feature_dates = pd.date_range(start=last_date, periods = steps+1, freq="W")[1:]

        return pd.DataFrame({
            "date": feature_dates,
            "predicted_sales": forecast_values
        }).set_index("date")

    def analyze_current_state(self, series: pd.Series, window: int=4):
        """Analiza dynamiki:
            Porównanie średniej sprzedaży z ostatniego okna
            do średniej z poprzedniego okna"""
        if len(series) < window *2:
            return {"status": "Nieznany (za mało danych)",
                    "velocity": 0.0, "current_avg":0.0, "previous_avg":0.0}
        
        # Podział na obecne i poprzednie
        current_period = series.iloc[-window:]
        previous_period = series.iloc[-2*window : -window]

        # Średnie
        curr_mean = current_period.mean()
        prev_mean = previous_period.mean()

        # Oblicznie % zmiany
        if prev_mean == 0:
            velocity = 1.0 if curr_mean > 0 else 0.0
        else:
            velocity = (curr_mean - prev_mean) / prev_mean

        if velocity > 0.10:
            status = "Wzrost"
        elif velocity < -0.10:
            status = "Spadek"
        else:
            status = "Stablinie"

        return {
            "status": status,
            "velocity": velocity,
            "current_avg": curr_mean,
            "previous_avg": prev_mean
        }

    def calculate_slope(self, series: pd.Series):
        """
        Oblicza przyśpiesznie trendu
        """
        n = len(series)
        if n < 8: # minimum danych
            return "Za mało danych"
        
        mid_point = n // 2

        # Nachylenie pierwszej połowy
        metrics_past = self.get_trend_metrics(series.iloc[:mid_point])
        slope_past = metrics_past["slope"]

        # Nachylenie drugiej połowy
        metrics_now = self.get_trend_metrics(series[mid_point:])
        slope_now = metrics_now["slope"]

        diff = slope_now - slope_past

        if diff > 0.05: return "Przyśpiesza"
        if diff < -0.05: return "Zwalnia"
        return "Stałe tempo"
    
    def generate_batch_ranking(self, min_sales_treshold: int =10):
        """
        Przetwarza uniklane produkty, generując tabelę KPI
        """
        unique_ids = self.data["index_id"].unique()
        results = []

        for index_id in unique_ids:
            series = self.prepare_series(index_id)

            if series is None or series.sum() < min_sales_treshold:
                continue
            
            try:
                metrics = self.get_trend_metrics(series)
                state = self.analyze_current_state(series)
                momentum = self.calculate_slope(series)

                results.append({
                    "index_id": index_id,
                    "total_sales": series.sum(),
                    "last_sales": series.iloc[-1],
                    "slope": round(metrics["slope"],3),
                    "velocity_pct": round(state["velocity"],3),
                    "trend_status": momentum,
                    "r2": round(metrics["r2"],2)
                })
            except Exception as e:
                continue

        ranking_df = pd.DataFrame(results)

        if not ranking_df.empty:
            ranking_df = ranking_df.sort_values(by="velocity_pct", ascending=False)
        return ranking_df

    def generate_batch_ranking_fast(self, windows_week: int=12, min_sales: int=10):
        max_date = self.data["date"].max()
        start_date = max_date - pd.Timedelta(weeks=windows_week)

        df_filtered = self.data[self.data["date"] > start_date].copy()

        df_weekly = df_filtered.set_index("date").groupby(
            ["index_id", pd.Grouper(freq="W")]
            )["sales"].sum().reset_index()
        
        min_date = df_weekly["date"].min()
        df_weekly["x"] = (df_weekly["date"] - min_date).dt.days /7

        df_weekly["xy"] = df_weekly["x"] * df_weekly["sales"]
        df_weekly["xx"] = df_weekly["x"] **2

        stats = df_weekly.groupby("index_id").agg(
            n=("sales", "count"),
            sum_x=("x", "sum"),
            sum_y=("sales", "sum"),
            sum_xy=("xy", "sum"),
            sum_xx=("xx", "sum"),
            total_sales=("sales", "sum")
        )
        stats = stats[stats["total_sales"] >= min_sales]
        #wzór n * sum(x^2) - (sum(x))^2
        denominator = (stats["n"] * stats["sum_xx"] - stats["sum_x"]**2)
        numerator = (stats["n"] * stats["sum_xy"] - stats["sum_x"] * stats["sum_y"])

        stats["slope"] = np.where(
            denominator != 0,
            numerator / denominator,
            0.0
        )
        check_date = max_date - pd.Timedelta(weeks=3)
        active_ids = df_filtered[df_filtered["date"]>check_date]["index_id"].unique()
        stats = stats[stats.index.isin(active_ids)]
        #uproszczone velocity liczone slope / średnia sprzedaż = % wzrostu
        avg_sales = stats["sum_y"] / stats["n"]
        stats["velocity_proxy"] = np.where(
            avg_sales > 0,
            stats["slope"] / avg_sales,
            0.0
        )

        
        result = stats[["total_sales","slope", "velocity_proxy"]].reset_index()
        result = result.sort_values(by="velocity_proxy", ascending=False)
        return result