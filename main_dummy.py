# === app.py ===
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from pathlib import Path

# --- API initialisatie ---
app = FastAPI(
    title="ObM Financiële Prognose API (Dummy Data)",
    description="Dummy-versie van de API voor Copilot-training, zonder vertrouwelijke data.",
    version="2.0"
)

# --- 1️⃣ Data inladen (dummy) ---
DATA_PATH = Path("data")

df_lvl3 = pd.read_csv("dummy_data/model_forecast_2025-12.csv")
df_full = pd.read_csv("dummy_data/model_forecast_full.csv")
df_importance = pd.read_csv("dummy_data/model_feature_importance.csv")

# --- Kolommen normaliseren ---
for df in [df_lvl3, df_full]:
    df.columns = df.columns.str.strip()
if "Jaar-maand" in df_full.columns:
    df_full["Jaar-maand"] = pd.to_datetime(df_full["Jaar-maand"]).dt.strftime("%Y-%m")

# --- 2️⃣ Pydantic-modellen ---
class ForecastResult(BaseModel):
    lvl3: str
    Realisatie: float
    Budget: float
    Afwijking_van_budget_euro: float
    Afwijking_van_budget_pct: float

class ForecastLvl4(BaseModel):
    Vestiging: str
    lvl3: str
    lvl4: str
    Jaar_maand: str
    Realisatie: float
    Budget: float
    Afwijking_euro: float
    Afwijking_pct: float

class Summary(BaseModel):
    lvl3: str
    Toelichting: str

class KPIStatus(BaseModel):
    Aantal_binnen_KPI: int
    Totaal_lvl3: int
    Percentage_binnen_KPI: float

class FeatureImportance(BaseModel):
    Feature: str
    Importance: float


# --- Helperfunctie ---
def calc_kpi(df: pd.DataFrame, kpi_threshold: float = 10.0):
    within = (df["Afwijking van budget (%)"].abs() < kpi_threshold).sum()
    total = len(df)
    return {
        "Aantal_binnen_KPI": int(within),
        "Totaal_lvl3": total,
        "Percentage_binnen_KPI": round((within / total) * 100, 2)
    }


# --- 3️⃣ API-endpoints ---

@app.get("/predict", response_model=List[ForecastResult])
def get_predictions():
    """Voorspelde realisaties per categorie (lvl3)"""
    return [
        {
            "lvl3": row["lvl3"],
            "Realisatie": float(row["Realisatie"]),
            "Budget": float(row["Budget"]),
            "Afwijking_van_budget_euro": float(row["Afwijking van budget (€)"]),
            "Afwijking_van_budget_pct": float(row["Afwijking van budget (%)"]),
        }
        for _, row in df_lvl3.iterrows()
    ]


@app.get("/details/lvl4", response_model=List[ForecastLvl4])
def get_lvl4_details():
    """Alle voorspellingen voor de maanden 2025-10 t/m 2025-12"""
    forecast_months = ["2025-10", "2025-11", "2025-12"]
    results = df_full[df_full["Jaar-maand"].isin(forecast_months)].copy()
    results = results.fillna(0)
    return [
        {
            "Vestiging": r["Vestiging"],
            "lvl3": r["lvl3"],
            "lvl4": r["lvl4"],
            "Jaar_maand": r["Jaar-maand"],
            "Realisatie": float(r["Realisatie"]),
            "Budget": float(r["Budget"]),
            "Afwijking_euro": float(r["Afwijking (€)"]),
            "Afwijking_pct": float(r["Afwijking (%)"]),
        }
        for _, r in results.iterrows()
    ]


@app.get("/details/lvl4/top", response_model=List[ForecastLvl4])
def get_top_lvl4(n: int = 5, sort_by: str = "euro"):
    """Top N grootboekposten met grootste afwijking (dummydata)"""
    df_dec = df_full[df_full["Jaar-maand"] == "2025-12"]
    key = "Afwijking (€)" if sort_by == "euro" else "Afwijking (%)"
    df_top = df_dec.sort_values(key, ascending=False).head(n)
    return [
        {
            "Vestiging": r["Vestiging"],
            "lvl3": r["lvl3"],
            "lvl4": r["lvl4"],
            "Jaar_maand": r["Jaar-maand"],
            "Realisatie": float(r["Realisatie"]),
            "Budget": float(r["Budget"]),
            "Afwijking_euro": float(r["Afwijking (€)"]),
            "Afwijking_pct": float(r["Afwijking (%)"]),
        }
        for _, r in df_top.iterrows()
    ]


@app.get("/summary", response_model=List[Summary])
def get_summary():
    """Tekstuele toelichting per kostenpost (dummydata)"""
    summaries = []
    for _, row in df_lvl3.iterrows():
        trend = "boven budget" if row["Afwijking van budget (€)"] > 0 else "onder budget"
        binnen = abs(row["Afwijking van budget (%)"]) < 10
        kpi = "✅ binnen KPI" if binnen else "⚠️ buiten KPI"
        summaries.append({
            "lvl3": row["lvl3"],
            "Toelichting": f"De categorie {row['lvl3']} ligt {abs(row['Afwijking van budget (%)']):.2f}% {trend} t.o.v. het budget ({kpi})."
        })
    return summaries


@app.get("/kpi", response_model=KPIStatus)
def get_kpi_status():
    """KPI-nalevingsoverzicht"""
    return calc_kpi(df_lvl3)


@app.get("/feature-importance", response_model=List[FeatureImportance])
def get_feature_importance():
    """Belangrijkste modelvariabelen (dummy)"""
    return df_importance.to_dict(orient="records")