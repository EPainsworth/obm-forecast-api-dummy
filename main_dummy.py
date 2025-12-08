# === app.py ===
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from typing import Optional
import pandas as pd
import numpy as np

# --- API initialisatie ---
app = FastAPI(
    title="ObM Financiële Prognose API",
    description="API voor financiële voorspellingen, afwijkingen en modeluitleg.",
    version="2.0"
)

# --- 1️⃣ Data inladen ---
df_lvl3 = pd.read_csv("dummy_data/model_forecast_2025-12_dummy.csv")
df_summary_fin = pd.read_csv("dummy_data/model_financial_summary_2025_dummy.csv")
df_summary_fin["lvl3"] = df_summary_fin["lvl3"].astype(str)

df_text_fin = pd.read_csv("dummy_data/model_financial_text_summary_2025_dummy.csv")
df_text_fin["lvl3"] = df_text_fin["lvl3"].astype(str)
df_text_fin["Toelichting"] = df_text_fin["Toelichting"].astype(str)



# Zorg dat categorische kolommen tekst zijn (voor API output)
str_cols = ["lvl3"]
for col in str_cols:
    if col in df_lvl3.columns:
        df_lvl3[col] = df_lvl3[col].astype(str)

# Volledige dataset (lvl4)
try:
    df_full = pd.read_csv("dummy_data/model_forecast_full_dummy.csv")

    # Kolomnamen opschonen (verwijder spaties, rare tekens)
    df_full.columns = df_full.columns.str.strip()

    # ✅ Kolomnamen standaardiseren (rename als ze anders heten)
    rename_map = {
        "Afwijking van budget (€)": "Afwijking (€)",
        "Afwijking van budget (%)": "Afwijking (%)",
        "Jaar_maand": "Jaar-maand",
        "jaar_maand": "Jaar-maand"
    }
    df_full = df_full.rename(columns=rename_map)

    # ✅ Tekstkolommen converteren naar str
    text_cols = ["Vestiging", "lvl3", "lvl4", "Jaar-maand"]
    for col in text_cols:
        if col in df_full.columns:
            df_full[col] = df_full[col].fillna("").astype(str)

    # ✅ Numerieke kolommen vullen met 0 om NaN’s te vermijden
    for col in ["Realisatie", "Budget"]:
        if col in df_full.columns:
            df_full[col] = df_full[col].fillna(0)

    # ✅ Afwijking berekenen als ontbrekend
    if "Afwijking (€)" not in df_full.columns:
        df_full["Afwijking (€)"] = df_full["Realisatie"] - df_full["Budget"]
    if "Afwijking (%)" not in df_full.columns:
        df_full["Afwijking (%)"] = np.where(
            df_full["Budget"] == 0,
            0,
            100 * (df_full["Afwijking (€)"] / df_full["Budget"])
        )

except FileNotFoundError:
    df_full = pd.DataFrame()
print("Kolommen in df_full:", list(df_full.columns))
print(df_full.head(3))

# Feature importance
try:
    df_importance = pd.read_csv("dummy_data/model_feature_importance_dummy.csv")
    df_importance["Feature"] = df_importance["Feature"].astype(str)
except FileNotFoundError:
    df_importance = pd.DataFrame({"Feature": [], "Importance": []})

# Zorg dat afwijking (%) correct berekend is
if "Afwijking van budget (€)" not in df_lvl3.columns:
    df_lvl3["Afwijking van budget (€)"] = df_lvl3["Realisatie"] - df_lvl3["Budget"]
if "Afwijking van budget (%)" not in df_lvl3.columns:
    df_lvl3["Afwijking van budget (%)"] = (df_lvl3["Afwijking van budget (€)"] / df_lvl3["Budget"]) * 100


# === 2️⃣ Pydantic-modellen ===
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

class FinancialSummaryRow(BaseModel):
    lvl3: str
    Realisatie_YTD_jan_apr: float
    Budget_YTD_jan_apr: float
    Verschil_YTD_euro: float
    Verschil_YTD_pct: float
    Realisatie_jaar_prognose: float
    Budget_jaar: float
    Verschil_jaar_euro: float
    Verschil_jaar_pct: float

class FinancialTextSummary(BaseModel):
    lvl3: str
    Toelichting: str


# === 3️⃣ Helpers ===
def calc_kpi(df: pd.DataFrame, kpi_threshold: float = 10.0):
    within = (df["Afwijking van budget (%)"].abs() < kpi_threshold).sum()
    total = len(df)
    return {
        "Aantal_binnen_KPI": int(within),
        "Totaal_lvl3": total,
        "Percentage_binnen_KPI": round((within / total) * 100, 2)
    }


# === 4️⃣ API-endpoints ===

## 📊 Hoofdniveau (lvl3)
@app.get("/predict", response_model=List[ForecastResult])
def get_predictions():
    """Geeft alle voorspelde realisaties per lvl3 terug"""
    return [
        {
            "lvl3": row["lvl3"],
            "Realisatie": round(row["Realisatie"], 2),
            "Budget": round(row["Budget"], 2),
            "Afwijking_van_budget_euro": round(row["Afwijking van budget (€)"], 2),
            "Afwijking_van_budget_pct": round(row["Afwijking van budget (%)"], 2),
        }
        for _, row in df_lvl3.iterrows()
    ]


## 🧾 Detailniveau (lvl4)
@app.get("/details/lvl4", response_model=List[ForecastLvl4])
def get_lvl4_details():
    """Geeft alleen de voorspelde maanden (2025-10 t/m 2025-12) terug"""
    if df_full.empty:
        return []

    # 🔹 Filter alleen op voorspelde maanden
    print("Unieke waarden in Jaar-maand:", df_full["Jaar-maand"].unique()[:15])
    print("Datatype van Jaar-maand:", df_full["Jaar-maand"].dtype)
    forecast_months = ["2025-10-01", "2025-11-01", "2025-12-01"]
    results = df_full[df_full["Jaar-maand"].isin(forecast_months)].copy()

    results = results.replace([np.inf, -np.inf], np.nan).fillna(0)

    return [
        {
            "Vestiging": r["Vestiging"],
            "lvl3": r["lvl3"],
            "lvl4": r["lvl4"],
            "Jaar_maand": r["Jaar-maand"],
            "Realisatie": round(r["Realisatie"], 2),
            "Budget": round(r["Budget"], 2),
            "Afwijking_euro": float(round(r.get("Afwijking (€)", 0) or 0, 2)),
            "Afwijking_pct": round(r["Afwijking (%)"], 2),
        }
        for _, r in results.iterrows()
    ]

## 🔝 Grootste afwijkingen per lvl4
@app.get("/details/lvl4/top", response_model=List[ForecastLvl4])
def get_top_lvl4(n: int = 10, sort_by: str = "euro"):
    """Geeft top N grootboekposten met de grootste afwijking in december 2025"""
    if df_full.empty:
        return []

    # 🔹 Filter alleen op de laatste voorspelde maand (december)
    print("Unieke waarden in Jaar-maand:", df_full["Jaar-maand"].unique()[:15])
    print("Datatype van Jaar-maand:", df_full["Jaar-maand"].dtype)
    df_dec = df_full[df_full["Jaar-maand"] == "2025-12-01"].copy()
    if df_dec.empty:
        return []

    key = "Afwijking (€)" if sort_by == "euro" else "Afwijking (%)"
    df_top = df_dec.sort_values(key, ascending=False).head(n)

    df_top = df_top.replace([np.inf, -np.inf], np.nan).fillna(0)

    return [
        {
            "Vestiging": r["Vestiging"],
            "lvl3": r["lvl3"],
            "lvl4": r["lvl4"],
            "Jaar_maand": r["Jaar-maand"],
            "Realisatie": round(r["Realisatie"], 2),
            "Budget": round(r["Budget"], 2),
            "Afwijking_euro": float(round(r.get("Afwijking (€)", 0) or 0, 2)),
            "Afwijking_pct": round(r["Afwijking (%)"], 2),
        }
        for _, r in df_top.iterrows()
    ]

## 📈 KPI-overzicht
@app.get("/kpi", response_model=KPIStatus)
def get_kpi_status():
    """Geeft overzicht van KPI-naleving (<10% afwijking van budget)"""
    return df_summary_fin()


## 🧠 Feature importance
@app.get("/feature-importance", response_model=List[FeatureImportance])
def get_feature_importance():
    """Geeft de echte variabelenbelangrijkheid uit het RandomForest-model"""
    if df_importance.empty:
        return []
    return df_importance.to_dict(orient="records")


## 📅 Trend per maand
@app.get("/trend")
def get_trend():
    """Toont trend van voorspelde realisatie vs. budget per maand op lvl3"""
    if df_full.empty:
        return {"message": "Trenddata niet beschikbaar."}
    trend = (
        df_full.groupby(["Jaar-maand", "lvl3"], as_index=False)[["Realisatie", "Budget"]]
        .sum()
        .assign(Afwijking_pct=lambda d: 100 * (d["Realisatie"] - d["Budget"]) / d["Budget"])
    )
    return trend.to_dict(orient="records")

@app.get("/summary", response_model=List[FinancialSummaryRow])
def get_financial_summary():
    rows = []
    for _, r in df_summary_fin.iterrows():
        rows.append({
            "lvl3": r["lvl3"],
            "Realisatie_YTD_jan_apr": r["Realisatie_YTD_jan_apr"],
            "Budget_YTD_jan_apr": r["Budget_YTD_jan_apr"],
            "Verschil_YTD_euro": r["Verschil_YTD_€"],
            "Verschil_YTD_pct": r["Verschil_YTD_%"],
            "Realisatie_jaar_prognose": r["Realisatie_jaar_prognose"],
            "Budget_jaar": r["Budget_jaar"],
            "Verschil_jaar_euro": r["Verschil_jaar_€"],
            "Verschil_jaar_pct": r["Verschil_jaar_%"],
        })
    return rows

@app.get("/summary/text", response_model=List[FinancialTextSummary])
def get_financial_text_summary():
    return df_text_fin.to_dict(orient="records")
