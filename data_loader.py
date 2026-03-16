"""
data_loader.py
══════════════
Carga todos los CSVs de una liga UNA SOLA VEZ y pre-calcula
todas las estadísticas necesarias para todos los módulos.

El objeto `LeagueData` es el que se pasa a cada módulo.
"""

import glob
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

DATA_ROOT = Path("data")


# ──────────────────────────────────────────────────────────────
#  CARGA Y LIMPIEZA
# ──────────────────────────────────────────────────────────────

def list_leagues() -> list[str]:
    if not DATA_ROOT.exists():
        return []
    return sorted(d.name for d in DATA_ROOT.iterdir() if d.is_dir())


def _load_csvs(league: str, max_seasons: Optional[int]) -> pd.DataFrame:
    folder = DATA_ROOT / league
    files  = sorted(glob.glob(str(folder / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No hay CSVs en: {folder}")
    if max_seasons:
        files = files[-max_seasons:]

    dfs = []
    for i, path in enumerate(files):
        df = pd.read_csv(path, low_memory=False)
        df["_season"]  = Path(path).stem
        df["_season_i"] = i           # índice para decaimiento temporal
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTR"])

    # Convertir columnas numéricas silenciosamente
    num_cols = ["FTHG","FTAG","HTHG","HTAG",
                "HS","AS","HST","AST",
                "HF","AF","HC","AC",
                "HY","AY","HR","AR"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0.0

    return df


# ──────────────────────────────────────────────────────────────
#  PRE-CÁLCULO DE ESTADÍSTICAS (se hace una sola vez)
# ──────────────────────────────────────────────────────────────

def _team_stats(df: pd.DataFrame, col_home: str, col_away: str) -> dict:
    """
    Para una métrica dada (ej. HST/AST) devuelve:
      - promedio del equipo como local
      - promedio del equipo como visitante
      - promedio global de la liga (home y away)
    """
    avg_home = df[col_home].mean() or 1.0
    avg_away = df[col_away].mean() or 1.0

    stats = {}
    for team in set(df["HomeTeam"]) | set(df["AwayTeam"]):
        as_home = df[df["HomeTeam"] == team]
        as_away = df[df["AwayTeam"] == team]
        stats[team] = {
            "home_avg": as_home[col_home].mean() if len(as_home) else avg_home,
            "away_avg": as_away[col_away].mean() if len(as_away) else avg_away,
            "home_def": as_home[col_away].mean() if len(as_home) else avg_away,  # concedidos como local
            "away_def": as_away[col_home].mean() if len(as_away) else avg_home,  # concedidos como visitante
        }
    return {"teams": stats, "league_home_avg": avg_home, "league_away_avg": avg_away}


def _compute_elo(df: pd.DataFrame, k: int = 32) -> dict[str, float]:
    elo: dict[str, float] = {}
    for _, r in df.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        elo.setdefault(h, 1500.0)
        elo.setdefault(a, 1500.0)
        exp_h = 1 / (1 + 10 ** ((elo[a] - elo[h] - 50) / 400))
        s_h = 1.0 if r["FTR"]=="H" else (0.0 if r["FTR"]=="A" else 0.5)
        elo[h] += k * (s_h - exp_h)
        elo[a] += k * ((1-s_h) - (1-exp_h))
    return {t: round(v) for t, v in elo.items()}


def _recent_form(df: pd.DataFrame, n: int = 10) -> dict[str, float]:
    """Fracción de puntos obtenidos en los últimos N partidos por equipo."""
    form = {}
    for team in set(df["HomeTeam"]) | set(df["AwayTeam"]):
        games = df[(df["HomeTeam"]==team)|(df["AwayTeam"]==team)].tail(n)
        if games.empty:
            form[team] = 0.33
            continue
        pts = sum(
            3 if (r["FTR"]=="H" and r["HomeTeam"]==team) or
                 (r["FTR"]=="A" and r["AwayTeam"]==team)
            else 1 if r["FTR"]=="D" else 0
            for _, r in games.iterrows()
        )
        form[team] = pts / (len(games) * 3)
    return form


# ──────────────────────────────────────────────────────────────
#  OBJETO PRINCIPAL
# ──────────────────────────────────────────────────────────────

class LeagueData:
    """
    Contiene el DataFrame completo y todas las estadísticas
    pre-calculadas. Se construye una sola vez por liga.
    """

    def __init__(self, league: str, max_seasons: Optional[int] = None):
        print(f"\n  Cargando '{league}'...", end=" ", flush=True)
        self.league      = league
        self.df          = _load_csvs(league, max_seasons)
        self.teams       = sorted(set(self.df["HomeTeam"]) | set(self.df["AwayTeam"]))
        self.n_matches   = len(self.df)
        self.seasons     = sorted(self.df["_season"].unique())

        print("pre-calculando estadísticas...", end=" ", flush=True)

        # Resultados (para predictor de ganador)
        self.result_stats = _team_stats(self.df, "FTHG", "FTAG")
        self.elo          = _compute_elo(self.df)
        self.form         = _recent_form(self.df)
        self.home_win_rate = (self.df["FTR"]=="H").sum() / max(self.n_matches, 1)

        # Tiros totales y a puerta
        self.shots_stats  = _team_stats(self.df, "HS",  "AS")
        self.sot_stats    = _team_stats(self.df, "HST", "AST")

        # Faltas y córners
        self.fouls_stats  = _team_stats(self.df, "HF", "AF")
        self.corners_stats = _team_stats(self.df, "HC", "AC")

        # Tarjetas
        self.yellow_stats = _team_stats(self.df, "HY", "AY")
        self.red_stats    = _team_stats(self.df, "HR", "AR")

        print(f"✓  {self.n_matches} partidos · {len(self.teams)} equipos · {len(self.seasons)} temporadas")

    def h2h(self, home: str, away: str) -> pd.DataFrame:
        """Todos los enfrentamientos directos entre dos equipos."""
        return self.df[
            ((self.df["HomeTeam"]==home) & (self.df["AwayTeam"]==away)) |
            ((self.df["HomeTeam"]==away) & (self.df["AwayTeam"]==home))
        ].copy()
