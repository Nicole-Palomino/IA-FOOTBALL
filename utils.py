"""
utils.py
════════
Funciones compartidas entre todos los módulos de predicción.
"""

import math
import pandas as pd


DECAY = 0.85   # Factor de decaimiento por antigüedad en H2H


# ──────────────────────────────────────────────────────────────
#  POISSON
# ──────────────────────────────────────────────────────────────

def poisson_pmf(lam: float, k: int) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def poisson_probs(lam_h: float, lam_a: float, max_val: int = 12):
    """
    P(home > away), P(home == away), P(home < away)
    usando distribución de Poisson bivariada.
    """
    p_h = p_d = p_a = 0.0
    for i in range(max_val + 1):
        for j in range(max_val + 1):
            p = poisson_pmf(lam_h, i) * poisson_pmf(lam_a, j)
            if   i > j: p_h += p
            elif i == j: p_d += p
            else:        p_a += p
    return p_h, p_d, p_a


def expected_value(lam_h: float, lam_a: float, max_val: int = 20) -> tuple[float, float]:
    """Retorna (E[home], E[away]) — goles/unidades esperadas."""
    return round(lam_h, 2), round(lam_a, 2)


# ──────────────────────────────────────────────────────────────
#  H2H PONDERADO
# ──────────────────────────────────────────────────────────────

def h2h_weighted(h2h_df: pd.DataFrame, home: str, col_home: str, col_away: str):
    """
    Promedio ponderado de una métrica en enfrentamientos directos.
    Los partidos más recientes tienen mayor peso (decaimiento exponencial).
    Devuelve (avg_home_metric, avg_away_metric).
    """
    if h2h_df.empty:
        return None, None

    rows  = h2h_df.reset_index(drop=True)
    n     = len(rows)
    total_w = sum_h = sum_a = 0.0

    for i, row in rows.iterrows():
        age = n - i
        w   = DECAY ** age
        is_home = row["HomeTeam"] == home
        v_h = row[col_home] if is_home else row[col_away]
        v_a = row[col_away] if is_home else row[col_home]
        sum_h   += v_h * w
        sum_a   += v_a * w
        total_w += w

    if total_w == 0:
        return None, None
    return round(sum_h / total_w, 2), round(sum_a / total_w, 2)


# ──────────────────────────────────────────────────────────────
#  LAMBDAS GENÉRICAS (sirve para goles, tiros, faltas, etc.)
# ──────────────────────────────────────────────────────────────

def compute_lambdas(stats: dict, home: str, away: str,
                    home_factor: float = 1.0) -> tuple[float, float]:
    """
    λ_home = ataque_home_local  × (defensa_away_como_visita / avg_liga) × home_factor
    λ_away = ataque_away_visita × (defensa_home_como_local  / avg_liga)
    """
    ts  = stats["teams"]
    avg_h = stats["league_home_avg"]
    avg_a = stats["league_away_avg"]

    att_h = ts.get(home, {}).get("home_avg", avg_h)
    def_a = ts.get(away, {}).get("away_def", avg_h)   # cuánto concede away como visitante

    att_a = ts.get(away, {}).get("away_avg", avg_a)
    def_h = ts.get(home, {}).get("home_def", avg_a)   # cuánto concede home como local

    lam_h = att_h * (def_a / avg_h) * home_factor
    lam_a = att_a * (def_h / avg_a)

    return max(round(lam_h, 3), 0.01), max(round(lam_a, 3), 0.01)


# ──────────────────────────────────────────────────────────────
#  NORMALIZACIÓN
# ──────────────────────────────────────────────────────────────

def normalize(h: float, d: float, a: float) -> tuple[float, float, float]:
    t = h + d + a or 1.0
    return round(h/t*100, 1), round(d/t*100, 1), round(a/t*100, 1)


def normalize2(h: float, a: float) -> tuple[float, float]:
    """Para métricas sin empate (ej. quién hace más tiros)."""
    t = h + a or 1.0
    return round(h/t*100, 1), round(a/t*100, 1)


# ──────────────────────────────────────────────────────────────
#  CONSOLA
# ──────────────────────────────────────────────────────────────

def bar(pct: float, width: int = 28) -> str:
    filled = int(round(pct / 100 * width))
    return "█" * filled + "░" * (width - filled)


def section(title: str):
    w = 56
    print()
    print("╔" + "═" * w + "╗")
    print("║  " + title.upper().ljust(w - 2) + "║")
    print("╚" + "═" * w + "╝")


def row3(label: str, home_val, draw_val, away_val, unit: str = "%"):
    """Imprime una fila de resultado con barra de progreso."""
    if isinstance(home_val, float):
        line = f"  {label:<20} {home_val:>5.1f}{unit}  {bar(home_val)}"
    else:
        line = f"  {label:<20} {home_val}"
    print(line)


def print_probs(home: str, away: str, ph: float, pd_: float, pa: float,
                label_draw: str = "Empate"):
    print(f"\n  {home[:22]:<22} {ph:>5.1f}%  {bar(ph)}")
    print(f"  {label_draw:<22} {pd_:>5.1f}%  {bar(pd_)}")
    print(f"  {away[:22]:<22} {pa:>5.1f}%  {bar(pa)}")


def print_metric_row(label: str, home_val: float, away_val: float,
                     home_name: str, away_name: str, unit: str = ""):
    ph, pa = normalize2(home_val, away_val)
    print(f"\n  {label}")
    print(f"    {home_name[:18]:<18}  {home_val:>5.1f}{unit}  {bar(ph, 20)}")
    print(f"    {away_name[:18]:<18}  {away_val:>5.1f}{unit}  {bar(pa, 20)}")
