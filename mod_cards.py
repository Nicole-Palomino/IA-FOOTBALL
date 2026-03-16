"""
mod_cards.py
════════════
Módulo: Predicción de tarjetas (amarillas y rojas).
Columnas usadas: HY, AY, HR, AR
"""

import math
from data_loader import LeagueData
from utils import (
    compute_lambdas, h2h_weighted,
    normalize2, bar, section
)


def _poisson_over(lam: float, threshold: float) -> float:
    """P(X > threshold) para una Poisson(lam). threshold puede ser x.5."""
    k_max = int(threshold)
    p_under = sum(
        math.exp(-lam) * (lam**k) / math.factorial(k)
        for k in range(k_max + 1)
    )
    return round((1 - p_under) * 100, 1)


def predict(ld: LeagueData, home: str, away: str,
            verbose: bool = True) -> dict:

    h2h = ld.h2h(home, away)

    # ── Amarillas ──────────────────────────────────
    # Visitante suele ver más amarillas (más presionado, árbitro)
    lam_hy, lam_ay = compute_lambdas(ld.yellow_stats, home, away, home_factor=0.92)
    h2h_hy, h2h_ay = h2h_weighted(h2h, home, "HY", "AY")

    if h2h_hy is not None:
        exp_hy = round(lam_hy * 0.55 + h2h_hy * 0.45, 2)
        exp_ay = round(lam_ay * 0.55 + h2h_ay * 0.45, 2)
    else:
        exp_hy, exp_ay = lam_hy, lam_ay

    # ── Rojas ──────────────────────────────────────
    lam_hr, lam_ar = compute_lambdas(ld.red_stats, home, away, home_factor=0.95)
    h2h_hr, h2h_ar = h2h_weighted(h2h, home, "HR", "AR")

    if h2h_hr is not None:
        exp_hr = round(lam_hr * 0.5 + h2h_hr * 0.5, 3)
        exp_ar = round(lam_ar * 0.5 + h2h_ar * 0.5, 3)
    else:
        exp_hr, exp_ar = lam_hr, lam_ar

    total_yellow = round(exp_hy + exp_ay, 1)
    total_red    = round(exp_hr + exp_ar, 2)

    # ── Over/Under amarillas ───────────────────────
    over_3_5_hy  = _poisson_over(exp_hy, 3)
    over_3_5_ay  = _poisson_over(exp_ay, 3)
    over_4_5_tot = _poisson_over(total_yellow, 4)
    under_4_5_tot = round(100 - over_4_5_tot, 1)

    # ── Probabilidad de al menos 1 roja ───────────
    p_no_red_h = math.exp(-exp_hr)
    p_no_red_a = math.exp(-exp_ar)
    p_any_red  = round((1 - p_no_red_h * p_no_red_a) * 100, 1)
    p_no_red   = round(100 - p_any_red, 1)

    # ── Quién ve más amarillas ──────────────────────
    ph_y, pa_y = normalize2(exp_hy, exp_ay)

    result = {
        "home": home, "away": away,
        "yellow": {
            "expected_home": exp_hy, "expected_away": exp_ay,
            "total": total_yellow,
            "prob_home_more": ph_y, "prob_away_more": pa_y,
            "home_over_3_5": over_3_5_hy,
            "away_over_3_5": over_3_5_ay,
            "total_over_4_5": over_4_5_tot,
            "total_under_4_5": under_4_5_tot,
        },
        "red": {
            "expected_home": exp_hr, "expected_away": exp_ar,
            "total": total_red,
            "prob_any_red": p_any_red,
            "prob_no_red":  p_no_red,
        },
        "h2h_count": len(h2h),
    }

    if verbose:
        _print(result)
    return result


def _print(r: dict):
    home, away = r["home"], r["away"]
    y = r["yellow"]
    red = r["red"]

    section(f"Tarjetas — {home} vs {away}")

    # Amarillas
    print(f"\n  ── TARJETAS AMARILLAS ──────────────────────────")
    print(f"    {home[:20]:<20}  {y['expected_home']:>4.1f} esperadas")
    print(f"    {away[:20]:<20}  {y['expected_away']:>4.1f} esperadas")
    print(f"    Total estimado del partido:  {y['total']}")

    ph, pa = y["prob_home_more"], y["prob_away_more"]
    print(f"    {home} ve más: {ph:.0f}%   {away} ve más: {pa:.0f}%")

    print(f"\n    {home[:18]} Over 3.5 amarillas:  {y['home_over_3_5']:>5.1f}%  {bar(y['home_over_3_5'])}")
    print(f"    {away[:18]} Over 3.5 amarillas:  {y['away_over_3_5']:>5.1f}%  {bar(y['away_over_3_5'])}")
    print(f"\n    Total Over  4.5 amarillas:  {y['total_over_4_5']:>5.1f}%  {bar(y['total_over_4_5'])}")
    print(f"    Total Under 4.5 amarillas:  {y['total_under_4_5']:>5.1f}%  {bar(y['total_under_4_5'])}")

    # Rojas
    print(f"\n  ── TARJETAS ROJAS ──────────────────────────────")
    print(f"    {home[:20]:<20}  {red['expected_home']:.3f} esperadas por partido")
    print(f"    {away[:20]:<20}  {red['expected_away']:.3f} esperadas por partido")
    print(f"    Total medio del partido:  {red['total']:.3f}")
    print(f"\n    Al menos 1 roja en el partido:  {red['prob_any_red']:>5.1f}%  {bar(red['prob_any_red'])}")
    print(f"    Partido sin rojas:              {red['prob_no_red']:>5.1f}%  {bar(red['prob_no_red'])}")

    print(f"\n  H2H analizados: {r['h2h_count']} enfrentamientos directos")
