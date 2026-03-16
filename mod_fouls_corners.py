"""
mod_fouls_corners.py
════════════════════
Módulo: Predicción de faltas y córners.
Columnas usadas: HF, AF, HC, AC
"""

from data_loader import LeagueData
from utils import (
    compute_lambdas, poisson_probs,
    h2h_weighted, normalize2,
    bar, section
)


def predict(ld: LeagueData, home: str, away: str,
            verbose: bool = True) -> dict:

    h2h = ld.h2h(home, away)

    # ── Faltas ─────────────────────────────────────
    # El local suele cometer menos faltas (árbitro local más permisivo?)
    lam_hf, lam_af = compute_lambdas(ld.fouls_stats, home, away, home_factor=0.97)
    h2h_hf, h2h_af = h2h_weighted(h2h, home, "HF", "AF")

    if h2h_hf is not None:
        exp_hf = round(lam_hf * 0.55 + h2h_hf * 0.45, 2)
        exp_af = round(lam_af * 0.55 + h2h_af * 0.45, 2)
    else:
        exp_hf, exp_af = lam_hf, lam_af

    # ── Córners ────────────────────────────────────
    # El local suele tener más córners (más ataque en casa)
    lam_hc, lam_ac = compute_lambdas(ld.corners_stats, home, away, home_factor=1.08)
    h2h_hc, h2h_ac = h2h_weighted(h2h, home, "HC", "AC")

    if h2h_hc is not None:
        exp_hc = round(lam_hc * 0.6 + h2h_hc * 0.4, 2)
        exp_ac = round(lam_ac * 0.6 + h2h_ac * 0.4, 2)
    else:
        exp_hc, exp_ac = lam_hc, lam_ac

    # ── Total esperado del partido ──────────────────
    total_fouls   = round(exp_hf + exp_af, 1)
    total_corners = round(exp_hc + exp_ac, 1)

    # ── Probabilidades de quién comete más / tiene más ──
    ph_f,  pa_f  = normalize2(exp_hf, exp_af)
    ph_c,  pa_c  = normalize2(exp_hc, exp_ac)

    # ── Over/Under córners (línea 9.5) ─────────────
    # Suma de Poisson: P(total > 9.5) ≈ 1 - P(total ≤ 9)
    import math
    lam_total_corners = exp_hc + exp_ac
    p_under_9 = sum(
        math.exp(-lam_total_corners) * (lam_total_corners**k) / math.factorial(k)
        for k in range(10)
    )
    over_9_5 = round((1 - p_under_9) * 100, 1)
    under_9_5 = round(p_under_9 * 100, 1)

    # Over/Under faltas (línea 20.5)
    lam_total_fouls = exp_hf + exp_af
    p_under_20 = sum(
        math.exp(-lam_total_fouls) * (lam_total_fouls**k) / math.factorial(k)
        for k in range(21)
    )
    over_20_5_f  = round((1 - p_under_20) * 100, 1)
    under_20_5_f = round(p_under_20 * 100, 1)

    result = {
        "home": home, "away": away,
        "fouls": {
            "expected_home": exp_hf, "expected_away": exp_af,
            "total": total_fouls,
            "prob_home_more": ph_f, "prob_away_more": pa_f,
            "over_20_5": over_20_5_f, "under_20_5": under_20_5_f,
        },
        "corners": {
            "expected_home": exp_hc, "expected_away": exp_ac,
            "total": total_corners,
            "prob_home_more": ph_c, "prob_away_more": pa_c,
            "over_9_5": over_9_5, "under_9_5": under_9_5,
        },
        "h2h_count": len(h2h),
    }

    if verbose:
        _print(result)
    return result


def _print(r: dict):
    home, away = r["home"], r["away"]
    f = r["fouls"]
    c = r["corners"]

    section(f"Faltas y Córners — {home} vs {away}")

    # Faltas
    print(f"\n  ── FALTAS ──────────────────────────────────────")
    print(f"    {home[:20]:<20}  {f['expected_home']:>5.1f} faltas esperadas")
    print(f"    {away[:20]:<20}  {f['expected_away']:>5.1f} faltas esperadas")
    print(f"    Total estimado del partido:  {f['total']}")
    ph, pa = f["prob_home_more"], f["prob_away_more"]
    print(f"    {home} comete más: {ph:.0f}%   {away} comete más: {pa:.0f}%")
    print(f"\n    Over  20.5 faltas:  {f['over_20_5']:>5.1f}%  {_bar(f['over_20_5'])}")
    print(f"    Under 20.5 faltas:  {f['under_20_5']:>5.1f}%  {_bar(f['under_20_5'])}")

    # Córners
    print(f"\n  ── CÓRNERS ─────────────────────────────────────")
    print(f"    {home[:20]:<20}  {c['expected_home']:>5.1f} córners esperados")
    print(f"    {away[:20]:<20}  {c['expected_away']:>5.1f} córners esperados")
    print(f"    Total estimado del partido:  {c['total']}")
    ph2, pa2 = c["prob_home_more"], c["prob_away_more"]
    print(f"    {home} tiene más: {ph2:.0f}%   {away} tiene más: {pa2:.0f}%")
    print(f"\n    Over  9.5 córners:  {c['over_9_5']:>5.1f}%  {_bar(c['over_9_5'])}")
    print(f"    Under 9.5 córners:  {c['under_9_5']:>5.1f}%  {_bar(c['under_9_5'])}")

    print(f"\n  H2H analizados: {r['h2h_count']} enfrentamientos directos")


def _bar(pct: float, w: int = 22) -> str:
    from utils import bar
    return bar(pct, w)
