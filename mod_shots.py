"""
mod_shots.py
════════════
Módulo: Predicción de tiros (totales y a puerta).
Columnas usadas: HS, AS, HST, AST
"""

from data_loader import LeagueData
from utils import (
    compute_lambdas, poisson_probs,
    h2h_weighted, normalize2,
    bar, section, print_metric_row
)


def predict(ld: LeagueData, home: str, away: str,
            verbose: bool = True) -> dict:

    h2h = ld.h2h(home, away)

    # ── Tiros totales ──────────────────────────────
    lam_hs, lam_as = compute_lambdas(ld.shots_stats, home, away, home_factor=1.05)
    h2h_hs, h2h_as = h2h_weighted(h2h, home, "HS", "AS")

    # Promedio ponderado con H2H si existe
    if h2h_hs is not None:
        exp_hs = round(lam_hs * 0.6 + h2h_hs * 0.4, 2)
        exp_as = round(lam_as * 0.6 + h2h_as * 0.4, 2)
    else:
        exp_hs, exp_as = lam_hs, lam_as

    # ── Tiros a puerta ─────────────────────────────
    lam_hst, lam_ast = compute_lambdas(ld.sot_stats, home, away, home_factor=1.05)
    h2h_hst, h2h_ast = h2h_weighted(h2h, home, "HST", "AST")

    if h2h_hst is not None:
        exp_hst = round(lam_hst * 0.6 + h2h_hst * 0.4, 2)
        exp_ast = round(lam_ast * 0.6 + h2h_ast * 0.4, 2)
    else:
        exp_hst, exp_ast = lam_hst, lam_ast

    # ── Probabilidad de quién tira más ─────────────
    ph_shots, pa_shots = normalize2(exp_hs, exp_as)
    ph_sot,   pa_sot   = normalize2(exp_hst, exp_ast)

    # ── Precisión de tiro (SOT / HS) ───────────────
    prec_h = round(exp_hst / max(exp_hs, 0.01) * 100, 1)
    prec_a = round(exp_ast / max(exp_as, 0.01) * 100, 1)

    result = {
        "home": home, "away": away,
        "shots": {
            "expected_home": exp_hs, "expected_away": exp_as,
            "prob_home_more": ph_shots, "prob_away_more": pa_shots,
        },
        "shots_on_target": {
            "expected_home": exp_hst, "expected_away": exp_ast,
            "prob_home_more": ph_sot,  "prob_away_more": pa_sot,
        },
        "precision": {
            "home_pct": prec_h, "away_pct": prec_a,
        },
        "h2h_count": len(h2h),
    }

    if verbose:
        _print(result)
    return result


def _print(r: dict):
    home, away = r["home"], r["away"]
    s   = r["shots"]
    sot = r["shots_on_target"]
    pre = r["precision"]

    section(f"Tiros — {home} vs {away}")

    print(f"\n  {'':22} {'LOCAL':>8}  {'VISITANTE':>8}")
    print(f"  {'─'*46}")

    # Tiros totales esperados
    ph, pa = s["prob_home_more"], s["prob_away_more"]
    print(f"\n  Tiros totales esperados")
    print(f"    {home[:20]:<20}  {s['expected_home']:>5.1f}  {bar(ph, 20)}")
    print(f"    {away[:20]:<20}  {s['expected_away']:>5.1f}  {bar(pa, 20)}")
    print(f"    → {home} tirará más: {ph:.0f}%   {away} tirará más: {pa:.0f}%")

    # Tiros a puerta esperados
    ph2, pa2 = sot["prob_home_more"], sot["prob_away_more"]
    print(f"\n  Tiros a puerta esperados")
    print(f"    {home[:20]:<20}  {sot['expected_home']:>5.1f}  {bar(ph2, 20)}")
    print(f"    {away[:20]:<20}  {sot['expected_away']:>5.1f}  {bar(pa2, 20)}")
    print(f"    → {home} más certero: {ph2:.0f}%   {away} más certero: {pa2:.0f}%")

    # Precisión de tiro
    print(f"\n  Precisión de tiro (SOT / tiros)")
    print(f"    {home[:20]:<20}  {pre['home_pct']:>5.1f}%")
    print(f"    {away[:20]:<20}  {pre['away_pct']:>5.1f}%")

    print(f"\n  H2H analizados: {r['h2h_count']} enfrentamientos directos")
