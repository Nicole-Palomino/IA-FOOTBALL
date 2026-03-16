"""
mod_winner.py
═════════════
Módulo: Predicción del ganador del partido.
Usa: FTR, FTHG, FTAG, HST, AST, Elo, Forma reciente.
"""

from data_loader import LeagueData
from utils import (
    compute_lambdas, poisson_probs, h2h_weighted,
    normalize, bar, section, print_probs
)

DEFAULT_WEIGHTS = {
    "h2h":      0.30,
    "form":     0.25,
    "home_adv": 0.15,
    "poisson":  0.20,
    "shots":    0.05,
    "elo":      0.05,
}


def predict(ld: LeagueData, home: str, away: str,
            weights: dict = None, verbose: bool = True) -> dict:

    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    tw = sum(w.values()) or 1.0
    w  = {k: v / tw for k, v in w.items()}

    h2h = ld.h2h(home, away)

    # ── H2H ponderado ──────────────────────────────
    h2h_h = h2h_d = h2h_a = 0.0
    if not h2h.empty:
        rows = h2h.reset_index(drop=True)
        n = len(rows)
        tw2 = 0.0
        for i, r in rows.iterrows():
            age = n - i
            ww  = 0.85 ** age
            is_home = r["HomeTeam"] == home
            if   r["FTR"] == "D":                               h2h_d += ww
            elif (r["FTR"]=="H" and is_home) or \
                 (r["FTR"]=="A" and not is_home):               h2h_h += ww
            else:                                               h2h_a += ww
            tw2 += ww
        if tw2: h2h_h /= tw2; h2h_d /= tw2; h2h_a /= tw2
    else:
        h2h_h, h2h_d, h2h_a = 0.40, 0.28, 0.32

    # ── Poisson ────────────────────────────────────
    lam_h, lam_a = compute_lambdas(ld.result_stats, home, away, home_factor=1.1)
    pp_h, pp_d, pp_a = poisson_probs(lam_h, lam_a)

    # ── Forma ──────────────────────────────────────
    form_h = ld.form.get(home, 0.33)
    form_a = ld.form.get(away, 0.33)

    # ── Elo ────────────────────────────────────────
    elo_h = ld.elo.get(home, 1500)
    elo_a = ld.elo.get(away, 1500)
    elo_exp_h = 1 / (1 + 10 ** ((elo_a - elo_h - 50) / 400))

    # ── Tiros a puerta ─────────────────────────────
    sot = ld.sot_stats["teams"]
    avg_hst = ld.sot_stats["league_home_avg"]
    avg_ast = ld.sot_stats["league_away_avg"]
    sh_ratio = sot.get(home, {}).get("home_avg", avg_hst) / avg_hst
    sa_ratio = sot.get(away, {}).get("away_avg", avg_ast) / avg_ast

    # ── Combinar ───────────────────────────────────
    s_h = (h2h_h * w["h2h"] + form_h * w["form"] + ld.home_win_rate * w["home_adv"]
           + pp_h * w["poisson"] + (sh_ratio/3) * w["shots"] + elo_exp_h * w["elo"])
    s_d = (h2h_d * w["h2h"] + 0.27 * w["form"] + 0.26 * w["home_adv"]
           + pp_d * w["poisson"] + 0.25 * w["shots"] + 0.27 * w["elo"])
    s_a = (h2h_a * w["h2h"] + form_a * w["form"] + (1-ld.home_win_rate) * w["home_adv"]
           + pp_a * w["poisson"] + (sa_ratio/3) * w["shots"] + (1-elo_exp_h) * w["elo"])

    ph, pd_, pa = normalize(s_h, s_d, s_a)

    result = {
        "home_win": ph, "draw": pd_, "away_win": pa,
        "details": {
            "lambda_home": lam_h, "lambda_away": lam_a,
            "elo_home": elo_h,    "elo_away": elo_a,
            "form_home": round(form_h*100,1), "form_away": round(form_a*100,1),
            "home_adv_pct": round(ld.home_win_rate*100,1),
            "h2h_count": len(h2h),
            "poisson_home": round(pp_h*100,1),
            "poisson_draw": round(pp_d*100,1),
            "poisson_away": round(pp_a*100,1),
        }
    }

    if verbose:
        _print(home, away, result)
    return result


def _print(home: str, away: str, r: dict):
    d = r["details"]
    section(f"Predicción de ganador — {home} vs {away}")
    print_probs(home, away, r["home_win"], r["draw"], r["away_win"])
    print()
    print(f"  {'Goles esperados':<22} {home[:14]} λ={d['lambda_home']}   {away[:14]} λ={d['lambda_away']}")
    print(f"  {'Elo':<22} {home[:14]} {d['elo_home']}   {away[:14]} {d['elo_away']}")
    print(f"  {'Forma reciente':<22} {home[:14]} {d['form_home']}%   {away[:14]} {d['form_away']}%")
    print(f"  {'Ventaja local':<22} {d['home_adv_pct']}% victorias locales en la liga")
    print(f"  {'H2H analizados':<22} {d['h2h_count']} enfrentamientos directos")
    print(f"  {'Poisson (H/D/A)':<22} {d['poisson_home']}% / {d['poisson_draw']}% / {d['poisson_away']}%")
