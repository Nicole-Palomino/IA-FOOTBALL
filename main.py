"""
main.py
═══════
Menú principal del Football Predictor.
Carga los datos UNA SOLA VEZ y luego permite elegir qué predecir
sin volver a leer los CSVs.

Ejecuta: python main.py
"""

from data_loader import LeagueData, list_leagues
import mod_winner
import mod_shots
import mod_fouls_corners
import mod_cards


# ──────────────────────────────────────────────────────────────
#  HELPERS DE MENÚ
# ──────────────────────────────────────────────────────────────

def pick(prompt, options, back=False):
    print(f"\n  {prompt}")
    print("  " + "─" * 42)
    for i, opt in enumerate(options, 1):
        print(f"  [{i:>2}] {opt}")
    if back:
        print("  [ 0] ← Volver")
    print()
    while True:
        try:
            raw = input("  → ").strip()
            if back and raw == "0":
                return None
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
            print(f"  Elige entre 1 y {len(options)}")
        except (ValueError, KeyboardInterrupt):
            pass


def pick_multi(prompt, options):
    print(f"\n  {prompt}")
    print("  " + "─" * 42)
    for i, opt in enumerate(options, 1):
        print(f"  [{i:>2}] {opt}")
    print("  [ 0] Todos")
    print()
    while True:
        try:
            raw = input("  → ").strip()
            if raw == "0":
                return options[:]
            indices = [int(x.strip()) - 1 for x in raw.split(",")]
            chosen  = [options[i] for i in indices if 0 <= i < len(options)]
            if chosen:
                return chosen
        except (ValueError, KeyboardInterrupt):
            pass


def pick_float(label, default):
    while True:
        try:
            raw = input(f"  {label} (Enter={default}): ").strip()
            if not raw:
                return default
            v = float(raw)
            if 0 <= v <= 1:
                return v
        except ValueError:
            pass
        print("  Ingresa un decimal entre 0.0 y 1.0")


def show_h2h(ld, home, away, n=8):
    h2h = ld.h2h(home, away).tail(n)
    if h2h.empty:
        print("\n  (Sin enfrentamientos directos en los datos)\n")
        return
    print(f"\n  ── Últimos {len(h2h)} H2H ────────────────────────────────")
    print(f"  {'Fecha':<12} {'Local':<20} {'Visit.':<20} {'Marc.':<8} Res")
    print("  " + "─" * 64)
    for _, r in h2h.iterrows():
        score = f"{int(r['FTHG'])}-{int(r['FTAG'])}"
        res   = {"H": "Local", "A": "Visita", "D": "Empate"}.get(r["FTR"], "?")
        print(f"  {str(r.get('Date',''))[:10]:<12} {r['HomeTeam']:<20} {r['AwayTeam']:<20} {score:<8} {res}")
    print()


# ──────────────────────────────────────────────────────────────
#  MÓDULOS DISPONIBLES
# ──────────────────────────────────────────────────────────────

MODULES = {
    "Ganador del partido  (H / Empate / A)":  "winner",
    "Tiros totales y a puerta":               "shots",
    "Faltas y corners":                       "fouls_corners",
    "Tarjetas amarillas y rojas":             "cards",
}


def run_modules(ld, home, away, chosen, weights=None):
    for label in chosen:
        key = MODULES[label]
        if   key == "winner":       mod_winner.predict(ld, home, away, weights=weights)
        elif key == "shots":        mod_shots.predict(ld, home, away)
        elif key == "fouls_corners":mod_fouls_corners.predict(ld, home, away)
        elif key == "cards":        mod_cards.predict(ld, home, away)


# ──────────────────────────────────────────────────────────────
#  FLUJO DE PREDICCIÓN (datos ya cargados)
# ──────────────────────────────────────────────────────────────

def prediction_loop(ld):
    """Permite predecir múltiples partidos sin recargar datos."""
    while True:
        home = pick("Equipo LOCAL (juega en casa):", ld.teams, back=True)
        if home is None:
            return  # volver al menú de liga

        away_opts = [t for t in ld.teams if t != home]
        away = pick("Equipo VISITANTE:", away_opts, back=True)
        if away is None:
            continue

        show_h2h(ld, home, away)

        # ── Qué predecir ────────────────────────────
        module_labels = list(MODULES.keys())
        chosen = pick_multi("¿Qué quieres predecir? (escribe números separados por coma):", module_labels)

        # ── Pesos para módulo ganador ────────────────
        weights = None
        if any(MODULES[c] == "winner" for c in chosen):
            print("\n  ── Pesos del modelo ganador ──────────────────────")
            print("  [1] Pesos por defecto   [2] Personalizar")
            if input("  → ").strip() == "2":
                print("\n  (0.0 = ignorar factor · 1.0 = peso máximo)\n")
                descs = {
                    "h2h":      "Historial directo H2H         ",
                    "form":     "Forma reciente (últ.10)       ",
                    "home_adv": "Ventaja de localía real       ",
                    "poisson":  "Modelo de goles (Poisson)     ",
                    "shots":    "Tiros a puerta                ",
                    "elo":      "Rating Elo dinámico           ",
                }
                weights = {}
                for k, desc in descs.items():
                    weights[k] = pick_float(desc, mod_winner.DEFAULT_WEIGHTS[k])

        # ── Ejecutar ─────────────────────────────────
        print(f"\n  Calculando {home} vs {away}...\n")
        run_modules(ld, home, away, chosen, weights)

        # ── Siguiente acción ─────────────────────────
        print("\n  ── Siguiente ───────────────────────────────────")
        print("  [1] Otro partido  (datos ya cargados, sin espera)")
        print("  [2] Cambiar liga")
        print("  [3] Salir")
        accion = input("\n  → ").strip()

        if   accion == "1": continue
        elif accion == "2": return "change_league"
        else:
            print("\n  Hasta pronto! ⚽\n")
            return "exit"


# ──────────────────────────────────────────────────────────────
#  MENÚ PRINCIPAL
# ──────────────────────────────────────────────────────────────

def main():
    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║   ⚽  Football Predictor  —  Motor multi-módulo ║")
    print("  ╠══════════════════════════════════════════════════╣")
    print("  ║  Los datos se cargan UNA VEZ por liga.          ║")
    print("  ║  Predice ganador, tiros, faltas y tarjetas.     ║")
    print("  ╚══════════════════════════════════════════════════╝")

    leagues = list_leagues()
    if not leagues:
        print("\n  No hay ligas en 'data/'.")
        print("  Crea subcarpetas con tus CSVs:")
        print("    data/premier_league/2023-24.csv")
        print("    data/la_liga/2022-23.csv\n")
        return

    while True:
        league = pick("Selecciona la liga:", leagues)
        if not league:
            break

        s_opts = ["Todas las temporadas", "Últimas 5", "Últimas 3", "Solo la más reciente"]
        s_map  = {"Todas las temporadas": None, "Últimas 5": 5, "Últimas 3": 3, "Solo la más reciente": 1}
        sel    = pick("¿Cuántas temporadas analizar?", s_opts)
        max_s  = s_map.get(sel)

        try:
            ld = LeagueData(league, max_s)
        except FileNotFoundError as e:
            print(f"\n  ⚠  {e}\n")
            continue

        print("  ✓ Listo. Sin más esperas para esta liga.\n")

        result = prediction_loop(ld)

        if result == "exit":
            return
        # result == "change_league" o None → vuelve al while y elige liga


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Saliendo...\n")
