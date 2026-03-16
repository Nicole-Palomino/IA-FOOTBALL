# ⚽ Football Predictor — Multi-módulo

Motor estadístico para predecir resultados de fútbol desde CSVs históricos.
**Los datos se cargan UNA SOLA VEZ** — luego puedes predecir múltiples partidos sin esperar.

---

## 📁 Estructura

```
football_predictor/
├── main.py              ← EJECUTA ESTO
├── data_loader.py       ← Carga CSVs y pre-calcula estadísticas
├── utils.py             ← Funciones compartidas (Poisson, Elo, etc.)
├── mod_winner.py        ← Módulo: ganador del partido
├── mod_shots.py         ← Módulo: tiros totales y a puerta
├── mod_fouls_corners.py ← Módulo: faltas y córners
├── mod_cards.py         ← Módulo: tarjetas amarillas y rojas
├── requirements.txt
└── data/
    ├── premier_league/
    │   ├── 2019-20.csv
    │   └── 2023-24.csv
    ├── la_liga/
    └── bundesliga/
```

---

## ⚙️ Instalación

```bash
pip install pandas numpy
```

---

## 🚀 Uso

```bash
python main.py
```

1. Elige la liga → los datos se cargan y pre-calculan **una sola vez**
2. Elige equipo local y visitante
3. Selecciona qué predecir (puedes elegir varios a la vez):
   - Ganador del partido
   - Tiros totales y a puerta
   - Faltas y córners
   - Tarjetas amarillas y rojas
4. Para el módulo ganador, ajusta los pesos si quieres
5. Predice otro partido **sin recargar** nada

---

## 🧠 Módulos

| Archivo               | Predice                            | Columnas usadas        |
|-----------------------|------------------------------------|------------------------|
| `mod_winner.py`       | H / Empate / A + probabilidades    | FTR, FTHG, FTAG, HST, AST |
| `mod_shots.py`        | Tiros esperados, precisión         | HS, AS, HST, AST       |
| `mod_fouls_corners.py`| Faltas, córners, over/under        | HF, AF, HC, AC         |
| `mod_cards.py`        | Amarillas, rojas, over/under       | HY, AY, HR, AR         |

---

## 📊 Columnas CSV mínimas requeridas

```
Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR,
HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR
```

Fuente de datos: https://www.football-data.co.uk (formato idéntico)
