---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python (ml-final)
  language: python
  name: ml-final
---

# Reproducibilidad y bibliografía

## Entorno

```{code-cell} ipython3
---
tags: [hide-input]
---
# Calcula y muestra los resultados de Entorno.
import sys, platform, json
sys.path.insert(0, "..")
from importlib.metadata import version
import pandas as pd
from src.config import SEED, RESULTADOS
libs = ["numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "xgboost", "optuna", "deap", "shap", "lime",
        "statsmodels", "arch", "faiss-cpu", "joblib", "jupyter-book"]
pd.DataFrame({"versión": {l: version(l) for l in libs} | {"python": platform.python_version()}})
```

Todas las versiones están fijadas en `requirements.txt` y en `environment.yml` (entorno
conda `ml-final`, Python 3.11). Los resultados se obtuvieron en Windows 11 con un
procesador Intel Core i7-1255U y 15,7 GB de memoria, sin GPU.

## Semillas

La semilla global es `SEED = 42`, definida una sola vez en `src/config.py`. La función
`fijar_semillas` la propaga a los generadores de Python y numpy; cada estimador la recibe en
`random_state` (scikit-learn, XGBoost e imbalanced-learn), el muestreador
TPE de Optuna en `seed` y el algoritmo genético la usa tanto en el generador de numpy que crea
la población inicial como en el módulo `random` que emplea DEAP en la selección, el cruce y la
mutación. La sensibilidad a la semilla se evaluó explícitamente con cinco semillas en el mejor benchmark
(sección de robustez del capítulo 3).

## Tabla maestra de experimentos

```{code-cell} ipython3
---
tags: [hide-input]
---
# Calcula y muestra los resultados de Tabla maestra de experimentos.
from src.analysis import tabla_pliegues
partes = []
for carpeta, tarea in (("clf_calamar", "clasificación"), ("reg_calamar", "regresión")):
    t = tabla_pliegues(carpeta); t["tarea"] = tarea
    partes.append(t)
largo = pd.concat(partes, ignore_index=True)
from src.analysis import cargar_carpeta
params = {(r["modelo"], r["balanceo"], r["optimizador"], r["pliegue"]): json.dumps(r["params"])
          for c in ("clf_calamar", "reg_calamar") for r in cargar_carpeta(c)}
largo["hiperparametros"] = [params[(a, b, c, d)] for a, b, c, d in zip(largo.modelo, largo.balanceo, largo.optimizador, largo.pliegue)]
largo["semilla"] = SEED
metricas = [c for c in ("pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy", "brier", "rmse", "mae", "r2", "mape") if c in largo]
agg = largo.groupby(["tarea", "modelo", "balanceo", "optimizador"]).agg(
    **{f"{m}_media": (m, "mean") for m in metricas}, **{f"{m}_desv": (m, "std") for m in metricas},
    tiempo_total_s=("t_opt", "sum"), evaluaciones=("n_eval", "sum"), pliegues=("pliegue", "nunique")).reset_index()
agg["semilla"] = SEED
largo.to_csv(RESULTADOS / "tabla_maestra_por_pliegue.csv", index=False)
agg.to_csv(RESULTADOS / "tabla_maestra.csv", index=False)
print(f"{len(agg)} combinaciones ({(agg.tarea == 'clasificación').sum()} de clasificación, "
      f"{(agg.tarea == 'regresión').sum()} de regresión) y {len(largo):,} registros por pliegue.")
agg.head(10)
```

La tabla maestra (`results/tabla_maestra.csv`) contiene una fila por combinación con la media y
la desviación de cada métrica del bucle externo, el tiempo total, el número de evaluaciones y la
semilla; `results/tabla_maestra_por_pliegue.csv` agrega los hiperparámetros elegidos en cada
pliegue externo. Las 112 combinaciones de clasificación y las 28 de regresión comparten el
mismo esquema.

## Cómo reproducir los resultados

El README del repositorio enumera los comandos en orden. Los cómputos costosos (la validación
anidada de los 140 modelos) se guardan como resultados versionados en
`results/`, y los cuadernos de este libro los leen y regeneran tablas y figuras; el código que
los produce está en `scripts/` y `src/` y se ejecuta de principio a fin con los comandos del
README. Las pruebas unitarias se ejecutan con `python -m pytest`.

## Bibliografía

```{bibliography}
:style: unsrt
```
