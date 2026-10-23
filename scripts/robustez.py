"""Sensibilidad del mejor benchmark de clasificación a la semilla de inicialización.

En cada pliegue externo se reutilizan los hiperparámetros que la optimización anidada eligió
para ese pliegue y solo se cambia la semilla, de modo que la variación medida es la del
algoritmo y no la de una nueva búsqueda.

Uso:  python -m scripts.robustez [n_jobs]
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTADOS, fijar_semillas
from src.experiment import cargar_datos
from src.models import construir_pipeline, puntaje_continuo

SAL = RESULTADOS / "robustez"
SEMILLAS = (1, 7, 42, 123, 2024)


def params_pliegue(carpeta: Path, patron: str) -> dict:
    return {pickle.load(open(f, "rb"))["pliegue"]: pickle.load(open(f, "rb"))["params"]
            for f in sorted(carpeta.glob(patron))}


def trabajo(fila, params: dict, pliegue: int, seed: int) -> dict:
    datos = cargar_datos()
    p = datos.pliegues[pliegue]
    X, y = datos.X.iloc[p["train"]], datos.y_clf.iloc[p["train"]]
    est = construir_pipeline("clf", fila.modelo, params, fila.balanceo, float(y.mean()), seed).fit(X, y)
    pr = average_precision_score(datos.y_clf.iloc[p["test"]], puntaje_continuo(est, datos.X.iloc[p["test"]]))
    return {"tipo": "semilla", "variante": "benchmark", "modelo": "benchmark", "pliegue": pliegue,
            "valor": None, "seed": seed, "pr_auc": pr}


if __name__ == "__main__":
    fijar_semillas()
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    SAL.mkdir(parents=True, exist_ok=True)
    mejores = pd.read_csv(RESULTADOS / "mejores_clf.csv")
    mejor = mejores.sort_values("pr_auc_mean", ascending=False).iloc[0]
    params = params_pliegue(RESULTADOS / "clf_calamar",
                            f"{mejor.modelo}__{mejor.balanceo}__{mejor.optimizador}__f*.pkl")
    tareas = [(mejor, params[k], k, s) for s in SEMILLAS for k in sorted(params)]
    res = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(delayed(trabajo)(*t) for t in tareas)
    df = pd.DataFrame(res)
    df.to_csv(SAL / "robustez.csv", index=False)
    print(df.groupby("seed").pr_auc.agg(["mean", "std"]))
