"""Validación anidada de los modelos LTSF-Linear con el protocolo de los demás modelos.

Usa exactamente las mismas particiones externas, la misma validación interna, la misma
semilla y el mismo presupuesto de evaluaciones que los clasificadores y regresores de
referencia, de modo que sus PR-AUC por bloque se comparan pareadas con las de ellos.

Los modelos LTSF-Linear (Zeng et al., 2023) pronostican el cambio logarítmico del caudal a
quince días, ``z = log(Q(t+15)/Q(t))``, a partir de la ventana de historia del caudal. Como son
pronosticadores puntuales, la probabilidad de banda se obtiene suponiendo un residuo gaussiano
cuya dispersión se estima fuera de muestra, en el último 15 % cronológico del entrenamiento, e
integrando esa distribución entre los límites de la banda trasladados a cada observación.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.metrics import average_precision_score

from .config import DATOS_PROC, RESULTADOS, SEED
from .dlinear import MODELOS_LTSF, datos_con_ventanas
from .experiment import Datos, cargar_datos, metricas_clf
from .optim import optimizar
from .spaces import Param
from .splits import cv_interna

ESPACIO_DLINEAR = [
    Param("longitud", "int", 15, 365, log=True),
    Param("kernel", "int", 5, 61),
    Param("alpha", "float", 1e-4, 10.0, log=True),
    Param("canales", "cat", opciones=("calamar", "ambos")),
]
# Linear y NLinear no descomponen la ventana, por lo que no tienen tamaño de media móvil
ESPACIO_LINEAR = [p for p in ESPACIO_DLINEAR if p.nombre != "kernel"]
ESPACIOS = {"dlinear": ESPACIO_DLINEAR, "nlinear": ESPACIO_LINEAR, "linear": ESPACIO_LINEAR}
LTSF = tuple(ESPACIOS)


def parametros_banda(estacion: str = "calamar") -> dict:
    par = json.loads((DATOS_PROC / "parametros.json").read_text(encoding="utf-8"))[estacion]
    return {"L": par["LIM"][0], "U": par["LIM"][1], "ref": f"{estacion}_cuad"}


class ModeloLTSF:
    """Pronosticador LTSF-Linear de z con residuo gaussiano y probabilidad de banda."""

    def __init__(self, variante: str, params: dict, banda: dict):
        self.variante, self.params, self.banda = variante, params, banda

    def fit(self, X, y, q_futuro):
        qref = np.sqrt(X[self.banda["ref"]].values)
        z = np.log(np.asarray(q_futuro) / qref)
        Modelo = MODELOS_LTSF[self.variante]
        corte = int(len(z) * 0.85)
        # La dispersión del residuo se estima fuera de muestra (último 15 % cronológico)
        aux = Modelo(**self.params).fit(X.iloc[:corte], z[:corte])
        self.sigma_ = float(np.std(z[corte:] - aux.predict(X.iloc[corte:])))
        self.m_ = Modelo(**self.params).fit(X, z)
        return self

    def predict_proba(self, X):
        b = self.banda
        qref = np.sqrt(X[b["ref"]].values)
        mu = self.m_.predict(X)
        l, u = np.log(b["L"] / qref), np.log(b["U"] / qref)
        p = np.clip(norm.cdf((u - mu) / self.sigma_) - norm.cdf((l - mu) / self.sigma_), 1e-7, 1 - 1e-7)
        return np.column_stack([1 - p, p])

    def predict_caudal(self, X):
        """Caudal previsto a quince días (mediana de la distribución lognormal)."""
        return np.sqrt(X[self.banda["ref"]].values) * np.exp(self.m_.predict(X))


def _objetivo(variante, banda, X, y, q):
    def f(params):
        punt = []
        for tr, va in cv_interna().split(X):
            m = ModeloLTSF(variante, params, banda).fit(X.iloc[tr], y.iloc[tr], q.iloc[tr].values)
            punt.append(average_precision_score(y.iloc[va], m.predict_proba(X.iloc[va])[:, 1]))
        return float(np.mean(punt))
    return f


def ejecutar_pliegue(datos: Datos, variante: str, metodo: str, pliegue: dict, presupuesto: int,
                     destino: Path, seed: int = SEED) -> Path:
    """Optimiza, reajusta y evalúa un modelo LTSF-Linear en un pliegue externo."""
    archivo = destino / f"{variante}__{metodo}__f{pliegue['pliegue']:02d}.pkl"
    if archivo.exists():
        return archivo
    banda = parametros_banda()
    tr, te = pliegue["train"], pliegue["test"]
    X, y, q = datos.X.iloc[tr], datos.y_clf.iloc[tr], datos.y_reg.iloc[tr]
    t0 = time.perf_counter()
    reg = optimizar(metodo, _objetivo(variante, banda, X, y, q), ESPACIOS[variante], presupuesto, seed)
    t_opt = time.perf_counter() - t0
    params, interno = reg.mejor()
    t1 = time.perf_counter()
    final = ModeloLTSF(variante, params, banda).fit(X, y, q.values)
    t_fit = time.perf_counter() - t1
    t2 = time.perf_counter()
    pred = final.predict_proba(datos.X.iloc[te])[:, 1]
    t_inf = time.perf_counter() - t2
    yte = datos.y_clf.iloc[te]
    res = {"tarea": "clf", "modelo": variante, "balanceo": "ninguno", "optimizador": metodo,
           "pliegue": pliegue["pliegue"], "bloque": pliegue["bloque"], "params": params,
           "puntaje_interno": interno, "metricas": metricas_clf(yte, pred), "t_opt": t_opt,
           "t_fit": t_fit, "t_inferencia": t_inf, "n_eval": len(reg.filas), "historial": reg.filas,
           "idx_test": te, "y_test": yte.values, "pred": pred, "seed": seed}
    destino.mkdir(parents=True, exist_ok=True)
    with open(archivo, "wb") as f:
        pickle.dump(res, f)
    return archivo


def ejecutar(variantes: list = LTSF, metodo: str = "bayes", presupuesto: int = 16, n_jobs: int = 6):
    """Lanza en paralelo las variantes × pliegues externos."""
    datos = datos_con_ventanas(cargar_datos())
    destino = RESULTADOS / "ltsf_calamar"
    tareas = [(v, p) for v in variantes for p in datos.pliegues]
    tareas.sort(key=lambda t: -len(t[1]["train"]))
    Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(ejecutar_pliegue)(datos, v, metodo, p, presupuesto, destino) for v, p in tareas)
