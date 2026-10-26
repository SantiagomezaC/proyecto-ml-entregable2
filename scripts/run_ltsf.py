"""Evaluación de los modelos LTSF-Linear (Zeng et al., 2023) con el mismo protocolo que los demás.

Uso:  python -m scripts.run_ltsf [n_jobs] [variantes]

Las variantes son ``linear``, ``nlinear`` y ``dlinear``, separadas por comas (por defecto, las
tres).

1. Validación anidada: las 12 particiones externas, la validación interna cronológica, la
   semilla y el presupuesto de 16 evaluaciones con TPE. Los resultados por pliegue se guardan en
   ``results/ltsf_calamar``.
2. Modelo final: optimización sobre todo el entrenamiento (1947-2009), reajuste y una única
   evaluación en la prueba reservada (2010-2026), en clasificación (PR-AUC) y en regresión
   (RMSE del caudal a quince días). Se guarda en ``results/final_ltsf.pkl``.
"""
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTADOS, SEED, fijar_semillas
from src.dlinear import datos_con_ventanas
from src.evaluation import ece
from src.experiment import cargar_datos, metricas_clf, metricas_reg
from src.ltsf import ESPACIOS, LTSF, ModeloLTSF, _objetivo, ejecutar, parametros_banda
from src.optim import optimizar


def final_ltsf(variante: str, datos, presupuesto: int = 16) -> dict:
    banda = parametros_banda()
    t0 = time.perf_counter()
    reg = optimizar("bayes", _objetivo(variante, banda, datos.X, datos.y_clf, datos.y_reg),
                    ESPACIOS[variante], presupuesto, SEED)
    params, interno = reg.mejor()
    t_opt = time.perf_counter() - t0
    modelo = ModeloLTSF(variante, params, banda).fit(datos.X, datos.y_clf, datos.y_reg.values)
    p = modelo.predict_proba(datos.X_test)[:, 1]
    q_pred = modelo.predict_caudal(datos.X_test)
    return {"modelo": variante, "optimizador": "bayes", "params": params, "puntaje_interno": interno,
            "t_opt": t_opt, "sigma": modelo.sigma_, "pred_test": p, "q_pred_test": q_pred,
            "metricas_test": metricas_clf(datos.y_clf_test, p), "ece_test": ece(datos.y_clf_test, p),
            "metricas_reg_test": metricas_reg(datos.y_reg_test.values, q_pred)}


if __name__ == "__main__":
    fijar_semillas()
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    variantes = sys.argv[2].split(",") if len(sys.argv) > 2 else list(LTSF)
    ejecutar(variantes, n_jobs=n_jobs)
    datos = datos_con_ventanas(cargar_datos())
    finales = {}
    for v in variantes:
        finales[v] = final_ltsf(v, datos)
        print(v, finales[v]["params"], {k: round(x, 4) for k, x in finales[v]["metricas_test"].items()},
              {k: round(x, 2) for k, x in finales[v]["metricas_reg_test"].items()})
    with open(RESULTADOS / "final_ltsf.pkl", "wb") as f:
        pickle.dump(finales, f)
