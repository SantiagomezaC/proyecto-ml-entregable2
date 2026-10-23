"""Comparación estadística de los benchmarks del Entregable 2.

Uso:  python -m scripts.estadistica
Genera results/estadistica/: tabla pliegues × modelo, Friedman y Nemenyi sobre las 112
combinaciones de clasificación, y en regresión el MCS, las pruebas dirigidas (DM-HLN,
Giacomini-White), Clark-West frente a la persistencia y SPA.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import cargar_carpeta, guardar_json, tabla_pliegues
from src.config import RESULTADOS, fijar_semillas
from src.experiment import cargar_datos
from src.stats_tests import (bootstrap_bloques, clark_west, d_cohen_pareado, diebold_mariano, friedman,
                             giacomini_white, holm, longitud_bloque, model_confidence_set, nemenyi, spa)

E = RESULTADOS / "estadistica"


def universo_clf():
    """Tabla pliegues × modelo con la mejor combinación de cada benchmark de clasificación."""
    d = tabla_pliegues("clf_calamar")
    mejores = pd.read_csv(RESULTADOS / "mejores_clf.csv")
    cols = {}
    for f in mejores.itertuples():
        g = d[(d.modelo == f.modelo) & (d.balanceo == f.balanceo) & (d.optimizador == f.optimizador)]
        cols[f.modelo] = g.set_index("pliegue").pr_auc
    return d, pd.DataFrame(cols).sort_index()


def main():
    fijar_semillas()
    E.mkdir(parents=True, exist_ok=True)
    d, tabla = universo_clf()
    tabla.to_csv(E / "universo_pliegues.csv")

    # Friedman y Nemenyi sobre las 112 combinaciones
    m112 = d.pivot_table(index="pliegue", columns="combo", values="pr_auc")
    f112 = friedman(m112)
    n112 = nemenyi(f112["rangos"], f112["n_bloques"])
    guardar_json({"chi2": f112["chi2"], "gl": f112["gl"], "p": f112["p"], "cd": n112["cd"],
                  "pares_signif": int(n112["pares"].significativo.sum()), "pares_total": int(len(n112["pares"])),
                  "rangos_top": f112["rangos"].head(15).to_dict()}, E / "cd_112.json")
    n112["pares"].to_parquet(E / "nemenyi_112.parquet", index=False)

    # Regresión: MCS sobre las 28 combinaciones con la pérdida cuadrática diaria fuera de muestra
    rr = cargar_carpeta("reg_calamar")
    perd = {}
    preds = {}
    for r in rr:
        clave = f"{r['modelo']}|{r['optimizador']}"
        perd.setdefault(clave, []).append(pd.Series((r["y_test"] - r["pred"]) ** 2, index=r["idx_test"]))
        preds.setdefault(clave, []).append(pd.Series(r["pred"], index=r["idx_test"]))
    perd = pd.DataFrame({k: pd.concat(v).sort_index() for k, v in perd.items()})
    preds = pd.DataFrame({k: pd.concat(v).sort_index() for k, v in preds.items()})
    perd.to_parquet(E / "perdidas_regresion.parquet")
    preds.to_parquet(E / "predicciones_regresion_oof.parquet")
    mcs = model_confidence_set(perd / 1e6, alpha=0.10)
    mcs.to_csv(E / "mcs_regresion_completo.csv")
    vista = mcs.reset_index().rename(columns={"index": "combinación"})
    vista.columns = ["Combinación" if c in ("index", "combinación") or c == vista.columns[0] else c for c in vista.columns]
    vista["RMSE (m³/s)"] = np.sqrt(vista.perdida_media * 1e6).round(1)
    vista = vista[["Combinación", "RMSE (m³/s)", "rango_perdida", "p_mcs", "en_conjunto"]].rename(
        columns={"rango_perdida": "Rango medio de pérdida", "p_mcs": "p-valor MCS", "en_conjunto": "¿En el conjunto?"})
    vista["Rango medio de pérdida"] = vista["Rango medio de pérdida"].round(2)
    vista["p-valor MCS"] = vista["p-valor MCS"].round(3)
    vista["¿En el conjunto?"] = vista["¿En el conjunto?"].map({True: "Sí", False: "No"})
    vista.to_csv(E / "mcs_regresion.csv", index=False)
    finalistas = mcs[mcs.en_conjunto].index.tolist()
    y_oof = pd.concat([pd.Series(r["y_test"], index=r["idx_test"]) for r in rr if
                       f"{r['modelo']}|{r['optimizador']}" == perd.columns[0]]).sort_index()
    filas = []
    orden = mcs.index.tolist()
    mejor = orden[0]
    # Persistencia (Q̂(t+15) = Q(t)): modelo restringido anidado en los modelos lineales
    datos = cargar_datos()
    q_pers = np.sqrt(datos.X.calamar_cuad.values)[y_oof.index.values]
    e_pers = y_oof.values - q_pers
    for b in orden[1:max(4, len(finalistas))]:
        ea, eb = y_oof.values - preds[mejor].values, y_oof.values - preds[b].values
        dm = diebold_mariano(ea, eb, h=15)
        gw = giacomini_white(ea, eb, h=15)
        dl = (ea ** 2 - eb ** 2)
        icb = bootstrap_bloques(np.mean, dl / 1e6, media_bloque=longitud_bloque(dl), n_boot=1000,
                                metodo="percentil")
        filas.append({"modelo A": mejor, "modelo B": b, "DM-HLN": dm["dm"], "p DM": dm["p"],
                      "GW": gw["estadistico"], "p GW": gw["p"],
                      "Δ pérdida media (10⁶ m⁶/s²)": icb["estimacion"],
                      "IC 95 % bootstrap estacionario": f"[{icb['ic_inf']:.3f}; {icb['ic_sup']:.3f}]",
                      "d de Cohen": d_cohen_pareado(dl)})
    cw_filas = []
    for f in finalistas:
        cw = clark_west(y_oof.values, q_pers, preds[f].values, h=15)
        cw_filas.append({"modelo general": f, "modelo restringido": "persistencia", "CW": cw["cw"], "p CW": cw["p"],
                         "RMSE general": float(np.sqrt(perd[f].mean())),
                         "RMSE persistencia": float(np.sqrt(np.mean(e_pers ** 2)))})
    cwd = pd.DataFrame(cw_filas)
    cwd["p CW (Holm)"] = holm(cwd["p CW"].values)
    cwd.to_csv(E / "regresion_clark_west.csv", index=False)
    cmp = pd.DataFrame(filas)
    cmp["p DM (Holm)"] = holm(cmp["p DM"].values)
    cmp["p GW (Holm)"] = holm(cmp["p GW"].values)
    cmp.to_csv(E / "regresion_dirigidas.csv", index=False)
    bench_ridge = [c for c in perd.columns if c.startswith("ridge|")][0]
    sp = spa(perd[bench_ridge].values / 1e6, perd.drop(columns=bench_ridge) / 1e6)
    guardar_json(sp, E / "spa_regresion.json")
    print(f"MCS (90 %): {len(finalistas)} de {perd.shape[1]} combinaciones; mejor: {mejor}")


if __name__ == "__main__":
    main()
