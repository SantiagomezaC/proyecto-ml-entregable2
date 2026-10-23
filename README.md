# Descarga efectiva en el río Magdalena: Entregable 2

Predicción, con quince días de anticipación, de si el caudal del río Magdalena en Calamar caerá
dentro de la banda de **descarga efectiva**: el rango de caudales que, a lo largo de las décadas,
transporta la mayor fracción del sedimento. Entregable 2 (modelos benchmark) del proyecto del
curso de Machine Learning (maestría, Universidad del Norte).

**Autores:** Kevin Clemente Rosario y Manuel Santiago Meza Castro

| Producto | Enlace |
|---|---|
| Informe técnico (Jupyter Book) | https://santiagomezac.github.io/proyecto-ml-entregable2/ |
| Repositorio | https://github.com/SantiagomezaC/proyecto-ml-entregable2 |

## Contenido

- **Datos:** ETL de siete series diarias del IDEAM (1940-2026), auditoría de la serie de
  transporte de sedimentos, análisis magnitud-frecuencia y EDA ampliado, con las correcciones
  del Entregable 1 (intervalos por bootstrap de bloques, ajuste de C, sin imputar estaciones que
  aún no existían). Base de modelado: 28.558 observaciones (Calamar y Puerto Salgar, 1947-2026).
- **140 modelos benchmark** con validación anidada cronológica (12 bloques externos de dos años,
  `TimeSeriesSplit` interno con separación de 30 días):
  - Clasificación: 7 modelos (KNN, Naive Bayes, regresión logística, árbol, Random Forest,
    XGBoost, SVM) × 4 balanceos (ninguno, SMOTE, ADASYN, ponderación por clase) × 4
    optimizadores (rejilla, aleatorio, bayesiano con Optuna-TPE, genético con DEAP) = 112.
  - Regresión: 7 modelos (KNN, Ridge, Lasso, árbol, Random Forest, XGBoost, SVR) × 4
    optimizadores = 28.
- **Optimización computacional:** complejidad teórica frente a tiempo empírico, variantes
  eficientes (KD-tree, Ball-tree y FAISS-HNSW; `hist` en XGBoost; Nyström en SVM),
  paralelización, perfilamiento y multifidelidad (*Successive Halving*).
- **Evaluación:** PR-AUC, ROC-AUC, F1, calibración (ECE, Platt, isotónica), robustez a la
  semilla; RMSE, MAE, R² y análisis de residuos en regresión.
- **Comparación estadística:** Friedman-Nemenyi, Wilcoxon-Holm, t corregida de Nadeau-Bengio,
  bootstrap estacionario; en regresión, MCS, Diebold-Mariano (HLN), Giacomini-White,
  Clark-West y SPA.
- **Interpretabilidad:** SHAP (TreeSHAP) global y local, y comparación con LIME en XGBoost.

## Estructura

```text
.
├── src/
│   ├── config.py           semilla global, rutas, partición y paleta
│   ├── etl.py              lectura DHIME, auditoría de transporte, descarga efectiva
│   ├── features.py         predictores y tareas de clasificación y regresión
│   ├── splits.py           particiones cronológicas de la validación anidada
│   ├── spaces.py           espacios de búsqueda comunes a los cuatro optimizadores
│   ├── models.py           catálogo de modelos, balanceo y pipelines
│   ├── optim.py            Grid, Random, TPE (Optuna) y genético (DEAP)
│   ├── experiment.py       motor de la validación anidada (resumible)
│   ├── stats_tests.py      pruebas estadísticas y bootstrap por bloques
│   ├── evaluation.py       calibración y valores de Shapley por permutaciones
│   ├── artifacts.py        serialización, metadata y prueba de ida y vuelta
│   ├── analysis.py         tablas de resultados para el libro
│   ├── data.py             formato de números y nombres legibles de los predictores
│   └── estilo.py           estilo de las figuras
├── scripts/                etl, run_benchmarks, final_models, robustez, computational,
│                           estadistica, ejecutar_libro
├── artifacts/              mejores modelos de clasificación (.joblib) y metadata.json
├── data/processed/         tablas de modelado y particiones
├── results/                tabla maestra, resultados por pliegue y análisis
├── tests/                  pruebas unitarias (pytest)
├── book/                   Jupyter Book (fuentes MyST en book/fuentes)
├── requirements.txt        versiones exactas
└── environment.yml
```

## Instalación

```bash
conda env create -f environment.yml
conda activate ml-final
python -m ipykernel install --user --name ml-final --display-name "Python (ml-final)"
```

Semilla global: `SEED = 42` en `src/config.py`, propagada a numpy, scikit-learn, XGBoost,
imbalanced-learn, Optuna y DEAP.

## Datos

Los archivos del IDEAM no se redistribuyen: la licencia del portal DHIME autoriza la descarga
para uso personal y no comercial. Descargue el caudal medio diario (`Q_MEDIA_D`) y el transporte
medio diario de sedimentos (`TR_KT_D_QS_D`) de las estaciones 29037020 (Calamar), 25027930
(Coyongal), 25027320 (San Roque), 25027020 (El Banco), 25027640 (Tres Cruces), 23037010
(Puerto Salgar) y 21237020 (Arrancaplumas) en http://dhime.ideam.gov.co y guarde los CSV en
`data/raw/` (o defina la variable de entorno `MAGDALENA_RAW`). Las tablas procesadas sí se
incluyen en `data/processed/`, de modo que el libro se puede regenerar sin los archivos crudos.

## Reproducción

```bash
python -m scripts.etl                      # tablas de modelado y particiones
python -m scripts.run_benchmarks clf       # 112 combinaciones (validación anidada)
python -m scripts.run_benchmarks reg       # 28 combinaciones
python -m scripts.final_models clf         # selección, reentrenamiento y artefactos
python -m scripts.final_models reg
python -m scripts.robustez                 # sensibilidad del mejor benchmark a la semilla
python -m scripts.computational            # complejidad, variantes eficientes, perfiles
python -m scripts.estadistica              # pruebas estadísticas
python -m pytest                           # pruebas unitarias
python -m scripts.ejecutar_libro           # ejecuta los cuadernos del libro
jupyter-book build book/
```

Los resultados por pliegue se guardan uno por archivo, de modo que una corrida interrumpida se
reanuda sin repetir trabajo. En un Intel Core i7-1255U con diez procesos, el bloque de
clasificación tarda unas cuatro horas y media. La tabla maestra de las 140 combinaciones queda
en `results/tabla_maestra.csv` (y por pliegue, con hiperparámetros, en
`results/tabla_maestra_por_pliegue.csv`).
