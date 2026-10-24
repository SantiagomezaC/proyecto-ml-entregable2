# Descarga efectiva en el río Magdalena

Autores: Kevin Clemente y Manuel Meza
Maestría · Universidad del Norte · Machine Learning, 2026-30

```{admonition} Entregable 2: modelos benchmark
:class: tip
- **Repositorio:** <https://github.com/SantiagomezaC/proyecto-ml-entregable2>
- **Libro en línea:** <https://santiagomezac.github.io/proyecto-ml-entregable2>
```

## Resumen ejecutivo

Este libro documenta el Entregable 2 del proyecto. A partir de 28.558 días de registro del
IDEAM (1947-2026) en Calamar y Puerto Salgar, se predice si el caudal del día *t* + 15 caerá en
la banda de ±15 % alrededor de la descarga efectiva de Calamar (9.954 m³/s), una tarea de
clasificación con 26 % de casos positivos; como tarea auxiliar de regresión se predice el propio
caudal a quince días.

Se entrenaron y compararon 140 combinaciones: siete clasificadores con cuatro técnicas de
balanceo y cuatro métodos de optimización de hiperparámetros (112), y siete regresores con los
cuatro optimizadores (28), todas con validación anidada cronológica en doce bloques externos de
dos años y el mismo presupuesto de evaluaciones. El mejor benchmark de clasificación es Random
Forest con ponderación por clase (PR-AUC externa de 0,910 ± 0,075), estadísticamente
indistinguible de XGBoost, SVM y KNN. El balanceo no mejora la discriminación y los cuatro
optimizadores alcanzan la misma calidad final, aunque TPE y el algoritmo genético llegan antes
que la rejilla. En regresión, SVR y XGBoost integran el conjunto de confianza de modelos y
reducen en un tercio el error de la persistencia. El estudio se completa con la optimización
computacional de cada familia, la calibración de probabilidades, la interpretación con SHAP y
LIME y la comparación estadística con pruebas adecuadas para datos dependientes.

## Problema

El río Magdalena drena cerca de la cuarta parte de Colombia y entrega al mar Caribe del orden
de 144 millones de toneladas de sedimento al año {cite}`restrepo2000`, uno de los rendimientos
sedimentarios más altos de Suramérica {cite}`milliman1992`. La mayor parte de ese trabajo
geomorfológico no la realizan las crecidas excepcionales ni los caudales bajos y frecuentes,
sino un rango intermedio de caudales, la descarga efectiva {cite}`wolman1960`, que en la
estación Calamar se ubica alrededor de 9.954 m³/s. La literatura usa la descarga efectiva como
un descriptor de largo plazo; este proyecto la convierte en un objetivo de predicción:
anticipar, con quince días de antelación, si el río estará dentro de la banda de ±15 % alrededor
de ese valor, información útil para programar purgas de embalses, dragados del canal navegable
y campañas de muestreo de sedimentos.

## Objetivos

1. Corregir el análisis del Entregable 1 según la retroalimentación recibida y ampliar el
   análisis exploratorio.
2. Entrenar, optimizar y comparar estadísticamente 140 combinaciones de modelos, técnicas de
   balanceo y métodos de optimización con validación anidada cronológica.
3. Medir y reducir el costo computacional de cada familia de modelos.
4. Interpretar los mejores modelos y documentar el experimento de forma reproducible.

## Mapa del libro

El capítulo 1 registra las correcciones al Entregable 1. El capítulo 2 reconstruye el ETL y
amplía el análisis exploratorio. El capítulo 3 y sus secciones presentan los modelos y sus
resultados: diseño experimental, clasificación, comparación de optimizadores, optimización
computacional, regresión, comparación estadística e interpretabilidad. El capítulo 4 documenta
el entorno, la semilla, la tabla maestra de experimentos y la bibliografía, y el capítulo 5
reúne las conclusiones.
