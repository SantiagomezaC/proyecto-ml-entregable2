# Anticipación de los periodos de descarga efectiva en el río Magdalena

**Evaluación de modelos de referencia para la predicción a quince días**

<div class="autores-libro">
  <div class="autor">
    <img src="_static/autor_kevin.jpg" alt="Fotografía de Kevin Clemente Rosario">
    <div><div class="nombre">Kevin Clemente Rosario</div><div class="rol">Geólogo · Estudiante de M.Sc.</div></div>
  </div>
  <div class="autor">
    <img src="_static/autor_manuel.jpg" alt="Fotografía de Manuel Santiago Meza Castro">
    <div><div class="nombre">Manuel Santiago Meza Castro</div><div class="rol">Estudiante de Geología y de M.Sc.</div></div>
  </div>
</div>
<p class="afiliacion">Universidad del Norte, Barranquilla, Colombia</p>

```{admonition} Recursos del estudio
:class: tip
- **Código y resultados:** <https://github.com/SantiagomezaC/proyecto-ml-entregable2>
- **Estudio completo**, con el modelo propuesto y la aplicación interactiva: <https://santiagomezac.github.io/proyecto-ml-final/>
```

## Resumen

Este libro documenta la evaluación de los modelos de referencia del estudio. A partir de 28.558 días de registro del
IDEAM (1947-2026) en Calamar y Puerto Salgar, se predice si el caudal del día *t* + 15 caerá en
la banda de ±15 % alrededor de la descarga efectiva de Calamar (9.954 m³/s), una tarea de
clasificación con 26 % de casos positivos; como tarea auxiliar de regresión se predice el propio
caudal a quince días.

Se entrenaron y compararon 140 combinaciones: siete clasificadores con cuatro técnicas de
balanceo y cuatro métodos de optimización de hiperparámetros (112), y siete regresores con los
cuatro optimizadores (28), todas con validación anidada cronológica en doce bloques externos de
dos años y el mismo presupuesto de evaluaciones. El mejor modelo de referencia de clasificación es Random
Forest con ponderación por clase (PR-AUC externa de 0,910 ± 0,075), estadísticamente
indistinguible de XGBoost, SVM y KNN. El balanceo no mejora la discriminación y los cuatro
optimizadores alcanzan la misma calidad final, aunque TPE y el algoritmo genético llegan antes
que la rejilla. En regresión, SVR y XGBoost integran el conjunto de confianza de modelos y
reducen en un tercio el error de la persistencia. El estudio se completa con la optimización
computacional de cada familia, la calibración de probabilidades, la interpretación con SHAP y
LIME y la comparación estadística con pruebas adecuadas para datos dependientes. Como contraste, se
evaluaron con el mismo protocolo los tres modelos de la familia LTSF-Linear de pronóstico de series
de tiempo (Linear, NLinear y DLinear): se ubican al nivel de la regresión logística y por debajo
de los clasificadores no lineales, lo que indica que lo que separa a los mejores modelos es la
capacidad de representar relaciones no lineales y no la forma de presentar la historia del caudal.

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

1. Revisar la metodología de la versión preliminar del estudio y ampliar el análisis exploratorio.
2. Entrenar, optimizar y comparar estadísticamente 140 combinaciones de modelos, técnicas de
   balanceo y métodos de optimización con validación anidada cronológica, y contrastarlas con
   modelos recientes de pronóstico de series de tiempo.
3. Medir y reducir el costo computacional de cada familia de modelos.
4. Interpretar los mejores modelos y documentar el experimento de forma reproducible.

## Organización del libro

El capítulo {doc}`02_datos` reconstruye el ETL y deriva del análisis exploratorio las decisiones
de preparación. {doc}`03_benchmarks` y sus secciones presentan los modelos y sus resultados:
diseño experimental, clasificación, comparación de optimizadores, optimización computacional,
regresión, comparación estadística, interpretabilidad y, como contraste, los modelos de pronóstico
de series de tiempo de la familia LTSF-Linear. {doc}`05_conclusiones` reúne los hallazgos y
{doc}`04_reproducibilidad` documenta el entorno, la semilla, la tabla maestra de experimentos y la
bibliografía. El {doc}`A_revisiones` registra los cambios metodológicos respecto de la versión
preliminar del estudio.

Este libro es la parte del estudio dedicada a los modelos de referencia. El estudio completo,
que propone un modelo de boosting distribucional para el mismo problema, lo valida
estadísticamente y lo pone a disposición en una aplicación interactiva, está en
<{LIBRO_E3}>.
