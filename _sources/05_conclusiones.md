# Conclusiones

## Hallazgos

El primer hallazgo proviene de las correcciones a la versión anterior del estudio. La imputación de estaciones que
aún no existían inflaba la información aparente del modelo; al restringir los predictores a las
estaciones que existen en todo el periodo (Calamar y Puerto Salgar, desde 1947) el problema
queda libre de imputaciones estructurales, conserva 28.558 observaciones y no pierde desempeño,
porque treinta años adicionales de registro aportan más que las estaciones del bajo Magdalena.
Los intervalos de confianza calculados con bootstrap por bloques resultaron unas cuatro veces más
anchos que los del informe original, que subestimaba la incertidumbre en esa proporción.

El segundo se refiere a los modelos. Con validación anidada cronológica en doce bloques de dos
años, el mejor benchmark de clasificación es Random Forest con ponderación por clase (PR-AUC
externa de 0,910 ± 0,075), pero Random Forest, XGBoost, SVM y KNN son estadísticamente
indistinguibles; Naive Bayes, el árbol de decisión y la regresión logística quedan por debajo,
con diferencias significativas tras la corrección de Holm. La dificultad de cada bloque se explica casi por completo por
la proporción de días positivos que exigen anticipar una entrada a la banda (correlación de
Spearman de −0,93 con la PR-AUC del bloque), lo que da un origen físico a la variabilidad entre
periodos. La ventaja de los modelos no lineales
sobre la regresión logística confirma la recomendación de explorarlos.

El tercero se refiere al balanceo. Ninguna técnica mejora la discriminación medida con PR-AUC:
la ponderación por clase y el sobremuestreo desplazan el punto de operación hacia una mayor
exhaustividad a costa de la precisión, y ADASYN la empeora, porque asigna el 81 % de su peso de
muestreo a positivos situados junto a los límites de la banda e interpola entre días contiguos
del mismo episodio.

El cuarto se refiere a la optimización de hiperparámetros. Con 16 evaluaciones por pliegue, la
rejilla, la búsqueda aleatoria, TPE y el algoritmo genético alcanzan la misma calidad final. TPE
y el algoritmo genético llegan antes que la rejilla, con un efecto grande, y solo
marginalmente antes que la búsqueda aleatoria. La estrategia de multifidelidad (*Successive
Halving*) ahorra entre la mitad y cuatro quintas partes del tiempo sin pérdida apreciable de
calidad.

El quinto se refiere al costo computacional. La SVC con núcleo RBF escala de forma casi
cuadrática con el tamaño de muestra, lo que justificó la aproximación de Nyström; el método por
histogramas acelera XGBoost unas catorce veces y el índice HNSW acelera las consultas de KNN unas
dieciocho veces, con una exhaustividad del 95 % de los vecinos exactos. La validación anidada de
los 112 clasificadores consumió unas 46 horas de CPU, resueltas en menos de cinco horas de reloj
con ejecución concurrente.

El sexto se refiere a la regresión y a la interpretación. SVR y XGBoost forman el conjunto de
confianza de modelos y reducen en un tercio el error de la persistencia. SHAP muestra que la
distancia del caudal actual a la descarga efectiva, el nivel del caudal y su tendencia de corto
plazo dominan la predicción, con un aporte marginal de Puerto Salgar. En el modelo de regresión,
el nivel actual del caudal y los armónicos estacionales concentran la mayor parte de la
importancia, y la distancia a la descarga efectiva pierde el peso que tiene en la clasificación,
porque el caudal futuro guarda una relación monótona con el caudal actual. En una muestra de 200 días
de prueba, LIME identifica las mismas variables relevantes que SHAP, pero cerca de los límites de
la banda invierte el sentido de algunas contribuciones secundarias. Los errores se concentran en los días que cruzan los límites de la banda por
un margen pequeño, donde la etiqueta binaria trata igual un caudal que roza el umbral y uno que
cae en el centro de la banda.

## Limitaciones

El conjunto de predictores carece de variables meteorológicas. Sin precipitación ni índices del
ENSO, ningún modelo dispone de la causa de los ascensos rápidos desde el estiaje, que concentran
los errores. La autocorrelación del caudal reduce el tamaño efectivo de muestra a una fracción de
los días disponibles, y con doce bloques externos la potencia de las pruebas no paramétricas es
limitada. El 89,6 % del registro de Calamar tiene nivel de aprobación preliminar. Finalmente, la
regulación por embalses y el cambio de uso del suelo implican que el pasado puede no representar
el futuro; la evaluación en el periodo 2010-2026 sugiere que la relación aprendida se mantiene,
pero no lo garantiza ante cambios mayores en la cuenca.