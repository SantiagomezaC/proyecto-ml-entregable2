"""Formato de textos y nombres legibles de los predictores para tablas y figuras del libro."""
from __future__ import annotations

import re

from .config import NOMBRES_LEGIBLES


def es(texto):
    """Convierte los números de un texto al formato español (1.234,5) sin tocar la prosa."""
    if not isinstance(texto, str):
        return texto
    texto = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", " ", texto)   # separador de miles provisional
    texto = re.sub(r"(?<=\d)\.(?=\d)", ",", texto)
    return texto.replace(" ", ".")


def etiqueta(col: str) -> str:
    """Nombre legible y unidad de cada predictor, para formularios, ejes y tablas."""
    est = {k: v for k, v in NOMBRES_LEGIBLES.items()}
    otros = {"sen_anual": "Seno del ciclo anual (–)", "cos_anual": "Coseno del ciclo anual (–)",
             "sen_semi": "Seno del ciclo semianual (–)", "cos_semi": "Coseno del ciclo semianual (–)",
             "dist_qef": "Distancia del caudal actual a Q_ef (m³/s)",
             "dist_qef_rel": "Distancia relativa a Q_ef (–)",
             "calamar_cuad": "Caudal actual en Calamar al cuadrado (m⁶/s²)",
             "dist_qef_lag7": "Distancia a Q_ef hace 7 días (m³/s)",
             "dist_qef_lag15": "Distancia a Q_ef hace 15 días (m³/s)"}
    if col in otros:
        return otros[col]
    if col == "y":
        return "Clase (1 = en la banda)"
    if col == "q_futuro":
        return "Caudal en Calamar en t+15 (m³/s)"
    if col in est:
        return f"Caudal en {est[col]} (m³/s)"
    m = re.match(r"(\w+?)_lag(\d+)$", col)
    if m and m.group(1) in est:
        return f"Caudal en {est[m.group(1)]} hace {m.group(2)} días (m³/s)"
    m = re.match(r"(\w+?)_(media|desv)(\d+)$", col)
    if m:
        tipo = "Media" if m.group(2) == "media" else "Desviación"
        return f"{tipo} móvil {m.group(3)} d en {est.get(m.group(1), m.group(1))} (m³/s)"
    m = re.match(r"(\w+?)_d(\d+)$", col)
    if m:
        return f"Cambio en {m.group(2)} d de {est.get(m.group(1), m.group(1))} (m³/s)"
    m = re.match(r"(\w+?)_t(\d+)$", col)
    if m:
        return f"Caudal en {est.get(m.group(1), m.group(1))} desfasado {m.group(2)} d (m³/s)"
    m = re.match(r"(\w+?)_rel$", col)
    if m:
        return f"Caudal relativo a la media anual en {est.get(m.group(1), m.group(1))} (–)"
    return col
