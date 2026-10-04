"""
iot_batch.py - Interfaz entre el prototipo (sensores IoT y cámara) y el artefacto
================================================================================

Carga por lotes (batch) al cierre del día, como recomendó el asesor: no se
trabaja en tiempo real. El software del prototipo exporta dos archivos y este
módulo los convierte en filas de los registros del artefacto.

  1. Lecturas de sensores  (lecturas_iot_AAAA-MM-DD.csv)
       fecha_hora, maquina_id, corriente_a, vibracion_mm_s
       opcionales: estacion, temperatura_c
     -> se detectan los paros (corriente por debajo del umbral) y se
        agregan a datos/paros.csv. La vibración alta genera alertas
        de mantenimiento predictivo.

  2. Detecciones de la cámara (detecciones_vision_AAAA-MM-DD.csv)
       fecha_hora, estacion_deteccion, clase_defecto, confianza
       opcionales: estacion_origen, accion
     -> las detecciones con confianza suficiente se agregan a
        datos/defectos.csv.

La producción del turno (unidades, operarios) la sigue anotando el supervisor
en datos/produccion.csv.

Uso desde la consola (también se puede programar en el Programador de tareas
de Windows para que corra solo al final del día):

    python iot_batch.py --lecturas entrada/lecturas_iot_2026-10-05.csv \
                        --vision entrada/detecciones_vision_2026-10-05.csv
    python iot_batch.py --demo 2026-10-05        # genera un día de prueba y lo procesa
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd

import motor

CARPETA_DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos")
CARPETA_SALIDAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "salidas")

IOT_DEF = {
    "turno1": "06:00-14:00",
    "turno2": "14:00-22:00",
    "pausas": "10:00-10:15, 12:30-12:45, 18:00-18:15, 20:30-20:45",  # refrigerio y limpieza
    "umbral_corriente": 0.5,      # A: por debajo, la máquina está detenida
    "min_paro": 3,                # min: paros más cortos se ignoran (microparos)
    "max_hueco": 5,               # min: más tiempo sin lecturas = hueco de datos
    "alerta_vibracion": 4.5,      # mm/s RMS: zona de alerta (referencia ISO 10816, máquinas pequeñas)
    "pct_alerta": 0.10,           # fracción de lecturas en marcha sobre el umbral para alertar
    "confianza_min": 0.50,        # confianza mínima de la cámara para registrar un defecto
}

COL_LECTURAS = ["fecha_hora", "maquina_id", "corriente_a", "vibracion_mm_s"]
COL_VISION = ["fecha_hora", "estacion_deteccion", "clase_defecto", "confianza"]


# ==========================================================================
# Utilidades
# ==========================================================================

def _rango(txt):
    a, b = [x.strip() for x in txt.split("-")]
    return motor._hora(a), motor._hora(b)


def _rangos(txt):
    return [_rango(x) for x in txt.split(",") if x.strip()]


def _hm(m):
    return f"{int(m) // 60:02d}:{int(m) % 60:02d}"


def _restar_pausas(ini, fin, pausas):
    """Quita de [ini, fin) los tramos que caen en una pausa planificada."""
    tramos = [(ini, fin)]
    for pa, pb in pausas:
        nuevos = []
        for a, b in tramos:
            if b <= pa or a >= pb:
                nuevos.append((a, b))
                continue
            if a < pa:
                nuevos.append((a, pa))
            if b > pb:
                nuevos.append((pb, b))
        tramos = nuevos
    return tramos


def _fecha_hora(serie):
    return pd.to_datetime(serie, errors="coerce", format="mixed", dayfirst=True)


# ==========================================================================
# 1. Sensores -> paros y alertas
# ==========================================================================

def procesar_lecturas(df, cfg=IOT_DEF):
    """Devuelve (paros_df, alertas_df, avisos[list])."""
    avisos = []
    falta = [c for c in COL_LECTURAS if c not in df.columns]
    if falta:
        raise ValueError("Faltan columnas en las lecturas: " + ", ".join(falta))

    d = df.copy()
    d["ts"] = _fecha_hora(d["fecha_hora"])
    d["corriente_a"] = pd.to_numeric(d["corriente_a"].astype(str).str.replace(",", "."), errors="coerce")
    d["vibracion_mm_s"] = pd.to_numeric(d["vibracion_mm_s"].astype(str).str.replace(",", "."), errors="coerce")
    malas = d["ts"].isna() | d["corriente_a"].isna()
    if malas.any():
        avisos.append(f"{int(malas.sum())} lecturas con fecha u hora ilegible o sin corriente se ignoraron.")
    d = d[~malas].sort_values(["maquina_id", "ts"])
    if "estacion" not in d.columns:
        d["estacion"] = ""

    turnos = [(1, *_rango(cfg["turno1"])), (2, *_rango(cfg["turno2"]))]
    pausas = _rangos(cfg["pausas"])
    d["fecha"] = d["ts"].dt.date
    d["min"] = d["ts"].dt.hour * 60 + d["ts"].dt.minute + d["ts"].dt.second / 60

    paros, alertas, huecos = [], [], 0
    for (fecha, maq), g in d.groupby(["fecha", "maquina_id"]):
        est = str(g["estacion"].iloc[0])
        for turno, t_ini, t_fin in turnos:
            gt = g[(g["min"] >= t_ini) & (g["min"] < t_fin)]
            if not len(gt):
                continue
            mins = gt["min"].to_numpy()
            parada = (gt["corriente_a"] < cfg["umbral_corriente"]).to_numpy()
            saltos = np.diff(mins)
            huecos += int((saltos > cfg["max_hueco"]).sum())

            # tramos consecutivos de máquina detenida
            i = 0
            while i < len(mins):
                if not parada[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < len(mins) and parada[j + 1] and mins[j + 1] - mins[j] <= cfg["max_hueco"]:
                    j += 1
                fin = mins[j + 1] if j + 1 < len(mins) else mins[j] + (saltos.min() if len(saltos) else 1)
                for a, b in _restar_pausas(mins[i], min(fin, t_fin), pausas):
                    if b - a >= cfg["min_paro"]:
                        paros.append(dict(fecha=fecha.strftime("%d/%m/%Y"), turno=turno, maquina_id=maq,
                                          estacion=est, hora_inicio=_hm(round(a)), hora_fin=_hm(round(b)),
                                          causa="detectado por sensor", tipo="sensor", planificado="no"))
                i = j + 1

            # alerta predictiva por vibración (solo con la máquina en marcha)
            marcha = gt[~parada]
            if len(marcha):
                frac = float((marcha["vibracion_mm_s"] > cfg["alerta_vibracion"]).mean())
                if frac >= cfg["pct_alerta"]:
                    alertas.append(dict(fecha=fecha.strftime("%d/%m/%Y"), turno=turno, maquina_id=maq,
                                        vibracion_p95=round(float(marcha["vibracion_mm_s"].quantile(0.95)), 2),
                                        lecturas_sobre_umbral=f"{frac:.0%}",
                                        accion="Inspeccionar rodamientos, correa y anclaje antes de que falle."))
    if huecos:
        avisos.append(f"{huecos} huecos de más de {cfg['max_hueco']} min sin lecturas. "
                      f"Revise la conexión del nodo o la batería.")
    return pd.DataFrame(paros), pd.DataFrame(alertas), avisos


# ==========================================================================
# 2. Cámara -> defectos
# ==========================================================================

def procesar_vision(df, cfg=IOT_DEF, turnos=None):
    avisos = []
    falta = [c for c in COL_VISION if c not in df.columns]
    if falta:
        raise ValueError("Faltan columnas en las detecciones: " + ", ".join(falta))
    d = df.copy()
    d["ts"] = _fecha_hora(d["fecha_hora"])
    d["confianza"] = pd.to_numeric(d["confianza"].astype(str).str.replace(",", "."), errors="coerce")
    malas = d["ts"].isna() | d["confianza"].isna()
    if malas.any():
        avisos.append(f"{int(malas.sum())} detecciones ilegibles se ignoraron.")
    d = d[~malas]
    bajas = d["confianza"] < cfg["confianza_min"]
    if bajas.any():
        avisos.append(f"{int(bajas.sum())} detecciones con confianza menor a {cfg['confianza_min']:.2f} "
                      f"no se registraron (posibles falsas alarmas).")
    d = d[~bajas]
    t2 = _rango(cfg["turno2"])[0]
    mins = d["ts"].dt.hour * 60 + d["ts"].dt.minute
    out = pd.DataFrame({
        "fecha": d["ts"].dt.strftime("%d/%m/%Y"),
        "turno": np.where(mins >= t2, 2, 1),
        "estacion_deteccion": d["estacion_deteccion"].astype(str),
        "estacion_origen": d["estacion_origen"].astype(str) if "estacion_origen" in d else "",
        "familia": d["familia"].astype(str) if "familia" in d else "",
        "clase_defecto": d["clase_defecto"].astype(str).str.strip().str.lower(),
        "accion": d["accion"].astype(str) if "accion" in d else "reproceso",
        "tiempo_reproceso_min": "",
    })
    return out.reset_index(drop=True), avisos


# ==========================================================================
# 3. Agregar a los registros del artefacto (sin duplicar)
# ==========================================================================

CLAVES = {
    "prod": ["fecha", "turno"],                                       # una fila por turno
    "paros": ["fecha", "maquina_id", "hora_inicio", "hora_fin"],      # un paro es único
    "defe": ["fecha", "turno", "estacion_deteccion", "clase_defecto"],  # se cuentan repeticiones
}


def _firma(df, cols):
    return df.reindex(columns=cols).fillna("").astype(str).apply(lambda c: c.str.strip()).agg("|".join, axis=1)


def agregar_a_registros(tipo, nuevos, carpeta=CARPETA_DATOS):
    """Agrega filas al CSV maestro sin duplicar lo que ya existe.
    Devuelve (agregadas, omitidas)."""
    if nuevos is None or not len(nuevos):
        return 0, 0
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, motor.ARCHIVOS[tipo])
    nuevos = nuevos.astype(str).reset_index(drop=True)
    if os.path.exists(ruta):
        actual, _ = motor.leer_csv(ruta)
    else:
        actual = pd.DataFrame(columns=motor.REQ[tipo] + motor.OPT[tipo])

    # Cada fila nueva "consume" una fila existente con la misma firma; así,
    # si se vuelve a cargar el mismo archivo, no se duplica nada, pero dos
    # prendas con el mismo defecto en el mismo turno sí se cuentan dos veces.
    existentes = _firma(actual, CLAVES[tipo]).value_counts().to_dict() if len(actual) else {}
    keep = []
    for k in _firma(nuevos, CLAVES[tipo]):
        if existentes.get(k, 0) > 0:
            existentes[k] -= 1
            keep.append(False)
        else:
            keep.append(True)
    mask = pd.Series(keep, index=nuevos.index)

    agregar = nuevos[mask]
    cols = list(dict.fromkeys(list(actual.columns) + list(agregar.columns)))
    final = pd.concat([actual.reindex(columns=cols), agregar.reindex(columns=cols)], ignore_index=True).fillna("")
    final.to_csv(ruta, index=False, encoding="utf-8-sig")
    return int(mask.sum()), int((~mask).sum())


# ==========================================================================
# 4. Reporte diario de eficiencia operativa
# ==========================================================================

def reporte_diario(raw, P, fecha, bl_ref=None):
    """Calcula A, P, Q y OEE del día indicado y los compara con la línea base."""
    f = fecha if not isinstance(fecha, str) else motor._fecha(fecha)
    sub = {}
    for k in ("prod", "paros", "defe"):
        df = raw[k]
        sub[k] = df[[motor._fecha(x) == f for x in df["fecha"]]] if len(df) else df
    if not len(sub["prod"]):
        return None
    sub["sep"] = raw.get("sep", ",")
    bl_dia = motor.linea_base(sub, P)
    fila = bl_dia["daily"].iloc[0]
    rep = {"fecha": f.strftime("%d/%m/%Y"), "A": fila["A"], "P": fila["P"], "Q": fila["Q"], "OEE": fila["OEE"],
           "unidades": fila["unidades"], "paro_min": fila["paro_no_planificado_min"],
           "campana": fila["campana"]}
    if bl_ref is not None:
        rep["OEE0"] = bl_ref["OEE0"]
        rep["dif_pts"] = (fila["OEE"] - bl_ref["OEE0"]) * 100
    rep["maquinas"] = bl_dia["maquinas"]
    rep["pareto"] = bl_dia["pareto"]
    return rep


def guardar_reporte(rep, alertas, carpeta=CARPETA_SALIDAS):
    os.makedirs(carpeta, exist_ok=True)
    f = datetime.strptime(rep["fecha"], "%d/%m/%Y").strftime("%Y-%m-%d")
    ruta = os.path.join(carpeta, f"reporte_diario_{f}.txt")
    pct = lambda x: "–" if x is None or not np.isfinite(x) else f"{x:.1%}"
    lineas = [
        f"REPORTE DIARIO DE EFICIENCIA OPERATIVA - {rep['fecha']}",
        "=" * 60,
        f"OEE del día ........ {pct(rep['OEE'])}" + (f"   (línea base {pct(rep['OEE0'])}, "
                                                    f"{rep['dif_pts']:+.1f} pts)" if "OEE0" in rep else ""),
        f"Disponibilidad A ... {pct(rep['A'])}",
        f"Rendimiento P ...... {pct(rep['P'])}",
        f"Calidad Q .......... {pct(rep['Q'])}",
        f"Unidades ........... {rep['unidades']:.0f}",
        f"Paro no planificado  {rep['paro_min']:.0f} min",
        "",
    ]
    if rep["campana"]:
        lineas.append("Día de campaña: no se compara con la línea base de días normales.")
    if len(rep["maquinas"]):
        m = rep["maquinas"].iloc[0]
        lineas.append(f"Máquina con más paro: {m['maquina_id']} ({m['minutos_paro']:.0f} min en {m['n_paros']:.0f} paros)")
    if len(rep["pareto"]):
        p = rep["pareto"].iloc[0]
        lineas.append(f"Defecto más frecuente: {p['clase']} ({p['n']:.0f} prendas, {p['pct']:.0%})")
    if alertas is not None and len(alertas):
        lineas.append("")
        lineas.append("ALERTAS DE MANTENIMIENTO PREDICTIVO")
        for _, a in alertas.iterrows():
            lineas.append(f"  - {a['maquina_id']} (turno {a['turno']}): vibración p95 {a['vibracion_p95']} mm/s. {a['accion']}")
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas) + "\n")
    return ruta, "\n".join(lineas)


# ==========================================================================
# 5. Día de prueba del prototipo
# ==========================================================================

def generar_dia_demo(fecha="2026-10-05", semilla=11):
    """Simula lo que exportaría el prototipo en un día: lecturas cada minuto
    de 5 máquinas y las detecciones de la cámara en inspección."""
    rng = np.random.default_rng(semilla)
    dia = pd.Timestamp(fecha)
    maquinas = ["C-01", "C-02", "C-03", "C-04", "C-05"]
    filas = []
    paros_reales = {
        "C-01": [(8 * 60 + 12, 8 * 60 + 31)],
        "C-04": [(11 * 60 + 5, 11 * 60 + 52), (16 * 60 + 40, 16 * 60 + 58)],
        "C-03": [(19 * 60 + 20, 19 * 60 + 29)],
    }
    pausas = _rangos(IOT_DEF["pausas"])
    for maq in maquinas:
        vib_base = 5.2 if maq == "C-04" else 2.4      # C-04 vibra más de lo normal
        for m in range(6 * 60, 22 * 60):
            detenida = any(a <= m < b for a, b in paros_reales.get(maq, [])) or any(a <= m < b for a, b in pausas)
            corr = round(float(rng.normal(0.15, 0.03)), 2) if detenida else round(float(rng.normal(2.1, 0.2)), 2)
            vib = round(float(rng.normal(0.3, 0.05)), 2) if detenida else round(float(abs(rng.normal(vib_base, 0.6))), 2)
            filas.append(dict(fecha_hora=(dia + pd.Timedelta(minutes=m)).strftime("%d/%m/%Y %H:%M"),
                              maquina_id=maq, estacion="C", corriente_a=corr, vibracion_mm_s=vib,
                              temperatura_c=round(float(rng.normal(38, 2)), 1)))
    lecturas = pd.DataFrame(filas)

    clases = ["costura saltada", "pretina descuadrada", "atraque faltante", "mancha", "basta despareja"]
    det = []
    for _ in range(int(rng.integers(14, 22))):
        m = int(rng.integers(6 * 60 + 30, 22 * 60))
        det.append(dict(fecha_hora=(dia + pd.Timedelta(minutes=m)).strftime("%d/%m/%Y %H:%M"),
                        estacion_deteccion="I", estacion_origen=str(rng.choice(["C", "D", "F", "G"], p=[.45, .2, .2, .15])),
                        familia="F1", clase_defecto=str(rng.choice(clases, p=[.34, .24, .18, .15, .09])),
                        confianza=round(float(rng.uniform(0.35, 0.98)), 2),
                        accion="merma" if rng.random() < 0.3 else "reproceso"))
    vision = pd.DataFrame(det).sort_values("fecha_hora")

    fs = dia.strftime("%d/%m/%Y")
    prod = pd.DataFrame([
        dict(fecha=fs, turno=1, familia="F1", unidades_producidas=139, unidades_defectuosas=2,
             unidades_reprocesadas=6, operarios_en_planta=34),
        dict(fecha=fs, turno=2, familia="F1", unidades_producidas=144, unidades_defectuosas=3,
             unidades_reprocesadas=5, operarios_en_planta=34),
    ])
    return lecturas, vision, prod


# ==========================================================================
# Consola
# ==========================================================================

def main():
    ap = argparse.ArgumentParser(description="Carga diaria del prototipo IoT y visión al artefacto")
    ap.add_argument("--lecturas", help="CSV exportado por los sensores")
    ap.add_argument("--vision", help="CSV exportado por la cámara")
    ap.add_argument("--demo", metavar="AAAA-MM-DD", help="genera y procesa un día de prueba")
    a = ap.parse_args()

    if a.demo:
        lect, vis, prod = generar_dia_demo(a.demo)
        os.makedirs("entrada", exist_ok=True)
        lect.to_csv(f"entrada/lecturas_iot_{a.demo}.csv", index=False)
        vis.to_csv(f"entrada/detecciones_vision_{a.demo}.csv", index=False)
        n, _ = agregar_a_registros("prod", prod)
        print(f"Día de prueba generado en entrada/ y {n} filas de producción agregadas.")
    else:
        lect = motor.leer_csv(a.lecturas)[0] if a.lecturas else None
        vis = motor.leer_csv(a.vision)[0] if a.vision else None

    alertas = None
    fechas_iot = set()
    if lect is not None:
        paros, alertas, av = procesar_lecturas(lect)
        n, o = agregar_a_registros("paros", paros)
        print(f"Paros detectados: {len(paros)} (agregados {n}, ya existían {o})")
        for x in av:
            print("  aviso:", x)
        fechas_iot |= set(_fecha_hora(lect["fecha_hora"]).dt.date.dropna())
    if vis is not None:
        defe, av = procesar_vision(vis)
        n, o = agregar_a_registros("defe", defe)
        print(f"Defectos de cámara: {len(defe)} (agregados {n}, ya existían {o})")
        for x in av:
            print("  aviso:", x)
        fechas_iot |= set(_fecha_hora(vis["fecha_hora"]).dt.date.dropna())
    if not fechas_iot:
        return

    # reporte del día cargado
    raw = {k: motor.leer_csv(os.path.join(CARPETA_DATOS, motor.ARCHIVOS[k]))[0]
           if os.path.exists(os.path.join(CARPETA_DATOS, motor.ARCHIVOS[k]))
           else pd.DataFrame(columns=motor.REQ[k]) for k in ("prod", "paros", "defe")}
    raw["sep"] = ","
    dia = max(fechas_iot)
    rep = reporte_diario(raw, motor.PLANTA_DEF, dia, None)
    if rep is None:
        print(f"\nFalta la producción del {dia:%d/%m/%Y} en datos/produccion.csv. "
              f"Anótela (una fila por turno) y vuelva a correr, o use el paso 5 de la app.")
        return
    try:
        bl = motor.linea_base(raw, motor.PLANTA_DEF)
        rep = reporte_diario(raw, motor.PLANTA_DEF, dia, bl)
    except Exception:
        pass
    ruta, texto = guardar_reporte(rep, alertas)
    print("\n" + texto)
    print(f"Reporte guardado en {ruta}")


if __name__ == "__main__":
    main()
