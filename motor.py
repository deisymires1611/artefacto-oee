"""
motor.py - Cálculos del Configurador OEE
=========================================

Integra en un solo módulo los checkpoints del artefacto:

    CP0  Línea base y validación de registros (OEE0, q0, u0)
    CP1  Catálogo de cámaras y sensores
    CP2  Configuración óptima (enumeración exhaustiva + regla voraz)
    CP3  Reglas de acción R1 a R7

No tiene interfaz: lo usan app.py (Streamlit) e iot_batch.py (carga diaria).
"""

import re
from copy import deepcopy

import numpy as np
import pandas as pd

# ==========================================================================
# PARÁMETROS POR DEFECTO
# ==========================================================================

PLANTA_DEF = {
    "minTurno": 480,         # minutos por turno
    "paradaPlan": 30,        # refrigerio, limpieza y arranque (min/turno)
    "turnosDia": 2,
    "diasMes": 24,
    "cicloIdeal": 110,       # s/prenda en la estación cuello de botella
    "umbralCampana": 45,     # operarios desde los que el día es de campaña
    "demanda": 10568,        # u/mes
    "demandaInsat": 930,     # u/mes que no se atienden
    "margen": 30.0,          # S/ por unidad
    "costoRep": 6.0,         # S/ por prenda reprocesada
}

PLANTA_CAMPOS = [
    ("minTurno", "Minutos por turno", "Duración total del turno"),
    ("paradaPlan", "Parada planificada (min/turno)", "Refrigerio, limpieza, arranque"),
    ("turnosDia", "Turnos por día", ""),
    ("diasMes", "Días hábiles por mes", ""),
    ("cicloIdeal", "Tiempo de ciclo ideal (s)", "Del estudio de tiempos, estación cuello de botella"),
    ("umbralCampana", "Operarios desde los que es campaña", "Esos días no entran en la línea base"),
    ("demanda", "Demanda mensual (unidades)", ""),
    ("demandaInsat", "Demanda no atendida (u/mes)", "Tope para valorar horas recuperadas"),
    ("margen", "Margen por unidad (S/)", "Precio menos costo variable"),
    ("costoRep", "Costo de reprocesar una prenda (S/)", ""),
]

CFG_DEF = {
    "Btot": 6000.0, "Bop": 80.0, "Tmax": 12.0, "Nest": 3, "Nmaq": 5,
    "deltaV": 0.70, "deltaM": 0.45, "rMinV": 0.80, "pMinV": 0.85, "rMinM": 0.70,
    "faMax": 0.10, "bV": 2500.0, "bM": 300.0, "c0": 400.0, "o0": 20.0, "nInsp": 3,
}

CFG_PRINCIPAL = [
    ("Btot", "Presupuesto total (S/)", "Inversión máxima"),
    ("Bop", "Gasto mensual máximo (S/)", "Internet, nube, repuestos"),
    ("Tmax", "Recuperación máxima (meses)", ""),
    ("Nest", "Estaciones candidatas a cámara", ""),
    ("Nmaq", "Máquinas críticas a sensar", "Las de más horas de paro"),
]

CFG_AVANZADO = [
    ("deltaV", "Fracción de defectos evitables al detectarlos (δv)", "Supuesto, 0 a 1"),
    ("deltaM", "Fracción de paro evitable con aviso (δm)", "Supuesto, 0 a 1"),
    ("rMinV", "Exhaustividad mínima de la cámara", "0 a 1"),
    ("pMinV", "Precisión mínima de la cámara", "0 a 1"),
    ("rMinM", "Exhaustividad mínima de los sensores", "0 a 1"),
    ("faMax", "Falsas alarmas máximas por máquina y día", ""),
    ("bV", "Tope de inversión por estación con cámara (S/)", ""),
    ("bM", "Tope de inversión por nodo de sensores (S/)", ""),
    ("c0", "Infraestructura común (S/)", "Panel, router, almacenamiento"),
    ("o0", "Gasto mensual común (S/)", ""),
    ("nInsp", "Inspecciones por prenda", "Puntos donde mira la cámara"),
]

# Catálogo de ejemplo. VALORES SUPUESTOS hasta tener cotizaciones y mediciones.
CAT_CAMARAS_DEF = [
    {"nombre": "Plantillas clásicas + cámara web + laptop existente", "costo": 350, "opex": 0, "recall": 0.55, "prec": 0.70, "lat": 0.2},
    {"nombre": "YOLOv8n + cámara web + Raspberry Pi 5", "costo": 900, "opex": 5, "recall": 0.82, "prec": 0.86, "lat": 0.5},
    {"nombre": "YOLOv8s + cámara web + Raspberry Pi 5", "costo": 950, "opex": 5, "recall": 0.86, "prec": 0.88, "lat": 1.3},
    {"nombre": "YOLOv8s + cámara industrial + laptop existente", "costo": 1600, "opex": 3, "recall": 0.88, "prec": 0.90, "lat": 0.4},
    {"nombre": "YOLOv8n + cámara industrial + Jetson Orin Nano", "costo": 2400, "opex": 8, "recall": 0.87, "prec": 0.90, "lat": 0.05},
    {"nombre": "YOLOv8m + cámara industrial + PC con GPU", "costo": 4200, "opex": 10, "recall": 0.90, "prec": 0.92, "lat": 0.03},
]

SENSOR_NOMBRE = {"vib": "Vibración", "temp": "Temperatura", "corr": "Corriente", "ciclos": "Ciclos"}

CAT_SENSORES_DEF = {
    "nodo": 60.0, "opexNodo": 2.0,
    "unit": {"vib": 15.0, "temp": 10.0, "corr": 35.0, "ciclos": 12.0},
    "subsets": [
        {"s": ["vib"], "recall": 0.68, "fa": 0.12},
        {"s": ["vib", "temp"], "recall": 0.74, "fa": 0.10},
        {"s": ["vib", "corr"], "recall": 0.76, "fa": 0.09},
        {"s": ["vib", "ciclos"], "recall": 0.70, "fa": 0.11},
        {"s": ["vib", "temp", "corr"], "recall": 0.79, "fa": 0.08},
        {"s": ["vib", "temp", "ciclos"], "recall": 0.75, "fa": 0.09},
        {"s": ["vib", "corr", "ciclos"], "recall": 0.77, "fa": 0.08},
        {"s": ["vib", "temp", "corr", "ciclos"], "recall": 0.80, "fa": 0.07},
    ],
}

REGLAS_DEF = [
    {"id": "R1", "ind": "Disponibilidad de la línea", "key": "A", "op": "<", "umbral": 85, "unidad": "%", "sev": "Crítica",
     "accion": "Revisar las órdenes de trabajo abiertas y atender primero la máquina {maq}, que acumula más horas de paro.",
     "resp": "Técnico de mantenimiento", "plazo": "Mismo turno"},
    {"id": "R2", "ind": "Horas de paro de la máquina más afectada", "key": "maxMaq", "op": ">", "umbral": 8, "unidad": "h/mes", "sev": "Aviso",
     "accion": "Programar mantenimiento preventivo de la máquina {maq} en la próxima parada planificada.",
     "resp": "Técnico de mantenimiento", "plazo": "48 horas"},
    {"id": "R3", "ind": "Tiempo medio de reparación más alto", "key": "maxMttr", "op": ">", "umbral": 40, "unidad": "min", "sev": "Aviso",
     "accion": "Revisar el stock de repuestos críticos (agujas, correas, bobinas) para la máquina {maqMttr}.",
     "resp": "Jefe de planta", "plazo": "1 semana"},
    {"id": "R4", "ind": "Tasa de defecto y reproceso", "key": "q", "op": ">", "umbral": 5, "unidad": "%", "sev": "Crítica",
     "accion": "Revisar la hoja de trabajo estandarizado de la estación {origen}, donde se originan más defectos.",
     "resp": "Jefe de planta", "plazo": "24 horas"},
    {"id": "R5", "ind": "Peso del defecto más frecuente", "key": "topClase", "op": ">", "umbral": 30, "unidad": "%", "sev": "Aviso",
     "accion": "Capacitación focalizada en el defecto \"{clase}\" para los operarios de la estación {origenClase}.",
     "resp": "Jefe de planta", "plazo": "1 semana"},
    {"id": "R6", "ind": "Rendimiento de la línea", "key": "P", "op": "<", "umbral": 70, "unidad": "%", "sev": "Aviso",
     "accion": "Revisar el balanceo de línea y actualizar los tiempos estándar con el estudio de tiempos.",
     "resp": "Jefe de planta", "plazo": "2 semanas"},
    {"id": "R7", "ind": "Problemas en los registros", "key": "bloqueos", "op": ">", "umbral": 0, "unidad": "problemas", "sev": "Crítica",
     "accion": "Corregir los registros señalados en Validaciones (paso 2) antes de tomar decisiones con estos datos.",
     "resp": "Responsable del registro", "plazo": "Antes de decidir"},
]

# Columnas de los tres registros
REQ = {
    "prod": ["fecha", "turno", "unidades_producidas", "unidades_defectuosas", "unidades_reprocesadas", "operarios_en_planta"],
    "paros": ["fecha", "maquina_id", "hora_inicio", "hora_fin", "planificado"],
    "defe": ["fecha", "clase_defecto"],
}
OPT = {
    "prod": ["familia"],
    "paros": ["turno", "estacion", "causa", "tipo"],
    "defe": ["turno", "estacion_deteccion", "estacion_origen", "familia", "accion", "tiempo_reproceso_min"],
}
ARCHIVOS = {"prod": "produccion.csv", "paros": "paros.csv", "defe": "defectos.csv"}
TITULOS = {"prod": "Producción", "paros": "Paros de máquina", "defe": "Defectos"}


def copia_defectos():
    """Copias independientes de los valores por defecto (para session_state)."""
    return (dict(PLANTA_DEF), dict(CFG_DEF), deepcopy(CAT_CAMARAS_DEF),
            deepcopy(CAT_SENSORES_DEF), deepcopy(REGLAS_DEF))


# ==========================================================================
# LECTURA DE ARCHIVOS
# ==========================================================================

def _norm_col(c):
    import unicodedata
    c = unicodedata.normalize("NFD", str(c).strip().lower())
    c = "".join(ch for ch in c if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", "_", c)


def leer_csv(origen):
    """Lee un CSV (ruta o archivo subido) aceptando coma o punto y coma y
    codificación UTF-8 o Latin-1. Devuelve (DataFrame, separador)."""
    if hasattr(origen, "getvalue"):
        crudo = origen.getvalue()
    else:
        with open(origen, "rb") as f:
            crudo = f.read()
    try:
        texto = crudo.decode("utf-8-sig")
    except UnicodeDecodeError:
        texto = crudo.decode("latin-1")
    primera = texto.split("\n", 1)[0]
    sep = ";" if primera.count(";") > primera.count(",") else ","
    from io import StringIO
    df = pd.read_csv(StringIO(texto), sep=sep, dtype=str, keep_default_na=False)
    df.columns = [_norm_col(c) for c in df.columns]
    df = df[~(df.apply(lambda r: all(str(x).strip() == "" for x in r), axis=1))]
    return df.reset_index(drop=True), sep


def revisar_columnas(tipo, df):
    """Devuelve un mensaje de error o None si el archivo es válido."""
    falta = [c for c in REQ[tipo] if c not in df.columns]
    if falta:
        return "Faltan columnas: " + ", ".join(falta)
    if not len(df):
        return "El archivo no tiene filas de datos."
    return None


def _fecha(s):
    s = str(s or "").strip()
    m = re.match(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})$", s)
    if m:
        d, mo, y = int(m[1]), int(m[2]), m[3]
        y = int("20" + y) if len(y) == 2 else int(y)
        try:
            return pd.Timestamp(year=y, month=mo, day=d).date()
        except ValueError:
            return None
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        try:
            return pd.Timestamp(year=int(m[1]), month=int(m[2]), day=int(m[3])).date()
        except ValueError:
            return None
    return None


def _hora(s):
    m = re.match(r"^(\d{1,2}):(\d{2})(?::\d{2})?$", str(s or "").strip())
    return int(m[1]) * 60 + int(m[2]) if m else None


def _num(s, sep=","):
    t = str(s if s is not None else "").strip()
    if t == "":
        return np.nan
    if sep == ";":
        t = t.replace(".", "").replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return np.nan


# ==========================================================================
# DATOS DE DEMOSTRACIÓN (reproducibles)
# ==========================================================================

def generar_demo(semilla=7):
    rng = np.random.default_rng(semilla)
    clases = ["costura saltada", "pretina descuadrada", "atraque faltante", "mancha", "basta despareja"]
    pesos = [0.34, 0.24, 0.18, 0.15, 0.09]
    origenes, pesos_o = ["C", "D", "F", "G"], [0.45, 0.20, 0.20, 0.15]
    maquinas, pesos_m = ["C-01", "C-02", "C-03", "C-04", "C-05"], [0.28, 0.14, 0.20, 0.26, 0.12]
    causas = ["rotura de aguja", "desajuste de tensión", "falla de motor", "atasco de hilo"]

    fechas, d = [], pd.Timestamp("2026-09-01")
    while len(fechas) < 24:
        if d.weekday() != 6:
            fechas.append(d)
        d += pd.Timedelta(days=1)

    hm = lambda m: f"{m // 60:02d}:{m % 60:02d}"
    prod, paros, defe = [], [], []
    for i, f in enumerate(fechas):
        fs = f.strftime("%d/%m/%Y")
        campana = i >= 17
        ops = int(rng.integers(50, 60)) if campana else int(rng.integers(30, 38))
        for turno in (1, 2):
            u = int(round(rng.normal(142, 15)))
            nd = int(rng.binomial(u, 0.062))
            merma = int(nd * 0.3)
            prod.append(dict(fecha=fs, turno=turno, familia="F1", unidades_producidas=u,
                             unidades_defectuosas=merma, unidades_reprocesadas=nd - merma,
                             operarios_en_planta=ops))
            for _ in range(nd):
                defe.append(dict(fecha=fs, turno=turno, estacion_deteccion="I",
                                 estacion_origen=rng.choice(origenes, p=pesos_o), familia="F1",
                                 clase_defecto=rng.choice(clases, p=pesos),
                                 accion="merma" if rng.random() < 0.3 else "reproceso",
                                 tiempo_reproceso_min=round(float(rng.normal(6, 2)), 1)))
        ocup = {}
        for _ in range(int(rng.poisson(2.8))):
            maq = rng.choice(maquinas, p=pesos_m)
            ini = 390 + int(rng.integers(0, 860))
            dur = int(round(rng.gamma(2.2, 14))) + 4
            if any(ini < b and ini + dur > a for a, b in ocup.get(maq, [])):
                continue
            ocup.setdefault(maq, []).append((ini, ini + dur))
            paros.append(dict(fecha=fs, turno=1 if ini < 840 else 2, maquina_id=maq, estacion="C",
                              hora_inicio=hm(ini), hora_fin=hm(ini + dur),
                              causa=rng.choice(causas), tipo="mecanica", planificado="no"))
    a_str = lambda rows: pd.DataFrame(rows).astype(str)
    return {"prod": a_str(prod), "paros": a_str(paros), "defe": a_str(defe), "sep": ","}


# ==========================================================================
# CP0 - LÍNEA BASE Y VALIDACIONES
# ==========================================================================

def linea_base(raw, P):
    """raw: dict con DataFrames 'prod', 'paros', 'defe' (texto) y 'sep'.
    P: parámetros de la planta. Devuelve un dict con todos los resultados."""
    V = []
    add = lambda nivel, txt: V.append({"nivel": nivel, "texto": txt})
    sep = raw.get("sep", ",")
    tp_turno = P["minTurno"] - P["paradaPlan"]
    h_mes = tp_turno / 60 * P["turnosDia"] * P["diasMes"]

    # --- producción -------------------------------------------------------
    prod, mal_num, mal_fecha = [], 0, 0
    for _, r in raw["prod"].iterrows():
        f = _fecha(r.get("fecha"))
        if not f:
            mal_fecha += 1
            continue
        o = dict(fecha=f, u=_num(r.get("unidades_producidas"), sep), d=_num(r.get("unidades_defectuosas"), sep),
                 r=_num(r.get("unidades_reprocesadas"), sep), ops=_num(r.get("operarios_en_planta"), sep))
        if any(not np.isfinite(o[k]) for k in ("u", "d", "r", "ops")):
            mal_num += 1
            continue
        prod.append(o)
    prod = pd.DataFrame(prod, columns=["fecha", "u", "d", "r", "ops"])
    if mal_fecha:
        add("Aviso", f"{mal_fecha} filas de producción tienen una fecha que no se pudo leer y se ignoraron.")
    if mal_num:
        add("Aviso", f"{mal_num} filas de producción tienen números vacíos o ilegibles y se ignoraron.")

    # --- paros ------------------------------------------------------------
    paros, paros_mal, paros_largos = [], 0, 0
    for _, r in raw["paros"].iterrows():
        f, a, b = _fecha(r.get("fecha")), _hora(r.get("hora_inicio")), _hora(r.get("hora_fin"))
        if not f or a is None or b is None or b - a <= 0:
            paros_mal += 1
            continue
        dur = b - a
        if dur > P["minTurno"]:
            paros_largos += 1
        plan = bool(re.match(r"^(si|sí|s|1|true|x)$", str(r.get("planificado", "")).strip(), re.I))
        paros.append(dict(fecha=f, maq=str(r.get("maquina_id") or "(sin código)").strip(),
                          est=str(r.get("estacion", "") or "").strip(), ini=a, fin=b, dur=dur, plan=plan))
    paros = pd.DataFrame(paros, columns=["fecha", "maq", "est", "ini", "fin", "dur", "plan"])
    if paros_mal:
        add("Bloquea", f"{paros_mal} paros tienen la hora de fin anterior o igual a la de inicio, o una hora ilegible. Revise la transcripción.")
    if paros_largos:
        add("Aviso", f"{paros_largos} paros duran más que un turno completo. Confirme que no sea un error de transcripción.")

    solapes = 0
    for _, g in paros.groupby(["fecha", "maq"]):
        g = g.sort_values("ini")
        solapes += int((g["ini"].values[1:] < g["fin"].values[:-1]).sum())
    if solapes:
        add("Aviso", f"{solapes} paros se superponen en la misma máquina. Puede haber registros duplicados.")

    # --- defectos ---------------------------------------------------------
    d = raw["defe"]
    defe = pd.DataFrame({
        "fecha": [_fecha(x) for x in d["fecha"]],
        "clase": [str(x or "(sin clase)").strip().lower() or "(sin clase)" for x in d["clase_defecto"]],
        "origen": [str(x).strip() for x in d["estacion_origen"]] if "estacion_origen" in d else "",
        "accion": [str(x).strip().lower() for x in d["accion"]] if "accion" in d else "",
    })
    defe = defe[defe["fecha"].notna()]

    # --- por día ----------------------------------------------------------
    dias = sorted(prod["fecha"].unique())
    np_paros = paros[~paros["plan"]]
    filas = []
    for f in dias:
        g = prod[prod["fecha"] == f]
        tp = len(g) * tp_turno
        tparo = min(np_paros.loc[np_paros["fecha"] == f, "dur"].sum(), tp)
        top = tp - tparo
        u, dd, rr = g["u"].sum(), g["d"].sum(), g["r"].sum()
        A = top / tp if tp else np.nan
        Pp = (u * P["cicloIdeal"] / 60) / top if top else np.nan
        Q = (u - dd - rr) / u if u else np.nan
        filas.append(dict(fecha=f, tiempo_programado_min=tp, paro_no_planificado_min=tparo, unidades=u,
                          defectuosas=dd, reprocesadas=rr, A=A, P=Pp, Q=Q, OEE=A * Pp * Q,
                          campana=bool(g["ops"].max() > P["umbralCampana"])))
    daily = pd.DataFrame(filas, columns=["fecha", "tiempo_programado_min", "paro_no_planificado_min", "unidades",
                                         "defectuosas", "reprocesadas", "A", "P", "Q", "OEE", "campana"])

    if len(dias) < 20:
        add("Bloquea", f"Solo hay {len(dias)} días con registro. Se necesitan al menos 20 para una línea base confiable.")
    else:
        add("Correcto", f"{len(dias)} días con registro de producción.")

    if len(dias) > 1:
        rango = pd.date_range(dias[0], dias[-1], freq="D")
        faltan = [x.date() for x in rango if x.weekday() != 6 and x.date() not in set(dias)]
        if faltan:
            ej = ", ".join(x.strftime("%d/%m/%Y") for x in faltan[:3])
            add("Aviso", f"{len(faltan)} días hábiles del periodo no tienen registro (por ejemplo {ej}).")

    n_def_arch, n_def_prod = len(defe), float(prod["d"].sum() + prod["r"].sum())
    if n_def_prod > 0:
        dif = abs(n_def_arch - n_def_prod) / n_def_prod
        if dif > 0.10:
            add("Aviso", f"El detalle de defectos ({n_def_arch} prendas) no coincide con lo declarado en producción "
                         f"({n_def_prod:.0f}). Diferencia de {dif:.0%}: probable subregistro.")
        else:
            add("Correcto", f"El detalle de defectos coincide con lo declarado en producción (diferencia de {dif:.0%}).")

    n_camp = int(daily["campana"].sum()) if len(daily) else 0
    if n_camp:
        add("Aviso", f"{n_camp} días con dotación de campaña (más de {P['umbralCampana']:.0f} operarios). "
                     f"Se excluyen de la línea base porque no son comparables.")
    p_alto = int((daily["P"] > 1).sum()) if len(daily) else 0
    if p_alto:
        add("Aviso", f"{p_alto} días dan un rendimiento mayor a 100%. El tiempo de ciclo ideal "
                     f"({P['cicloIdeal']:.0f} s) parece demasiado alto.")

    normales = daily[~daily["campana"]] if len(daily) else daily
    usar_normales = len(normales) >= 5
    base = normales if usar_normales else daily
    etiqueta = "días normales" if usar_normales else "todos los días (hay menos de 5 días normales)"

    A0, P0, Q0 = (base[k].mean() if len(base) else np.nan for k in ("A", "P", "Q"))
    OEE0 = A0 * P0 * Q0
    u_b = base["unidades"].sum() if len(base) else 0
    q0 = (base["defectuosas"].sum() + base["reprocesadas"].sum()) / u_b if u_b else np.nan

    ic = ic_bootstrap(base["OEE"]) if len(base) else (np.nan, np.nan)

    # --- máquinas ---------------------------------------------------------
    nd = max(len(dias), 1)
    t_op_total = nd * P["turnosDia"] * tp_turno
    if len(np_paros):
        maq = np_paros.groupby("maq").agg(estacion=("est", "first"), n_paros=("dur", "size"),
                                          minutos_paro=("dur", "sum")).reset_index().rename(columns={"maq": "maquina_id"})
        maq["mttr_min"] = maq["minutos_paro"] / maq["n_paros"]
        maq["h_mes"] = maq["minutos_paro"] / 60 * P["diasMes"] / nd
        maq["mtbf_h"] = (t_op_total - maq["minutos_paro"]) / 60 / maq["n_paros"]
        maq = maq.sort_values("h_mes", ascending=False).reset_index(drop=True)
    else:
        maq = pd.DataFrame(columns=["maquina_id", "estacion", "n_paros", "minutos_paro", "mttr_min", "h_mes", "mtbf_h"])
    u0 = float(maq["h_mes"].sum()) if len(maq) else 0.0

    # --- pareto -----------------------------------------------------------
    if len(defe):
        par = defe.groupby("clase").size().reset_index(name="n").sort_values("n", ascending=False).reset_index(drop=True)
        par["pct"] = par["n"] / par["n"].sum()
        par["acum"] = par["pct"].cumsum()
    else:
        par = pd.DataFrame(columns=["clase", "n", "pct", "acum"])

    def origen_de(clase=None):
        x = defe if clase is None else defe[defe["clase"] == clase]
        x = x[x["origen"].astype(str) != ""] if len(x) else x
        if not len(x):
            return "(sin dato de origen)"
        return x["origen"].value_counts().index[0]

    bloqueos = sum(1 for v in V if v["nivel"] == "Bloquea")
    return dict(
        V=V, daily=daily, base=base, base_etiqueta=etiqueta, A0=A0, P0=P0, Q0=Q0, OEE0=OEE0, q0=q0, u0=u0,
        ic=ic, maquinas=maq, pareto=par, Hmes=h_mes, tp_turno=tp_turno, dias=dias, n_camp=n_camp,
        bloqueos=bloqueos, origen=origen_de(None), origen_clase=origen_de(par.iloc[0]["clase"]) if len(par) else "",
        periodo=(dias[0], dias[-1]) if dias else None,
    )


def ic_bootstrap(valores, n=1000, semilla=42):
    """Intervalo de confianza al 95 % por remuestreo de días."""
    v = np.asarray(pd.Series(valores).dropna(), dtype=float)
    if len(v) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(semilla)
    medias = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return tuple(np.percentile(medias, [2.5, 97.5]))


# ==========================================================================
# CP2 - CONFIGURADOR
# ==========================================================================

def costo_subset(ss, CS):
    return CS["nodo"] + sum(CS["unit"].get(k, 0) for k in ss["s"])


def nombre_subset(ss):
    return " + ".join(SENSOR_NOMBRE.get(k, k) for k in ss["s"])


def _frontera(items):
    for a in items:
        a["eff"] = not any(
            b is not a and b["costo"] <= a["costo"] and b["recall"] >= a["recall"]
            and (b["costo"] < a["costo"] or b["recall"] > a["recall"]) for b in items)
    return items


def configurar(bl, P, C, cat_v, CS):
    TK = bl["Hmes"] * 3600 / P["demanda"] if P["demanda"] else np.inf
    lat_max = TK / C["nInsp"] if C["nInsp"] else np.inf
    loss_q = 1 - bl["Q0"]
    maq = bl["maquinas"]
    u_all = float(maq["h_mes"].sum()) if len(maq) else 0.0
    n_est = int(C["Nest"])
    n_maq_real = int(min(C["Nmaq"], len(maq) or C["Nmaq"]))
    u_top = float(maq["h_mes"].head(int(C["Nmaq"])).sum()) if len(maq) else 0.0
    loss_acrit = (1 - bl["A0"]) * (u_top / u_all) if u_all > 0 else 0.0

    v_eval = []
    for j, v in enumerate(cat_v):
        why = []
        if not v["recall"] >= C["rMinV"]:
            why.append(f"exhaustividad {v['recall']:.2f} menor al mínimo {C['rMinV']:.2f}")
        if not v["prec"] >= C["pMinV"]:
            why.append(f"precisión {v['prec']:.2f} menor al mínimo {C['pMinV']:.2f}")
        if not v["lat"] <= lat_max:
            why.append(f"tarda {v['lat']:.1f} s por imagen y el máximo es {lat_max:.1f} s")
        if not v["costo"] <= C["bV"]:
            why.append(f"cuesta S/ {v['costo']:,.0f} por estación y el tope es S/ {C['bV']:,.0f}")
        v_eval.append({"j": j, **v, "ok": not why, "why": why})

    s_eval = []
    for i, ss in enumerate(CS["subsets"]):
        costo, why = costo_subset(ss, CS), []
        if not ss["recall"] >= C["rMinM"]:
            why.append(f"exhaustividad {ss['recall']:.2f} menor al mínimo {C['rMinM']:.2f}")
        if not ss["fa"] <= C["faMax"]:
            why.append(f"{ss['fa']:.2f} falsas alarmas por día, el máximo es {C['faMax']:.2f}")
        if not costo <= C["bM"]:
            why.append(f"cuesta S/ {costo:,.0f} por nodo y el tope es S/ {C['bM']:,.0f}")
        s_eval.append({"i": i, "nombre": nombre_subset(ss), "costo": costo, "recall": ss["recall"],
                       "fa": ss["fa"], "ok": not why, "why": why})

    vF, sF = _frontera(v_eval), _frontera(s_eval)

    def evaluar(v, kv, s, km):
        z = 1 if (kv > 0 or km > 0) else 0
        rV = v["recall"] if v else 0
        rM = s["recall"] if s else 0
        frac_m = km / n_maq_real if n_maq_real else 0
        dQ = loss_q * C["deltaV"] * rV * (kv / n_est if n_est else 0)
        dA = loss_acrit * C["deltaM"] * rM * frac_m
        A = min(1, bl["A0"] + dA)
        Q = min(1, bl["Q0"] + dQ)
        oee = A * bl["P0"] * Q
        capex = C["c0"] * z + kv * (v["costo"] if v else 0) + km * (s["costo"] if s else 0)
        opex = C["o0"] * z + kv * (v["opex"] if v else 0) + km * CS["opexNodo"]
        ahorro_q = P["demanda"] * loss_q * C["deltaV"] * rV * (kv / n_est if n_est else 0) * P["costoRep"]
        h_rec = loss_acrit * bl["Hmes"] * C["deltaM"] * rM * frac_m
        u_rec = min(h_rec * 3600 / TK, P["demandaInsat"]) if np.isfinite(TK) and TK > 0 else 0
        ahorro_a = u_rec * P["margen"]
        ahorro = ahorro_q + ahorro_a
        payback = capex / (ahorro - opex) if ahorro > opex else np.inf
        eps = (oee - bl["OEE0"]) * 100 / (capex / 1000) if capex > 0 else 0
        return dict(v=v, kv=kv, s=s, km=km, A=A, Q=Q, OEE=oee, dOEE=oee - bl["OEE0"], capex=capex, opex=opex,
                    ahorro=ahorro, ahorro_q=ahorro_q, ahorro_a=ahorro_a, u_rec=u_rec, payback=payback, eps=eps)

    def factible(e):
        return (e["kv"] + e["km"] > 0) and e["capex"] <= C["Btot"] and e["opex"] <= C["Bop"] and e["payback"] <= C["Tmax"]

    # enumeración exhaustiva
    v_opts = [None] + [v for v in vF if v["ok"]]
    s_opts = [None] + [s for s in sF if s["ok"]]
    best, n, n_fact = None, 0, 0
    razones = {"capex": 0, "opex": 0, "payback": 0}
    for v in v_opts:
        for kv in (range(1, n_est + 1) if v else [0]):
            for s in s_opts:
                for km in (range(1, n_maq_real + 1) if s else [0]):
                    if not v and not s:
                        continue
                    e = evaluar(v, kv, s, km)
                    n += 1
                    if e["capex"] > C["Btot"]:
                        razones["capex"] += 1
                        continue
                    if e["opex"] > C["Bop"]:
                        razones["opex"] += 1
                        continue
                    if e["payback"] > C["Tmax"]:
                        razones["payback"] += 1
                        continue
                    n_fact += 1
                    if (best is None or e["dOEE"] > best["dOEE"] + 1e-9
                            or (abs(e["dOEE"] - best["dOEE"]) < 1e-9 and e["capex"] < best["capex"])):
                        best = e

    # regla voraz: orden sugerido de compra si el presupuesto llega por partes
    ratio = lambda e: e["dOEE"] / e["capex"] if e["capex"] > 0 else 0
    v_good = [v for v in vF if v["ok"]]
    s_good = [s for s in sF if s["ok"]]
    v_star = max(v_good, key=lambda v: ratio(evaluar(v, 1, None, 0)), default=None)
    s_star = max(s_good, key=lambda s: ratio(evaluar(None, 0, s, 1)), default=None)
    kv = km = 0
    pasos = []
    cur = evaluar(None, 0, None, 0)
    while True:
        cands = []
        if v_star and kv < n_est:
            cands.append(("v", evaluar(v_star, kv + 1, s_star, km)))
        if s_star and km < n_maq_real:
            cands.append(("s", evaluar(v_star, kv, s_star, km + 1)))
        ok = [(t, e, (e["dOEE"] - cur["dOEE"]) / max(1, e["capex"] - cur["capex"]))
              for t, e in cands if e["capex"] <= C["Btot"] and e["opex"] <= C["Bop"]]
        ok.sort(key=lambda x: -x[2])
        if not ok or ok[0][1]["dOEE"] <= cur["dOEE"]:
            break
        t, e, _ = ok[0]
        if t == "v":
            kv += 1
        else:
            km += 1
        pasos.append(dict(compra=len(pasos) + 1, tipo="Cámara" if t == "v" else "Nodo de sensores",
                          equipo=v_star["nombre"] if t == "v" else nombre_subset(CS["subsets"][s_star["i"]]),
                          ganancia_pts=(e["dOEE"] - cur["dOEE"]) * 100, costo=e["capex"] - cur["capex"],
                          inversion_acum=e["capex"], oee_alcanzado=e["OEE"]))
        cur = e
    voraz = cur if pasos else None

    return dict(TK=TK, lat_max=lat_max, loss_q=loss_q, loss_acrit=loss_acrit, n_maq_real=n_maq_real,
                vF=vF, sF=sF, best=best, voraz=voraz, voraz_ok=bool(voraz and factible(voraz)),
                pasos=pd.DataFrame(pasos), n=n, n_fact=n_fact, razones=razones)


# ==========================================================================
# CP3 - REGLAS DE ACCIÓN
# ==========================================================================

def evaluar_reglas(bl, reglas):
    maq = bl["maquinas"]
    top = maq.iloc[0] if len(maq) else None
    mttr_top = maq.sort_values("mttr_min", ascending=False).iloc[0] if len(maq) else None
    par = bl["pareto"]
    vals = {
        "A": bl["A0"] * 100, "P": bl["P0"] * 100, "q": bl["q0"] * 100,
        "maxMaq": float(top["h_mes"]) if top is not None else 0.0,
        "maxMttr": float(mttr_top["mttr_min"]) if mttr_top is not None else 0.0,
        "topClase": float(par.iloc[0]["pct"]) * 100 if len(par) else 0.0,
        "bloqueos": bl["bloqueos"],
    }
    ctx = {
        "maq": top["maquina_id"] if top is not None else "(sin datos)",
        "maqMttr": mttr_top["maquina_id"] if mttr_top is not None else "(sin datos)",
        "origen": bl["origen"], "clase": par.iloc[0]["clase"] if len(par) else "",
        "origenClase": bl["origen_clase"],
    }
    salida = []
    for r in reglas:
        v = vals.get(r["key"], np.nan)
        fire = bool(v < r["umbral"]) if r["op"] == "<" else bool(v > r["umbral"])
        texto = re.sub(r"\{(\w+)\}", lambda m: str(ctx.get(m[1], "")), r["accion"])
        salida.append({**r, "valor": v, "disparada": fire, "texto": texto})
    return salida
