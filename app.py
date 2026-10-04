"""
app.py - Configurador OEE (artefacto de la tesis)
=================================================

Un solo programa que integra todos los checkpoints:

    Paso 1  Datos de planta          (carga y parámetros)
    Paso 2  Línea base               (CP0: OEE0, q0, u0, validaciones)
    Paso 3  Configuración            (CP1 catálogo + CP2 optimización)
    Paso 4  Acciones                 (CP3 reglas R1-R7 con responsable y plazo)
    Paso 5  Carga diaria del prototipo (interfaz IoT y cámara, por lotes)

Para abrirlo:  doble clic en INICIAR_ARTEFACTO.bat
o en la consola:  streamlit run app.py
"""

import copy
import json
import os
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import iot_batch as ib
import motor

BASE = os.path.dirname(os.path.abspath(__file__))
DATOS = os.path.join(BASE, "datos")
SALIDAS = os.path.join(BASE, "salidas")
REG_ACCIONES = os.path.join(SALIDAS, "registro_acciones.json")
os.makedirs(SALIDAS, exist_ok=True)

st.set_page_config(page_title="Configurador OEE · Confección", page_icon="🧵", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1200px;}
div[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
.chip {display:inline-block; padding:3px 10px; border-radius:999px; font-size:13px; margin-right:6px;
       font-weight:500; border:1px solid rgba(128,128,128,.25);}
.chip.ok {background:#E3F1E8; color:#2F7A4D; border-color:transparent;}
.chip.warn {background:#FBEFD9; color:#9A5B0A; border-color:transparent;}
.chip.crit {background:#F8E3E3; color:#A93232; border-color:transparent;}
.reco {border:2px dashed #B8812F; border-radius:8px; padding:14px 18px; margin:6px 0 14px 0;}
</style>
""", unsafe_allow_html=True)


# ==========================================================================
# Formato
# ==========================================================================

def pct(x, d=1):
    return "–" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def soles(x):
    return "–" if x is None or not np.isfinite(x) else f"S/ {x:,.0f}"


def fx(x, d=1):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def fecha_txt(f):
    return f.strftime("%d/%m/%Y") if f else "–"


# ==========================================================================
# Estado inicial
# ==========================================================================

def iniciar_estado():
    P, C, cat_v, cat_s, reglas = motor.copia_defectos()
    ss = st.session_state
    ss.setdefault("P", P)
    ss.setdefault("C", C)
    ss.setdefault("cat_v", cat_v)
    ss.setdefault("cat_s", cat_s)
    ss.setdefault("reglas", reglas)
    ss.setdefault("iot_cfg", dict(ib.IOT_DEF))
    ss.setdefault("subidos", {})
    ss.setdefault("iot", None)
    hay_carpeta = all(os.path.exists(os.path.join(DATOS, f)) for f in motor.ARCHIVOS.values())
    ss.setdefault("fuente", "carpeta" if hay_carpeta else "demo")


iniciar_estado()
ss = st.session_state


def cargar_raw():
    """Devuelve (raw, etiqueta) según la fuente elegida.
    No cambia la opción elegida por el usuario: si faltan archivos, solo
    muestra la demostración mientras tanto (así no se borran los ya subidos)."""
    if ss.fuente == "carpeta":
        try:
            raw = {k: motor.leer_csv(os.path.join(DATOS, f))[0] for k, f in motor.ARCHIVOS.items()}
            raw["sep"] = ","
            return raw, "Registros de la carpeta datos/"
        except FileNotFoundError:
            pass
    if ss.fuente == "subidos" and len(ss.subidos) == 3:
        raw = {k: v["df"] for k, v in ss.subidos.items()}
        raw["sep"] = ss.subidos["prod"]["sep"]
        return raw, "Registros cargados"
    return motor.generar_demo(), "Datos de demostración"


# ==========================================================================
# Cabecera
# ==========================================================================

c1, c2 = st.columns([3, 2])
with c1:
    st.caption("HERRAMIENTA DE SOPORTE A DECISIONES · MYPE DE CONFECCIÓN DE DENIM")
    st.title("Configurador OEE")
chips_slot = c2.empty()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1 · Datos de planta", "2 · Línea base", "3 · Configuración", "4 · Acciones", "5 · Carga diaria del prototipo"])


# ==========================================================================
# PASO 1 - DATOS DE PLANTA
# ==========================================================================

with tab1:
    st.subheader("Registros de planta")
    opciones = {"demo": "Datos de demostración", "carpeta": "Registros de la carpeta datos/", "subidos": "Subir mis archivos CSV"}
    elegido = st.radio("¿Con qué datos quiere trabajar?", list(opciones), format_func=opciones.get,
                       index=list(opciones).index(ss.fuente), horizontal=True)
    if elegido != ss.fuente:
        ss.fuente = elegido
        st.rerun()

    if ss.fuente == "carpeta":
        st.info(f"Se leen los archivos **produccion.csv**, **paros.csv** y **defectos.csv** de la carpeta `{DATOS}`. "
                "La carga diaria del paso 5 agrega filas a estos mismos archivos.")
        if not all(os.path.exists(os.path.join(DATOS, f)) for f in motor.ARCHIVOS.values()):
            st.warning("La carpeta datos/ todavía no tiene los tres archivos.")

    if ss.fuente == "subidos":
        st.caption("Guarde cada registro desde Excel como **CSV UTF-8 (delimitado por comas)**. "
                   "Se necesitan al menos 20 días hábiles.")
        for k, titulo in motor.TITULOS.items():
            col_a, col_b = st.columns([2, 3])
            archivo = col_a.file_uploader(titulo, type=["csv"], key=f"up_{k}")
            if archivo is not None:
                try:
                    df, sep = motor.leer_csv(archivo)
                    err = motor.revisar_columnas(k, df)
                except Exception:
                    df, sep, err = None, ",", "No se pudo leer el archivo. ¿Lo guardó como CSV?"
                if err:
                    ss.subidos.pop(k, None)
                    col_b.error(f"Error. {err}")
                else:
                    ss.subidos[k] = {"df": df, "sep": sep, "nombre": archivo.name}
                    col_b.success(f"Listo · {len(df)} filas · {archivo.name}")
            elif k in ss.subidos:
                col_b.success(f"Listo · {len(ss.subidos[k]['df'])} filas · {ss.subidos[k]['nombre']}")
        faltan = [motor.TITULOS[k] for k in motor.TITULOS if k not in ss.subidos]
        if faltan:
            st.warning("Faltan archivos válidos: " + ", ".join(faltan) + ". Mientras tanto se muestran datos de demostración.")
        elif st.button("Guardar estos registros en la carpeta datos/ (para seguir sumando días)"):
            os.makedirs(DATOS, exist_ok=True)
            for k, v in ss.subidos.items():
                v["df"].to_csv(os.path.join(DATOS, motor.ARCHIVOS[k]), index=False, encoding="utf-8-sig")
            st.success("Registros guardados. Ahora puede elegir 'Registros de la carpeta datos/'.")

    if ss.fuente == "demo":
        if st.button("Crear la carpeta datos/ con estos datos de demostración"):
            os.makedirs(DATOS, exist_ok=True)
            demo = motor.generar_demo()
            for k, f in motor.ARCHIVOS.items():
                demo[k].to_csv(os.path.join(DATOS, f), index=False, encoding="utf-8-sig")
            ss.fuente = "carpeta"
            st.rerun()

    st.divider()
    st.subheader("Parámetros de la planta")
    st.caption("Reemplace los valores de ejemplo por los de su planta. Los resultados se recalculan solos.")
    cols = st.columns(3)
    for i, (k, etiqueta, ayuda) in enumerate(motor.PLANTA_CAMPOS):
        ss.P[k] = cols[i % 3].number_input(etiqueta, min_value=0.0, value=float(ss.P[k]), step=1.0,
                                           help=ayuda or None, key=f"pl_{k}")
    if st.button("Restaurar parámetros de ejemplo"):
        for k in motor.PLANTA_DEF:
            ss.pop(f"pl_{k}", None)
        ss.P = dict(motor.PLANTA_DEF)
        st.rerun()

    st.divider()
    st.subheader("Formato de los archivos")
    st.caption("Descargue la plantilla, ábrala en Excel y llene una fila por registro.")
    cols = st.columns(3)
    for i, (k, titulo) in enumerate(motor.TITULOS.items()):
        encabezados = ",".join(motor.REQ[k] + motor.OPT[k]) + "\n"
        cols[i].markdown(f"**{titulo}**  \nObligatorias: `{', '.join(motor.REQ[k])}`")
        cols[i].download_button(f"Descargar plantilla de {titulo.lower()}", encabezados.encode("utf-8-sig"),
                                file_name=f"plantilla_{motor.ARCHIVOS[k]}", mime="text/csv", key=f"dl_{k}")


# ==========================================================================
# Cálculo (se usa en todos los pasos)
# ==========================================================================

raw, etiqueta_fuente = cargar_raw()
P = ss.P
bl = motor.linea_base(raw, P)
hay_datos = len(bl["daily"]) > 0

chips = [f'<span class="chip {"warn" if ss.fuente == "demo" else "ok"}">● {etiqueta_fuente}</span>']
if hay_datos:
    chips.append(f'<span class="chip crit">● {bl["bloqueos"]} problema(s) en los registros</span>' if bl["bloqueos"]
                 else '<span class="chip ok">● Registros sin bloqueos</span>')
chips.append('<span class="chip">● Catálogo con valores de ejemplo</span>')
chips_slot.markdown("<div style='text-align:right;padding-top:2.2rem'>" + "".join(chips) + "</div>", unsafe_allow_html=True)


# ==========================================================================
# PASO 2 - LÍNEA BASE
# ==========================================================================

with tab2:
    if not hay_datos:
        st.error("No hay días válidos para calcular. Revise los registros en el paso 1.")
    else:
        if ss.fuente == "demo":
            st.info("**Está viendo datos de demostración.** Cargue sus registros en el paso 1 para calcular la línea base real.")
        if bl["bloqueos"]:
            st.error(f"**No use estos resultados para decidir todavía.** Hay {bl['bloqueos']} problema(s) en los "
                     "registros. Vea Validaciones al final de esta página.")
        per = f"{fecha_txt(bl['periodo'][0])} al {fecha_txt(bl['periodo'][1])}"
        st.subheader("Línea base")
        st.caption(f"Periodo {per} · calculada con {len(bl['base'])} {bl['base_etiqueta']}")

        m = st.columns(6)
        m[0].metric("OEE actual (OEE₀)", pct(bl["OEE0"]), help=f"IC 95 %: {pct(bl['ic'][0])} a {pct(bl['ic'][1])}")
        m[1].metric("Disponibilidad A₀", pct(bl["A0"]))
        m[2].metric("Rendimiento P₀", pct(bl["P0"]))
        m[3].metric("Calidad Q₀", pct(bl["Q0"]))
        m[4].metric("Defecto y reproceso q₀", pct(bl["q0"], 2))
        m[5].metric("Paro no planificado u₀", f"{fx(bl['u0'])} h/mes")
        st.caption(f"Intervalo de confianza al 95 % del OEE₀ (remuestreo de días): {pct(bl['ic'][0])} a {pct(bl['ic'][1])}")

        st.markdown("#### OEE por día")
        dd = bl["daily"].copy()
        dd["Fecha"] = pd.to_datetime(dd["fecha"]).dt.strftime("%d/%m")
        dd["Tipo de día"] = np.where(dd["campana"], "Campaña (excluido)", "Normal")
        barras = alt.Chart(dd).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("Fecha:N", sort=None, title="Día"),
            y=alt.Y("OEE:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]), title="OEE"),
            color=alt.Color("Tipo de día:N", scale=alt.Scale(domain=["Normal", "Campaña (excluido)"],
                                                              range=["#3B4F8C", "#C9CFDD"])),
            tooltip=[alt.Tooltip("Fecha:N"), alt.Tooltip("Tipo de día:N"),
                     alt.Tooltip("OEE:Q", format=".1%"), alt.Tooltip("A:Q", format=".1%"),
                     alt.Tooltip("P:Q", format=".1%"), alt.Tooltip("Q:Q", format=".1%")])
        linea = alt.Chart(pd.DataFrame({"y": [bl["OEE0"]]})).mark_rule(color="#B8812F", strokeDash=[5, 4], size=2).encode(y="y:Q")
        st.altair_chart((barras + linea).properties(height=260), width="stretch")
        st.caption("La línea punteada es el OEE₀. Los días de campaña no entran en el cálculo.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Dónde se pierde calidad")
            par = bl["pareto"].copy()
            if len(par):
                par_v = par.rename(columns={"clase": "Defecto", "n": "Prendas"})
                par_v["Peso"] = par_v["pct"].map(lambda x: pct(x, 0))
                par_v["Acumulado"] = par_v["acum"].map(lambda x: pct(x, 0))
                st.dataframe(par_v[["Defecto", "Prendas", "Peso", "Acumulado"]], hide_index=True, width="stretch")
                n80 = int((par["acum"] <= 0.8).sum() + 1)
                st.caption(f"{min(n80, len(par))} clases explican cerca del 80 % de los defectos.")
            else:
                st.write("Sin registros de defectos.")
        with c2:
            st.markdown("#### Dónde se pierde disponibilidad")
            mq = bl["maquinas"]
            if len(mq):
                mv = pd.DataFrame({
                    "Máquina": mq["maquina_id"], "Paros": mq["n_paros"].astype(int),
                    "Horas de paro/mes": mq["h_mes"].round(1), "MTBF (h)": mq["mtbf_h"].round(0),
                    "MTTR (min)": mq["mttr_min"].round(0),
                    "Candidata a sensor": ["Sí" if i < ss.C["Nmaq"] else "" for i in range(len(mq))]})
                st.dataframe(mv, hide_index=True, width="stretch")
            else:
                st.write("Sin paros no planificados.")

        st.markdown("#### Validaciones de los registros")
        for v in bl["V"]:
            {"Correcto": st.success, "Aviso": st.warning, "Bloquea": st.error}[v["nivel"]](f"**{v['nivel']}.** {v['texto']}")

        resumen = (
            f"LÍNEA BASE ({per}, {len(bl['base'])} {bl['base_etiqueta']})\n"
            f"OEE0 = {pct(bl['OEE0'])} (IC95% {pct(bl['ic'][0])} a {pct(bl['ic'][1])})\n"
            f"A0 = {pct(bl['A0'])} · P0 = {pct(bl['P0'])} · Q0 = {pct(bl['Q0'])}\n"
            f"q0 = {pct(bl['q0'], 2)} · u0 = {fx(bl['u0'])} h/mes\n"
            + (f"Máquina crítica: {bl['maquinas'].iloc[0]['maquina_id']} ({bl['maquinas'].iloc[0]['h_mes']:.1f} h/mes)\n" if len(bl['maquinas']) else "")
            + (f"Defecto dominante: {bl['pareto'].iloc[0]['clase']} ({pct(bl['pareto'].iloc[0]['pct'], 0)})" if len(bl['pareto']) else ""))
        with st.expander("Copiar resumen de la línea base"):
            st.code(resumen, language=None)
        out = bl["daily"].copy()
        st.download_button("Descargar línea base por día (CSV)", out.to_csv(index=False).encode("utf-8-sig"),
                           "linea_base_por_dia.csv", "text/csv")


# ==========================================================================
# PASO 3 - CONFIGURACIÓN
# ==========================================================================

with tab3:
    if not hay_datos:
        st.error("Primero se necesita una línea base válida (paso 2).")
    else:
        st.subheader("Presupuesto y restricciones")
        cols = st.columns(5)
        for i, (k, etiqueta, ayuda) in enumerate(motor.CFG_PRINCIPAL):
            entero = k in ("Nest", "Nmaq")
            ss.C[k] = cols[i].number_input(etiqueta, min_value=0 if entero else 0.0,
                                           value=int(ss.C[k]) if entero else float(ss.C[k]),
                                           step=1 if entero else 100.0 if k in ("Btot",) else 1.0,
                                           help=ayuda or None, key=f"cf_{k}")
        with st.expander("Supuestos y mínimos exigidos (no cambiar salvo indicación del responsable del proyecto)"):
            cols = st.columns(3)
            for i, (k, etiqueta, ayuda) in enumerate(motor.CFG_AVANZADO):
                entero = k == "nInsp"
                ss.C[k] = cols[i % 3].number_input(etiqueta, min_value=0 if entero else 0.0,
                                                   value=int(ss.C[k]) if entero else float(ss.C[k]),
                                                   step=1 if entero else 0.01 if ss.C[k] <= 1 else 10.0,
                                                   help=ayuda or None, key=f"cfa_{k}")

        with st.expander("Catálogo de cámaras y sensores (actualizar con cotizaciones reales)"):
            st.markdown("**Catálogo de cámaras** · costo por estación instalada")
            dv = st.data_editor(pd.DataFrame(ss.cat_v), num_rows="dynamic", width="stretch", key="ed_v",
                                column_config={
                                    "nombre": "Opción", "costo": st.column_config.NumberColumn("Costo (S/)", min_value=0),
                                    "opex": st.column_config.NumberColumn("Gasto mensual (S/)", min_value=0),
                                    "recall": st.column_config.NumberColumn("Exhaustividad", min_value=0.0, max_value=1.0, format="%.2f"),
                                    "prec": st.column_config.NumberColumn("Precisión", min_value=0.0, max_value=1.0, format="%.2f"),
                                    "lat": st.column_config.NumberColumn("s por imagen", min_value=0.0, format="%.2f")})
            ss.cat_v = [r for r in dv.dropna(subset=["nombre", "costo", "recall", "prec", "lat"]).fillna({"opex": 0}).to_dict("records")]

            st.markdown("**Catálogo de sensores** · nodo base + sensores por máquina")
            cs = ss.cat_s
            cols = st.columns(6)
            cs["nodo"] = cols[0].number_input("Nodo base (S/)", 0.0, value=float(cs["nodo"]), key="cs_nodo")
            cs["opexNodo"] = cols[1].number_input("Gasto mensual por nodo (S/)", 0.0, value=float(cs["opexNodo"]), key="cs_opex")
            for i, k in enumerate(["vib", "temp", "corr", "ciclos"]):
                cs["unit"][k] = cols[i + 2].number_input(f"{motor.SENSOR_NOMBRE[k]} (S/)", 0.0, value=float(cs["unit"][k]), key=f"cs_{k}")
            ds = pd.DataFrame([{"Sensores": motor.nombre_subset(s), "Exhaustividad": s["recall"],
                                "Falsas alarmas/día": s["fa"]} for s in cs["subsets"]])
            ds = st.data_editor(ds, disabled=["Sensores"], hide_index=True, width="stretch", key="ed_s",
                                column_config={"Exhaustividad": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, format="%.2f"),
                                               "Falsas alarmas/día": st.column_config.NumberColumn(min_value=0.0, format="%.2f")})
            for s, (_, r) in zip(cs["subsets"], ds.iterrows()):
                s["recall"], s["fa"] = float(r["Exhaustividad"]), float(r["Falsas alarmas/día"])
            if st.button("Restaurar catálogo de ejemplo"):
                ss.cat_v = [dict(x) for x in motor.CAT_CAMARAS_DEF]
                ss.cat_s = copy.deepcopy(motor.CAT_SENSORES_DEF)
                for k in ["ed_v", "ed_s", "cs_nodo", "cs_opex"] + [f"cs_{k}" for k in motor.SENSOR_NOMBRE]:
                    ss.pop(k, None)
                st.rerun()

        cf = motor.configurar(bl, P, ss.C, ss.cat_v, ss.cat_s)
        st.info(f"**Tiempo disponible por imagen: {fx(cf['lat_max'])} s.** Con un takt de {fx(cf['TK'])} s por prenda y "
                f"{int(ss.C['nInsp'])} inspecciones por prenda, casi cualquier equipo alcanza: la elección depende del costo.")

        st.subheader("Configuración recomendada")
        b = cf["best"]
        if b is None:
            peor = max(cf["razones"], key=cf["razones"].get)
            nombre = {"capex": "el presupuesto total", "opex": "el gasto mensual máximo", "payback": "la recuperación máxima"}[peor]
            st.error(f"Ninguna combinación cumple todas las restricciones. La que más descarta opciones es **{nombre}**. "
                     "Suba ese valor o revise el catálogo.")
            reco_txt = "Sin configuración factible."
        else:
            partes = []
            if b["v"]:
                partes.append(f"{b['kv']} cámara(s) «{b['v']['nombre']}»")
            if b["s"]:
                partes.append(f"{b['km']} nodo(s) de sensores «{b['s']['nombre']}» en las máquinas críticas")
            reco_txt = "Instalar " + " y ".join(partes) + "."
            st.markdown(f"<div class='reco'><b>{reco_txt}</b></div>", unsafe_allow_html=True)
            m = st.columns(6)
            m[0].metric("OEE esperado", pct(b["OEE"]), f"{b['dOEE'] * 100:+.1f} pts")
            m[1].metric("Inversión", soles(b["capex"]))
            m[2].metric("Gasto mensual", soles(b["opex"]))
            m[3].metric("Ahorro mensual", soles(b["ahorro"]),
                        help=f"Calidad {soles(b['ahorro_q'])} + paros {soles(b['ahorro_a'])}")
            m[4].metric("Recuperación", f"{fx(b['payback'])} meses")
            m[5].metric("Eficiencia ε", f"{fx(b['eps'], 2)} pts/S/1,000")
            st.caption(f"Se evaluaron {cf['n']} combinaciones; {cf['n_fact']} cumplen el presupuesto, el gasto mensual y la recuperación.")
            if b["payback"] < 1:
                st.warning("La recuperación sale en menos de un mes porque domina el ahorro por paros. Para el escenario "
                           "conservador, ponga la demanda no atendida en 0 (paso 1).")

        st.markdown("#### Por dónde empezar si el dinero llega por partes")
        if len(cf["pasos"]):
            pv = cf["pasos"].copy()
            pv = pd.DataFrame({"Compra": pv["compra"], "Tipo": pv["tipo"], "Equipo": pv["equipo"],
                               "Sube (pts)": pv["ganancia_pts"].round(2), "Costo": pv["costo"].map(soles),
                               "Inversión acumulada": pv["inversion_acum"].map(soles),
                               "OEE alcanzado": pv["oee_alcanzado"].map(pct)})
            st.dataframe(pv, hide_index=True, width="stretch")
        else:
            st.write("No hay compras que mejoren el OEE dentro del presupuesto.")

        st.markdown("#### Costo frente a desempeño")

        def grafico_frontera(items, umbral, sel_nombre, titulo):
            df = pd.DataFrame([{"Opción": f"{i + 1}. {it['nombre']}", "Costo": it["costo"], "Exhaustividad": it["recall"],
                                "Estado": ("Recomendada" if it["nombre"] == sel_nombre else
                                           "En la frontera" if it["ok"] and it["eff"] else
                                           "Dominada" if it["ok"] else "No cumple"),
                                "Motivo": "; ".join(it["why"])} for i, it in enumerate(items)])
            puntos = alt.Chart(df).mark_point(size=140, filled=True).encode(
                x=alt.X("Costo:Q", title="Costo por unidad instalada (S/)"),
                y=alt.Y("Exhaustividad:Q", scale=alt.Scale(zero=False)),
                color=alt.Color("Estado:N", scale=alt.Scale(domain=["Recomendada", "En la frontera", "Dominada", "No cumple"],
                                                            range=["#B8812F", "#3B4F8C", "#9AA3C7", "#B0B4BE"])),
                shape=alt.Shape("Estado:N", scale=alt.Scale(domain=["Recomendada", "En la frontera", "Dominada", "No cumple"],
                                                            range=["diamond", "circle", "circle", "cross"])),
                tooltip=["Opción", "Costo", "Exhaustividad", "Estado", "Motivo"])
            regla = alt.Chart(pd.DataFrame({"y": [umbral]})).mark_rule(strokeDash=[4, 4], color="#A93232").encode(y="y:Q")
            return (puntos + regla).properties(height=280, title=titulo)

        g1, g2 = st.columns(2)
        g1.altair_chart(grafico_frontera(cf["vF"], ss.C["rMinV"], b["v"]["nombre"] if b and b["v"] else None, "Cámaras"),
                        width="stretch")
        g2.altair_chart(grafico_frontera(cf["sF"], ss.C["rMinM"], b["s"]["nombre"] if b and b["s"] else None, "Nodos de sensores"),
                        width="stretch")
        st.caption("La línea roja es el mínimo de exhaustividad exigido. Pase el cursor sobre un punto para ver el motivo si no cumple.")

        with st.expander("Copiar recomendación"):
            txt = reco_txt
            if b:
                txt += (f"\nOEE esperado {pct(b['OEE'])} ({b['dOEE'] * 100:+.1f} pts sobre {pct(bl['OEE0'])}). "
                        f"Inversión {soles(b['capex'])}, gasto mensual {soles(b['opex'])}, ahorro mensual {soles(b['ahorro'])}, "
                        f"recuperación {fx(b['payback'])} meses.")
            st.code(txt, language=None)


# ==========================================================================
# PASO 4 - ACCIONES
# ==========================================================================

def leer_registro():
    try:
        with open(REG_ACCIONES, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def guardar_registro(reg):
    with open(REG_ACCIONES, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=1)


with tab4:
    if not hay_datos:
        st.error("Primero se necesita una línea base válida (paso 2).")
    else:
        evaluadas = motor.evaluar_reglas(bl, ss.reglas)
        fin_periodo = fecha_txt(bl["periodo"][1])
        reg = leer_registro()
        ahora = datetime.now().strftime("%Y-%m-%d %H:%M")
        cambios = False
        for r in evaluadas:
            if r["disparada"]:
                clave = f"{r['id']}|{fin_periodo}|{etiqueta_fuente}"
                if clave not in reg:
                    reg[clave] = {"regla": r["id"], "accion": r["texto"], "responsable": r["resp"], "plazo": r["plazo"],
                                  "severidad": r["sev"], "periodo": fin_periodo, "disparada": ahora, "atendida": None}
                    cambios = True
        if cambios:
            guardar_registro(reg)

        disp = [r for r in evaluadas if r["disparada"]]
        disp.sort(key=lambda r: 0 if r["sev"] == "Crítica" else 1)
        claves_act = [f"{r['id']}|{fin_periodo}|{etiqueta_fuente}" for r in disp]
        propias = {k: v for k, v in reg.items() if k.endswith("|" + etiqueta_fuente)}
        total = len(propias)
        atendidas = [v for v in propias.values() if v["atendida"]]
        tiempos = [(datetime.strptime(v["atendida"], "%Y-%m-%d %H:%M") - datetime.strptime(v["disparada"], "%Y-%m-%d %H:%M")).total_seconds() / 60
                   for v in atendidas]

        m = st.columns(4)
        m[0].metric("Acciones disparadas", len(disp))
        m[1].metric("Críticas", sum(r["sev"] == "Crítica" for r in disp))
        m[2].metric("Tasa de acción", pct(len(atendidas) / total, 0) if total else "–",
                    help="Acciones atendidas sobre acciones disparadas (histórico)")
        m[3].metric("Tiempo hasta actuar", f"{fx(np.mean(tiempos), 0)} min" if tiempos else "–")

        st.subheader("Qué hacer ahora")
        if not disp:
            st.success("Todos los indicadores están en rango. No hay acciones pendientes.")
        for r, clave in zip(disp, claves_act):
            ent = reg.get(clave, {})
            with st.container(border=True):
                a, b2 = st.columns([5, 1])
                tag = "🔴 Crítica" if r["sev"] == "Crítica" else "🟠 Aviso"
                a.markdown(f"**{tag} · {r['id']} · {r['ind']}** ({fx(r['valor'])} {r['unidad']} {r['op']} {r['umbral']} {r['unidad']})  \n"
                           f"{r['texto']}  \n"
                           f"<span style='opacity:.7'>Responsable: {r['resp']} · Plazo: {r['plazo']}"
                           + (f" · Atendida el {ent['atendida']}" if ent.get("atendida") else "") + "</span>",
                           unsafe_allow_html=True)
                hecho = b2.checkbox("Marcar atendida", value=bool(ent.get("atendida")), key=f"chk_{clave}")
                if hecho != bool(ent.get("atendida")):
                    reg[clave]["atendida"] = ahora if hecho else None
                    guardar_registro(reg)
                    st.rerun()

        st.subheader("Reglas y umbrales")
        tabla = pd.DataFrame([{"Regla": r["id"], "Indicador": r["ind"], "Condición": r["op"], "Umbral": r["umbral"],
                               "Unidad": r["unidad"], "Valor actual": round(float(r["valor"]), 2),
                               "Estado": "Disparada" if r["disparada"] else "En rango",
                               "Responsable": r["resp"], "Plazo": r["plazo"]} for r in evaluadas])
        ed = st.data_editor(tabla, hide_index=True, width="stretch", key="ed_reglas",
                            disabled=[c for c in tabla.columns if c not in ("Umbral", "Responsable", "Plazo")])
        for r, (_, f) in zip(ss.reglas, ed.iterrows()):
            r["umbral"], r["resp"], r["plazo"] = float(f["Umbral"]), f["Responsable"], f["Plazo"]
        if st.button("Restaurar umbrales"):
            ss.reglas = [dict(x) for x in motor.REGLAS_DEF]
            ss.pop("ed_reglas", None)
            st.rerun()

        if propias:
            hist = pd.DataFrame(propias.values())[["regla", "severidad", "accion", "responsable", "plazo", "periodo", "disparada", "atendida"]]
            with st.expander("Registro de acciones (se guarda en salidas/registro_acciones.json)"):
                st.dataframe(hist, hide_index=True, width="stretch")
                st.download_button("Descargar registro de acciones (CSV)", hist.to_csv(index=False).encode("utf-8-sig"),
                                   "registro_acciones.csv", "text/csv")


# ==========================================================================
# PASO 5 - CARGA DIARIA DEL PROTOTIPO (IoT + cámara, por lotes)
# ==========================================================================

with tab5:
    st.subheader("Carga diaria del prototipo")
    st.caption("Al cierre del día, cargue lo que exportaron los sensores y la cámara. El artefacto detecta los paros, "
               "registra los defectos y genera el reporte diario. No trabaja en tiempo real: es una carga por lotes.")

    with st.expander("Parámetros de la interfaz IoT"):
        ic = ss.iot_cfg
        c = st.columns(3)
        ic["turno1"] = c[0].text_input("Horario turno 1", ic["turno1"])
        ic["turno2"] = c[1].text_input("Horario turno 2", ic["turno2"])
        ic["pausas"] = c[2].text_input("Pausas planificadas", ic["pausas"], help="Rangos separados por coma")
        c = st.columns(3)
        ic["umbral_corriente"] = c[0].number_input("Corriente bajo la cual la máquina está detenida (A)", 0.0, value=float(ic["umbral_corriente"]), step=0.1)
        ic["min_paro"] = c[1].number_input("Duración mínima de un paro (min)", 0.0, value=float(ic["min_paro"]), step=1.0)
        ic["max_hueco"] = c[2].number_input("Hueco máximo sin lecturas (min)", 0.0, value=float(ic["max_hueco"]), step=1.0)
        c = st.columns(3)
        ic["alerta_vibracion"] = c[0].number_input("Vibración de alerta (mm/s RMS)", 0.0, value=float(ic["alerta_vibracion"]), step=0.1)
        ic["pct_alerta"] = c[1].number_input("Fracción de lecturas sobre el umbral para alertar", 0.0, 1.0, value=float(ic["pct_alerta"]), step=0.05)
        ic["confianza_min"] = c[2].number_input("Confianza mínima de la cámara", 0.0, 1.0, value=float(ic["confianza_min"]), step=0.05)

    c1, c2 = st.columns(2)
    f_lect = c1.file_uploader("Lecturas de sensores (CSV)", type=["csv"], key="up_lect",
                              help="Columnas: " + ", ".join(ib.COL_LECTURAS) + " (opcionales: estacion, temperatura_c)")
    f_vis = c2.file_uploader("Detecciones de la cámara (CSV)", type=["csv"], key="up_vis",
                             help="Columnas: " + ", ".join(ib.COL_VISION) + " (opcionales: estacion_origen, accion)")
    b1, b2 = st.columns(2)
    procesar = b1.button("Procesar archivos del día", type="primary", disabled=not (f_lect or f_vis))
    demo_dia = b2.button("Probar con un día simulado del prototipo")

    if procesar or demo_dia:
        try:
            if demo_dia:
                lect, vis, prod_dia = ib.generar_dia_demo()
            else:
                lect = motor.leer_csv(f_lect)[0] if f_lect else None
                vis = motor.leer_csv(f_vis)[0] if f_vis else None
                prod_dia = None
            res = {"avisos": []}
            if lect is not None:
                res["paros"], res["alertas"], av = ib.procesar_lecturas(lect, ss.iot_cfg)
                res["avisos"] += av
            if vis is not None:
                res["defe"], av = ib.procesar_vision(vis, ss.iot_cfg)
                res["avisos"] += av
            fechas = set()
            for k in ("paros", "defe"):
                if k in res and len(res[k]):
                    fechas |= set(res[k]["fecha"])
            if lect is not None:
                fechas |= set(pd.to_datetime(lect["fecha_hora"], format="mixed", dayfirst=True, errors="coerce").dt.strftime("%d/%m/%Y").dropna())
            res["fecha"] = sorted(fechas)[-1] if fechas else datetime.now().strftime("%d/%m/%Y")
            if prod_dia is None:
                prod_dia = pd.DataFrame([dict(fecha=res["fecha"], turno=t, familia="F1", unidades_producidas=0,
                                              unidades_defectuosas=0, unidades_reprocesadas=0, operarios_en_planta=0) for t in (1, 2)])
            res["prod"] = prod_dia
            ss.iot = res
            ss.ultimo_reporte = None
        except ValueError as e:
            st.error(str(e))
            ss.iot = None

    res = ss.iot
    if res:
        st.markdown(f"#### Resultado del {res['fecha']}")
        for a in res["avisos"]:
            st.warning(a)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Paros detectados por los sensores**")
            if "paros" in res and len(res["paros"]):
                st.dataframe(res["paros"][["turno", "maquina_id", "hora_inicio", "hora_fin"]], hide_index=True, width="stretch")
            else:
                st.write("Ningún paro detectado.")
        with c2:
            st.markdown("**Defectos detectados por la cámara**")
            if "defe" in res and len(res["defe"]):
                st.dataframe(res["defe"].groupby("clase_defecto").size().rename("prendas").reset_index(),
                             hide_index=True, width="stretch")
            else:
                st.write("Ningún defecto registrado.")
        if "alertas" in res and len(res["alertas"]):
            st.markdown("**Alertas de mantenimiento predictivo (vibración)**")
            st.dataframe(res["alertas"], hide_index=True, width="stretch")

        st.markdown("**Producción del día** (la anota el supervisor; complete o corrija antes de guardar)")
        prod_ed = st.data_editor(res["prod"], hide_index=True, width="stretch", key=f"prod_{res['fecha']}")

        if st.button("Agregar a los registros y generar el reporte diario", type="primary"):
            if not all(os.path.exists(os.path.join(DATOS, f)) for f in motor.ARCHIVOS.values()):
                st.error("Primero cree la carpeta datos/ con sus registros (paso 1).")
            elif (pd.to_numeric(prod_ed["unidades_producidas"], errors="coerce").fillna(0) <= 0).any():
                st.error("Complete las unidades producidas de cada turno.")
            else:
                n_p = ib.agregar_a_registros("prod", prod_ed)
                n_pa = ib.agregar_a_registros("paros", res.get("paros"))
                n_d = ib.agregar_a_registros("defe", res.get("defe"))
                raw_c = {k: motor.leer_csv(os.path.join(DATOS, f))[0] for k, f in motor.ARCHIVOS.items()}
                raw_c["sep"] = ","
                rep = ib.reporte_diario(raw_c, P, res["fecha"], bl)
                ruta, texto = ib.guardar_reporte(rep, res.get("alertas"))
                ss.ultimo_reporte = {
                    "msg": f"Se agregaron {n_p[0]} filas de producción, {n_pa[0]} paros y {n_d[0]} defectos "
                           f"(omitidos por estar repetidos: {n_p[1] + n_pa[1] + n_d[1]}).",
                    "rep": {k: rep[k] for k in ("OEE", "A", "P", "Q", "dif_pts")}, "texto": texto, "ruta": ruta}
                ss.fuente = "carpeta"
                ss.iot = None
                st.rerun()   # recalcula los pasos 2 a 4 con el día nuevo

    if ss.get("ultimo_reporte"):
        ur = ss.ultimo_reporte
        st.success(ur["msg"])
        rp = ur["rep"]
        m = st.columns(4)
        m[0].metric("OEE del día", pct(rp["OEE"]), f"{rp['dif_pts']:+.1f} pts vs línea base")
        m[1].metric("Disponibilidad", pct(rp["A"]))
        m[2].metric("Rendimiento", pct(rp["P"]))
        m[3].metric("Calidad", pct(rp["Q"]))
        st.code(ur["texto"], language=None)
        st.caption(f"Reporte guardado en {ur['ruta']}. Los pasos 2, 3 y 4 ya incluyen este día.")

    with st.expander("¿Cómo llegan los datos del prototipo a este programa?"):
        st.markdown(
            "1. Los nodos de sensores (corriente y vibración) guardan una lectura por minuto en la tarjeta o en la "
            "laptop de planta. La cámara guarda una fila por cada prenda con defecto.\n"
            "2. Al cierre del día, el software del prototipo exporta dos archivos CSV con las columnas indicadas arriba.\n"
            "3. Se cargan aquí (o se programa `python iot_batch.py --lecturas ... --vision ...` en el Programador de "
            "tareas de Windows) y el artefacto los convierte en filas de **paros.csv** y **defectos.csv**.\n"
            "4. Si se recarga el mismo archivo, las filas repetidas se omiten.\n\n"
            "A futuro, la recomendación es que esta carga sea automática mediante una interfaz directa con el software IoT.")
