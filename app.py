"""
=====================================================================================
 SIMULADOR Y OPTIMIZADOR DE TIEMPOS - MÉTODO CPM / PERT
 Proyecto: Mantenimiento de Turbinas a Gas
 Empresa: Turbinas E&M
 Autores base del diagrama: Camila Giraldo, Loren Delgado, Juan Grisales
=====================================================================================

 DESCRIPCIÓN:
 Aplicación Streamlit de un solo archivo que reconstruye la red PERT/CPM extraída
 del diagrama original (33 nodos, Tiempo total = 237.8 horas, Ruta crítica =
 A-B-C-F-G-J-K-Q-O-T-Y-AC-AB-AA-AD-AE) y permite:
   1. Visualizar el estado actual del proyecto (red + Gantt + métricas).
   2. Simular escenarios "What-If" alterando duraciones de tareas en tiempo real.
   3. Ejecutar una simulación de Montecarlo (distribución triangular O-M-P) para
      estimar el riesgo de atraso del proyecto.

 EJECUCIÓN:
   1. Instalar dependencias:
        pip install streamlit networkx pandas numpy plotly
   2. Ejecutar:
        streamlit run app.py
=====================================================================================
"""

import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict

# =====================================================================================
# 1. CONFIGURACIÓN DE PÁGINA
# =====================================================================================
st.set_page_config(
    page_title="CPM Turbinas E&M",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================================
# 2. DATOS BASE EXTRAÍDOS DEL DIAGRAMA PERT/CPM (IMAGEN ORIGINAL)
#    Duración (tiempo probable "M") de cada actividad, en horas.
#    Red del tipo "Actividad en el Nodo" (AON).
# =====================================================================================
BASE_DURATIONS = {
    "INICIO": 0.0,
    "A": 6.0, "B": 5.0, "C": 5.0, "D": 3.0, "E": 8.0, "F": 9.6, "G": 7.2,
    "H": 12.0, "I": 24.0, "J": 20.0, "K": 28.0, "L": 24.0, "M": 24.0,
    "N": 24.0, "O": 36.0, "P": 48.0, "Q": 6.0, "R": 48.0, "S": 12.9,
    "T": 44.5, "U": 3.0, "V": 7.3, "W": 9.1, "X": 29.0, "Y": 19.5,
    "Z": 4.0, "AA": 9.0, "AB": 14.0, "AC": 18.0, "AD": 4.0, "AE": 6.0,
    "FIN": 0.0,
}

# Dependencias (predecesor -> sucesor) reconstruidas fielmente del diagrama
EDGES = [
    ("INICIO", "A"), ("A", "B"), ("B", "C"),
    ("C", "E"), ("C", "F"),
    ("F", "G"),
    ("E", "L"), ("E", "M"),
    ("G", "D"), ("G", "J"), ("G", "H"),
    ("L", "H"),
    ("D", "N"),
    ("M", "R"),
    ("H", "I"),
    ("I", "R"), ("I", "S"), ("I", "O"),
    ("J", "K"),
    ("K", "Q"), ("K", "W"),
    ("Q", "O"), ("Q", "P"),
    ("O", "T"),
    ("W", "X"), ("N", "X"),
    ("X", "Z"),
    ("P", "Y"), ("Z", "Y"), ("T", "Y"),
    ("S", "AB"), ("V", "AB"),
    ("R", "U"), ("U", "V"),
    ("Y", "AC"),
    ("AC", "AB"),
    ("AB", "AA"),
    ("AA", "AD"),
    ("AD", "AE"),
    ("AE", "FIN"),
]

ORIGINAL_TOTAL = 237.8
ORIGINAL_CRITICAL_PATH = ["A", "B", "C", "F", "G", "J", "K", "Q", "O", "T",
                           "Y", "AC", "AB", "AA", "AD", "AE"]
TASKS = [n for n in BASE_DURATIONS if n not in ("INICIO", "FIN")]


# =====================================================================================
# 3. CÁLCULO DE TIEMPOS OPTIMISTA (O) Y PESIMISTA (P)
#    O = M - 15%   |   P = M + 25%
# =====================================================================================
def compute_ope(durations):
    o, p = {}, {}
    for k, v in durations.items():
        if v == 0:
            o[k], p[k] = 0.0, 0.0
        else:
            o[k] = round(v * 0.85, 4)
            p[k] = round(v * 1.25, 4)
    return o, p


BASE_O, BASE_P = compute_ope(BASE_DURATIONS)


# =====================================================================================
# 4. ESTRUCTURA DE RED (TOPOLOGÍA FIJA) - SE CALCULA UNA SOLA VEZ
# =====================================================================================
@st.cache_data
def get_topology():
    G = nx.DiGraph()
    G.add_nodes_from(BASE_DURATIONS.keys())
    G.add_edges_from(EDGES)
    topo = list(nx.topological_sort(G))
    preds = {n: list(G.predecessors(n)) for n in G.nodes}
    succs = {n: list(G.successors(n)) for n in G.nodes}
    return topo, preds, succs


TOPO_ORDER, PRED, SUCC = get_topology()


def build_weighted_graph(durations):
    """Construye el grafo dirigido con el peso de cada arco = duración del nodo origen
    (truco estándar para usar algoritmos de camino más largo de NetworkX en redes AON)."""
    G = nx.DiGraph()
    for n, d in durations.items():
        G.add_node(n, duration=d)
    for u, v in EDGES:
        G.add_edge(u, v, weight=durations[u])
    return G


def compute_cpm(durations):
    """Pasada hacia adelante y hacia atrás clásica de CPM."""
    ES, EF = {}, {}
    for n in TOPO_ORDER:
        preds = PRED[n]
        ES[n] = max([EF[p] for p in preds], default=0.0)
        EF[n] = ES[n] + durations[n]

    total = EF["FIN"]

    LF, LS = {}, {}
    for n in reversed(TOPO_ORDER):
        succs = SUCC[n]
        LF[n] = min([LS[s] for s in succs], default=total)
        LS[n] = LF[n] - durations[n]

    slack = {n: round(LS[n] - ES[n], 6) for n in TOPO_ORDER}

    G = build_weighted_graph(durations)
    critical_path = nx.dag_longest_path(G, weight="weight")
    # quitamos INICIO y FIN para mostrar solo las actividades reales
    critical_path_display = [n for n in critical_path if n not in ("INICIO", "FIN")]

    return ES, EF, LS, LF, slack, total, critical_path_display


# =====================================================================================
# 5. SIDEBAR - SIMULADOR "WHAT-IF" (INYECCIÓN DE RETRASOS)
# =====================================================================================
st.sidebar.title("🛠️ Simulador What-If")
st.sidebar.caption("Modifica la duración de una o varias tareas y observa el impacto "
                    "inmediato sobre el proyecto.")

if "ms_tasks" not in st.session_state:
    st.session_state["ms_tasks"] = []

if st.sidebar.button("🔄 Restablecer valores originales", use_container_width=True):
    st.session_state["ms_tasks"] = []
    for t in TASKS:
        key = f"slider_{t}"
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

selected_tasks = st.sidebar.multiselect(
    "Selecciona las tareas a modificar:",
    options=sorted(TASKS),
    key="ms_tasks",
)

overrides = {}
if selected_tasks:
    st.sidebar.markdown("---")
    for t in selected_tasks:
        max_val = round(BASE_P[t] * 1.6, 1) if BASE_P[t] > 0 else 10.0
        overrides[t] = st.sidebar.slider(
            f"Duración de **{t}** (h)  —  original: {BASE_DURATIONS[t]} h",
            min_value=0.0,
            max_value=max(max_val, 1.0),
            value=float(BASE_DURATIONS[t]),
            step=0.1,
            key=f"slider_{t}",
        )

current_durations = {**BASE_DURATIONS, **overrides}

st.sidebar.markdown("---")
st.sidebar.caption(
    "ℹ️ Los tiempos Optimista (O) y Pesimista (P) usados en la Simulación de "
    "Montecarlo se calculan siempre sobre los datos originales del diagrama "
    "(O = M−15%, P = M+25%), independientemente de los ajustes What-If."
)

# =====================================================================================
# 6. CÁLCULO CPM CON LOS DATOS ACTUALES (ORIGINALES + OVERRIDES)
# =====================================================================================
ES, EF, LS, LF, SLACK, TOTAL, CRITICAL_PATH = compute_cpm(current_durations)
project_changed = abs(TOTAL - ORIGINAL_TOTAL) > 1e-6
path_changed = CRITICAL_PATH != ORIGINAL_CRITICAL_PATH

# =====================================================================================
# 7. ENCABEZADO
# =====================================================================================
st.title("⚙️ Simulador y Optimizador CPM — Mantenimiento de Turbinas a Gas")
st.caption("Empresa Turbinas E&M · Investigación de Operaciones")

tab_dash, tab_net, tab_mc = st.tabs(
    ["📊 Dashboard Principal", "🕸️ Red & Diagrama de Gantt", "🎲 Simulación de Montecarlo"]
)

# =====================================================================================
# 8. DASHBOARD PRINCIPAL
# =====================================================================================
with tab_dash:
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "⏱️ Tiempo total del proyecto",
        f"{TOTAL:,.1f} h",
        delta=f"{TOTAL - ORIGINAL_TOTAL:+.1f} h" if project_changed else "Sin cambios",
        delta_color="inverse" if project_changed else "off",
    )
    col2.metric("📏 Tiempo original (diagrama)", f"{ORIGINAL_TOTAL:,.1f} h")
    col3.metric(
        "🎯 Tareas en ruta crítica",
        f"{len(CRITICAL_PATH)}",
        delta="Ruta cambió" if path_changed else "Ruta sin cambios",
        delta_color="inverse" if path_changed else "off",
    )

    st.markdown("### 🔴 Ruta crítica actual")
    st.success(" → ".join(CRITICAL_PATH))

    if path_changed:
        st.warning(
            "⚠️ La ruta crítica **cambió** respecto a la original debido a los ajustes "
            "What-If aplicados en la barra lateral.\n\n"
            f"**Original:** {' → '.join(ORIGINAL_CRITICAL_PATH)}\n\n"
            f"**Actual:** {' → '.join(CRITICAL_PATH)}"
        )
    else:
        st.info("✅ La ruta crítica se mantiene igual a la original del diagrama.")

    if overrides:
        st.markdown("### 🧪 Resumen de ajustes What-If aplicados")
        df_over = pd.DataFrame(
            [
                {
                    "Tarea": t,
                    "Duración original (h)": BASE_DURATIONS[t],
                    "Duración simulada (h)": v,
                    "Δ (h)": round(v - BASE_DURATIONS[t], 2),
                    "¿Es crítica ahora?": "Sí" if t in CRITICAL_PATH else "No",
                }
                for t, v in overrides.items()
            ]
        )
        st.dataframe(df_over, use_container_width=True, hide_index=True)

    st.markdown("### 📋 Tabla completa de tiempos CPM (ES, EF, LS, LF, Holgura)")
    rows = []
    for n in TOPO_ORDER:
        rows.append(
            {
                "Tarea": n,
                "Duración (h)": round(current_durations[n], 2),
                "ES (Inicio temprano)": round(ES[n], 2),
                "EF (Fin temprano)": round(EF[n], 2),
                "LS (Inicio tardío)": round(LS[n], 2),
                "LF (Fin tardío)": round(LF[n], 2),
                "Holgura": round(SLACK[n], 2),
                "Crítica": "🔴 Sí" if n in CRITICAL_PATH or n in ("INICIO", "FIN") and n in CRITICAL_PATH else ("🔴 Sí" if n in CRITICAL_PATH else "⚪ No"),
            }
        )
    df_cpm = pd.DataFrame(rows)
    st.dataframe(df_cpm, use_container_width=True, hide_index=True, height=420)

# =====================================================================================
# 9. RED + GANTT
# =====================================================================================
with tab_net:
    st.subheader("🕸️ Diagrama de red del proyecto")
    st.caption("Los nodos y arcos en rojo representan la ruta crítica actual. "
               "Pasa el cursor sobre cada nodo para ver sus tiempos ES/EF/LS/LF/Holgura.")

    # --- Posicionamiento de nodos: eje X = Inicio Temprano (ES), eje Y = carril ---
    buckets = defaultdict(list)
    for n in TOPO_ORDER:
        buckets[round(ES[n], 1)].append(n)

    pos = {}
    for x_val, nodes_in_bucket in buckets.items():
        nodes_in_bucket = sorted(nodes_in_bucket)
        offset = len(nodes_in_bucket) / 2.0
        for i, n in enumerate(nodes_in_bucket):
            pos[n] = (x_val, i - offset)

    crit_set = set(CRITICAL_PATH) | {"INICIO", "FIN"} if CRITICAL_PATH else set()
    # INICIO y FIN siempre son parte conceptual de la ruta crítica (duración 0)
    crit_set = set(CRITICAL_PATH) | {"INICIO", "FIN"}

    edge_x, edge_y = [], []
    edge_x_c, edge_y_c = [], []
    annotations = []
    for u, v in EDGES:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        is_critical_edge = (u in crit_set and v in crit_set)
        if is_critical_edge:
            edge_x_c += [x0, x1, None]
            edge_y_c += [y0, y1, None]
        else:
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        annotations.append(
            dict(
                ax=x0, ay=y0, x=x1, y=y1,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1,
                arrowwidth=2 if is_critical_edge else 1,
                arrowcolor="#d62728" if is_critical_edge else "#b0b0b0",
                opacity=0.85,
            )
        )

    fig_net = go.Figure()
    fig_net.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                  line=dict(color="#c9c9c9", width=1),
                                  hoverinfo="none", showlegend=False))
    fig_net.add_trace(go.Scatter(x=edge_x_c, y=edge_y_c, mode="lines",
                                  line=dict(color="#d62728", width=1),
                                  hoverinfo="none", showlegend=False))

    node_x, node_y, node_text, node_color, node_hover, node_size = [], [], [], [], [], []
    for n in TOPO_ORDER:
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_text.append(n)
        is_crit = n in crit_set
        node_color.append("#d62728" if is_crit else "#1f77b4")
        node_size.append(34 if is_crit else 26)
        node_hover.append(
            f"<b>{n}</b><br>Duración: {current_durations[n]:.1f} h"
            f"<br>ES: {ES[n]:.1f} | EF: {EF[n]:.1f}"
            f"<br>LS: {LS[n]:.1f} | LF: {LF[n]:.1f}"
            f"<br>Holgura: {SLACK[n]:.1f}"
            f"<br>{'🔴 Crítica' if is_crit else '⚪ No crítica'}"
        )

    fig_net.add_trace(
        go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            marker=dict(size=node_size, color=node_color, line=dict(width=1.5, color="white")),
            text=node_text, textposition="middle center",
            textfont=dict(color="white", size=10, family="Arial Black"),
            hovertext=node_hover, hoverinfo="text", showlegend=False,
        )
    )

    fig_net.update_layout(
        annotations=annotations,
        xaxis=dict(title="Inicio temprano (horas)", showgrid=True, zeroline=False),
        yaxis=dict(visible=False),
        height=650,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_net, use_container_width=True)

    st.markdown("---")
    st.subheader("📅 Diagrama de Gantt (CPM)")
    st.caption("Barras ordenadas por inicio temprano (ES). Rojo = ruta crítica.")

    order = sorted(TASKS, key=lambda n: (ES[n], n))
    fig_gantt = go.Figure()
    for n in order:
        is_crit = n in crit_set
        fig_gantt.add_trace(
            go.Bar(
                x=[current_durations[n]],
                y=[n],
                base=[ES[n]],
                orientation="h",
                marker_color="#d62728" if is_crit else "#1f77b4",
                name=n,
                showlegend=False,
                hovertemplate=(
                    f"<b>{n}</b><br>ES: {ES[n]:.1f} h<br>EF: {EF[n]:.1f} h"
                    f"<br>Duración: {current_durations[n]:.1f} h"
                    f"<br>Holgura: {SLACK[n]:.1f} h<extra></extra>"
                ),
            )
        )
    fig_gantt.update_layout(
        barmode="stack",
        xaxis_title="Horas desde el inicio del proyecto",
        yaxis=dict(categoryorder="array", categoryarray=list(reversed(order)), title=""),
        height=850,
        margin=dict(l=40, r=20, t=20, b=40),
        bargap=0.25,
    )
    fig_gantt.add_vline(x=TOTAL, line_dash="dash", line_color="#d62728",
                         annotation_text=f"Fin del proyecto: {TOTAL:.1f} h",
                         annotation_position="top right")
    st.plotly_chart(fig_gantt, use_container_width=True)

# =====================================================================================
# 10. SIMULACIÓN DE MONTECARLO
# =====================================================================================
with tab_mc:
    st.subheader("🎲 Análisis de riesgo — Simulación de Montecarlo")
    st.markdown(
        "Cada tarea se modela con una **distribución triangular** definida por sus "
        "tiempos **Optimista (O)**, **Más probable (M)** y **Pesimista (P)**, "
        "calculados a partir de los datos originales del diagrama:\n"
        "- **O = M − 15%**\n"
        "- **P = M + 25%**"
    )

    with st.expander("Ver tabla de tiempos O-M-P por tarea"):
        df_omp = pd.DataFrame(
            [
                {"Tarea": t, "Optimista (O)": BASE_O[t], "Probable (M)": BASE_DURATIONS[t],
                 "Pesimista (P)": BASE_P[t]}
                for t in TASKS
            ]
        )
        st.dataframe(df_omp, use_container_width=True, hide_index=True)

    n_iter = 1000
    run_mc = st.button(f"▶️ Ejecutar Simulación de Riesgo ({n_iter} iteraciones)",
                        type="primary", use_container_width=True)

    if run_mc:
        rng = np.random.default_rng(42)
        samples = {}
        for n in TOPO_ORDER:
            o, m, p = BASE_O[n], BASE_DURATIONS[n], BASE_P[n]
            if m == 0:
                samples[n] = np.zeros(n_iter)
            else:
                samples[n] = rng.triangular(o, m, p, size=n_iter)

        EF_sim = {}
        for n in TOPO_ORDER:
            preds = PRED[n]
            if preds:
                es_n = np.maximum.reduce([EF_sim[pr] for pr in preds])
            else:
                es_n = np.zeros(n_iter)
            EF_sim[n] = es_n + samples[n]

        totals = EF_sim["FIN"]
        st.session_state["mc_totals"] = totals

    if "mc_totals" in st.session_state:
        totals = st.session_state["mc_totals"]

        prob_on_time = float(np.mean(totals <= ORIGINAL_TOTAL) * 100)
        mean_t = float(np.mean(totals))
        std_t = float(np.std(totals))
        p90 = float(np.percentile(totals, 90))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"✅ P(Terminar ≤ {ORIGINAL_TOTAL} h)", f"{prob_on_time:.1f} %")
        c2.metric("Media simulada", f"{mean_t:,.1f} h")
        c3.metric("Desviación estándar", f"{std_t:,.1f} h")
        c4.metric("Percentil 90 (P90)", f"{p90:,.1f} h")

        fig_hist = px.histogram(
            x=totals, nbins=50,
            labels={"x": "Duración total del proyecto (horas)"},
            title=f"Distribución de duración total del proyecto ({n_iter} iteraciones)",
            color_discrete_sequence=["#1f77b4"],
        )
        fig_hist.add_vline(
            x=ORIGINAL_TOTAL, line_dash="dash", line_color="#d62728",
            annotation_text=f"Plazo original: {ORIGINAL_TOTAL} h",
            annotation_position="top right",
        )
        fig_hist.add_vline(
            x=mean_t, line_dash="dot", line_color="#2ca02c",
            annotation_text=f"Media: {mean_t:.1f} h",
            annotation_position="top left",
        )
        fig_hist.update_layout(height=550, yaxis_title="Frecuencia")
        st.plotly_chart(fig_hist, use_container_width=True)

        if prob_on_time >= 70:
            st.success(
                f"🟢 Riesgo bajo: existe un **{prob_on_time:.1f}%** de probabilidad de "
                f"finalizar el mantenimiento en {ORIGINAL_TOTAL} h o menos."
            )
        elif prob_on_time >= 40:
            st.warning(
                f"🟡 Riesgo moderado: solo un **{prob_on_time:.1f}%** de probabilidad de "
                f"cumplir el plazo de {ORIGINAL_TOTAL} h. Se recomienda monitorear las "
                f"tareas críticas."
            )
        else:
            st.error(
                f"🔴 Riesgo alto: apenas un **{prob_on_time:.1f}%** de probabilidad de "
                f"cumplir el plazo de {ORIGINAL_TOTAL} h. Se recomienda revisar la "
                f"planeación del mantenimiento."
            )
    else:
        st.info("Presiona el botón para ejecutar la simulación de Montecarlo.")
```
