# ==========================================
#  ruleta_fiscal_evento2.py
#  Ruleta fiscal: Louvain + Glow + Play/Pause
#  ROTACIÓN SOLO CUANDO CAMBIA EL PAÍS (NO GIRO INFINITO)
#  (Arregla Duplicate callback outputs: 1 solo callback para la animación)
# ==========================================

import math
import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain

from typing import Optional, Dict, List, Tuple, Set

from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
import plotly.graph_objects as go
import plotly.express as px
import os

# ----------------------------
# CONFIG
# ----------------------------
CSV_PATH = "TFG_panel_balanced.csv"

ISO_COL = "country_iso2"
NAME_COL = "country"
YEAR_COL = "year"

DEBT_COL = "debt"
DEFICIT_COL = "deficit"

THR_DEFAULT = 0.40
WINDOW_OPTIONS = [4, 6, 8, 10, 12]

INTERVAL_YEAR_MS = 900

# Animación de giro (solo cuando cambia país)
INTERVAL_TURN_MS = 35
TURN_FRAMES = 30

BG = "#0b1020"
FG = "#e8ecff"
EDGE_COL = "rgba(255,255,255,0.78)"
EDGE_GLOW = "rgba(255,255,255,0.18)"

DIM_NODE = "rgba(232,236,255,0.10)"
DIM_TEXT = "rgba(232,236,255,0.18)"

PANEL_BG = "rgba(255,255,255,0.03)"
PANEL_BORDER = "1px solid rgba(255,255,255,0.08)"


# ----------------------------
# DATA PREP
# ----------------------------
def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[YEAR_COL] = df[YEAR_COL].astype(int)
    df = df.sort_values([ISO_COL, YEAR_COL]).reset_index(drop=True)

    # ΔDeuda
    df["d_debt"] = df.groupby(ISO_COL)[DEBT_COL].diff()

    # Déficit: negativo = déficit; positivo = superávit
    # presión = max(-deficit, 0)
    df["deficit_pressure"] = (-df[DEFICIT_COL]).clip(lower=0)
    df["d_deficit_pressure"] = df.groupby(ISO_COL)["deficit_pressure"].diff()

    return df


def window_slice(df: pd.DataFrame, end_year: int, window: int) -> pd.DataFrame:
    start_year = end_year - window + 1
    return df[(df[YEAR_COL] >= start_year) & (df[YEAR_COL] <= end_year)].copy()


def corr_matrix(sub: pd.DataFrame, metric: str) -> pd.DataFrame:
    piv = sub.pivot(index=YEAR_COL, columns=ISO_COL, values=metric)
    return piv.corr()


def build_graph_from_corr(C: pd.DataFrame, thr: float) -> nx.Graph:
    G = nx.Graph()
    for n in C.columns:
        G.add_node(n)

    cols = list(C.columns)
    for i, a in enumerate(cols):
        for b in cols[i+1:]:
            w = C.loc[a, b]
            if pd.notna(w) and abs(w) >= thr:
                G.add_edge(a, b, weight=float(abs(w)), signed_weight=float(w))
    return G


def louvain_partition(G: nx.Graph) -> Dict[str, int]:
    if G.number_of_edges() == 0:
        return {n: i for i, n in enumerate(G.nodes())}
    return community_louvain.best_partition(G, weight="weight", random_state=42)


# ----------------------------
# EASING
# ----------------------------
def ease_in_out(t: float) -> float:
    # t in [0,1]
    return 0.5 - 0.5 * math.cos(math.pi * t)


# ----------------------------
# WHEEL POSITIONS
# - selected fijo arriba (0,1)
# - el resto rota con rotation_phase
# ----------------------------
def wheel_positions(nodes: List[str], selected: Optional[str], rotation_phase: float) -> Dict[str, Tuple[float, float]]:
    nodes = sorted(nodes)
    n = len(nodes)
    if n == 0:
        return {}

    pos: Dict[str, Tuple[float, float]] = {}

    if selected in nodes and n > 1:
        pos[selected] = (0.0, 1.0)
        others = [x for x in nodes if x != selected]
        m = len(others)
        for i, node in enumerate(others):
            a = rotation_phase + 2 * math.pi * i / m
            pos[node] = (math.cos(a), math.sin(a))
    else:
        for i, node in enumerate(nodes):
            a = rotation_phase + 2 * math.pi * i / n
            pos[node] = (math.cos(a), math.sin(a))

    return pos


# ----------------------------
# FIGURE
# ----------------------------
def make_ruleta_figure(
    nodes: List[str],
    part: Dict[str, int],
    iso_to_name: Dict[str, str],
    selected: Optional[str],
    neighbors: Set[str],
    corr_of_selected: Dict[str, float],
    thr: float,
    title: str,
    rotation_phase: float
) -> go.Figure:

    pos = wheel_positions(nodes, selected, rotation_phase)
    fig = go.Figure()

    # edges solo desde selected a neighbors
    if selected and selected in pos:
        x0, y0 = pos[selected]
        for nb in sorted(neighbors):
            if nb not in pos:
                continue
            x1, y1 = pos[nb]
            w = abs(corr_of_selected.get(nb, 0.0))
            if w < thr:
                continue

            width = 1.4 + 6.2 * min(1.0, w)

            # glow
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode="lines",
                line=dict(width=width + 6, color=EDGE_GLOW),
                hoverinfo="skip",
                showlegend=False
            ))
            # main
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode="lines",
                line=dict(width=width, color=EDGE_COL),
                hoverinfo="skip",
                showlegend=False
            ))

    # colores por comunidad
    comm_ids = sorted(set(part.values()))
    palette = px.colors.qualitative.Set3
    comm_color = {c: palette[i % len(palette)] for i, c in enumerate(comm_ids)}

    xs_dim, ys_dim, text_dim, hover_dim = [], [], [], []
    xs_lit, ys_lit, text_lit, hover_lit, color_lit, size_lit = [], [], [], [], [], []

    xs_halo, ys_halo, halo_size, halo_color = [], [], [], []

    for n in nodes:
        if n not in pos:
            continue
        x, y = pos[n]
        comm = part.get(n, -1)
        cname = iso_to_name.get(n, n)
        hover = f"<b>{cname}</b><br>ISO: {n}<br>Comunidad: {comm}"

        is_dim = (selected is not None) and (n != selected) and (n not in neighbors)

        if is_dim:
            xs_dim.append(x); ys_dim.append(y)
            text_dim.append(n)
            hover_dim.append(hover)
        else:
            xs_lit.append(x); ys_lit.append(y)
            text_lit.append(n)
            hover_lit.append(hover)

            base_col = comm_color.get(comm, "rgba(200,200,200,1)")
            color_lit.append(base_col)

            if n == selected:
                size = 28
            elif n in neighbors:
                size = 18
            else:
                size = 14
            size_lit.append(size)

            if n == selected:
                xs_halo.append(x); ys_halo.append(y)
                halo_size.append(size + 22)
                halo_color.append("rgba(255,255,255,0.18)")
            elif n in neighbors:
                xs_halo.append(x); ys_halo.append(y)
                halo_size.append(size + 16)
                halo_color.append("rgba(255,255,255,0.10)")

    if xs_halo:
        fig.add_trace(go.Scatter(
            x=xs_halo, y=ys_halo,
            mode="markers",
            marker=dict(size=halo_size, color=halo_color, line=dict(width=0)),
            hoverinfo="skip",
            showlegend=False
        ))

    fig.add_trace(go.Scatter(
        x=xs_dim, y=ys_dim,
        mode="markers+text",
        text=text_dim,
        textposition="top center",
        textfont=dict(color=DIM_TEXT),
        marker=dict(size=10, color=DIM_NODE),
        hovertext=hover_dim,
        hoverinfo="text",
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=xs_lit, y=ys_lit,
        mode="markers+text",
        text=text_lit,
        textposition="top center",
        textfont=dict(color=FG),
        marker=dict(
            size=size_lit,
            color=color_lit,
            line=dict(width=1.2, color="rgba(255,255,255,0.65)")
        ),
        hovertext=hover_lit,
        hoverinfo="text",
        showlegend=False,
        customdata=text_lit
    ))

    fig.update_layout(
        title=title,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        xaxis=dict(visible=False, range=[-1.25, 1.25]),
        yaxis=dict(visible=False, range=[-1.25, 1.25]),
        height=820,
        margin=dict(l=30, r=30, t=60, b=30),
    )
    return fig


# ----------------------------
# APP
# ----------------------------
df = load_and_prepare(CSV_PATH)
iso_to_name = df[[ISO_COL, NAME_COL]].drop_duplicates().set_index(ISO_COL)[NAME_COL].to_dict()

min_year = int(df[YEAR_COL].min())
max_year = int(df[YEAR_COL].max())
all_countries = sorted(df[ISO_COL].dropna().unique().tolist())

app = Dash(__name__)
server = app.server
app.title = "Ruleta fiscal (evento2)"

app.layout = html.Div(style={"backgroundColor": BG, "color": FG, "minHeight": "100vh", "padding": "18px"}, children=[

    html.H2("Red fiscal europea — Ruleta interactiva (Louvain)", style={"marginBottom": "8px"}),

    # Stores
    dcc.Store(id="store_selected_country", data=None),
    dcc.Store(id="store_playing_year", data=False),

    # Rotación
    dcc.Store(id="store_rotation_phase", data=0.0),
    dcc.Store(id="store_turn_active", data=False),
    dcc.Store(id="store_turn_step", data=0),
    dcc.Store(id="store_turn_from", data=0.0),
    dcc.Store(id="store_turn_to", data=0.0),
    dcc.Store(id="store_selected_prev", data=None),

    html.Div(style={"display": "grid", "gridTemplateColumns": "1.2fr 0.8fr 0.9fr 0.6fr", "gap": "14px", "alignItems": "center"}, children=[

        html.Div(children=[
            html.Div("Año (fin de ventana)", style={"fontSize": "12px", "opacity": 0.9}),
            dcc.Dropdown(
                id="year",
                options=[{"label": str(y), "value": y} for y in range(min_year, max_year + 1)],
                value=max_year,
                clearable=False
            )
        ]),

        html.Div(children=[
            html.Div("Ventana (años)", style={"fontSize": "12px", "opacity": 0.9}),
            dcc.Dropdown(
                id="window",
                options=[{"label": str(w), "value": w} for w in WINDOW_OPTIONS],
                value=8,
                clearable=False
            )
        ]),

        html.Div(children=[
            html.Div("Métrica", style={"fontSize": "12px", "opacity": 0.9}),
            dcc.RadioItems(
                id="metric",
                options=[
                    {"label": "ΔDeuda (d_debt)", "value": "d_debt"},
                    {"label": "ΔPresión de déficit (d_deficit_pressure)", "value": "d_deficit_pressure"},
                ],
                value="d_debt",
                labelStyle={"display": "block", "marginRight": "10px"},
            ),
        ]),

        html.Div(style={"display": "flex", "justifyContent": "flex-end", "gap": "10px"}, children=[
            html.Button("▶ Play", id="btn_play", n_clicks=0, style={"padding": "6px 10px"}),
            html.Button("⏸ Pause", id="btn_pause", n_clicks=0, style={"padding": "6px 10px"}),
        ]),
    ]),

    html.Div(id="status_line", style={"marginTop": "10px", "opacity": 0.9, "fontSize": "13px"}),

    html.Div(style={"display": "grid", "gridTemplateColumns": "2.2fr 1fr", "gap": "14px", "marginTop": "10px"}, children=[

        dcc.Graph(
            id="wheel",
            config={"displayModeBar": True, "scrollZoom": False},
            style={"backgroundColor": BG, "borderRadius": "10px"}
        ),

        html.Div(style={"backgroundColor": PANEL_BG, "border": PANEL_BORDER,
                        "borderRadius": "10px", "padding": "12px"}, children=[

            html.H4("Panel del país", style={"marginTop": "0px"}),

            html.Div("Seleccionar país (opcional)", style={"fontSize": "12px", "opacity": 0.85}),
            dcc.Dropdown(
                id="country_dropdown",
                options=[{"label": f"{iso_to_name.get(c,c)} ({c})", "value": c} for c in all_countries],
                value=None,
                placeholder="(o haz click en la ruleta)",
                clearable=True
            ),

            html.Div(style={"height": "10px"}),

            html.Div("Filtrar vecinos por comunidad (del año actual)", style={"fontSize": "12px", "opacity": 0.85}),
            dcc.Dropdown(
                id="community_filter",
                options=[{"label": "Todas", "value": "ALL"}],
                value="ALL",
                clearable=False
            ),

            html.Div(style={"height": "14px"}),

            html.Div("Top conexiones (vecinos) por correlación:", style={"fontSize": "12px", "opacity": 0.85}),
            html.Div(id="neighbors_list", style={"fontSize": "12px", "whiteSpace": "pre-line", "marginTop": "8px"}),

            html.Div(style={"height": "10px"}),

            html.Div("Tip: click en un país para fijarlo. Hover: nombre completo + comunidad.", style={"fontSize": "11px", "opacity": 0.6}),
        ])
    ]),

    # intervals
    dcc.Interval(id="interval_year", interval=INTERVAL_YEAR_MS, n_intervals=0, disabled=True),
    dcc.Interval(id="interval_turn", interval=INTERVAL_TURN_MS, n_intervals=0, disabled=True),
])


# ----------------------------
# PLAY/PAUSE YEAR
# ----------------------------
@app.callback(
    Output("store_playing_year", "data"),
    Output("interval_year", "disabled"),
    Input("btn_play", "n_clicks"),
    Input("btn_pause", "n_clicks"),
    State("store_playing_year", "data"),
    prevent_initial_call=True
)
def set_play_pause(n_play, n_pause, playing):
    trig = ctx.triggered_id
    if trig == "btn_play":
        return True, False
    if trig == "btn_pause":
        return False, True
    return playing, (not bool(playing))


# ----------------------------
# AUTO-ADVANCE YEAR WHEN PLAYING
# ----------------------------
@app.callback(
    Output("year", "value"),
    Input("interval_year", "n_intervals"),
    State("store_playing_year", "data"),
    State("year", "value"),
)
def tick_year(n, playing, year):
    if not playing:
        return no_update
    y = int(year)
    return min_year if y >= max_year else (y + 1)


# ----------------------------
# SELECT COUNTRY (click or dropdown)
# ----------------------------
@app.callback(
    Output("store_selected_country", "data"),
    Output("country_dropdown", "value"),
    Input("wheel", "clickData"),
    Input("country_dropdown", "value"),
    State("store_selected_country", "data"),
)
def set_selected(clickData, dropdown_value, stored):
    trig = ctx.triggered_id

    if trig == "wheel" and clickData and "points" in clickData and clickData["points"]:
        pt = clickData["points"][0]
        iso = pt.get("customdata") or pt.get("text")
        if iso in all_countries:
            return iso, iso
        return stored, stored

    if trig == "country_dropdown":
        if dropdown_value in all_countries:
            return dropdown_value, dropdown_value
        return None, None

    return stored, stored


# ----------------------------
# UN SOLO CALLBACK PARA LA ANIMACIÓN DE GIRO
# - Detecta cambio de país
# - Si cambia: inicializa animación y habilita interval_turn
# - Si interval_turn hace tick: avanza fase, y al final lo deshabilita
# ----------------------------
@app.callback(
    Output("store_rotation_phase", "data"),
    Output("store_turn_active", "data"),
    Output("store_turn_step", "data"),
    Output("store_turn_from", "data"),
    Output("store_turn_to", "data"),
    Output("store_selected_prev", "data"),
    Output("interval_turn", "disabled"),
    Input("store_selected_country", "data"),
    Input("interval_turn", "n_intervals"),
    State("store_rotation_phase", "data"),
    State("store_turn_active", "data"),
    State("store_turn_step", "data"),
    State("store_turn_from", "data"),
    State("store_turn_to", "data"),
    State("store_selected_prev", "data"),
)
def turn_controller(selected, n_tick, phase_now, active, step, ph_from, ph_to, prev_selected):
    trig = ctx.triggered_id

    phase_now = float(phase_now)
    active = bool(active)
    step = int(step)
    ph_from = float(ph_from)
    ph_to = float(ph_to)

    # 1) evento: cambio país
    if trig == "store_selected_country":
        # si no cambia, nada
        if selected == prev_selected:
            return phase_now, active, step, ph_from, ph_to, prev_selected, (not active)

        # si selected None -> no animamos
        if selected is None:
            return phase_now, False, 0, phase_now, phase_now, None, True

        # inicializa animación
        from_phase = phase_now
        # giro corto "bonito" hacia delante (120º)
        target = (from_phase + 2 * math.pi / 3) % (2 * math.pi)

        return phase_now, True, 0, from_phase, target, selected, False

    # 2) ticks del interval_turn
    if trig == "interval_turn":
        if not active:
            return phase_now, False, 0, ph_from, ph_to, prev_selected, True

        t = step / max(1, (TURN_FRAMES - 1))
        tt = ease_in_out(t)

        a = ph_from
        b = ph_to

        # distancia angular corta
        diff = (b - a + math.pi) % (2 * math.pi) - math.pi
        ph = (a + diff * tt) % (2 * math.pi)

        step_next = step + 1
        if step_next >= TURN_FRAMES:
            return (ph_to % (2 * math.pi)), False, 0, ph_to, ph_to, prev_selected, True

        return ph, True, step_next, ph_from, ph_to, prev_selected, False

    return phase_now, active, step, ph_from, ph_to, prev_selected, (not active)


# ----------------------------
# MAIN UPDATE (figure + panel)
# ----------------------------
@app.callback(
    Output("wheel", "figure"),
    Output("status_line", "children"),
    Output("community_filter", "options"),
    Output("community_filter", "value"),
    Output("neighbors_list", "children"),
    Input("year", "value"),
    Input("window", "value"),
    Input("metric", "value"),
    Input("store_selected_country", "data"),
    Input("community_filter", "value"),
    Input("store_rotation_phase", "data"),
)
def update_all(year, window, metric, selected_country, community_value, rotation_phase):
    year = int(year)
    window = int(window)

    sub = window_slice(df, year, window)
    C = corr_matrix(sub, metric)

    G = build_graph_from_corr(C, THR_DEFAULT)
    part = louvain_partition(G)

    nodes = list(C.columns)
    selected = selected_country if (selected_country in nodes) else None

    neighbors_all: List[Tuple[str, float]] = []
    corr_of_selected: Dict[str, float] = {}

    if selected is not None and selected in C.columns:
        s = C[selected].drop(index=selected, errors="ignore")
        for nb, val in s.items():
            if pd.notna(val) and abs(val) >= THR_DEFAULT:
                corr_of_selected[nb] = float(val)
                neighbors_all.append((nb, float(val)))

    neighbors_all.sort(key=lambda t: abs(t[1]), reverse=True)

    comms_present = []
    if selected is not None:
        seen = set()
        for nb, _ in neighbors_all:
            comm = part.get(nb, None)
            if comm is None:
                continue
            if comm not in seen:
                seen.add(comm)
                comms_present.append(comm)
        comms_present = sorted(comms_present)

    options = [{"label": "Todas", "value": "ALL"}]
    for c in comms_present:
        options.append({"label": f"Comunidad {c}", "value": str(c)})

    if community_value is None:
        community_value = "ALL"
    if community_value != "ALL":
        valid = set([o["value"] for o in options])
        if community_value not in valid:
            community_value = "ALL"

    neighbors_filtered: Set[str] = set()
    if selected is not None:
        for nb, val in neighbors_all:
            if community_value == "ALL":
                neighbors_filtered.add(nb)
            else:
                if str(part.get(nb, -999)) == str(community_value):
                    neighbors_filtered.add(nb)

    if selected is None:
        neighbors_text = "Selecciona un país (click o desplegable) para ver sus conexiones."
        comm_of_selected = "-"
    else:
        comm_of_selected = part.get(selected, "-")
        if len(neighbors_filtered) == 0:
            neighbors_text = "No hay vecinos (con el umbral actual / filtro de comunidades)."
        else:
            lines = []
            for nb, val in neighbors_all:
                if nb not in neighbors_filtered:
                    continue
                sign = "+" if val >= 0 else "-"
                cname = iso_to_name.get(nb, nb)
                lines.append(f"• {cname} ({nb}) | Comunidad {part.get(nb,'-')} | corr={abs(val):.2f} (signo {sign})")
                if len(lines) >= 12:
                    break
            neighbors_text = "\n".join(lines)

    start_y = year - window + 1
    metric_name = "Δdebt" if metric == "d_debt" else "Δpresión déficit"

    if selected is None:
        status = f"Sin país seleccionado | Ventana {start_y}–{year} (thr={THR_DEFAULT}) | Métrica: {metric_name}"
    else:
        status = f"País seleccionado: {iso_to_name.get(selected, selected)} ({selected}) | Comunidad: {comm_of_selected} | Ventana {start_y}–{year} (thr={THR_DEFAULT})"

    fig_title = f"Ruleta fiscal | {metric_name} | Ventana {start_y}–{year} (thr={THR_DEFAULT})"
    if selected is not None:
        fig_title += f" | País: {selected}"

    fig = make_ruleta_figure(
        nodes=nodes,
        part=part,
        iso_to_name=iso_to_name,
        selected=selected,
        neighbors=neighbors_filtered,
        corr_of_selected=corr_of_selected,
        thr=THR_DEFAULT,
        title=fig_title,
        rotation_phase=float(rotation_phase)
    )

    return fig, status, options, community_value, neighbors_text


if __name__ == "__main__":
    # Si tienes otro Dash en el mismo puerto, cámbialo (ej: 8051)
    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )
