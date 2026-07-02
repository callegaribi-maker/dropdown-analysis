"""
Dropdown Analysis - Visualizador de sinais sincronizados (cinemática + IMU)
============================================================================
App Streamlit para explorar arquivos .xlsx com sinais de cinemática
(posição/velocidade/aceleração por eixo) e IMU (acelerômetro/giroscópio),
segmentados em ciclos de teste a partir de uma coluna de referência.

Como rodar localmente:
    pip install -r requirements.txt
    streamlit run app.py

Deploy no Streamlit Community Cloud: aponte para este repositório / app.py.
"""

import io
import re
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import find_peaks

st.set_page_config(page_title="Dropdown Analysis - Sinais Sincronizados", layout="wide")

# ----------------------------------------------------------------------------
# Parsing / categorização de colunas
# ----------------------------------------------------------------------------

IMU_MAP = {
    "ACC_X": ("IMU - Acelerômetro", "X"),
    "ACC_Y": ("IMU - Acelerômetro", "Y"),
    "ACC_Z": ("IMU - Acelerômetro", "Z"),
    "GYR_X": ("IMU - Giroscópio", "X"),
    "GYR_Y": ("IMU - Giroscópio", "Y"),
    "GYR_Z": ("IMU - Giroscópio", "Z"),
}


def categorize_column(col_name: str):
    """Classifica uma coluna em (grupo, eixo).

    Grupos possíveis:
      - IMU - Acelerômetro / IMU - Giroscópio
      - Cinemática - Posição / Velocidade / Aceleração
    """
    if col_name in IMU_MAP:
        return IMU_MAP[col_name]

    m = re.match(r"^(.*?)\s+v\(([XYZ])\)$", col_name)
    if m:
        return ("Cinemática - Velocidade", m.group(2))

    m = re.match(r"^(.*?)\s+a\(([XYZ])\)$", col_name)
    if m:
        return ("Cinemática - Aceleração", m.group(2))

    m = re.match(r"^(.*?)\s+([XYZ])$", col_name)
    if m:
        return ("Cinemática - Posição", m.group(2))

    return (None, None)


@st.cache_data(show_spinner=False)
def load_workbook(file_bytes: bytes):
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    sheets = {}
    for name in xls.sheet_names:
        df = xls.parse(name)
        sheets[name] = df
    return sheets


def build_catalog(df: pd.DataFrame):
    """Retorna dict {grupo: {eixo: nome_da_coluna}} para um dataframe."""
    catalog = {}
    for col in df.columns[1:]:  # pula a coluna de tempo
        grupo, eixo = categorize_column(str(col))
        if grupo is None:
            continue
        catalog.setdefault(grupo, {})[eixo] = col
    return catalog


def time_column(df: pd.DataFrame) -> str:
    return df.columns[0]


# ----------------------------------------------------------------------------
# Segmentação de ciclos
# ----------------------------------------------------------------------------

def detect_cycles(signal: np.ndarray, method: str, distance: int, prominence: float):
    """Detecta índices que marcam início/fim de ciclos no sinal de referência."""
    if method == "Picos (máximos)":
        idx, _ = find_peaks(signal, distance=distance, prominence=prominence)
    elif method == "Vales (mínimos)":
        idx, _ = find_peaks(-signal, distance=distance, prominence=prominence)
    elif method == "Cruzamento por zero (subindo)":
        idx = np.where((signal[:-1] < 0) & (signal[1:] >= 0))[0]
    elif method == "Cruzamento por zero (descendo)":
        idx = np.where((signal[:-1] > 0) & (signal[1:] <= 0))[0]
    else:
        idx = np.array([], dtype=int)
    return idx


CYCLE_COLORS = ["rgba(99,110,250,0.10)", "rgba(239,85,59,0.10)"]


def add_cycle_shading(fig: go.Figure, mark_times: np.ndarray):
    """Desenha faixas alternadas + linhas verticais nos limites de cada ciclo."""
    for i in range(len(mark_times) - 1):
        fig.add_vrect(
            x0=mark_times[i], x1=mark_times[i + 1],
            fillcolor=CYCLE_COLORS[i % 2], line_width=0, layer="below",
        )
    for mt in mark_times:
        fig.add_vline(x=mt, line_dash="dash", line_color="red", opacity=0.5)


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

st.title("📊 Dropdown Analysis — Sinais Sincronizados")
st.caption(
    "Carregue o arquivo .xlsx com os sinais sincronizados. Use os menus (dropdowns) "
    "para escolher a região do corpo, o tipo de dispositivo/sinal e o eixo. "
    "Os ciclos de teste são segmentados a partir de uma coluna de referência."
)

uploaded = st.file_uploader("Arquivo .xlsx de sinais sincronizados", type=["xlsx"])

if uploaded is None:
    st.info("Envie um arquivo .xlsx para começar (ex: sinais_sincronizados_*.xlsx).")
    st.stop()

sheets = load_workbook(uploaded.getvalue())
sheet_names = list(sheets.keys())

# ---- Sidebar: segmentação de ciclos ----------------------------------------
st.sidebar.header("🔁 Segmentação de ciclos")

ref_sheet = st.sidebar.selectbox(
    "Aba de referência", sheet_names,
    index=sheet_names.index("L5") if "L5" in sheet_names else 0,
)
ref_df = sheets[ref_sheet]
ref_cols = list(ref_df.columns[1:])

# coluna D = 4ª coluna da planilha original (índice 3) -> índice 2 após remover Tempo
default_ref_idx = 2 if len(ref_cols) > 2 else 0
ref_col = st.sidebar.selectbox(
    "Coluna de referência (padrão: coluna D)", ref_cols, index=default_ref_idx
)

method = st.sidebar.selectbox(
    "Método de segmentação",
    ["Picos (máximos)", "Vales (mínimos)", "Cruzamento por zero (subindo)", "Cruzamento por zero (descendo)"],
)

min_distance = st.sidebar.slider("Distância mínima entre marcos (amostras)", 5, 300, 50)
prominence = st.sidebar.slider("Proeminência mínima (picos/vales)", 0.0, 2.0, 0.05, step=0.01)

t = ref_df[time_column(ref_df)].to_numpy()
ref_signal = ref_df[ref_col].to_numpy(dtype=float)
marks_idx = detect_cycles(ref_signal, method, min_distance, prominence)
n_cycles = max(len(marks_idx) - 1, 0)

mark_times = t[marks_idx] if len(marks_idx) else np.array([])

# ---- Main: gráfico de referência (sempre visível, segmentado) --------------
st.subheader("🔁 Sinal de referência — ciclos segmentados")
fig_ref = go.Figure()
add_cycle_shading(fig_ref, mark_times)
fig_ref.add_trace(go.Scatter(x=t, y=ref_signal, mode="lines", name=ref_col, line=dict(color="#1f77b4")))
if len(marks_idx):
    fig_ref.add_trace(go.Scatter(
        x=mark_times, y=ref_signal[marks_idx],
        mode="markers", name="início/fim de ciclo", marker=dict(color="red", size=8),
    ))
fig_ref.update_layout(
    title=f"{ref_sheet} — {ref_col} ({n_cycles} ciclo(s) detectado(s))",
    xaxis_title="Tempo (s)", yaxis_title=ref_col,
    height=320, margin=dict(l=10, r=10, t=40, b=10),
)
st.plotly_chart(fig_ref, use_container_width=True)

st.divider()

# ---- Main: seleção de região do corpo ---------------------------------------
body_sheet = st.selectbox("Região do corpo / aba", sheet_names, key="body_sheet")

df = sheets[body_sheet]
catalog = build_catalog(df)
tcol = time_column(df)
df_t = df[tcol].to_numpy()

# Grupos de aceleração/velocidade angular a empilhar por eixo, nesta ordem:
ACCEL_GROUPS = ["Cinemática - Aceleração", "IMU - Acelerômetro", "IMU - Giroscópio"]
GROUP_COLORS = {
    "Cinemática - Aceleração": "#2ca02c",
    "IMU - Acelerômetro": "#1f77b4",
    "IMU - Giroscópio": "#d62728",
}

st.subheader(f"📈 Canais de aceleração/giroscópio — {body_sheet}")
st.caption(
    "Para cada eixo (X, Y, Z): aceleração cinemática, acelerômetro (ACC) e giroscópio (GYR), "
    "um abaixo do outro. Role a página para ver os 3 eixos. Troque a região acima para atualizar tudo."
)

for axis in ["X", "Y", "Z"]:
    st.markdown(f"**Eixo {axis}**")
    for grp in ACCEL_GROUPS:
        colname = catalog.get(grp, {}).get(axis)
        if colname is None:
            continue
        fig = go.Figure()
        add_cycle_shading(fig, mark_times)
        fig.add_trace(go.Scatter(
            x=df_t, y=df[colname], mode="lines", name=colname,
            line=dict(color=GROUP_COLORS[grp]),
        ))
        fig.update_layout(
            title=f"{grp} — Eixo {axis} ({colname})",
            xaxis_title="Tempo (s)", yaxis_title=colname,
            height=260, margin=dict(l=10, r=10, t=35, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("")

# ---- Extra: posição e velocidade (opcional) ---------------------------------
with st.expander("➕ Ver também: posição e velocidade (cinemática)"):
    other_groups = [g for g in catalog.keys() if g not in ACCEL_GROUPS]
    if other_groups:
        col_a, col_b = st.columns(2)
        with col_a:
            extra_group = st.selectbox("Dispositivo / tipo de sinal", other_groups, key="extra_group")
        axes_available = sorted(catalog.get(extra_group, {}).keys())
        with col_b:
            extra_axis = st.selectbox("Eixo", axes_available + ["Todos"], key="extra_axis")
        axes_to_plot = axes_available if extra_axis == "Todos" else [extra_axis]
        for ax in axes_to_plot:
            colname = catalog[extra_group][ax]
            fig = go.Figure()
            add_cycle_shading(fig, mark_times)
            fig.add_trace(go.Scatter(x=df_t, y=df[colname], mode="lines", name=colname))
            fig.update_layout(
                title=f"{body_sheet} — {extra_group} — Eixo {ax} ({colname})",
                xaxis_title="Tempo (s)", yaxis_title=colname,
                height=320, margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Nenhum outro grupo de sinal disponível nesta aba.")

# ---- Exportação em lote -----------------------------------------------------
st.divider()
st.subheader("📦 Exportar todos os gráficos (todas as combinações)")
st.caption("Gera um PNG para cada combinação de região × dispositivo × eixo, em um .zip.")

if st.button("Gerar pacote de gráficos (.zip)"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        progress = st.progress(0.0, text="Gerando gráficos...")
        combos = []
        for sn in sheet_names:
            cat = build_catalog(sheets[sn])
            for grp, axes in cat.items():
                for ax_name, col in axes.items():
                    combos.append((sn, grp, ax_name, col))
        total = len(combos) or 1
        for i, (sn, grp, ax_name, col) in enumerate(combos):
            d = sheets[sn]
            fig = go.Figure()
            add_cycle_shading(fig, mark_times)
            fig.add_trace(go.Scatter(x=d[time_column(d)], y=d[col], mode="lines", name=col))
            fig.update_layout(
                title=f"{sn} — {grp} — Eixo {ax_name} ({col})",
                xaxis_title="Tempo (s)", yaxis_title=col,
                width=1000, height=450,
            )
            safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", f"{sn}_{grp}_{ax_name}")
            png_bytes = fig.to_image(format="png")
            zf.writestr(f"{sn}/{safe_name}.png", png_bytes)
            progress.progress((i + 1) / total, text=f"Gerando gráficos... ({i+1}/{total})")
    buf.seek(0)
    st.download_button(
        "⬇️ Baixar graficos.zip", data=buf, file_name="graficos_dropdown_analysis.zip",
        mime="application/zip",
    )

st.caption(
    "Dica: para exportar imagens estáticas (PNG), este app usa Kaleido "
    "(incluído no requirements.txt)."
)
