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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.signal import butter, detrend, filtfilt, find_peaks

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


def _butter_lowpass(cutoff_hz: float, fs: float, order: int):
    nyq = fs / 2.0
    wn = min(max(cutoff_hz / nyq, 1e-4), 0.99)
    return butter(order, wn, btype="low")


def filter_dataframe(df: pd.DataFrame, kinem_cutoff_hz: float, imu_cutoff_hz: float, order: int) -> pd.DataFrame:
    """Detrend + filtro Butterworth passa-baixa (zero-fase, via filtfilt) em todas as
    colunas de sinal (todas menos a de tempo). Cinemática (posição/velocidade/aceleração)
    usa um corte próprio (mais baixo), diferente do IMU (ACC/GYR)."""
    tcol = time_column(df)
    t_arr = df[tcol].to_numpy(dtype=float)
    dt = np.median(np.diff(t_arr)) if len(t_arr) > 1 else 0.01
    fs = 1.0 / dt if dt > 0 else 100.0

    b_kinem, a_kinem = _butter_lowpass(kinem_cutoff_hz, fs, order)
    b_imu, a_imu = _butter_lowpass(imu_cutoff_hz, fs, order)

    out = df.copy()
    for col in df.columns[1:]:
        grupo, _ = categorize_column(str(col))
        b, a = (b_kinem, a_kinem) if (grupo or "").startswith("Cinemática") else (b_imu, a_imu)
        min_len = 3 * (max(len(a), len(b)))
        sig = df[col].to_numpy(dtype=float)
        sig = detrend(sig)
        if len(sig) > min_len:
            sig = filtfilt(b, a, sig)
        out[col] = sig
    return out


# ----------------------------------------------------------------------------
# Segmentação de trials (manual, a partir de vales/picos detectados)
# ----------------------------------------------------------------------------

DESCIDA_COLOR = "rgba(255,127,14,0.18)"
SUBIDA_COLOR = "rgba(44,160,44,0.18)"
PLATEAU_COLOR = "rgba(150,150,150,0.25)"


def add_trial_shading(fig: go.Figure, sel_starts, sel_ends, valley_times: np.ndarray, t_first: float):
    """Cada CICLO completo = platô (cinza) + descida (laranja) + subida (verde), nessa
    ordem, sem nenhum trecho fora dessas 3 fases. O platô do ciclo i é o intervalo entre
    o fim do ciclo anterior (ou o início da gravação, no ciclo 1) e o início da descida."""
    n = len(sel_starts)
    for i in range(n):
        platform_start = sel_ends[i - 1] if i > 0 else t_first
        d_start = sel_starts[i]
        d_end = sel_ends[i]
        inside = valley_times[(valley_times > d_start) & (valley_times < d_end)]
        v = inside[0] if len(inside) else (d_start + d_end) / 2
        if platform_start < d_start:
            fig.add_vrect(
                x0=platform_start, x1=d_start, fillcolor=PLATEAU_COLOR, line_width=0, layer="below",
                annotation_text="platô", annotation_position="top", annotation_font_size=10,
            )
        fig.add_vrect(x0=d_start, x1=v, fillcolor=DESCIDA_COLOR, line_width=0, layer="below")
        fig.add_vrect(x0=v, x1=d_end, fillcolor=SUBIDA_COLOR, line_width=0, layer="below")
        fig.add_vline(x=v, line_dash="dot", line_color="orange", opacity=0.8)
    for s in sel_starts:
        fig.add_vline(x=s, line_dash="dash", line_color="#1f77b4", opacity=0.6)
    for e in sel_ends:
        fig.add_vline(x=e, line_dash="dash", line_color="#2ca02c", opacity=0.6)


def find_plateau_edges(is_flat: np.ndarray, idx: int):
    """Expande a partir de idx enquanto o sinal estiver 'plano', retornando (esquerda, direita)."""
    n = len(is_flat)
    left = right = idx
    while left > 0 and is_flat[left - 1]:
        left -= 1
    while right < n - 1 and is_flat[right + 1]:
        right += 1
    return left, right


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

st.title("📊 Dropdown Analysis — Sinais Sincronizados")
st.caption(
    "Carregue o arquivo .xlsx. No gráfico de referência, todos os vales aparecem marcados "
    "(▽ laranja) e os pontos no platô do topo (◇/★) podem ser clicados para marcar o início "
    "e o fim de cada trial. Depois, navegue trial a trial e veja Deslocamento, Velocidade e "
    "Aceleração (cinemática) e ACC/GYR (IMU) por eixo."
)

uploaded = st.file_uploader("Arquivo .xlsx de sinais sincronizados", type=["xlsx"])

if uploaded is None:
    st.info("Envie um arquivo .xlsx para começar (ex: sinais_sincronizados_*.xlsx).")
    st.stop()

sheets_raw = load_workbook(uploaded.getvalue())
sheet_names = list(sheets_raw.keys())

# ---- Sidebar: filtro do sinal -----------------------------------------------
st.sidebar.header("🧹 Filtro do sinal")
use_filter = st.sidebar.checkbox(
    "Aplicar filtro passa-baixa (detrend + Butterworth + filtfilt)", value=True
)
kinem_cutoff = st.sidebar.slider("Corte Kinem (Hz)", 0.2, 10.0, 1.0, step=0.1)
imu_cutoff = st.sidebar.slider("Corte ACC/GYR (Hz)", 0.5, 10.0, 2.0, step=0.5)
filter_order = st.sidebar.slider("Ordem do filtro", 2, 8, 4)

if use_filter:
    sheets = {name: filter_dataframe(df, kinem_cutoff, imu_cutoff, filter_order) for name, df in sheets_raw.items()}
    st.sidebar.caption(
        f"Filtro ativo: Kinem {kinem_cutoff:.1f} Hz, ACC/GYR {imu_cutoff:.1f} Hz, ordem {filter_order} "
        f"(Butterworth passa-baixa, zero-fase)."
    )
else:
    sheets = sheets_raw
    st.sidebar.caption("Filtro desativado — usando sinal bruto.")

# ---- Sidebar: detecção de candidatos (vales e picos/platôs) -----------------
st.sidebar.header("🔁 Segmentação de trials")

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

min_distance = st.sidebar.slider("Distância mínima entre marcos (amostras)", 5, 300, 50)
prominence = st.sidebar.slider("Proeminência mínima (vales/picos)", 0.0, 2.0, 0.05, step=0.01)
plateau_frac = st.sidebar.slider(
    "Sensibilidade do platô (menor = platô mais estreito)", 0.01, 0.30, 0.05, step=0.01
)

t = ref_df[time_column(ref_df)].to_numpy()
ref_signal = ref_df[ref_col].to_numpy(dtype=float)
n_samples = len(ref_signal)

valleys_idx, _ = find_peaks(-ref_signal, distance=min_distance, prominence=prominence)
peaks_idx, _ = find_peaks(ref_signal, distance=min_distance, prominence=prominence)
valley_times = t[valleys_idx]

# Detecta o platô (região "plana", derivada baixa) em torno de cada pico e nas duas
# bordas do registro. O platô fica ENTRE o marco de "fim de trial" e o marco de
# "início do próximo trial" e não entra na análise (não é sombreado nem incluído
# na janela de um trial).
deriv = np.gradient(ref_signal, t)
max_abs_deriv = np.max(np.abs(deriv)) if n_samples else 1.0
is_flat = np.abs(deriv) < plateau_frac * (max_abs_deriv if max_abs_deriv > 0 else 1.0)

# Platô inicial/final: procura o ponto mais alto na região antes do 1º vale / depois
# do último vale (em vez de checar só a própria borda, que pode ter ruído) e expande
# a partir dali — assim o platô do começo/fim da gravação é sempre considerado.
pre_end = valleys_idx[0] if len(valleys_idx) else n_samples - 1
pre_peak = int(np.argmax(ref_signal[:pre_end + 1])) if pre_end > 0 else 0
if is_flat[pre_peak]:
    _, r0 = find_plateau_edges(is_flat, pre_peak)
    start0 = t[r0]
else:
    start0 = t[pre_peak]

post_start = valleys_idx[-1] if len(valleys_idx) else 0
post_peak = post_start + int(np.argmax(ref_signal[post_start:]))
if is_flat[post_peak]:
    l_last, _ = find_plateau_edges(is_flat, post_peak)
    end_last = t[l_last]
else:
    end_last = t[post_peak]

start_times = [start0]
end_times = []
for p in peaks_idx:
    left, right = find_plateau_edges(is_flat, p)
    end_times.append(t[left])
    start_times.append(t[right])
end_times.append(end_last)

start_times = np.array(start_times)
end_times = np.array(end_times)

# Reseta a seleção manual sempre que os candidatos mudarem (nova coluna/aba/sensibilidade)
sig_key = (
    ref_sheet, ref_col, min_distance, prominence, plateau_frac,
    use_filter, kinem_cutoff, imu_cutoff, filter_order, len(start_times), len(end_times),
)
if st.session_state.get("peaks_sig_key") != sig_key:
    st.session_state.peaks_sig_key = sig_key
    st.session_state.start_mask = np.ones(len(start_times), dtype=bool)
    st.session_state.end_mask = np.ones(len(end_times), dtype=bool)
    st.session_state.trial_idx = 1
    st.session_state.last_click_sig = ()

start_mask = st.session_state.start_mask
end_mask = st.session_state.end_mask

st.sidebar.caption(
    f"{len(valley_times)} vale(s) · {len(start_times)} marco(s) de início · "
    f"{len(end_times)} marco(s) de fim"
)
col_sa, col_sb = st.sidebar.columns(2)
with col_sa:
    if st.button("Marcar todos", use_container_width=True):
        st.session_state.start_mask = np.ones(len(start_times), dtype=bool)
        st.session_state.end_mask = np.ones(len(end_times), dtype=bool)
        st.rerun()
with col_sb:
    if st.button("Limpar", use_container_width=True):
        st.session_state.start_mask = np.zeros(len(start_times), dtype=bool)
        st.session_state.end_mask = np.zeros(len(end_times), dtype=bool)
        st.rerun()

# ---- Main: gráfico de referência interativo ---------------------------------
st.subheader("🔁 Sinal de referência — clique para marcar início/fim do trial")
st.caption(
    "O teste tem 3 fases por ciclo: descida (laranja, início→vale), subida (verde, vale→fim) "
    "e platô (cinza, fase separada, fora da análise). Cada fase tem 2 marcações: descida vai de "
    "▲ até ▽ (vale), subida vai de ▽ até ■. ▲ azul = início do trial = fim do platô anterior. "
    "■ verde = fim do trial = início do próximo platô. Clique num marcador (▲/■) para incluir/excluir."
)

sel_starts = sorted(start_times[start_mask].tolist())
sel_ends = sorted(end_times[end_mask].tolist())
trial_pairs = list(zip(sel_starts, sel_ends))
n_trials = len(trial_pairs)

fig_ref = go.Figure()
add_trial_shading(fig_ref, sel_starts, sel_ends, valley_times, t[0])
fig_ref.add_trace(go.Scatter(
    x=[None], y=[None], mode="markers", marker=dict(size=12, color=PLATEAU_COLOR, symbol="square"),
    name="platô (fase separada)",
))
fig_ref.add_trace(go.Scatter(x=t, y=ref_signal, mode="lines", name=ref_col, line=dict(color="#1f77b4")))

trace_idx = 2  # 0 = legenda do platô (dummy), 1 = linha do sinal
VALLEY_TRACE_INDEX = None
if len(valley_times):
    fig_ref.add_trace(go.Scatter(
        x=valley_times, y=ref_signal[valleys_idx], mode="markers", name="vales",
        marker=dict(color="orange", symbol="triangle-down", size=10),
    ))
    VALLEY_TRACE_INDEX = trace_idx
    trace_idx += 1

START_TRACE_INDEX = trace_idx
start_y = np.interp(start_times, t, ref_signal)
colors_s = np.where(start_mask, "#1f77b4", "lightgray").tolist()
sizes_s = np.where(start_mask, 14, 9).tolist()
fig_ref.add_trace(go.Scatter(
    x=start_times, y=start_y, mode="markers", name="início do trial (clique p/ alternar)",
    marker=dict(color=colors_s, symbol="triangle-up", size=sizes_s, line=dict(width=1, color="black")),
))
trace_idx += 1

END_TRACE_INDEX = trace_idx
end_y = np.interp(end_times, t, ref_signal)
colors_e = np.where(end_mask, "#2ca02c", "lightgray").tolist()
sizes_e = np.where(end_mask, 14, 9).tolist()
fig_ref.add_trace(go.Scatter(
    x=end_times, y=end_y, mode="markers", name="fim do trial (clique p/ alternar)",
    marker=dict(color=colors_e, symbol="square", size=sizes_e, line=dict(width=1, color="black")),
))

fig_ref.update_layout(
    title=f"{ref_sheet} — {ref_col} ({n_trials} trial(s) definido(s))",
    xaxis_title="Tempo (s)", yaxis_title=ref_col,
    height=380, margin=dict(l=10, r=10, t=40, b=10),
)

event = st.plotly_chart(
    fig_ref, use_container_width=True, on_select="rerun", key="ref_chart",
    selection_mode=("points",),
)

if event and event.get("selection", {}).get("points"):
    pts = event["selection"]["points"]
    click_sig = tuple(sorted((p.get("curve_number"), p.get("point_index")) for p in pts))
    if click_sig and click_sig != st.session_state.get("last_click_sig"):
        for curve_number, idx in click_sig:
            if curve_number == START_TRACE_INDEX and idx is not None and 0 <= idx < len(st.session_state.start_mask):
                st.session_state.start_mask[idx] = not st.session_state.start_mask[idx]
            elif curve_number == END_TRACE_INDEX and idx is not None and 0 <= idx < len(st.session_state.end_mask):
                st.session_state.end_mask[idx] = not st.session_state.end_mask[idx]
        st.session_state.last_click_sig = click_sig
        st.rerun()

# ---- Faixa de ciclos (todos na mesma cor, numerados) ------------------------
if n_trials:
    fig_cycles = go.Figure()
    for i, (s, e) in enumerate(trial_pairs, start=1):
        fig_cycles.add_shape(
            type="rect", x0=s, x1=e, y0=0, y1=1,
            fillcolor="rgba(31,119,180,0.45)", line=dict(width=1, color="#1f77b4"),
        )
        fig_cycles.add_annotation(
            x=(s + e) / 2, y=0.5, text=f"Ciclo {i}", showarrow=False,
            font=dict(color="white", size=12),
        )
    fig_cycles.update_xaxes(range=[t[0], t[-1]], title="Tempo (s)")
    fig_cycles.update_yaxes(visible=False, range=[0, 1])
    fig_cycles.update_layout(
        height=90, margin=dict(l=10, r=10, t=10, b=30), showlegend=False, plot_bgcolor="white",
    )
    st.plotly_chart(fig_cycles, use_container_width=True)

st.divider()

# ---- Região do corpo (único dropdown desta seção) ---------------------------
st.subheader("⚙️ Região")
body_sheet = st.selectbox("Região do corpo / aba", sheet_names, key="body_sheet")

# Cinemática: sempre as 3 (Posição/Deslocamento, Velocidade, Aceleração), sem dropdown.
KINEM_GROUP_MAP = {
    "Posição": "Cinemática - Posição",
    "Velocidade": "Cinemática - Velocidade",
    "Aceleração": "Cinemática - Aceleração",
}
KINEM_ORDER = ["Posição", "Velocidade", "Aceleração"]

# Nomes e unidades para exibição (cinemática em cm, IMU com nomes físicos)
KINEM_LABEL_MAP = {"Posição": "Deslocamento", "Velocidade": "Velocidade", "Aceleração": "Aceleração"}
KINEM_UNIT_MAP = {"Posição": "cm", "Velocidade": "cm/s", "Aceleração": "cm/s²"}

IMU_LABELS = {
    "IMU - Acelerômetro": ("Aceleração Linear", "m/s²"),
    "IMU - Giroscópio": ("Velocidade Angular", "°/s"),
}
# Mapeamento anatômico dos eixos — diferente entre Kinem (sistema óptico) e o
# celular (ACC/GYR), e no celular o mapeamento do ACC/GYR também muda conforme a
# região (Joelho vs L5), porque a orientação do celular no corpo é diferente:
#   Kinem:        Z = Vertical, Y = Anteroposterior (AP), X = Mediolateral (ML)
#   ACC/GYR Joelho: Y = Vertical, Z = Mediolateral (ML),   X = Anteroposterior (AP)
#   ACC/GYR L5:     Y = Vertical, Z = Anteroposterior (AP), X = Mediolateral (ML)
KINEM_AXIS_LABEL = {"X": "ML", "Y": "AP", "Z": "Vertical"}
IMU_AXIS_LABEL_JOELHO = {"X": "AP", "Y": "Vertical", "Z": "ML"}
IMU_AXIS_LABEL_L5 = {"X": "ML", "Y": "Vertical", "Z": "AP"}


def get_imu_axis_label(region_name):
    return IMU_AXIS_LABEL_L5 if "l5" in region_name.lower() else IMU_AXIS_LABEL_JOELHO


IMU_AXIS_LABEL = get_imu_axis_label(body_sheet)

# Cor por DIREÇÃO anatômica (não pelo eixo bruto) — assim Vertical é sempre a
# mesma cor tanto no Kinem (Z) quanto no celular (Y), e o mesmo vale para AP e ML.
DIR_COLORS = {"Vertical": "#2ca02c", "AP": "#1f77b4", "ML": "#d62728"}


def axis_direction(is_kinem, axis):
    mapping = KINEM_AXIS_LABEL if is_kinem else IMU_AXIS_LABEL
    return mapping[axis]


def axis_color(is_kinem, axis):
    return DIR_COLORS[axis_direction(is_kinem, axis)]


def axis_name(is_kinem, axis):
    return f"{axis} ({axis_direction(is_kinem, axis)})"

df = sheets[body_sheet]
catalog = build_catalog(df)
tcol = time_column(df)
df_t = df[tcol].to_numpy()

st.divider()

# ---- Helpers de ciclo/fases por trial ---------------------------------------
if n_trials == 0:
    st.info("Mantenha pelo menos um par início/fim marcado no gráfico acima para definir um trial.")
    st.stop()

IMU_ROWS = ["IMU - Acelerômetro", "IMU - Giroscópio"]
AXES = ["X", "Y", "Z"]
acc_label, acc_unit = IMU_LABELS["IMU - Acelerômetro"]
gyr_label, gyr_unit = IMU_LABELS["IMU - Giroscópio"]


def trial_bounds(trial_idx):
    """Ciclo completo = platô (do fim do ciclo anterior, ou início da gravação, até o
    início da descida) + descida + subida."""
    cycle_start = sel_ends[trial_idx - 2] if trial_idx > 1 else t[0]
    d_start = sel_starts[trial_idx - 1]
    cycle_end = sel_ends[trial_idx - 1]
    valley_in_cycle = valley_times[(valley_times > d_start) & (valley_times < cycle_end)]
    v_trial = valley_in_cycle[0] if len(valley_in_cycle) else (d_start + cycle_end) / 2
    return cycle_start, d_start, v_trial, cycle_end


def make_helpers(cycle_start, d_start, v_trial, cycle_end):
    def norm_t(x):
        return (x - cycle_start) / (cycle_end - cycle_start) if (cycle_end - cycle_start) != 0 else 0.0

    def add_phase_shading_subplot(fig, row, col):
        if cycle_start < d_start:
            fig.add_vrect(x0=norm_t(cycle_start), x1=norm_t(d_start), fillcolor=PLATEAU_COLOR, line_width=0, layer="below", row=row, col=col)
        fig.add_vrect(x0=norm_t(d_start), x1=norm_t(v_trial), fillcolor=DESCIDA_COLOR, line_width=0, layer="below", row=row, col=col)
        fig.add_vrect(x0=norm_t(v_trial), x1=norm_t(cycle_end), fillcolor=SUBIDA_COLOR, line_width=0, layer="below", row=row, col=col)

    def add_event_lines_subplot(fig, row, col):
        fig.add_vline(x=norm_t(d_start), line_dash="dash", line_color="#1f77b4", opacity=0.9, row=row, col=col)
        fig.add_vline(x=norm_t(v_trial), line_dash="dot", line_color="orange", opacity=0.9, row=row, col=col)
        fig.add_vline(x=norm_t(cycle_end), line_dash="dash", line_color="#2ca02c", opacity=0.9, row=row, col=col)

    return norm_t, add_phase_shading_subplot, add_event_lines_subplot


# Tamanho de figura para células realmente quadradas: o plotly consome uma fração
# do espaço em "gaps" entre subplots (horizontal_spacing/vertical_spacing), então
# largura e altura totais precisam compensar isso — não basta usar cell*cols e
# cell*rows direto, senão o resultado fica mais alto que largo (ou o contrário).
# Além disso, limitamos a largura total (MAX_WIDTH): figuras muito largas (ex:
# 2000px+ para 5-6 trials) ficam maiores que a área visível do Streamlit, que
# comprime só a largura ao exibir — isso destrói o quadrado mesmo com a conta
# certa. Por isso o tamanho da célula encolhe (mantendo-se quadrada) quando há
# muitas colunas, para a figura inteira caber sem ser redimensionada.
CELL_PX = 300
MAX_WIDTH = 1100
MARGIN = dict(l=10, r=10, t=60, b=10)
H_SPACING = 0.06
V_SPACING = 0.12


def square_fig_size(rows, cols):
    col_frac = (1 - H_SPACING * (cols - 1)) / cols
    row_frac = (1 - V_SPACING * (rows - 1)) / rows if rows > 1 else 1.0
    max_plot_w = MAX_WIDTH - MARGIN["l"] - MARGIN["r"]
    cell = min(CELL_PX, max_plot_w * col_frac)
    plot_w = cell / col_frac
    plot_h = cell / row_frac
    width = int(round(plot_w)) + MARGIN["l"] + MARGIN["r"]
    height = int(round(plot_h)) + MARGIN["t"] + MARGIN["b"]
    return width, height


# ---- Seção 1: Cinemática — 1 trial por vez, eixos sempre juntos -------------
st.subheader(f"📈 {body_sheet} — Cinemática")
st.caption(
    "Deslocamento, Velocidade e Aceleração lado a lado, cada um com X, Y, Z juntos no mesmo "
    "gráfico. Fundo cinza = platô, laranja = descida, verde = subida. No Kinem: "
    "Z = Vertical, Y = Anteroposterior (AP), X = Mediolateral (ML)."
)

kinem_trial_idx = st.selectbox(
    "Trial (só afeta a Cinemática)", list(range(1, n_trials + 1)), key="kinem_trial_idx"
)

k_cycle_start, k_d_start, k_v_trial, k_cycle_end = trial_bounds(kinem_trial_idx)
k_norm_t, k_add_phase, k_add_events = make_helpers(k_cycle_start, k_d_start, k_v_trial, k_cycle_end)
k_trial_mask = (df_t >= k_cycle_start) & (df_t <= k_cycle_end)

fig_kinem = make_subplots(
    rows=1, cols=3,
    subplot_titles=[
        f"{KINEM_LABEL_MAP['Posição']} (X, Y, Z)",
        f"{KINEM_LABEL_MAP['Velocidade']} (X, Y, Z)",
        f"{KINEM_LABEL_MAP['Aceleração']} (X, Y, Z)",
    ],
    shared_xaxes=True, horizontal_spacing=H_SPACING,
)
for col_i, choice in enumerate(KINEM_ORDER, start=1):
    grp = KINEM_GROUP_MAP[choice]
    label = KINEM_LABEL_MAP[choice]
    unit = KINEM_UNIT_MAP[choice]
    has_trace = False
    for axis in AXES:
        colname = catalog.get(grp, {}).get(axis)
        if colname is None:
            continue
        fig_kinem.add_trace(
            go.Scatter(
                x=k_norm_t(df_t[k_trial_mask]), y=df[colname].to_numpy()[k_trial_mask],
                mode="lines", line=dict(color=axis_color(True, axis)), name=axis_name(True, axis),
                showlegend=(col_i == 1), legendgroup=axis_name(True, axis),
            ),
            row=1, col=col_i,
        )
        has_trace = True
    if has_trace:
        # IMPORTANTE: o traço precisa existir ANTES do add_vrect/add_vline com row/col.
        k_add_phase(fig_kinem, 1, col_i)
        k_add_events(fig_kinem, 1, col_i)
        fig_kinem.update_yaxes(title_text=f"{label} ({unit})", row=1, col=col_i)

fig_kinem.update_xaxes(showgrid=False, range=[0, 1], title_text="Fração do ciclo (0–1)")
fig_kinem.update_yaxes(showgrid=False)
_kw, _kh = square_fig_size(1, 3)
fig_kinem.update_layout(width=_kw, height=_kh, margin=MARGIN, plot_bgcolor="white")
st.plotly_chart(fig_kinem, use_container_width=False, key="kinem_chart")

st.divider()

# ---- Seção 2: ACC/GYR — matriz 2 (ACC, GYR) × N trials, eixos sempre juntos -
st.subheader(f"📈 {body_sheet} — ACC / GYR — todos os {n_trials} trials")
_imu_dir_desc = ", ".join(f"{ax} = {IMU_AXIS_LABEL[ax]}" for ax in AXES)
st.caption(
    f"Cada coluna é um trial (1 a {n_trials}); linhas: {acc_label} e {gyr_label}, sempre com "
    f"X, Y, Z juntos no mesmo gráfico. No celular (ACC/GYR) em {body_sheet}: {_imu_dir_desc}."
)

imu_titles = []
for grp in IMU_ROWS:
    for i in range(1, n_trials + 1):
        imu_titles.append(f"Trial {i}")

fig_imu = make_subplots(
    rows=2, cols=n_trials, subplot_titles=imu_titles, shared_xaxes=True,
    horizontal_spacing=H_SPACING, vertical_spacing=V_SPACING,
)

for row_i, grp in enumerate(IMU_ROWS, start=1):
    label, unit = IMU_LABELS[grp]
    for col_j, trial_idx in enumerate(range(1, n_trials + 1), start=1):
        cycle_start, d_start, v_trial, cycle_end = trial_bounds(trial_idx)
        norm_t, add_phase, add_events = make_helpers(cycle_start, d_start, v_trial, cycle_end)
        trial_mask = (df_t >= cycle_start) & (df_t <= cycle_end)

        has_trace = False
        for axis in AXES:
            colname = catalog.get(grp, {}).get(axis)
            if colname is None:
                continue
            fig_imu.add_trace(
                go.Scatter(
                    x=norm_t(df_t[trial_mask]), y=df[colname].to_numpy()[trial_mask],
                    mode="lines", line=dict(color=axis_color(False, axis)), name=axis_name(False, axis),
                    showlegend=(row_i == 1 and col_j == 1), legendgroup=axis_name(False, axis),
                ),
                row=row_i, col=col_j,
            )
            has_trace = True
        if has_trace:
            add_phase(fig_imu, row_i, col_j)
            add_events(fig_imu, row_i, col_j)
            if col_j == 1:
                fig_imu.update_yaxes(title_text=f"{label} ({unit})", row=row_i, col=col_j)

fig_imu.update_xaxes(showgrid=False, range=[0, 1], title_text="Fração do ciclo (0–1)")
fig_imu.update_yaxes(showgrid=False)
_iw, _ih = square_fig_size(2, n_trials)
fig_imu.update_layout(width=_iw, height=_ih, margin=MARGIN, plot_bgcolor="white")
st.plotly_chart(fig_imu, use_container_width=False, key="imu_matrix")

st.divider()

# ---- Seção 3: média de todos os trials, com sombra de desvio padrão --------
st.subheader(f"📈 {body_sheet} — Média de todos os trials (sombra = ±1 desvio padrão)")
_kinem_dir_desc = ", ".join(f"{ax} = {KINEM_AXIS_LABEL[ax]}" for ax in AXES)
st.caption(
    f"Cada gráfico combina os {n_trials} trials: linha = média, sombra = ±1 desvio padrão, por "
    "direção anatômica (Vertical/AP/ML — mesma cor em todos os gráficos). Inclui Cinemática "
    "(Deslocamento, Velocidade, Aceleração) e IMU (ACC, GYR). Tempo normalizado (0–1) por ciclo "
    f"antes de calcular a média. No Kinem: {_kinem_dir_desc}. No celular (ACC/GYR) em "
    f"{body_sheet}: {_imu_dir_desc}."
)

GRID = np.linspace(0.0, 1.0, 101)


def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def ensemble_mean_std(grp, axis):
    curves = []
    for trial_idx in range(1, n_trials + 1):
        cycle_start, d_start, v_trial, cycle_end = trial_bounds(trial_idx)
        norm_t, _, _ = make_helpers(cycle_start, d_start, v_trial, cycle_end)
        trial_mask = (df_t >= cycle_start) & (df_t <= cycle_end)
        colname = catalog.get(grp, {}).get(axis)
        if colname is None:
            continue
        x_trial = norm_t(df_t[trial_mask])
        y_trial = df[colname].to_numpy()[trial_mask]
        if len(x_trial) < 2:
            continue
        order = np.argsort(x_trial)
        curves.append(np.interp(GRID, x_trial[order], y_trial[order]))
    if not curves:
        return None, None
    arr = np.vstack(curves)
    return arr.mean(axis=0), arr.std(axis=0)


AVG_GROUPS = [
    (KINEM_LABEL_MAP["Posição"], KINEM_GROUP_MAP["Posição"], KINEM_UNIT_MAP["Posição"]),
    (KINEM_LABEL_MAP["Velocidade"], KINEM_GROUP_MAP["Velocidade"], KINEM_UNIT_MAP["Velocidade"]),
    (KINEM_LABEL_MAP["Aceleração"], KINEM_GROUP_MAP["Aceleração"], KINEM_UNIT_MAP["Aceleração"]),
    (acc_label, "IMU - Acelerômetro", acc_unit),
    (gyr_label, "IMU - Giroscópio", gyr_unit),
]

fig_avg = make_subplots(
    rows=1, cols=len(AVG_GROUPS),
    subplot_titles=[f"{label} (X, Y, Z)" for label, _, _ in AVG_GROUPS],
    shared_xaxes=True, horizontal_spacing=H_SPACING,
)

for col_i, (label, grp, unit) in enumerate(AVG_GROUPS, start=1):
    is_kinem = grp.startswith("Cinemática")
    for axis in AXES:
        mean_y, std_y = ensemble_mean_std(grp, axis)
        if mean_y is None:
            continue
        color = axis_color(is_kinem, axis)
        direction = axis_direction(is_kinem, axis)
        upper = mean_y + std_y
        lower = mean_y - std_y
        fig_avg.add_trace(
            go.Scatter(
                x=np.concatenate([GRID, GRID[::-1]]), y=np.concatenate([upper, lower[::-1]]),
                fill="toself", fillcolor=hex_to_rgba(color, 0.2),
                line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False,
            ),
            row=1, col=col_i,
        )
        # Cor = direção anatômica (Vertical/AP/ML), consistente entre Kinem e IMU,
        # então 1 legenda só (pela direção) já vale pro gráfico inteiro.
        fig_avg.add_trace(
            go.Scatter(
                x=GRID, y=mean_y, mode="lines", line=dict(color=color),
                name=direction, showlegend=(col_i == 1), legendgroup=direction,
            ),
            row=1, col=col_i,
        )
    fig_avg.update_yaxes(title_text=f"{label} ({unit})", row=1, col=col_i)

fig_avg.update_xaxes(showgrid=False, range=[0, 1], title_text="Fração do ciclo (0–1)")
fig_avg.update_yaxes(showgrid=False)
_aw, _ah = square_fig_size(1, len(AVG_GROUPS))
fig_avg.update_layout(width=_aw, height=_ah, margin=MARGIN, plot_bgcolor="white")
st.plotly_chart(fig_avg, use_container_width=False, key="avg_chart")
