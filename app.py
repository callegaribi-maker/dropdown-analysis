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
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
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

# ---- Correção de calibração conhecida (Joelho) ------------------------------
# No sensor do Joelho, os canais ACC_X (mapeado como AP) e ACC_Y (mapeado como Vertical)
# saem com o sinal invertido em relação à cinemática (sistema óptico): mesma forma de
# onda, sinal trocado. Confirmado em 2 gravações/sujeitos distintos (mesmo padrão nos
# dois), então é tratado como um erro sistemático de montagem/calibração do sensor nessa
# região — não de sincronização — e corrigido aqui na entrada, antes de qualquer filtro,
# gráfico ou cálculo, pra propagar certo em todo o app. ACC_Z (ML) já vem com o sinal
# correto e não é alterado. GYR não é alterado (a inversão encontrada foi só no ACC).
JOELHO_ACC_SIGN_FIX = ("ACC_X", "ACC_Y")
if "Joelho" in sheets_raw:
    _jo_df = sheets_raw["Joelho"].copy()
    for _col in JOELHO_ACC_SIGN_FIX:
        if _col in _jo_df.columns:
            _jo_df[_col] = -_jo_df[_col]
    sheets_raw["Joelho"] = _jo_df

# ---- Correção de unidade do giroscópio (rad/s -> °/s) -----------------------
# O sensor do celular (GYR_X/Y/Z) sai em radianos/segundo (padrão do sensor de giroscópio
# do Android/iOS), não em graus/segundo — os valores brutos são muito pequenos (ex.: pico
# de ~2.5 em vez de ~145) pra serem °/s durante um movimento como esse. Convertido aqui na
# entrada, uma vez, pra todas as abas — assim todo o app (gráficos de ACC/GYR, resultante,
# e o cálculo do ângulo de inclinação) já trabalha em °/s de verdade.
GYR_COLS = ("GYR_X", "GYR_Y", "GYR_Z")
for _name, _df in sheets_raw.items():
    _df2 = _df.copy()
    for _col in GYR_COLS:
        if _col in _df2.columns:
            _df2[_col] = np.degrees(_df2[_col])
    sheets_raw[_name] = _df2

sheet_names = list(sheets_raw.keys())

# ---- Sidebar: orientação do sensor (topo — mostra L5 e Joelho juntos) ------
_static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
st.sidebar.header("📱 Orientação do sensor")
for _region_label, _fname in (("L5 (lombar)", "orientacao_l5.png"), ("Joelho", "orientacao_joelho.png")):
    _orient_file = os.path.join(_static_dir, _fname)
    if os.path.exists(_orient_file):
        st.sidebar.image(_orient_file, caption=f"Celular em {_region_label}", use_container_width=True)
    else:
        st.sidebar.caption(f"Imagem de orientação ({_region_label}) não encontrada.")
st.sidebar.divider()

# ---- Sidebar: segmentação de trials (menus recolhidos para ocupar menos) ---
with st.sidebar.expander("🔁 Segmentação de trials", expanded=False):
    # Referência escondida por padrão (raramente muda) — abas/coluna já vêm com um
    # default sensato (L5, coluna D), então fica num sub-expander recolhido.
    with st.expander("Referência (avançado)", expanded=False):
        ref_sheet = st.selectbox(
            "Aba de referência", sheet_names,
            index=sheet_names.index("L5") if "L5" in sheet_names else 0,
        )
        _ref_cols_raw = list(sheets_raw[ref_sheet].columns[1:])
        # coluna D = 4ª coluna da planilha original (índice 3) -> índice 2 após remover Tempo
        _default_ref_idx = 2 if len(_ref_cols_raw) > 2 else 0
        ref_col = st.selectbox(
            "Coluna de referência (padrão: coluna D)", _ref_cols_raw, index=_default_ref_idx
        )

    min_distance = st.slider("Distância mínima entre marcos (amostras)", 5, 300, 50)
    prominence = st.slider("Proeminência mínima (vales/picos)", 0.0, 2.0, 0.05, step=0.01)
    plateau_frac = st.slider(
        "Sensibilidade do platô (menor = platô mais estreito)", 0.01, 0.30, 0.05, step=0.01
    )

# ---- Sidebar: filtro do sinal (menu recolhido) ------------------------------
with st.sidebar.expander("🧹 Filtro do sinal", expanded=False):
    use_filter = st.checkbox(
        "Aplicar filtro passa-baixa (detrend + Butterworth + filtfilt)", value=True
    )
    kinem_cutoff = st.slider("Corte Kinem (Hz)", 0.2, 10.0, 1.0, step=0.1)
    imu_cutoff = st.slider("Corte ACC/GYR (Hz)", 0.5, 10.0, 1.0, step=0.5)
    filter_order = st.slider("Ordem do filtro", 2, 8, 4)

if use_filter:
    sheets = {name: filter_dataframe(df, kinem_cutoff, imu_cutoff, filter_order) for name, df in sheets_raw.items()}
    st.sidebar.caption(
        f"Filtro ativo: Kinem {kinem_cutoff:.1f} Hz, ACC/GYR {imu_cutoff:.1f} Hz, ordem {filter_order} "
        f"(Butterworth passa-baixa, zero-fase)."
    )
else:
    sheets = sheets_raw
    st.sidebar.caption("Filtro desativado — usando sinal bruto.")

ref_df = sheets[ref_sheet]
ref_cols = list(ref_df.columns[1:])

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

st.caption(
    "ℹ️ Correção aplicada em todas as abas: o giroscópio (GYR_X/Y/Z) do celular sai em "
    "radianos/segundo (padrão do sensor do Android/iOS) e foi convertido pra graus/segundo "
    "(°/s) — os valores brutos eram pequenos demais (pico de ~2,5 em vez de ~145) pra serem "
    "°/s durante um movimento como esse. Todos os gráficos e cálculos abaixo já usam °/s."
)

if body_sheet == "Joelho":
    st.caption(
        "ℹ️ Correção aplicada: ACC_X (AP) e ACC_Y (Vertical) do Joelho tiveram o sinal "
        "invertido antes de qualquer gráfico/cálculo — validado como erro sistemático de "
        "montagem/calibração em 2 gravações distintas (mesma forma de onda, sinal trocado "
        "em relação à cinemática). ACC_Z (ML) não foi alterado."
    )

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
# Sem limite de largura: célula sempre no mesmo tamanho (300px), e se não couber
# na tela o Streamlit mostra barra de rolagem horizontal em vez de encolher.
CELL_PX = 300
MARGIN = dict(l=10, r=10, t=95, b=10)
H_SPACING = 0.06
V_SPACING = 0.12
# Legenda sempre no topo-ESQUERDA da figura inteira (não no topo-direita, que é o
# padrão do plotly) — assim ela cai dentro da janela inicialmente visível mesmo em
# figuras muito largas com rolagem horizontal (senão fica "escondida" lá na direita).
LEGEND_TOP_LEFT = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)


def square_fig_size(rows, cols):
    col_frac = (1 - H_SPACING * (cols - 1)) / cols
    row_frac = (1 - V_SPACING * (rows - 1)) / rows if rows > 1 else 1.0
    plot_w = CELL_PX / col_frac
    plot_h = CELL_PX / row_frac
    width = int(round(plot_w)) + MARGIN["l"] + MARGIN["r"]
    height = int(round(plot_h)) + MARGIN["t"] + MARGIN["b"]
    return width, height


def render_scrollable(fig, total_width, total_height, visible_width):
    """Mostra a figura numa janela mais estreita (visible_width) com barra de
    rolagem horizontal própria, em vez de deixar as colunas extras cortadas ou
    depender da rolagem da página inteira."""
    html = fig.to_html(include_plotlyjs="cdn", full_html=False)
    wrapped = (
        f'<div style="width:{visible_width}px; overflow-x:auto; overflow-y:hidden; '
        f'border:1px solid #eee; border-radius:4px;">'
        f'<div style="width:{total_width}px;">{html}</div></div>'
    )
    components.html(wrapped, height=total_height + 25, scrolling=False)


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
fig_kinem.update_layout(width=_kw, height=_kh, margin=MARGIN, plot_bgcolor="white", legend=LEGEND_TOP_LEFT)
st.plotly_chart(fig_kinem, use_container_width=False, key="kinem_chart")

# ---- Padrão de deslocamento por trial (fase de descida): direção anatômica ---
st.subheader(f"📐 {body_sheet} — Padrão de deslocamento na descida (Vertical / AP / ML)")
st.caption(
    "Deslocamento líquido (posição no fim da descida − posição no início da descida) em "
    "cada direção anatômica. Convenção: Vertical negativo = desce, positivo = sobe; AP "
    "positivo = anterior, negativo = posterior; ML positivo = lateral, negativo = medial. "
    "Repetido em todos os trials → padrão consistente (não é ruído); CV baixo = repetições "
    "parecidas entre si."
)

DISP_DIRECTION_WORDS = {
    "Vertical": {"pos": "sobe", "neg": "desce"},
    "AP": {"pos": "anterior", "neg": "posterior"},
    "ML": {"pos": "lateral", "neg": "medial"},
}

pos_catalog = catalog.get("Cinemática - Posição", {})
disp_rows = []
for trial_idx in range(1, n_trials + 1):
    _, d_start, v_trial, _ = trial_bounds(trial_idx)
    mask_desc = (df_t >= d_start) & (df_t <= v_trial)
    row = {"Trial": trial_idx}
    for axis in AXES:
        colname = pos_catalog.get(axis)
        if colname is None:
            continue
        direction = KINEM_AXIS_LABEL[axis]
        sig = df[colname].to_numpy()[mask_desc]
        if len(sig) < 2:
            continue
        net = float(sig[-1] - sig[0])
        word = DISP_DIRECTION_WORDS[direction]["pos"] if net >= 0 else DISP_DIRECTION_WORDS[direction]["neg"]
        row[f"Δ {direction}"] = round(net, 4)
        row[f"{direction} (direção)"] = word
    disp_rows.append(row)

if disp_rows:
    disp_df = pd.DataFrame(disp_rows).set_index("Trial")
    st.dataframe(disp_df, use_container_width=True)

    summary_rows = []
    for direction in ["Vertical", "AP", "ML"]:
        col = f"Δ {direction}"
        if col not in disp_df.columns:
            continue
        vals = disp_df[col].to_numpy(dtype=float)
        mean_v = vals.mean()
        std_v = vals.std()
        cv = (100 * std_v / abs(mean_v)) if mean_v != 0 else float("nan")
        summary_rows.append({"Direção": direction, "Média": round(mean_v, 4), "Desvio": round(std_v, 4), "CV (%)": round(cv, 1)})
    st.caption("Consistência entre trials (quanto menor o CV, mais repetido o padrão):")
    st.dataframe(pd.DataFrame(summary_rows).set_index("Direção"), use_container_width=True)

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
fig_imu.update_layout(width=_iw, height=_ih, margin=MARGIN, plot_bgcolor="white", legend=LEGEND_TOP_LEFT)
st.caption(f"Mostrando 3 trials por vez ({_kw}px) — arraste a barra de rolagem abaixo do gráfico para ver os demais.")
render_scrollable(fig_imu, _iw, _ih, visible_width=_kw)

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


# Divisão de fases (platô/descida/subida) para os gráficos de média — como cada
# trial normaliza o próprio ciclo (0–1) mas a duração de cada fase varia um pouco
# de trial a trial, usamos a média das frações de cada fase entre os trials para
# desenhar uma única divisão representativa em todos os gráficos de resultante.
def average_phase_fracs():
    d_fracs, v_fracs = [], []
    for trial_idx in range(1, n_trials + 1):
        cycle_start, d_start, v_trial, cycle_end = trial_bounds(trial_idx)
        span = cycle_end - cycle_start
        if span <= 0:
            continue
        d_fracs.append((d_start - cycle_start) / span)
        v_fracs.append((v_trial - cycle_start) / span)
    if not d_fracs:
        return 0.0, 0.5
    return float(np.mean(d_fracs)), float(np.mean(v_fracs))


AVG_D_FRAC, AVG_V_FRAC = average_phase_fracs()


def add_avg_phase_shading(fig, row, col):
    if AVG_D_FRAC > 0:
        fig.add_vrect(x0=0, x1=AVG_D_FRAC, fillcolor=PLATEAU_COLOR, line_width=0, layer="below", row=row, col=col)
    fig.add_vrect(x0=AVG_D_FRAC, x1=AVG_V_FRAC, fillcolor=DESCIDA_COLOR, line_width=0, layer="below", row=row, col=col)
    fig.add_vrect(x0=AVG_V_FRAC, x1=1.0, fillcolor=SUBIDA_COLOR, line_width=0, layer="below", row=row, col=col)


def add_avg_event_lines(fig, row, col):
    fig.add_vline(x=AVG_D_FRAC, line_dash="dash", line_color="#1f77b4", opacity=0.9, row=row, col=col)
    fig.add_vline(x=AVG_V_FRAC, line_dash="dot", line_color="orange", opacity=0.9, row=row, col=col)
    fig.add_vline(x=1.0, line_dash="dash", line_color="#2ca02c", opacity=0.9, row=row, col=col)


# Layout 2 linhas x 3 colunas: Velocidade Angular (GYR) embaixo de Velocidade,
# Aceleração Linear (ACC) embaixo de Aceleração. Deslocamento não tem par no IMU.
AVG_GRID = [
    [
        (KINEM_LABEL_MAP["Posição"], KINEM_GROUP_MAP["Posição"], KINEM_UNIT_MAP["Posição"]),
        (KINEM_LABEL_MAP["Velocidade"], KINEM_GROUP_MAP["Velocidade"], KINEM_UNIT_MAP["Velocidade"]),
        (KINEM_LABEL_MAP["Aceleração"], KINEM_GROUP_MAP["Aceleração"], KINEM_UNIT_MAP["Aceleração"]),
    ],
    [
        None,
        (gyr_label, "IMU - Giroscópio", gyr_unit),
        (acc_label, "IMU - Acelerômetro", acc_unit),
    ],
]
AVG_ROWS, AVG_COLS = len(AVG_GRID), len(AVG_GRID[0])

fig_avg = make_subplots(
    rows=AVG_ROWS, cols=AVG_COLS,
    subplot_titles=[
        (f"{cell[0]} (X, Y, Z)" if cell else "Orientação do sensor")
        for row in AVG_GRID for cell in row
    ],
    shared_xaxes=True, horizontal_spacing=H_SPACING, vertical_spacing=V_SPACING,
)

for row_i, row in enumerate(AVG_GRID, start=1):
    for col_i, cell in enumerate(row, start=1):
        if cell is None:
            # Espaço livre (Deslocamento não tem par de IMU) — fica em branco; a
            # imagem de orientação do celular agora está na barra lateral, maior.
            fig_avg.update_xaxes(visible=False, showgrid=False, row=row_i, col=col_i)
            fig_avg.update_yaxes(visible=False, showgrid=False, row=row_i, col=col_i)
            continue
        label, grp, unit = cell
        is_kinem = grp.startswith("Cinemática")
        has_trace = False
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
                row=row_i, col=col_i,
            )
            # Cor = direção anatômica (Vertical/AP/ML), consistente entre Kinem e IMU,
            # então 1 legenda só (pela direção) já vale pro gráfico inteiro.
            fig_avg.add_trace(
                go.Scatter(
                    x=GRID, y=mean_y, mode="lines", line=dict(color=color),
                    name=direction, showlegend=(row_i == 1 and col_i == 1), legendgroup=direction,
                ),
                row=row_i, col=col_i,
            )
            has_trace = True
        if has_trace:
            # IMPORTANTE: o traço precisa existir ANTES do add_vrect/add_vline com row/col.
            add_avg_phase_shading(fig_avg, row_i, col_i)
            add_avg_event_lines(fig_avg, row_i, col_i)
        fig_avg.update_yaxes(title_text=f"{label} ({unit})", row=row_i, col=col_i)

fig_avg.update_xaxes(showgrid=False, range=[0, 1], title_text="Fração do ciclo (0–1)")
fig_avg.update_yaxes(showgrid=False)
_aw, _ah = square_fig_size(AVG_ROWS, AVG_COLS)
fig_avg.update_layout(width=_aw, height=_ah, margin=MARGIN, plot_bgcolor="white", legend=LEGEND_TOP_LEFT)
st.plotly_chart(fig_avg, use_container_width=False, key="avg_chart")

st.divider()

ALPHA_COMP = 0.96  # peso do giroscópio no filtro complementar (perto de 1 = confia + no giro)
_TILT_LIGHT_CUTOFF_HZ = 5.0  # filtro leve (sem detrend) só p/ tirar ruído, preservando a gravidade


def _light_lowpass(sig, cutoff_hz, fs, order=2):
    b, a = _butter_lowpass(cutoff_hz, fs, order)
    min_len = 3 * max(len(a), len(b))
    return filtfilt(b, a, sig) if len(sig) > min_len else sig


# ---- L5 + Joelho juntos — comparação da inclinação frontal e sagital -------
# As duas montagens do celular são perpendiculares entre si (Y = Vertical nas duas, mas
# X/Z trocam de papel entre AP e ML) — por isso cada região usa o eixo bruto de giroscópio
# correto pra ela (ver IMU_AXIS_LABEL_L5 / IMU_AXIS_LABEL_JOELHO). Calculamos as duas juntas
# (independente da região selecionada acima) e sobrepomos no mesmo gráfico, lado a lado.
st.subheader("🔗 L5 + Joelho — inclinação frontal e sagital, comparadas")
st.caption(
    "As duas curvas no mesmo eixo de tempo normalizado, pra ver a relação entre o tronco/pelve "
    "(L5) e o joelho durante a descida — por exemplo, se a queda pélvica de um lado acompanha "
    "(ou não) o valgo dinâmico do joelho, ou se o tronco inclina pra frente junto com o joelho "
    "avançando sobre o pé. Não é o ângulo do joelho (isso precisaria de 2 sensores no mesmo "
    "segmento) — é o quanto cada ponto onde o celular está preso tomba (medial/lateral, ou "
    "anterior/posterior) em relação à vertical, estimado por filtro complementar ACC + GYR."
)


def _explicacao_frontal():
    st.markdown(
        f"""
**De qual articulação / movimento é esse ângulo, exatamente:**

Não é um ângulo articular (não é "quanto o joelho dobrou" nem um ângulo entre coxa e perna
— isso exigiria 2 sensores, um em cada segmento, pra comparar a orientação de um contra o
outro). É a **inclinação do próprio celular** (e do pedaço de corpo onde ele está preso — L5
ou coxa/perna, na altura do joelho) **em relação à vertical**, olhando só o plano frontal
(o plano de "de frente pro corpo", que separa lado direito de esquerdo — por isso "ML":
Medial/Lateral). Em outras palavras: o quanto aquele ponto do corpo tomba pro lado (pra dentro
= medial, ou pra fora = lateral) durante o movimento, comparado a como ele estava no começo do
ciclo. É um proxy de valgo/varo dinâmico **local**, não a medida clínica completa (que usaria
2 segmentos) — mas segue o mesmo raciocínio: se o ponto perto do joelho está tombando bastante
pra dentro durante a descida, é sinal de valgo dinâmico ali.

**Por que não dá pra usar só o acelerômetro, nem só o giroscópio:**

- O **acelerômetro** sozinho consegue estimar a inclinação do celular (e do segmento onde ele
  está preso) em relação à vertical, porque em repouso ele mede o vetor gravidade: se o celular
  está na vertical, toda a gravidade aparece no eixo Vertical; se ele inclina pro lado, parte
  dessa gravidade "vaza" pro eixo ML. O ângulo sai de `arctan(ACC_ML / ACC_Vertical)`. O
  problema: isso só é confiável quando o segmento está **parado ou se movendo devagar** —
  durante a descida em si (movimento rápido), o acelerômetro também sente a aceleração do
  próprio movimento, misturada com a gravidade, e o ângulo calculado fica errado (picos falsos).
- O **giroscópio** sozinho mede velocidade angular (°/s) e dá pra integrar no tempo pra virar
  ângulo. Isso funciona bem durante o movimento rápido (sem o problema acima), mas tem um defeito
  conhecido: qualquer pequeno erro de leitura vai se acumulando a cada instante da integração, e
  o ângulo **desvia (drift)** com o tempo — depois de alguns segundos já não representa mais o
  ângulo real.
- O **filtro complementar** combina os dois: usa o giroscópio pra seguir os movimentos rápidos
  com precisão (sem atraso), e deixa o acelerômetro "puxar de volta" bem devagar qualquer desvio
  acumulado, funcionando como uma âncora de longo prazo. Fórmula aplicada a cada instante *i*:

  `ângulo[i] = α × (ângulo[i-1] + giro[i] × Δt) + (1 − α) × ângulo_acelerômetro[i]`

  com α perto de 1 (aqui α = {ALPHA_COMP:.2f}) — ou seja, confia quase todo no giroscópio a cada
  passo, mas puxa levemente pro valor do acelerômetro o suficiente pra não acumular erro.
- É por isso que o **momento quase parado no fundo do agachamento** é tão útil: é exatamente ali
  que o acelerômetro sozinho já é confiável (pouca aceleração de movimento, quase só gravidade),
  então o filtro complementar tem uma "âncora" boa bem no ponto que mais importa clinicamente (o
  pico de inclinação / valgo).
- O eixo do giroscópio usado é sempre o que corresponde à rotação em torno do eixo
  Anteroposterior (AP) do celular naquela região — é essa rotação que mistura Vertical e ML, ou
  seja, é ela que "sente" o segmento inclinando pro lado. Como a orientação física do celular no
  corpo muda entre L5 e Joelho (são perpendiculares entre si), esse eixo bruto de giroscópio
  (X, Y ou Z) também muda — o app escolhe automaticamente o eixo correto pra cada região.
- O ângulo é sempre calculado **em relação ao início de cada ciclo** (começa em 0°), pra remover
  qualquer desvio fixo de como o celular foi colocado — o que importa aqui é a **variação** de
  inclinação durante o movimento, não um ângulo anatômico absoluto calibrado. Positivo = lateral,
  negativo = medial (mesma convenção do ML).
- Esse cálculo usa o sinal **bruto** de ACC/GYR (não o filtrado/detrend da barra lateral), porque
  detrend removeria justamente o componente de gravidade que a estimativa de ângulo precisa.
- Se o celular exportar **aceleração linear** (gravidade já removida pelo próprio sensor/app,
  em vez do acelerômetro bruto), não existe componente de gravidade nenhum pra usar como âncora
  — isso não dá pra corrigir por cálculo depois. Quando o app detecta essa situação (magnitude
  do vetor ACC muito abaixo do esperado pra gravidade), ele troca sozinho, automaticamente, para
  uma estimativa só por integração do giroscópio (reiniciada em 0° a cada ciclo). Fica mais
  sujeita a desvio (drift), mas como cada ciclo dura só alguns segundos, ainda é uma estimativa
  utilizável — só não tem a correção extra que a gravidade daria.

**Por que esse ângulo tende a ser MENOR que o ângulo articular real:**

- É a inclinação de **1 segmento só** em relação à vertical, não a diferença relativa entre
  2 segmentos (o que seria o ângulo articular de verdade — ex.: coxa vs perna, ou o
  quadril-joelho-tornozelo usado na avaliação clínica em vídeo). Quando os dois segmentos se
  movem em direções diferentes (comum no valgo dinâmico — quadril aduzindo enquanto o pé fica
  fixo no chão), o ângulo articular total soma as duas contribuições e costuma ficar maior do
  que a inclinação de qualquer um dos segmentos isolados.
- O ângulo é zerado no início de cada ciclo — qualquer inclinação que já existisse **antes**
  da descida (ex.: um pequeno desvio postural de base) não entra na conta. Um goniômetro ou
  marcador óptico mediria o ângulo total desde uma posição neutra; aqui só medimos a
  **variação** durante o movimento.
- Quando falta o componente de gravidade e o app cai no modo só-giroscópio, a integração
  tende a ficar mais conservadora, e o filtro leve (5 Hz) também atenua picos rápidos —
  isso pode empurrar a estimativa ainda mais pra baixo.
- Na prática: trate os valores como um indicador **relativo** (bom pra comparar repetição
  com repetição, sessão com sessão, ou lado com lado), não como substituto de uma medição
  goniométrica ou de vídeo 2D calibrada — se o gráfico mostra 5°, o ângulo articular real
  provavelmente é maior que isso.
"""
    )


def _explicacao_sagital():
    st.markdown(
        f"""
A lógica é idêntica à da inclinação frontal (ML) — só troca qual eixo entra em cada papel:

- **Acelerômetro:** em vez de `arctan(ACC_ML / ACC_Vertical)`, aqui é
  `arctan(ACC_AP / ACC_Vertical)` — o quanto a gravidade "vaza" pro eixo Anteroposterior (AP)
  em vez do Mediolateral (ML).
- **Giroscópio:** a rotação que inclina o segmento pra frente/trás é em torno do eixo
  Mediolateral (ML) — é essa rotação que mistura Vertical e AP (o oposto da inclinação
  frontal, onde a rotação relevante é em torno do AP). O app escolhe automaticamente o eixo
  bruto certo (X, Y ou Z) pra cada região, do mesmo jeito que faz pra inclinação frontal.
- O resto é igual: filtro complementar (α = {ALPHA_COMP:.2f}), ângulo relativo ao início de
  cada ciclo, e troca automática pra giroscópio puro quando falta o componente de gravidade
  no ACC (mesma checagem de magnitude). A mesma ressalva da inclinação frontal vale aqui: é
  a inclinação de 1 segmento só, tende a ser **menor** que o ângulo articular real (ex.: o
  ângulo verdadeiro de flexão/extensão do joelho), e serve melhor como indicador relativo
  entre repetições/sessões do que como valor anatômico absoluto.
"""
    )


def compute_tilt_curve_for_region(region_name):
    if region_name not in sheets or region_name not in sheets_raw:
        return None
    _df_r = sheets[region_name]
    _catalog_r = build_catalog(_df_r)
    _imu_axis_r = get_imu_axis_label(region_name)
    _ap_r = next((ax for ax in AXES if _imu_axis_r[ax] == "AP"), None)
    _ml_r = next((ax for ax in AXES if _imu_axis_r[ax] == "ML"), None)
    _vert_r = next((ax for ax in AXES if _imu_axis_r[ax] == "Vertical"), None)
    gyr_ap_c = _catalog_r.get("IMU - Giroscópio", {}).get(_ap_r) if _ap_r else None
    acc_ml_c = _catalog_r.get("IMU - Acelerômetro", {}).get(_ml_r) if _ml_r else None
    acc_vert_c = _catalog_r.get("IMU - Acelerômetro", {}).get(_vert_r) if _vert_r else None
    if not (gyr_ap_c and acc_ml_c and acc_vert_c):
        return None

    _raw_r = sheets_raw[region_name]
    _t_r = _df_r[time_column(_df_r)].to_numpy()
    _dt_r = float(np.median(np.diff(_t_r)))
    _fs_r = 1.0 / _dt_r if _dt_r > 0 else 100.0

    acc_ml_f = _light_lowpass(_raw_r[acc_ml_c].to_numpy(dtype=float), _TILT_LIGHT_CUTOFF_HZ, _fs_r)
    acc_vert_f = _light_lowpass(_raw_r[acc_vert_c].to_numpy(dtype=float), _TILT_LIGHT_CUTOFF_HZ, _fs_r)
    gyr_ap_f = _light_lowpass(_raw_r[gyr_ap_c].to_numpy(dtype=float), _TILT_LIGHT_CUTOFF_HZ, _fs_r)
    theta_acc_f = np.degrees(np.arctan2(acc_ml_f, acc_vert_f))
    grav_mag_r = float(np.median(np.sqrt(acc_ml_f**2 + acc_vert_f**2)))
    use_anchor_r = grav_mag_r >= 3.0

    curves_r = []
    for trial_idx in range(1, n_trials + 1):
        cycle_start, d_start, v_trial, cycle_end = trial_bounds(trial_idx)
        norm_t, _, _ = make_helpers(cycle_start, d_start, v_trial, cycle_end)
        trial_mask_r = (_t_r >= cycle_start) & (_t_r <= cycle_end)
        if trial_mask_r.sum() < 3:
            continue
        theta_acc = theta_acc_f[trial_mask_r] - theta_acc_f[trial_mask_r][0]
        gyr_ap = gyr_ap_f[trial_mask_r]
        theta = np.zeros(len(theta_acc))
        for i in range(1, len(theta)):
            theta_gyro = theta[i - 1] + gyr_ap[i] * _dt_r
            theta[i] = ALPHA_COMP * theta_gyro + (1 - ALPHA_COMP) * theta_acc[i] if use_anchor_r else theta_gyro
        x_trial = norm_t(_t_r[trial_mask_r])
        oi = np.argsort(x_trial)
        curves_r.append(np.interp(GRID, x_trial[oi], theta[oi]))

    if not curves_r:
        return None
    arr_r = np.vstack(curves_r)
    return {
        "mean": arr_r.mean(axis=0), "std": arr_r.std(axis=0), "n": len(curves_r),
        "grav_mag": grav_mag_r, "use_anchor": use_anchor_r,
    }


REGION_COMPARE_COLORS = {"L5": "#1f77b4", "Joelho": "#d62728"}


def _build_combo_figure(results, y_title, chart_title):
    fig = go.Figure()
    for region, res in results.items():
        if res is None:
            continue
        color = REGION_COMPARE_COLORS.get(region, "#7f7f7f")
        m, s = res["mean"], res["std"]
        fig.add_trace(go.Scatter(
            x=np.concatenate([GRID, GRID[::-1]]), y=np.concatenate([m + s, (m - s)[::-1]]),
            fill="toself", fillcolor=hex_to_rgba(color, 0.15),
            line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False,
        ))
        anchor_note = "" if res["use_anchor"] else " (só giro, sem âncora)"
        fig.add_trace(go.Scatter(
            x=GRID, y=m, mode="lines", line=dict(color=color, width=2.5),
            name=f"{region}{anchor_note}",
        ))
    if AVG_D_FRAC > 0:
        fig.add_vrect(x0=0, x1=AVG_D_FRAC, fillcolor=PLATEAU_COLOR, line_width=0, layer="below")
    fig.add_vrect(x0=AVG_D_FRAC, x1=AVG_V_FRAC, fillcolor=DESCIDA_COLOR, line_width=0, layer="below")
    fig.add_vrect(x0=AVG_V_FRAC, x1=1.0, fillcolor=SUBIDA_COLOR, line_width=0, layer="below")
    fig.update_xaxes(showgrid=False, range=[0, 1], title_text="Fração do ciclo (0–1)")
    fig.update_yaxes(showgrid=False, title_text=y_title)
    fig.update_layout(
        title=chart_title, height=420, margin=dict(l=55, r=20, t=70, b=50),
        plot_bgcolor="white", legend=LEGEND_TOP_LEFT,
    )
    return fig


def compute_tilt_curve_for_region_ap(region_name):
    if region_name not in sheets or region_name not in sheets_raw:
        return None
    _df_r = sheets[region_name]
    _catalog_r = build_catalog(_df_r)
    _imu_axis_r = get_imu_axis_label(region_name)
    _ap_r = next((ax for ax in AXES if _imu_axis_r[ax] == "AP"), None)
    _ml_r = next((ax for ax in AXES if _imu_axis_r[ax] == "ML"), None)
    _vert_r = next((ax for ax in AXES if _imu_axis_r[ax] == "Vertical"), None)
    gyr_ml_c = _catalog_r.get("IMU - Giroscópio", {}).get(_ml_r) if _ml_r else None
    acc_ap_c = _catalog_r.get("IMU - Acelerômetro", {}).get(_ap_r) if _ap_r else None
    acc_vert_c = _catalog_r.get("IMU - Acelerômetro", {}).get(_vert_r) if _vert_r else None
    if not (gyr_ml_c and acc_ap_c and acc_vert_c):
        return None

    _raw_r = sheets_raw[region_name]
    _t_r = _df_r[time_column(_df_r)].to_numpy()
    _dt_r = float(np.median(np.diff(_t_r)))
    _fs_r = 1.0 / _dt_r if _dt_r > 0 else 100.0

    acc_ap_f = _light_lowpass(_raw_r[acc_ap_c].to_numpy(dtype=float), _TILT_LIGHT_CUTOFF_HZ, _fs_r)
    acc_vert_f = _light_lowpass(_raw_r[acc_vert_c].to_numpy(dtype=float), _TILT_LIGHT_CUTOFF_HZ, _fs_r)
    gyr_ml_f = _light_lowpass(_raw_r[gyr_ml_c].to_numpy(dtype=float), _TILT_LIGHT_CUTOFF_HZ, _fs_r)
    theta_acc_f = np.degrees(np.arctan2(acc_ap_f, acc_vert_f))
    grav_mag_r = float(np.median(np.sqrt(acc_ap_f**2 + acc_vert_f**2)))
    use_anchor_r = grav_mag_r >= 3.0

    curves_r = []
    for trial_idx in range(1, n_trials + 1):
        cycle_start, d_start, v_trial, cycle_end = trial_bounds(trial_idx)
        norm_t, _, _ = make_helpers(cycle_start, d_start, v_trial, cycle_end)
        trial_mask_r = (_t_r >= cycle_start) & (_t_r <= cycle_end)
        if trial_mask_r.sum() < 3:
            continue
        theta_acc = theta_acc_f[trial_mask_r] - theta_acc_f[trial_mask_r][0]
        gyr_ml = gyr_ml_f[trial_mask_r]
        theta = np.zeros(len(theta_acc))
        for i in range(1, len(theta)):
            theta_gyro = theta[i - 1] + gyr_ml[i] * _dt_r
            theta[i] = ALPHA_COMP * theta_gyro + (1 - ALPHA_COMP) * theta_acc[i] if use_anchor_r else theta_gyro
        x_trial = norm_t(_t_r[trial_mask_r])
        oi = np.argsort(x_trial)
        curves_r.append(np.interp(GRID, x_trial[oi], theta[oi]))

    if not curves_r:
        return None
    arr_r = np.vstack(curves_r)
    return {
        "mean": arr_r.mean(axis=0), "std": arr_r.std(axis=0), "n": len(curves_r),
        "grav_mag": grav_mag_r, "use_anchor": use_anchor_r,
    }


_combo_results = {region: compute_tilt_curve_for_region(region) for region in ("L5", "Joelho") if region in sheet_names}
_combo_ap_results = {region: compute_tilt_curve_for_region_ap(region) for region in ("L5", "Joelho") if region in sheet_names}

col_frontal, col_sagital = st.columns(2)

with col_frontal:
    if all(_combo_results.get(r) for r in ("L5", "Joelho") if r in sheet_names):
        fig_combo = _build_combo_figure(
            _combo_results,
            "Δ ângulo (°) — positivo = lateral, negativo = medial",
            "Inclinação frontal (ML)",
        )
        st.plotly_chart(fig_combo, use_container_width=True, key="tilt_combo_chart")
    else:
        st.caption("Não foi possível calcular a comparação frontal — faltam colunas de ACC/GYR em uma das duas abas.")

with col_sagital:
    if all(_combo_ap_results.get(r) for r in ("L5", "Joelho") if r in sheet_names):
        fig_combo_ap = _build_combo_figure(
            _combo_ap_results,
            "Δ ângulo (°) — positivo = anterior, negativo = posterior",
            "Inclinação sagital (AP)",
        )
        st.plotly_chart(fig_combo_ap, use_container_width=True, key="tilt_combo_ap_chart")
    else:
        st.caption("Não foi possível calcular a comparação sagital — faltam colunas de ACC/GYR em uma das duas abas.")

with st.expander("O que é o ângulo frontal (ML), e como ele é calculado? (clique para abrir)", expanded=False):
    _explicacao_frontal()

with st.expander("E o ângulo sagital (AP) — em que muda? (clique para abrir)", expanded=False):
    _explicacao_sagital()
