"""
Dropdown Analysis - Comparação entre duas pessoas (L5 x Joelho, A x B)
=======================================================================
App Streamlit derivado de app.py: carrega DOIS arquivos .xlsx (sinais sincronizados
de duas pessoas diferentes), segmenta os ciclos de cada uma independentemente e
sobrepõe, no MESMO gráfico, as 4 combinações possíveis:

    Pessoa A - L5      Pessoa A - Joelho
    Pessoa B - L5      Pessoa B - Joelho

Assim dá pra ver ao mesmo tempo (1) a diferença entre as pessoas (A x B) e (2) a
diferença entre as regiões dentro de cada pessoa (L5 x Joelho de A, comparada com
L5 x Joelho de B) — cor = pessoa, tracejado = região (L5 sólido, Joelho tracejado).

Cada curva é a RESULTANTE entre os ciclos daquela pessoa/região: média ± desvio
padrão, tempo normalizado (0-1) por ciclo. A segmentação em fases (platô / descida /
subida) aparece em TODOS os gráficos: o fundo colorido (cinza/laranja/verde) usa a
média das fases entre as duas pessoas (referência visual única, pra não conflitar),
e linhas verticais pontilhadas, na cor de cada pessoa, marcam o início da descida e o
vale exatos dela.

Como rodar localmente:
    pip install -r requirements.txt
    streamlit run app_comparacao_pessoas.py
"""

import io
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.signal import butter, detrend, filtfilt, find_peaks

st.set_page_config(page_title="Dropdown Analysis - Comparação entre pessoas", layout="wide")

# ----------------------------------------------------------------------------
# Parsing / categorização de colunas (igual ao app.py)
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


def build_catalog(df: pd.DataFrame):
    catalog = {}
    for col in df.columns[1:]:
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


def _light_lowpass(sig, cutoff_hz, fs, order=2):
    b, a = _butter_lowpass(cutoff_hz, fs, order)
    min_len = 3 * max(len(a), len(b))
    return filtfilt(b, a, sig) if len(sig) > min_len else sig


@st.cache_data(show_spinner=False)
def load_workbook(file_bytes: bytes):
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    return {name: xls.parse(name) for name in xls.sheet_names}


JOELHO_ACC_SIGN_FIX = ("ACC_X", "ACC_Y")
GYR_COLS = ("GYR_X", "GYR_Y", "GYR_Z")


def apply_known_corrections(sheets_raw: dict) -> dict:
    """Mesmas correções conhecidas do app.py: inversão de sinal do ACC no Joelho e
    conversão do giroscópio de rad/s -> °/s em todas as abas."""
    sheets_raw = {k: v.copy() for k, v in sheets_raw.items()}
    if "Joelho" in sheets_raw:
        jo = sheets_raw["Joelho"].copy()
        for col in JOELHO_ACC_SIGN_FIX:
            if col in jo.columns:
                jo[col] = -jo[col]
        sheets_raw["Joelho"] = jo
    for name, df in sheets_raw.items():
        df2 = df.copy()
        for col in GYR_COLS:
            if col in df2.columns:
                df2[col] = np.degrees(df2[col])
        sheets_raw[name] = df2
    return sheets_raw


def find_plateau_edges(is_flat: np.ndarray, idx: int):
    n = len(is_flat)
    left = right = idx
    while left > 0 and is_flat[left - 1]:
        left -= 1
    while right < n - 1 and is_flat[right + 1]:
        right += 1
    return left, right


def detect_cycles(ref_df: pd.DataFrame, ref_col: str, min_distance: int, prominence: float, plateau_frac: float):
    """Extraído de app.py: detecta vales/picos no sinal de referência e devolve os
    marcos de início/fim de cada ciclo (platô + descida + subida)."""
    t = ref_df[time_column(ref_df)].to_numpy()
    ref_signal = ref_df[ref_col].to_numpy(dtype=float)
    n_samples = len(ref_signal)

    valleys_idx, _ = find_peaks(-ref_signal, distance=min_distance, prominence=prominence)
    peaks_idx, _ = find_peaks(ref_signal, distance=min_distance, prominence=prominence)
    valley_times = t[valleys_idx]

    deriv = np.gradient(ref_signal, t)
    max_abs_deriv = np.max(np.abs(deriv)) if n_samples else 1.0
    is_flat = np.abs(deriv) < plateau_frac * (max_abs_deriv if max_abs_deriv > 0 else 1.0)

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

    return np.array(start_times), np.array(end_times), valley_times, t


def make_trial_bounds_fn(sel_starts, sel_ends, valley_times, t0):
    """trial_bounds(i) -> (cycle_start, d_start, v_trial, cycle_end), fechada sobre os
    marcos de UMA pessoa (mesma lógica de app.py)."""
    def trial_bounds(trial_idx):
        cycle_start = sel_ends[trial_idx - 2] if trial_idx > 1 else t0
        d_start = sel_starts[trial_idx - 1]
        cycle_end = sel_ends[trial_idx - 1]
        valley_in_cycle = valley_times[(valley_times > d_start) & (valley_times < cycle_end)]
        v_trial = valley_in_cycle[0] if len(valley_in_cycle) else (d_start + cycle_end) / 2
        return cycle_start, d_start, v_trial, cycle_end
    return trial_bounds


GRID = np.linspace(0.0, 1.0, 101)


def ensemble_mean_std(df, df_t, catalog, trial_bounds_fn, n_trials, grp, axis):
    curves = []
    for trial_idx in range(1, n_trials + 1):
        cycle_start, d_start, v_trial, cycle_end = trial_bounds_fn(trial_idx)
        span = cycle_end - cycle_start
        if span <= 0:
            continue
        trial_mask = (df_t >= cycle_start) & (df_t <= cycle_end)
        colname = catalog.get(grp, {}).get(axis)
        if colname is None:
            continue
        x_trial = (df_t[trial_mask] - cycle_start) / span
        y_trial = df[colname].to_numpy()[trial_mask]
        if len(x_trial) < 2:
            continue
        order = np.argsort(x_trial)
        curves.append(np.interp(GRID, x_trial[order], y_trial[order]))
    if not curves:
        return None, None
    arr = np.vstack(curves)
    return arr.mean(axis=0), arr.std(axis=0)


def average_phase_fracs(trial_bounds_fn, n_trials):
    d_fracs, v_fracs = [], []
    for trial_idx in range(1, n_trials + 1):
        cycle_start, d_start, v_trial, cycle_end = trial_bounds_fn(trial_idx)
        span = cycle_end - cycle_start
        if span <= 0:
            continue
        d_fracs.append((d_start - cycle_start) / span)
        v_fracs.append((v_trial - cycle_start) / span)
    if not d_fracs:
        return 0.0, 0.5
    return float(np.mean(d_fracs)), float(np.mean(v_fracs))


def phase_duration_stats(trial_bounds_fn, n_trials):
    """Duração (em segundos) de cada fase do ciclo, média ± DP entre os ciclos."""
    plato, desc, sub, total = [], [], [], []
    for trial_idx in range(1, n_trials + 1):
        cycle_start, d_start, v_trial, cycle_end = trial_bounds_fn(trial_idx)
        plato.append(d_start - cycle_start)
        desc.append(v_trial - d_start)
        sub.append(cycle_end - v_trial)
        total.append(cycle_end - cycle_start)

    def _stat(vals):
        arr = np.array(vals, dtype=float)
        return (float(arr.mean()), float(arr.std())) if len(arr) else (float("nan"), float("nan"))

    return {
        "platô": _stat(plato), "descida": _stat(desc),
        "subida": _stat(sub), "ciclo total": _stat(total),
    }


def net_displacement_stats(df, df_t, catalog, trial_bounds_fn, n_trials, direction):
    """Deslocamento líquido (posição final - inicial da descida) numa direção
    anatômica (Vertical/AP/ML) — média ± DP e CV(%) entre os ciclos. Usado tanto pra
    'profundidade' (Vertical) quanto pra consistência/assimetria entre repetições."""
    axis = next((ax for ax, d in KINEM_AXIS_LABEL.items() if d == direction), None)
    colname = catalog.get("Cinemática - Posição", {}).get(axis) if axis else None
    if colname is None:
        return None
    deltas = []
    for trial_idx in range(1, n_trials + 1):
        cycle_start, d_start, v_trial, cycle_end = trial_bounds_fn(trial_idx)
        mask = (df_t >= d_start) & (df_t <= v_trial)
        sig = df[colname].to_numpy()[mask]
        if len(sig) < 2:
            continue
        deltas.append(float(sig[-1] - sig[0]))
    if not deltas:
        return None
    arr = np.array(deltas)
    mean_v, std_v = float(arr.mean()), float(arr.std())
    cv = (100 * std_v / abs(mean_v)) if mean_v != 0 else float("nan")
    return {"mean": mean_v, "std": std_v, "cv": cv, "n": len(arr)}


def phase_amplitude_stats(df, df_t, catalog, trial_bounds_fn, n_trials, grp):
    """Amplitude (pico-a-pico da resultante/norma vetorial dos eixos X/Y/Z
    disponíveis) de uma variável (grp), calculada separadamente em cada fase do
    ciclo (platô/descida/subida) — média ± DP entre os ciclos."""
    cols = catalog.get(grp, {})
    axes_present = [ax for ax in ("X", "Y", "Z") if ax in cols]
    if not axes_present:
        return None
    phases = {"platô": [], "descida": [], "subida": []}
    for trial_idx in range(1, n_trials + 1):
        cycle_start, d_start, v_trial, cycle_end = trial_bounds_fn(trial_idx)
        bounds = {"platô": (cycle_start, d_start), "descida": (d_start, v_trial), "subida": (v_trial, cycle_end)}
        for phase_name, (t0, t1) in bounds.items():
            if t1 <= t0:
                continue
            mask = (df_t >= t0) & (df_t <= t1)
            if mask.sum() < 2:
                continue
            vecs = np.vstack([df[cols[ax]].to_numpy()[mask] for ax in axes_present])
            resultant = np.sqrt((vecs ** 2).sum(axis=0))
            phases[phase_name].append(float(resultant.max() - resultant.min()))
    out = {}
    for phase_name, vals in phases.items():
        if vals:
            arr = np.array(vals)
            out[phase_name] = (float(arr.mean()), float(arr.std()))
        else:
            out[phase_name] = (float("nan"), float("nan"))
    return out


DESCIDA_COLOR = "rgba(255,127,14,0.18)"
SUBIDA_COLOR = "rgba(44,160,44,0.18)"
PLATEAU_COLOR = "rgba(150,150,150,0.25)"

KINEM_AXIS_LABEL = {"X": "ML", "Y": "AP", "Z": "Vertical"}
IMU_AXIS_LABEL_JOELHO = {"X": "AP", "Y": "Vertical", "Z": "ML"}
IMU_AXIS_LABEL_L5 = {"X": "ML", "Y": "Vertical", "Z": "AP"}


def get_imu_axis_label(region_name):
    return IMU_AXIS_LABEL_L5 if "l5" in region_name.lower() else IMU_AXIS_LABEL_JOELHO


def axis_direction(is_kinem, axis, imu_axis_label):
    mapping = KINEM_AXIS_LABEL if is_kinem else imu_axis_label
    return mapping[axis]


def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


AXES = ["X", "Y", "Z"]
SIGNAL_ROWS = [
    ("Cinemática - Posição", ("Deslocamento", "cm"), True),
    ("Cinemática - Velocidade", ("Velocidade", "cm/s"), True),
    ("Cinemática - Aceleração", ("Aceleração", "cm/s²"), True),
    ("IMU - Acelerômetro", ("Aceleração Linear", "m/s²"), False),
    ("IMU - Giroscópio", ("Velocidade Angular", "°/s"), False),
]
DIRECTIONS = ["Vertical", "AP", "ML"]

PERSON_COLORS = {"A": "#1f77b4", "B": "#d62728"}  # A = azul, B = vermelho
REGION_DASH = {"L5": "solid", "Joelho": "dash"}

H_SPACING = 0.06
V_SPACING = 0.10
CELL_PX = 260
MARGIN = dict(l=10, r=10, t=50, b=10)


def square_fig_size(rows, cols):
    col_frac = (1 - H_SPACING * (cols - 1)) / cols
    row_frac = (1 - V_SPACING * (rows - 1)) / rows if rows > 1 else 1.0
    plot_w = CELL_PX / col_frac
    plot_h = CELL_PX / row_frac
    width = int(round(plot_w)) + MARGIN["l"] + MARGIN["r"]
    height = int(round(plot_h)) + MARGIN["t"] + MARGIN["b"]
    return width, height


# ----------------------------------------------------------------------------
# UI — upload
# ----------------------------------------------------------------------------

st.title("🆚 Dropdown Analysis — Comparação entre duas pessoas (L5 × Joelho, A × B)")
st.caption(
    "Carregue os dois arquivos .xlsx (um por pessoa). Cada um é segmentado em ciclos "
    "de forma independente (mesmos parâmetros de referência/filtro para os dois). Os "
    "gráficos mostram, na MESMA figura, as 4 combinações — Pessoa A/L5, Pessoa A/Joelho, "
    "Pessoa B/L5, Pessoa B/Joelho — cor = pessoa, tracejado = região. Cada curva é a "
    "RESULTANTE entre os ciclos (média ± desvio padrão), tempo normalizado (0–1)."
)

col_up1, col_up2 = st.columns(2)
with col_up1:
    file_a = st.file_uploader("Arquivo .xlsx — Pessoa A", type=["xlsx"], key="file_a")
    name_a = st.text_input("Nome da Pessoa A", value="Pessoa A", key="name_a")
with col_up2:
    file_b = st.file_uploader("Arquivo .xlsx — Pessoa B", type=["xlsx"], key="file_b")
    name_b = st.text_input("Nome da Pessoa B", value="Pessoa B", key="name_b")

if file_a is None or file_b is None:
    st.info("Envie os dois arquivos .xlsx (sinais sincronizados) para começar.")
    st.stop()

sheet_names_probe = list(load_workbook(file_a.getvalue()).keys())

with st.sidebar.expander("🔁 Segmentação de ciclos", expanded=True):
    ref_sheet = st.selectbox(
        "Aba de referência (usada só p/ achar os ciclos)", sheet_names_probe,
        index=sheet_names_probe.index("L5") if "L5" in sheet_names_probe else 0,
    )
    min_distance = st.slider("Distância mínima entre marcos (amostras)", 5, 300, 50)
    prominence = st.slider("Proeminência mínima (vales/picos)", 0.0, 2.0, 0.05, step=0.01)
    plateau_frac = st.slider(
        "Sensibilidade do platô (menor = platô mais estreito)", 0.01, 0.30, 0.05, step=0.01
    )

with st.sidebar.expander("🧹 Filtro do sinal", expanded=False):
    use_filter = st.checkbox("Aplicar filtro passa-baixa (detrend + Butterworth + filtfilt)", value=True)
    kinem_cutoff = st.slider("Corte Kinem (Hz)", 0.2, 10.0, 1.0, step=0.1)
    imu_cutoff = st.slider("Corte ACC/GYR (Hz)", 0.5, 10.0, 1.0, step=0.5)
    filter_order = st.slider("Ordem do filtro", 2, 8, 4)


def prepare_person(file_obj):
    sheets_raw = load_workbook(file_obj.getvalue())
    sheets_raw = apply_known_corrections(sheets_raw)
    if use_filter:
        sheets = {n: filter_dataframe(df, kinem_cutoff, imu_cutoff, filter_order) for n, df in sheets_raw.items()}
    else:
        sheets = sheets_raw
    return sheets_raw, sheets


raw_a, sheets_a = prepare_person(file_a)
raw_b, sheets_b = prepare_person(file_b)

if ref_sheet not in sheets_a or ref_sheet not in sheets_b:
    st.error(f"A aba de referência '{ref_sheet}' não existe em um dos dois arquivos.")
    st.stop()

ref_cols_common = [c for c in sheets_a[ref_sheet].columns[1:] if c in sheets_b[ref_sheet].columns[1:]]
default_ref_idx = 2 if len(ref_cols_common) > 2 else 0
ref_col = st.sidebar.selectbox(
    "Coluna de referência (padrão: coluna D)", ref_cols_common, index=default_ref_idx
)

REGIONS = [r for r in ("L5", "Joelho") if r in sheets_a and r in sheets_b]


def build_person_context(key, label, sheets_raw, sheets):
    ref_df = sheets[ref_sheet]
    start_times, end_times, valley_times, t = detect_cycles(
        ref_df, ref_col, min_distance, prominence, plateau_frac
    )
    sel_starts = sorted(start_times.tolist())
    sel_ends = sorted(end_times.tolist())
    n_trials = min(len(sel_starts), len(sel_ends))
    trial_bounds_fn = make_trial_bounds_fn(sel_starts, sel_ends, valley_times, t[0])
    d_frac, v_frac = average_phase_fracs(trial_bounds_fn, n_trials)
    return {
        "key": key, "label": label, "color": PERSON_COLORS[key],
        "sheets_raw": sheets_raw, "sheets": sheets,
        "t": t, "ref_signal": ref_df[ref_col].to_numpy(dtype=float),
        "valley_times": valley_times, "sel_starts": sel_starts, "sel_ends": sel_ends,
        "n_trials": n_trials, "trial_bounds_fn": trial_bounds_fn,
        "d_frac": d_frac, "v_frac": v_frac,
    }


ctx_a = build_person_context("A", name_a or "Pessoa A", raw_a, sheets_a)
ctx_b = build_person_context("B", name_b or "Pessoa B", raw_b, sheets_b)
PERSON_CTXS = (ctx_a, ctx_b)

# d_frac/v_frac só dependem da pessoa (a segmentação vem sempre da mesma aba de
# referência e é aplicada por tempo às duas regiões) — por isso são iguais para L5 e
# Joelho da mesma pessoa. Para desenhar UM sombreado de fase por painel (em vez de 2
# conflitantes), usamos a média entre as duas pessoas como referência visual; as linhas
# pontilhadas coloridas continuam marcando o valor exato de cada pessoa.
OVERALL_D_FRAC = float(np.mean([ctx_a["d_frac"], ctx_b["d_frac"]]))
OVERALL_V_FRAC = float(np.mean([ctx_a["v_frac"], ctx_b["v_frac"]]))


def add_phase_bg(fig_obj, row=None, col=None):
    kwargs = {} if row is None else dict(row=row, col=col)
    if OVERALL_D_FRAC > 0:
        fig_obj.add_vrect(x0=0, x1=OVERALL_D_FRAC, fillcolor=PLATEAU_COLOR, line_width=0, layer="below", **kwargs)
    fig_obj.add_vrect(x0=OVERALL_D_FRAC, x1=OVERALL_V_FRAC, fillcolor=DESCIDA_COLOR, line_width=0, layer="below", **kwargs)
    fig_obj.add_vrect(x0=OVERALL_V_FRAC, x1=1.0, fillcolor=SUBIDA_COLOR, line_width=0, layer="below", **kwargs)

st.sidebar.caption(
    f"{ctx_a['label']}: {ctx_a['n_trials']} ciclo(s) detectado(s) · "
    f"{ctx_b['label']}: {ctx_b['n_trials']} ciclo(s) detectado(s)."
)

# ---- Checagem visual da segmentação de cada pessoa --------------------------
st.subheader("🔁 Checagem da segmentação — sinal de referência por pessoa")
st.caption(
    f"Confira se os ciclos ficaram bem marcados em '{ref_sheet}' / '{ref_col}' antes de "
    "olhar as comparações abaixo. Se a segmentação estiver ruim, ajuste os parâmetros "
    "na barra lateral (valem para as duas pessoas)."
)

fig_check = make_subplots(rows=1, cols=2, subplot_titles=[ctx_a["label"], ctx_b["label"]])
for col_i, ctx in enumerate(PERSON_CTXS, start=1):
    fig_check.add_trace(
        go.Scatter(x=ctx["t"], y=ctx["ref_signal"], mode="lines", line=dict(color=ctx["color"]), showlegend=False),
        row=1, col=col_i,
    )
    for i in range(ctx["n_trials"]):
        s, e = ctx["sel_starts"][i], ctx["sel_ends"][i]
        platform_start = ctx["sel_ends"][i - 1] if i > 0 else ctx["t"][0]
        inside = ctx["valley_times"][(ctx["valley_times"] > s) & (ctx["valley_times"] < e)]
        v = inside[0] if len(inside) else (s + e) / 2
        if platform_start < s:
            fig_check.add_vrect(x0=platform_start, x1=s, fillcolor=PLATEAU_COLOR, line_width=0, layer="below", row=1, col=col_i)
        fig_check.add_vrect(x0=s, x1=v, fillcolor=DESCIDA_COLOR, line_width=0, layer="below", row=1, col=col_i)
        fig_check.add_vrect(x0=v, x1=e, fillcolor=SUBIDA_COLOR, line_width=0, layer="below", row=1, col=col_i)
fig_check.update_xaxes(title_text="Tempo (s)")
fig_check.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="white")
st.plotly_chart(fig_check, use_container_width=True)

if ctx_a["n_trials"] == 0 or ctx_b["n_trials"] == 0:
    st.warning("Não foi possível detectar nenhum ciclo em uma das duas pessoas — ajuste os parâmetros de segmentação.")
    st.stop()

# ---- Bloco de interpretação: duração das fases + profundidade da descida ----
_dur_a = phase_duration_stats(ctx_a["trial_bounds_fn"], ctx_a["n_trials"])
_dur_b = phase_duration_stats(ctx_b["trial_bounds_fn"], ctx_b["n_trials"])


def _fmt_dur(stat):
    mean_v, std_v = stat
    return f"{mean_v:.2f}s (±{std_v:.2f})"


def _fmt_pct_diff(stat_a, stat_b, label_a, label_b):
    a, b = stat_a[0], stat_b[0]
    if a == 0 and b == 0:
        return "sem diferença"
    maior, menor = (a, b) if a >= b else (b, a)
    quem_maior = label_a if a >= b else label_b
    pct = 100 * (maior - menor) / menor if menor != 0 else float("inf")
    return f"{quem_maior} é {pct:.0f}% maior"


_depth_lines = []
for region in REGIONS:
    dep_a = net_displacement_stats(
        ctx_a["sheets"][region], ctx_a["sheets"][region][time_column(ctx_a["sheets"][region])].to_numpy(),
        build_catalog(ctx_a["sheets"][region]), ctx_a["trial_bounds_fn"], ctx_a["n_trials"], "Vertical",
    )
    dep_b = net_displacement_stats(
        ctx_b["sheets"][region], ctx_b["sheets"][region][time_column(ctx_b["sheets"][region])].to_numpy(),
        build_catalog(ctx_b["sheets"][region]), ctx_b["trial_bounds_fn"], ctx_b["n_trials"], "Vertical",
    )
    if dep_a is None or dep_b is None:
        continue
    quem_desce_mais = ctx_a["label"] if abs(dep_a["mean"]) > abs(dep_b["mean"]) else ctx_b["label"]
    _depth_lines.append(
        f"- **{region}**: {ctx_a['label']} = {dep_a['mean']:.3f} (±{dep_a['std']:.3f}), "
        f"{ctx_b['label']} = {dep_b['mean']:.3f} (±{dep_b['std']:.3f}) — {quem_desce_mais} desce mais nessa região."
    )

st.info(
    f"**Duração das fases (média ± DP entre os ciclos, e diferença % entre as pessoas):**\n\n"
    f"- Platô: {ctx_a['label']} {_fmt_dur(_dur_a['platô'])} · {ctx_b['label']} {_fmt_dur(_dur_b['platô'])} "
    f"— {_fmt_pct_diff(_dur_a['platô'], _dur_b['platô'], ctx_a['label'], ctx_b['label'])}\n"
    f"- Descida: {ctx_a['label']} {_fmt_dur(_dur_a['descida'])} · {ctx_b['label']} {_fmt_dur(_dur_b['descida'])} "
    f"— {_fmt_pct_diff(_dur_a['descida'], _dur_b['descida'], ctx_a['label'], ctx_b['label'])}\n"
    f"- Subida: {ctx_a['label']} {_fmt_dur(_dur_a['subida'])} · {ctx_b['label']} {_fmt_dur(_dur_b['subida'])} "
    f"— {_fmt_pct_diff(_dur_a['subida'], _dur_b['subida'], ctx_a['label'], ctx_b['label'])}\n"
    f"- Ciclo total: {ctx_a['label']} {_fmt_dur(_dur_a['ciclo total'])} · {ctx_b['label']} {_fmt_dur(_dur_b['ciclo total'])} "
    f"— {_fmt_pct_diff(_dur_a['ciclo total'], _dur_b['ciclo total'], ctx_a['label'], ctx_b['label'])}\n\n"
    f"**Quanto desce (deslocamento líquido Vertical na descida, unidade da coluna de "
    f"Kinemática — confira se é cm ou m no seu sistema de captura):**\n\n"
    + "\n".join(_depth_lines) +
    "\n\n_Interpretação automática, calculada a partir dos ciclos detectados — não substitui "
    "avaliação clínica._"
)

st.divider()

# ----------------------------------------------------------------------------
# Comparação resultante — A x B e L5 x Joelho, tudo na mesma figura
# ----------------------------------------------------------------------------

st.subheader("📊 Comparação resultante — Pessoa A × Pessoa B e L5 × Joelho")
legend_bits = " / ".join(
    f"{ctx['label']} = {'azul' if ctx['key']=='A' else 'vermelho'}" for ctx in PERSON_CTXS
)
st.caption(
    f"Cor = pessoa ({legend_bits}); tracejado = região (L5 sólido, Joelho tracejado). "
    "Sombra = ±1 desvio padrão entre os ciclos daquela pessoa/região. Fundo cinza = "
    "platô, laranja = descida, verde = subida (usando a média das fases entre as duas "
    "pessoas, já que a divisão de cada uma pode variar um pouco); as linhas verticais "
    "pontilhadas, na cor de cada pessoa, marcam o início da descida e o vale exatos "
    "dela. Cada painel já dá pra ler nos dois sentidos: A×B (compare as cores) e "
    "L5×Joelho (compare sólido×tracejado, dentro da mesma cor)."
)

fig = make_subplots(
    rows=len(SIGNAL_ROWS), cols=len(DIRECTIONS),
    subplot_titles=[f"{lbl} — {d}" for (_, (lbl, _), _) in SIGNAL_ROWS for d in DIRECTIONS],
    shared_xaxes=True, horizontal_spacing=H_SPACING, vertical_spacing=V_SPACING,
)

legend_shown = set()
for row_i, (grp, (label, unit), is_kinem) in enumerate(SIGNAL_ROWS, start=1):
    for col_i, direction in enumerate(DIRECTIONS, start=1):
        for ctx in PERSON_CTXS:
            for region in REGIONS:
                df_r = ctx["sheets"][region]
                catalog_r = build_catalog(df_r)
                df_t_r = df_r[time_column(df_r)].to_numpy()
                imu_axis_r = get_imu_axis_label(region)
                axis = next((ax for ax in AXES if axis_direction(is_kinem, ax, imu_axis_r) == direction), None)
                if axis is None:
                    continue
                mean_y, std_y = ensemble_mean_std(df_r, df_t_r, catalog_r, ctx["trial_bounds_fn"], ctx["n_trials"], grp, axis)
                if mean_y is None:
                    continue
                color = ctx["color"]
                dash = REGION_DASH[region]
                upper, lower = mean_y + std_y, mean_y - std_y
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([GRID, GRID[::-1]]), y=np.concatenate([upper, lower[::-1]]),
                        fill="toself", fillcolor=hex_to_rgba(color, 0.12),
                        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False,
                    ),
                    row=row_i, col=col_i,
                )
                trace_key = f"{ctx['label']} - {region}"
                fig.add_trace(
                    go.Scatter(
                        x=GRID, y=mean_y, mode="lines",
                        line=dict(color=color, width=2.2, dash=dash),
                        name=trace_key, legendgroup=trace_key,
                        showlegend=trace_key not in legend_shown,
                    ),
                    row=row_i, col=col_i,
                )
                legend_shown.add(trace_key)
        # sombreado de fase (platô/descida/subida) + linha pontilhada exata de cada
        # pessoa — precisa vir DEPOIS dos traços do painel (senão o plotly ignora
        # o row/col do add_vrect/add_vline).
        add_phase_bg(fig, row=row_i, col=col_i)
        for ctx in PERSON_CTXS:
            fig.add_vline(x=ctx["d_frac"], line_dash="dot", line_color=ctx["color"], opacity=0.6, row=row_i, col=col_i)
            fig.add_vline(x=ctx["v_frac"], line_dash="dot", line_color=ctx["color"], opacity=0.6, row=row_i, col=col_i)
        fig.update_yaxes(title_text=f"{label} ({unit})" if col_i == 1 else "", row=row_i, col=col_i)

fig.update_xaxes(showgrid=False, range=[0, 1], title_text="Fração do ciclo (0–1)")
fig.update_yaxes(showgrid=False)
w, h = square_fig_size(len(SIGNAL_ROWS), len(DIRECTIONS))
fig.update_layout(
    width=w, height=h, margin=MARGIN, plot_bgcolor="white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)

col_chart, col_amp = st.columns([3, 2])
with col_chart:
    st.plotly_chart(fig, use_container_width=False, key="compare_main")

with col_amp:
    _amp_rows = []
    for region in REGIONS:
        for grp, (label, unit), is_kinem in SIGNAL_ROWS:
            df_a_r = ctx_a["sheets"][region]
            df_b_r = ctx_b["sheets"][region]
            amp_a = phase_amplitude_stats(
                df_a_r, df_a_r[time_column(df_a_r)].to_numpy(), build_catalog(df_a_r),
                ctx_a["trial_bounds_fn"], ctx_a["n_trials"], grp,
            )
            amp_b = phase_amplitude_stats(
                df_b_r, df_b_r[time_column(df_b_r)].to_numpy(), build_catalog(df_b_r),
                ctx_b["trial_bounds_fn"], ctx_b["n_trials"], grp,
            )
            if amp_a is None or amp_b is None:
                continue
            for phase_name in ("platô", "descida", "subida"):
                ma, sa = amp_a[phase_name]
                mb, sb = amp_b[phase_name]
                _amp_rows.append({
                    "Região": region, "Fase": phase_name, "Variável": f"{label} ({unit})",
                    f"{ctx_a['label']} — média": round(ma, 3) if not np.isnan(ma) else None,
                    f"{ctx_a['label']} — DP": round(sa, 3) if not np.isnan(sa) else None,
                    f"{ctx_b['label']} — média": round(mb, 3) if not np.isnan(mb) else None,
                    f"{ctx_b['label']} — DP": round(sb, 3) if not np.isnan(sb) else None,
                })

    st.info(
        "**Amplitude por fase (pico-a-pico da resultante dos 3 eixos), média ± DP "
        "entre os ciclos** — para cada variável (Deslocamento, Velocidade, Aceleração, "
        "ACC, GYR), em L5 e Joelho, separada por platô/descida/subida:"
    )
    if _amp_rows:
        st.dataframe(pd.DataFrame(_amp_rows), use_container_width=True, hide_index=True, height=520)
    else:
        st.caption("Sem dados suficientes para calcular a amplitude por fase.")

# ---- Bloco de interpretação: consistência entre repetições (CV) -------------
_cv_lines = []
for region in REGIONS:
    for direction in ("Vertical", "AP", "ML"):
        if region == "L5" and direction in ("AP", "ML"):
            continue  # descartado: deslocamento do L5 nesses eixos é perto de zero, CV vira ruído
        stat_a = net_displacement_stats(
            ctx_a["sheets"][region], ctx_a["sheets"][region][time_column(ctx_a["sheets"][region])].to_numpy(),
            build_catalog(ctx_a["sheets"][region]), ctx_a["trial_bounds_fn"], ctx_a["n_trials"], direction,
        )
        stat_b = net_displacement_stats(
            ctx_b["sheets"][region], ctx_b["sheets"][region][time_column(ctx_b["sheets"][region])].to_numpy(),
            build_catalog(ctx_b["sheets"][region]), ctx_b["trial_bounds_fn"], ctx_b["n_trials"], direction,
        )
        if stat_a is None or stat_b is None:
            continue
        quem_mais_variavel = ctx_a["label"] if stat_a["cv"] > stat_b["cv"] else ctx_b["label"]
        _cv_lines.append(
            f"- **{region} — {direction}**: {ctx_a['label']} CV={stat_a['cv']:.1f}% · "
            f"{ctx_b['label']} CV={stat_b['cv']:.1f}% — {quem_mais_variavel} repete de forma menos consistente."
        )

st.info(
    "**Consistência entre repetições (CV do deslocamento líquido na descida — "
    "quanto maior, menos consistente/controlado o movimento):**\n\n"
    + "\n".join(_cv_lines) +
    "\n\n_CV alto pode indicar fadiga, falta de controle motor ou variação real entre "
    "tentativas — vale olhar as curvas acima pra confirmar antes de concluir algo clínico._"
)

st.divider()

# ----------------------------------------------------------------------------
# Inclinação (frontal / sagital) — mesma lógica de app.py, generalizada p/ 2 pessoas
# ----------------------------------------------------------------------------

ALPHA_COMP = 0.96
_TILT_LIGHT_CUTOFF_HZ = 5.0


def compute_tilt_curve(sheets_raw, sheets, region, trial_bounds_fn, n_trials, plane):
    """plane = 'frontal' (ML, rotação em torno do AP) ou 'sagital' (AP, rotação em
    torno do ML) — mesmo cálculo (filtro complementar ACC+GYR) de app.py."""
    if region not in sheets or region not in sheets_raw:
        return None
    df_r = sheets[region]
    catalog_r = build_catalog(df_r)
    imu_axis_r = get_imu_axis_label(region)
    ap_r = next((ax for ax in AXES if imu_axis_r[ax] == "AP"), None)
    ml_r = next((ax for ax in AXES if imu_axis_r[ax] == "ML"), None)
    vert_r = next((ax for ax in AXES if imu_axis_r[ax] == "Vertical"), None)

    horiz_ax = ml_r if plane == "frontal" else ap_r
    gyr_ax = ap_r if plane == "frontal" else ml_r
    if horiz_ax is None or gyr_ax is None or vert_r is None:
        return None
    gyr_col = catalog_r.get("IMU - Giroscópio", {}).get(gyr_ax)
    acc_h_col = catalog_r.get("IMU - Acelerômetro", {}).get(horiz_ax)
    acc_v_col = catalog_r.get("IMU - Acelerômetro", {}).get(vert_r)
    if not (gyr_col and acc_h_col and acc_v_col):
        return None

    raw_r = sheets_raw[region]
    t_r = df_r[time_column(df_r)].to_numpy()
    dt_r = float(np.median(np.diff(t_r)))
    fs_r = 1.0 / dt_r if dt_r > 0 else 100.0

    acc_h_f = _light_lowpass(raw_r[acc_h_col].to_numpy(dtype=float), _TILT_LIGHT_CUTOFF_HZ, fs_r)
    acc_v_f = _light_lowpass(raw_r[acc_v_col].to_numpy(dtype=float), _TILT_LIGHT_CUTOFF_HZ, fs_r)
    gyr_f = _light_lowpass(raw_r[gyr_col].to_numpy(dtype=float), _TILT_LIGHT_CUTOFF_HZ, fs_r)
    theta_acc_f = np.degrees(np.arctan2(acc_h_f, acc_v_f))
    grav_mag_r = float(np.median(np.sqrt(acc_h_f ** 2 + acc_v_f ** 2)))
    use_anchor_r = grav_mag_r >= 3.0

    curves_r = []
    for trial_idx in range(1, n_trials + 1):
        cycle_start, d_start, v_trial, cycle_end = trial_bounds_fn(trial_idx)
        span = cycle_end - cycle_start
        if span <= 0:
            continue
        trial_mask_r = (t_r >= cycle_start) & (t_r <= cycle_end)
        if trial_mask_r.sum() < 3:
            continue
        theta_acc = theta_acc_f[trial_mask_r] - theta_acc_f[trial_mask_r][0]
        gyr_h = gyr_f[trial_mask_r]
        theta = np.zeros(len(theta_acc))
        for i in range(1, len(theta)):
            theta_gyro = theta[i - 1] + gyr_h[i] * dt_r
            theta[i] = ALPHA_COMP * theta_gyro + (1 - ALPHA_COMP) * theta_acc[i] if use_anchor_r else theta_gyro
        x_trial = (t_r[trial_mask_r] - cycle_start) / span
        oi = np.argsort(x_trial)
        curves_r.append(np.interp(GRID, x_trial[oi], theta[oi]))

    if not curves_r:
        return None
    arr_r = np.vstack(curves_r)
    return {"mean": arr_r.mean(axis=0), "std": arr_r.std(axis=0), "n": len(curves_r), "use_anchor": use_anchor_r}


def build_tilt_combo_figure(plane, y_title, chart_title):
    fig_t = go.Figure()
    any_trace = False
    for ctx in PERSON_CTXS:
        for region in REGIONS:
            res = compute_tilt_curve(ctx["sheets_raw"], ctx["sheets"], region, ctx["trial_bounds_fn"], ctx["n_trials"], plane)
            if res is None:
                continue
            any_trace = True
            color = ctx["color"]
            dash = REGION_DASH[region]
            m, s = res["mean"], res["std"]
            fig_t.add_trace(go.Scatter(
                x=np.concatenate([GRID, GRID[::-1]]), y=np.concatenate([m + s, (m - s)[::-1]]),
                fill="toself", fillcolor=hex_to_rgba(color, 0.12),
                line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False,
            ))
            anchor_note = "" if res["use_anchor"] else " (só giro)"
            fig_t.add_trace(go.Scatter(
                x=GRID, y=m, mode="lines", line=dict(color=color, width=2.2, dash=dash),
                name=f"{ctx['label']} - {region}{anchor_note}",
            ))
    if not any_trace:
        return None
    add_phase_bg(fig_t)
    for ctx in PERSON_CTXS:
        fig_t.add_vline(x=ctx["d_frac"], line_dash="dot", line_color=ctx["color"], opacity=0.6)
        fig_t.add_vline(x=ctx["v_frac"], line_dash="dot", line_color=ctx["color"], opacity=0.6)
    fig_t.update_xaxes(showgrid=False, range=[0, 1], title_text="Fração do ciclo (0–1)")
    fig_t.update_yaxes(showgrid=False, title_text=y_title)
    fig_t.update_layout(
        title=dict(text=chart_title, y=0.98, yanchor="top"),
        height=440, margin=dict(l=55, r=20, t=48, b=90), plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
    )
    return fig_t


st.subheader("📐 Inclinação (frontal / sagital) — Pessoa A × Pessoa B, L5 × Joelho")
st.caption(
    "Estimativa por filtro complementar (ACC + GYR) de o quanto cada ponto do corpo "
    "tomba em relação à sua posição no início do ciclo — não é o ângulo articular do "
    "joelho (precisaria de 2 sensores no mesmo segmento), é a inclinação de 1 sensor "
    "só. Útil como indicador relativo entre pessoas/regiões/repetições. Ver explicação "
    "completa do cálculo no app.py original."
)

def _tilt_peak_summary(plane):
    """Pra cada pessoa/região, pico da curva média (maior magnitude, positiva ou
    negativa) e em que fração do ciclo ele acontece."""
    out = {}
    for ctx in PERSON_CTXS:
        for region in REGIONS:
            res = compute_tilt_curve(ctx["sheets_raw"], ctx["sheets"], region, ctx["trial_bounds_fn"], ctx["n_trials"], plane)
            if res is None:
                continue
            m = res["mean"]
            peak_idx = int(np.argmax(np.abs(m)))
            out[(ctx["key"], region)] = {"peak": float(m[peak_idx]), "frac": peak_idx / 100.0, "std": float(res["std"][peak_idx])}
    return out


col_frontal, col_sagital = st.columns(2)
with col_frontal:
    fig_frontal = build_tilt_combo_figure(
        "frontal", "Δ ângulo (°) — positivo = lateral, negativo = medial", "Inclinação frontal (ML)"
    )
    if fig_frontal is not None:
        st.plotly_chart(fig_frontal, use_container_width=True, key="tilt_frontal")
        _peaks_f = _tilt_peak_summary("frontal")
        if ("A", "Joelho") in _peaks_f and ("B", "Joelho") in _peaks_f:
            pa, pb = _peaks_f[("A", "Joelho")], _peaks_f[("B", "Joelho")]
            quem_mais_valgo = ctx_a["label"] if abs(pa["peak"]) > abs(pb["peak"]) else ctx_b["label"]
            st.info(
                "**Valgo dinâmico (pico de inclinação medial do Joelho):**\n\n"
                f"- {ctx_a['label']}: {pa['peak']:.1f}° (±{pa['std']:.1f}) em ~{pa['frac']*100:.0f}% do ciclo\n"
                f"- {ctx_b['label']}: {pb['peak']:.1f}° (±{pb['std']:.1f}) em ~{pb['frac']*100:.0f}% do ciclo\n\n"
                f"**{quem_mais_valgo}** tem o pico de inclinação medial maior — indicador de mais "
                "valgo dinâmico do joelho nesse teste.\n\n"
                "_Proxy por 1 sensor (não é o ângulo articular real do joelho) — não substitui "
                "avaliação clínica._"
            )
    else:
        st.caption("Não foi possível calcular a inclinação frontal — faltam colunas de ACC/GYR necessárias.")

with col_sagital:
    fig_sagital = build_tilt_combo_figure(
        "sagital", "Δ ângulo (°) — positivo = anterior, negativo = posterior", "Inclinação sagital (AP)"
    )
    if fig_sagital is not None:
        st.plotly_chart(fig_sagital, use_container_width=True, key="tilt_sagital")
        _div_lines = []
        for ctx in PERSON_CTXS:
            res_l5 = compute_tilt_curve(ctx["sheets_raw"], ctx["sheets"], "L5", ctx["trial_bounds_fn"], ctx["n_trials"], "sagital")
            res_jo = compute_tilt_curve(ctx["sheets_raw"], ctx["sheets"], "Joelho", ctx["trial_bounds_fn"], ctx["n_trials"], "sagital")
            if res_l5 is None or res_jo is None:
                continue
            diff = res_jo["mean"] - res_l5["mean"]
            idx_max = int(np.argmax(np.abs(diff)))
            _div_lines.append(
                f"- **{ctx['label']}**: divergência máxima de {diff[idx_max]:.1f}° em ~{idx_max}% do ciclo "
                f"(L5={res_l5['mean'][idx_max]:.1f}°, Joelho={res_jo['mean'][idx_max]:.1f}°)"
            )
        if _div_lines:
            st.info(
                "**Divergência L5 × Joelho no plano sagital** (o quanto o joelho se inclina de "
                "forma diferente do tronco/pelve — esperado ser grande, já que o joelho flexiona "
                "bem mais que o tronco numa descida):\n\n"
                + "\n".join(_div_lines) +
                "\n\n_Divergência parecida entre pessoas é normal aqui; a comparação mais "
                "informativa costuma ser no plano frontal (compensação lateral), não no sagital._"
            )
    else:
        st.caption("Não foi possível calcular a inclinação sagital — faltam colunas de ACC/GYR necessárias.")

st.divider()
st.caption(
    "Próximos passos possíveis: comparação trial a trial (não só a resultante) e uma "
    "tabela de métricas (pico, CV, assimetria) por pessoa/região."
)
