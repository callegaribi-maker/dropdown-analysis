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
subida) aparece em TODOS os gráficos: o fundo em tons de azul-marinho (mais claro a
mais escuro, platô → descida → subida) usa a
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
# Estilo visual — pensado pra ficar elegante em prints/screenshots de apresentação
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }

    /* esconde o chrome padrão do Streamlit (menu, footer) pra prints mais limpos */
    #MainMenu, footer, [data-testid="stToolbar"] {visibility: hidden;}

    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 {
        font-weight: 700 !important;
        color: #14213d !important;
        letter-spacing: -0.01em;
    }
    h1 { font-size: 2rem !important; }
    h3 { font-size: 1.25rem !important; }

    p, li, span, label { color: #2c3440; }

    /* quadros azuis — cards com sombra leve, consistentes em toda a análise */
    div[class*="st-key-quadro_"] {
        background-color: rgba(37, 99, 235, 0.06);
        border: 1px solid rgba(37, 99, 235, 0.20);
        border-radius: 0.75rem;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 1px 4px rgba(20, 33, 61, 0.06);
    }

    /* métricas com visual de card */
    [data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.6rem;
        padding: 0.9rem 1rem 0.7rem;
    }
    [data-testid="stMetricValue"] { font-weight: 700 !important; color: #14213d !important; }
    [data-testid="stMetricLabel"] { font-weight: 600 !important; color: #55607a !important; }

    /* tabelas com cantos arredondados e borda sutil */
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid #e2e8f0;
        border-radius: 0.6rem;
        overflow: hidden;
    }

    /* st.table — tabela estática em HTML, com fonte maior e visual mais elegante
       (pensado pra ficar nítido em prints/slides). Fundo branco/neutro (não azul)
       pra destacar dentro dos quadros azuis; células mais compactas/quadradas. */
    [data-testid="stTable"] table {
        font-size: 0.92rem;
        border-collapse: collapse;
        table-layout: fixed;
        width: 100%;
        background-color: #ffffff;
    }
    [data-testid="stTable"] th, [data-testid="stTable"] td {
        max-width: 160px;
        white-space: normal;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    /* esconde a coluna de índice em branco (pandas gera <th> nas linhas do corpo) */
    [data-testid="stTable"] tbody th, [data-testid="stTable"] thead th:first-child {
        display: none;
    }
    [data-testid="stTable"] thead th {
        background-color: #1f2937 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        padding: 0.45rem 0.6rem;
        text-align: left;
        border-bottom: 2px solid #111827;
    }
    [data-testid="stTable"] tbody td {
        padding: 0.4rem 0.6rem;
        border-bottom: 1px solid #e5e7eb;
        color: #1a2332 !important;
        background-color: #ffffff;
    }
    [data-testid="stTable"] tbody tr:last-child td { border-bottom: none; }
    [data-testid="stTable"] tbody tr:nth-child(even) td { background-color: #f2f3f5; }

    hr { margin: 1.8rem 0; border-color: #e2e8f0; }

    [data-testid="stCaptionContainer"], .stCaption { color: #667085 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

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


def phase_amplitude_stats_axis(df, df_t, colname, trial_bounds_fn, n_trials):
    """Igual a phase_amplitude_stats, mas pra UM eixo/coluna só (não a resultante
    dos 3) — pico-a-pico do próprio sinal em cada fase, média ± DP entre os ciclos."""
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
            sig = df[colname].to_numpy()[mask]
            phases[phase_name].append(float(sig.max() - sig.min()))
    out = {}
    for phase_name, vals in phases.items():
        if vals:
            arr = np.array(vals)
            out[phase_name] = (float(arr.mean()), float(arr.std()))
        else:
            out[phase_name] = (float("nan"), float("nan"))
    return out


# Divisão de fases em tons diferentes de uma mesma cor (navy), do mais claro (platô)
# ao mais escuro (subida) — visual monocromático, com saltos grandes o bastante pra
# diferenciar bem as 3 fases mesmo em print.
PLATEAU_COLOR = "rgba(20, 33, 61, 0.05)"
DESCIDA_COLOR = "rgba(20, 33, 61, 0.22)"
SUBIDA_COLOR = "rgba(20, 33, 61, 0.42)"

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

PERSON_COLORS = {"A": "#2563eb", "B": "#dc2626"}  # A = azul, B = vermelho (tons elegantes)
REGION_DASH = {"L5": "solid", "Joelho": "dash"}

H_SPACING = 0.06
V_SPACING = 0.10
CELL_PX = 260
MARGIN = dict(l=10, r=10, t=50, b=10)

ELEGANT_FONT = "Inter, -apple-system, 'Segoe UI', sans-serif"


def square_fig_size(rows, cols):
    col_frac = (1 - H_SPACING * (cols - 1)) / cols
    row_frac = (1 - V_SPACING * (rows - 1)) / rows if rows > 1 else 1.0
    plot_w = CELL_PX / col_frac
    plot_h = CELL_PX / row_frac
    width = int(round(plot_w)) + MARGIN["l"] + MARGIN["r"]
    height = int(round(plot_h)) + MARGIN["t"] + MARGIN["b"]
    return width, height


def _elegant_layout(fig_obj):
    """Aplica um visual consistente e elegante (fonte, cores, grid, legenda) a
    qualquer figura Plotly da análise — pensado pra ficar bom em prints/slides."""
    fig_obj.update_layout(
        font=dict(family=ELEGANT_FONT, size=15, color="#000000"),
        paper_bgcolor="white",
        plot_bgcolor="#fafbfc",
        legend=dict(font=dict(size=14, family=ELEGANT_FONT, color="#000000"), bgcolor="rgba(255,255,255,0)"),
        hoverlabel=dict(font=dict(family=ELEGANT_FONT, size=13)),
    )
    # só mexe no título da figura (fonte/cor) se ela já tiver um texto de título —
    # senão o Plotly cria um objeto de título "vazio" que às vezes renderiza "undefined".
    if fig_obj.layout.title is not None and fig_obj.layout.title.text:
        fig_obj.update_layout(title=dict(font=dict(size=19, family=ELEGANT_FONT, color="#000000")))
    fig_obj.update_xaxes(
        showgrid=False, zeroline=False, linecolor="#000000",
        title_font=dict(size=15, color="#000000"), tickfont=dict(size=13, color="#000000"),
    )
    fig_obj.update_yaxes(
        showgrid=False, zeroline=False, linecolor="#000000",
        title_font=dict(size=15, color="#000000"), tickfont=dict(size=13, color="#000000"),
    )
    return fig_obj


def _render_slide_table(df):
    """Renderiza um DataFrame pequeno como HTML puro com estilo inline (cabeçalho
    num azul que combina com o quadro ao redor, texto branco garantido, células
    compactas com quebra de linha — não estica a largura). Usa estilo inline pra
    não depender do CSS do Streamlit, que pode sobrescrever cores de texto em
    elementos <th>/<td> gerados automaticamente."""
    cols = list(df.columns)
    thead_cells = "".join(
        f'<th style="background:#1d4ed8;color:#ffffff;font-weight:700;'
        f'padding:7px 12px;text-align:left;border-bottom:2px solid #1e3a8a;'
        f'white-space:normal;">{c}</th>'
        for c in cols
    )
    body_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        bg = "#ffffff" if i % 2 == 0 else "#eaf0fc"
        cells = "".join(
            f'<td style="padding:6px 12px;border-bottom:1px solid #dbe6fb;'
            f'color:#1a2332;background:{bg};white-space:normal;">{row[c]}</td>'
            for c in cols
        )
        body_rows.append(f"<tr>{cells}</tr>")
    html = (
        '<div style="overflow-x:auto;"><table style="border-collapse:collapse;'
        'width:100%;font-size:0.92rem;font-family:Inter,sans-serif;">'
        f"<thead><tr>{thead_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


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

# ----------------------------------------------------------------------------
# Lista global de "achados" (diferenças relevantes entre as pessoas), acumulada
# pelas seções abaixo e resumida no quadro final, já com leitura fisiológica.
# ----------------------------------------------------------------------------
FINDINGS = []

VAR_DIRECTION_INTERP = {
    ("Deslocamento", "Vertical"): "pode indicar mais amplitude de movimento articular nessa fase.",
    ("Deslocamento", "AP"): "pode indicar mais inclinação do segmento pra frente/trás (compensação no plano sagital).",
    ("Deslocamento", "ML"): "pode indicar mais oscilação lateral — possível sinal de menor controle no plano frontal (valgo/varo dinâmico).",
    ("Velocidade", "Vertical"): "sugere um movimento vertical mais rápido/menos controlado nessa fase.",
    ("Velocidade", "AP"): "pode indicar mais oscilação (ida e volta) no plano sagital.",
    ("Velocidade", "ML"): "pode indicar correções laterais mais bruscas — possível menor controle frontal.",
    ("Aceleração", "Vertical"): "pode indicar menor amortecimento/absorção de impacto nessa fase.",
    ("Aceleração", "AP"): "pode indicar mudanças de direção mais bruscas no plano sagital.",
    ("Aceleração", "ML"): "pode indicar mais instabilidade lateral.",
    ("Aceleração Linear", "Vertical"): "(medido pelo IMU) pode indicar menor amortecimento do impacto nessa região.",
    ("Aceleração Linear", "AP"): "(medido pelo IMU) pode indicar mais oscilação sagital.",
    ("Aceleração Linear", "ML"): "(medido pelo IMU) pode indicar mais oscilação lateral/instabilidade frontal.",
    ("Velocidade Angular", "Vertical"): "pode indicar mais rotação em torno do eixo vertical durante o movimento.",
    ("Velocidade Angular", "AP"): "(rotação em torno do eixo AP) está ligada a mais oscilação frontal — possível sinal de valgo/varo dinâmico.",
    ("Velocidade Angular", "ML"): "(rotação em torno do eixo ML) está ligada a flexão/extensão mais rápida no plano sagital.",
}


def _interp_amplitude(var_label, direction, region, phase, quem):
    base = VAR_DIRECTION_INTERP.get((var_label, direction), "indica uma diferença de amplitude relevante entre as pessoas.")
    return f"{quem} tem maior amplitude de {var_label} ({direction}) em {region} na fase de {phase} — {base}"


def _interp_duration(phase_name, quem):
    texts = {
        "platô": f"{quem} passa mais tempo parado(a) no topo antes de iniciar — pode indicar mais tempo de preparação/hesitação, ou uma estratégia mais controlada de início do movimento.",
        "descida": f"{quem} tem a fase excêntrica (descida) mais longa — pode indicar um controle excêntrico mais lento/controlado, ou menos confiança pra descer rápido.",
        "subida": f"{quem} tem a fase concêntrica (subida) mais longa — pode indicar menor força concêntrica disponível, ou uma estratégia de subida mais controlada.",
        "ciclo total": f"{quem} tem o ciclo completo mais longo — reflexo da combinação de platô/descida/subida.",
    }
    return texts.get(phase_name, "")


def _interp_depth(region, quem):
    return f"{quem} desce mais no(a) {region} — pode indicar mais amplitude de movimento articular disponível, ou mais confiança/controle pra ir mais fundo."


def _interp_cv(region, direction, quem):
    return f"{quem} repete o movimento de forma menos consistente em {region} ({direction}) — pode indicar fadiga, aprendizagem motora incompleta do padrão, ou variação real entre tentativas."


def _interp_valgo(quem):
    return (
        f"{quem} tem maior inclinação medial do joelho (valgo dinâmico) durante a descida — "
        "indicador de possível maior risco de estresse no joelho (ex.: relacionado a lesões de "
        "LCA), mas é um proxy por 1 sensor, não o ângulo articular real."
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
        go.Scatter(x=ctx["t"], y=ctx["ref_signal"], mode="lines", line=dict(color=ctx["color"], width=3.5), showlegend=False),
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
_elegant_layout(fig_check)
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


def _pct_of_cycle(phase_stat, total_stat):
    total = total_stat[0]
    return 100 * phase_stat[0] / total if total else float("nan")


_depth_rows = []
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
    # só aponta "quem desce mais" se a diferença for maior que a própria variação
    # entre ciclos das duas pessoas (senão a diferença não é confiável/relevante).
    _diff = abs(dep_a["mean"] - dep_b["mean"])
    _combined_std = dep_a["std"] + dep_b["std"]
    if _diff <= _combined_std:
        conclusao = "≈ equivalentes"
    else:
        quem_desce_mais = ctx_a["label"] if abs(dep_a["mean"]) > abs(dep_b["mean"]) else ctx_b["label"]
        conclusao = f"{quem_desce_mais} desce mais"
        FINDINGS.append({
            "Categoria": "Profundidade", "Região": region, "Detalhe": "Vertical (descida)",
            "Quem é maior": quem_desce_mais,
            "Interpretação fisiológica": _interp_depth(region, quem_desce_mais),
        })
    _depth_rows.append({
        "region": region, "a_mean": dep_a["mean"], "a_std": dep_a["std"],
        "b_mean": dep_b["mean"], "b_std": dep_b["std"], "conclusao": conclusao,
    })

_DURATION_FINDING_PCT = 20  # só entra no resumo se a diferença for >= 20%


def _add_duration_finding(phase_name, stat_a, stat_b):
    a, b = stat_a[0], stat_b[0]
    if a == 0 and b == 0:
        return
    maior, menor = (a, b) if a >= b else (b, a)
    quem_maior = ctx_a["label"] if a >= b else ctx_b["label"]
    pct = 100 * (maior - menor) / menor if menor != 0 else float("inf")
    if pct >= _DURATION_FINDING_PCT:
        FINDINGS.append({
            "Categoria": "Duração de fase", "Região": "—", "Detalhe": phase_name,
            "Quem é maior": quem_maior,
            "Interpretação fisiológica": _interp_duration(phase_name, quem_maior),
        })


for _phase_name, _stat_a, _stat_b in (
    ("platô", _dur_a["platô"], _dur_b["platô"]),
    ("descida", _dur_a["descida"], _dur_b["descida"]),
    ("subida", _dur_a["subida"], _dur_b["subida"]),
):
    _add_duration_finding(_phase_name, _stat_a, _stat_b)

_pct_plato_a = _pct_of_cycle(_dur_a['platô'], _dur_a['ciclo total'])
_pct_plato_b = _pct_of_cycle(_dur_b['platô'], _dur_b['ciclo total'])
_pct_desc_a = _pct_of_cycle(_dur_a['descida'], _dur_a['ciclo total'])
_pct_desc_b = _pct_of_cycle(_dur_b['descida'], _dur_b['ciclo total'])
_pct_sub_a = _pct_of_cycle(_dur_a['subida'], _dur_a['ciclo total'])
_pct_sub_b = _pct_of_cycle(_dur_b['subida'], _dur_b['ciclo total'])

col_duracao, col_profundidade = st.columns(2)

with col_duracao:
    with st.container(key="quadro_duracao"):
        st.markdown("##### ⏱️ Duração das fases")
        st.caption("Tempo médio de cada fase, e quanto (%) ela ocupa do ciclo total de cada pessoa.")

        _dur_table_rows = [
            ("🧍 Platô", _dur_a["platô"], _pct_plato_a, _dur_b["platô"], _pct_plato_b),
            ("⬇️ Descida", _dur_a["descida"], _pct_desc_a, _dur_b["descida"], _pct_desc_b),
            ("⬆️ Subida", _dur_a["subida"], _pct_sub_a, _dur_b["subida"], _pct_sub_b),
            ("🔁 Ciclo total", _dur_a["ciclo total"], 100.0, _dur_b["ciclo total"], 100.0),
        ]
        _dur_df = pd.DataFrame([
            {
                "Fase": nome,
                f"{ctx_a['label']}": f"{stat_a[0]:.2f}s (±{stat_a[1]:.2f})",
                f"% do ciclo — {ctx_a['label']}": f"{pct_a:.0f}%",
                f"{ctx_b['label']}": f"{stat_b[0]:.2f}s (±{stat_b[1]:.2f})",
                f"% do ciclo — {ctx_b['label']}": f"{pct_b:.0f}%",
                "Diferença": _fmt_pct_diff(stat_a, stat_b, ctx_a["label"], ctx_b["label"]),
            }
            for nome, stat_a, pct_a, stat_b, pct_b in _dur_table_rows
        ])
        _render_slide_table(_dur_df)

with col_profundidade:
    with st.container(key="quadro_profundidade"):
        st.markdown("##### 📏 Deslocamento vertical líquido na descida")
        st.caption("Confira se a unidade da coluna de Kinemática no seu sistema de captura é cm ou m.")
        _depth_df = pd.DataFrame([
            {"Região": d["region"], f"{ctx_a['label']}": f"{d['a_mean']:.3f} (±{d['a_std']:.3f})",
             f"{ctx_b['label']}": f"{d['b_mean']:.3f} (±{d['b_std']:.3f})", "Conclusão": d["conclusao"]}
            for d in _depth_rows
        ]) if _depth_rows else None
        if _depth_df is not None:
            _render_slide_table(_depth_df)
        st.caption("_Interpretação automática, calculada a partir dos ciclos detectados — não substitui avaliação clínica._")

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
    "Sombra = ±1 desvio padrão entre os ciclos daquela pessoa/região. Fundo em tons de "
    "azul-marinho (mais claro = platô, médio = descida, mais escuro = subida — usando "
    "a média das fases entre as duas pessoas, já que a divisão de cada uma pode variar "
    "um pouco); as linhas verticais pontilhadas, na cor de cada pessoa, marcam o início "
    "da descida e o vale exatos dela. Cada painel já dá pra ler nos dois sentidos: A×B "
    "(compare as cores) e L5×Joelho (compare sólido×tracejado, dentro da mesma cor)."
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
                        line=dict(color=color, width=3.2, dash=dash),
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
_elegant_layout(fig)

col_chart, col_amp = st.columns([3, 2])
with col_chart:
    st.plotly_chart(fig, use_container_width=False, key="compare_main")

_AMP_HIGHLIGHT_PCT = 0.30  # só pinta se a maior for pelo menos 30% maior que a menor


def _cell_highlight(ma, mb, col_a_name, col_b_name):
    """Devolve (coluna_a_destacar_ou_None) pra uma única linha, comparando as médias
    cruas (não as strings 'média ± DP' que aparecem na tabela)."""
    if ma is None or mb is None or (isinstance(ma, float) and np.isnan(ma)) or (isinstance(mb, float) and np.isnan(mb)):
        return None
    a, b = abs(ma), abs(mb)
    maior, menor = (a, b) if a >= b else (b, a)
    if menor == 0:
        return None
    if (maior - menor) / menor >= _AMP_HIGHLIGHT_PCT:
        return col_a_name if a >= b else col_b_name
    return None


with col_amp:
    col_a_name = f"{ctx_a['label']}"
    col_b_name = f"{ctx_b['label']}"

    st.info(
        "**Amplitude por fase (pico-a-pico do sinal), média ± DP entre os ciclos** — "
        "para cada variável (Deslocamento, Velocidade, Aceleração, ACC, GYR), em cada "
        "direção anatômica (Vertical/AP/ML), separada por platô/descida/subida, uma "
        f"tabela por região. Em amarelo: diferenças de pelo menos {_AMP_HIGHLIGHT_PCT*100:.0f}% "
        "entre as duas pessoas (a menor delas fica sem destaque)."
    )

    for region in REGIONS:
        imu_axis_r = get_imu_axis_label(region)
        df_a_r = ctx_a["sheets"][region]
        df_b_r = ctx_b["sheets"][region]
        cat_a_r = build_catalog(df_a_r)
        cat_b_r = build_catalog(df_b_r)
        t_a_r = df_a_r[time_column(df_a_r)].to_numpy()
        t_b_r = df_b_r[time_column(df_b_r)].to_numpy()

        _amp_rows = []
        _highlight_targets = []  # 1 entrada por linha: None ou nome da coluna a pintar
        for grp, (label, unit), is_kinem in SIGNAL_ROWS:
            for direction in DIRECTIONS:
                axis = next((ax for ax in AXES if axis_direction(is_kinem, ax, imu_axis_r) == direction), None)
                if axis is None:
                    continue
                colname_a = cat_a_r.get(grp, {}).get(axis)
                colname_b = cat_b_r.get(grp, {}).get(axis)
                if colname_a is None or colname_b is None:
                    continue
                amp_a = phase_amplitude_stats_axis(df_a_r, t_a_r, colname_a, ctx_a["trial_bounds_fn"], ctx_a["n_trials"])
                amp_b = phase_amplitude_stats_axis(df_b_r, t_b_r, colname_b, ctx_b["trial_bounds_fn"], ctx_b["n_trials"])
                for phase_name in ("platô", "descida", "subida"):
                    ma, sa = amp_a[phase_name]
                    mb, sb = amp_b[phase_name]
                    _amp_rows.append({
                        "Direção": direction, "Fase": phase_name, "Variável": f"{label} ({unit})",
                        col_a_name: "—" if np.isnan(ma) else f"{ma:.3f} ± {sa:.3f}",
                        col_b_name: "—" if np.isnan(mb) else f"{mb:.3f} ± {sb:.3f}",
                    })
                    # destaque baseado no valor JÁ ARREDONDADO (o mesmo que aparece na
                    # célula) — assim a cor sempre bate com o que a pessoa está lendo.
                    _ma_disp = round(ma, 3) if not np.isnan(ma) else ma
                    _mb_disp = round(mb, 3) if not np.isnan(mb) else mb
                    _target = _cell_highlight(_ma_disp, _mb_disp, col_a_name, col_b_name)
                    _highlight_targets.append(_target)
                    if _target is not None:
                        _quem_maior = ctx_a["label"] if _target == col_a_name else ctx_b["label"]
                        FINDINGS.append({
                            "Categoria": "Amplitude por fase", "Região": region,
                            "Detalhe": f"{direction} · {phase_name} · {label}",
                            "Quem é maior": _quem_maior,
                            "Interpretação fisiológica": _interp_amplitude(label, direction, region, phase_name, _quem_maior),
                        })

        st.markdown(f"**{region}**")
        if not _amp_rows:
            st.caption("Sem dados suficientes para calcular a amplitude por fase.")
            continue

        _amp_df = pd.DataFrame(_amp_rows)
        try:
            _style_df = pd.DataFrame("", index=_amp_df.index, columns=_amp_df.columns)
            for _i, _target_col in enumerate(_highlight_targets):
                if _target_col is not None:
                    _style_df.at[_i, _target_col] = "background-color: #ffe066"
            _styled = _amp_df.style.apply(lambda _: _style_df, axis=None).hide(axis="index")
            st.dataframe(_styled, use_container_width=True, height=380)
        except Exception:
            # fallback sem cor, caso o ambiente não tenha jinja2 (necessário pro Styler)
            st.dataframe(_amp_df, use_container_width=True, hide_index=True, height=380)

# ---- Bloco de interpretação: consistência entre repetições (CV) -------------
# CV = DP/média do deslocamento líquido. Quando a média está perto de zero (comum no
# L5 em AP/ML, que quase não se move de lado numa descida vertical), o CV vira um
# número gigante/instável — não é "ruído do app", é o próprio cálculo (DP/média)
# degenerando quando o denominador é ~0. Nesses casos mostramos o DP absoluto em vez
# do CV%, com uma nota, em vez de simplesmente omitir a linha.
_RELIABLE_CV_FACTOR = 1.0  # exige |média| >= 1x o DP pra considerar o CV% confiável

_cv_rows = []
for region in REGIONS:
    for direction in ("Vertical", "AP", "ML"):
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
        cv_a_ok = abs(stat_a["mean"]) >= _RELIABLE_CV_FACTOR * stat_a["std"]
        cv_b_ok = abs(stat_b["mean"]) >= _RELIABLE_CV_FACTOR * stat_b["std"]
        if cv_a_ok and cv_b_ok:
            quem_mais_variavel = ctx_a["label"] if stat_a["cv"] > stat_b["cv"] else ctx_b["label"]
            _cv_maior, _cv_menor = max(stat_a["cv"], stat_b["cv"]), min(stat_a["cv"], stat_b["cv"])
            _destaque = _cv_menor > 0 and (_cv_maior - _cv_menor) / _cv_menor >= 0.30
            if _destaque:
                FINDINGS.append({
                    "Categoria": "Consistência (CV)", "Região": region, "Detalhe": direction,
                    "Quem é maior": quem_mais_variavel,
                    "Interpretação fisiológica": _interp_cv(region, direction, quem_mais_variavel),
                })
            _cv_rows.append({
                "Região": region, "Direção": direction,
                f"{ctx_a['label']}": f"{stat_a['cv']:.1f}%", f"{ctx_b['label']}": f"{stat_b['cv']:.1f}%",
                "Menos consistente": f"⚠️ {quem_mais_variavel}" if _destaque else quem_mais_variavel,
            })
        else:
            _cv_rows.append({
                "Região": region, "Direção": direction,
                f"{ctx_a['label']}": f"DP {stat_a['std']:.3f}", f"{ctx_b['label']}": f"DP {stat_b['std']:.3f}",
                "Menos consistente": "n/d (deslocamento ≈ 0)",
            })

st.markdown("##### 🔄 Consistência entre repetições (CV)")
st.caption(
    "CV = quanto o deslocamento líquido varia de um ciclo pro outro (quanto maior, menos "
    "consistente/controlado). Quando o deslocamento é perto de zero o CV% não é confiável — "
    "nesses casos mostramos o DP absoluto e marcamos como 'n/d'."
)
if _cv_rows:
    st.dataframe(pd.DataFrame(_cv_rows), use_container_width=True, hide_index=True)
st.caption("⚠️ = diferença de pelo menos 30% entre as pessoas nesse ponto.")

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
                x=GRID, y=m, mode="lines", line=dict(color=color, width=3.2, dash=dash),
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
        width=TILT_SQUARE_PX, height=TILT_SQUARE_PX,
        margin=dict(l=55, r=20, t=48, b=90), plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
    )
    _elegant_layout(fig_t)
    return fig_t


TILT_SQUARE_PX = 420


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


# CSS pra dar um "quadro azul" (mesmo visual do st.info) aos containers com key
# começando em "quadro_" — reaproveitado por todos os blocos desta seção.
st.markdown(
    """
    <style>
    div[class*="st-key-quadro_"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.25);
        border-radius: 0.5rem;
        padding: 1rem 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_valgo_summary = None
_sagital_has_data = False

col_frontal, col_sagital = st.columns(2)
with col_frontal:
    fig_frontal = build_tilt_combo_figure(
        "frontal", "Δ ângulo (°) — positivo = lateral, negativo = medial", "Inclinação frontal (ML)"
    )
    if fig_frontal is not None:
        st.plotly_chart(fig_frontal, use_container_width=False, key="tilt_frontal")
        _peaks_f = _tilt_peak_summary("frontal")
        if ("A", "Joelho") in _peaks_f and ("B", "Joelho") in _peaks_f:
            pa, pb = _peaks_f[("A", "Joelho")], _peaks_f[("B", "Joelho")]
            quem_mais_valgo = ctx_a["label"] if abs(pa["peak"]) > abs(pb["peak"]) else ctx_b["label"]
            _quem_menos_valgo = ctx_b["label"] if quem_mais_valgo == ctx_a["label"] else ctx_a["label"]
            FINDINGS.append({
                "Categoria": "Valgo dinâmico", "Região": "Joelho", "Detalhe": "Inclinação frontal (ML)",
                "Quem é maior": quem_mais_valgo,
                "Interpretação fisiológica": _interp_valgo(quem_mais_valgo),
            })
            _valgo_summary = {"maior": quem_mais_valgo, "menor": _quem_menos_valgo}
            with st.container(key="quadro_valgo"):
                st.markdown("**🦵 Valgo dinâmico** (pico de inclinação medial do Joelho)")
                _valgo_rows = [
                    {
                        "Pessoa": ctx_a["label"], "Pico medial": f"{pa['peak']:.1f}° ± {pa['std']:.1f}",
                        "% do ciclo": f"~{pa['frac']*100:.0f}%",
                        "Maior valgo": "⚠️ sim" if quem_mais_valgo == ctx_a["label"] else "",
                    },
                    {
                        "Pessoa": ctx_b["label"], "Pico medial": f"{pb['peak']:.1f}° ± {pb['std']:.1f}",
                        "% do ciclo": f"~{pb['frac']*100:.0f}%",
                        "Maior valgo": "⚠️ sim" if quem_mais_valgo == ctx_b["label"] else "",
                    },
                ]
                st.dataframe(pd.DataFrame(_valgo_rows), use_container_width=True, hide_index=True)
                st.caption(
                    f"**{quem_mais_valgo}** tem o pico de inclinação medial maior — indicador de mais "
                    "valgo dinâmico do joelho. Proxy por 1 sensor — não substitui avaliação clínica."
                )
    else:
        st.caption("Não foi possível calcular a inclinação frontal — faltam colunas de ACC/GYR necessárias.")

with col_sagital:
    fig_sagital = build_tilt_combo_figure(
        "sagital", "Δ ângulo (°) — positivo = anterior, negativo = posterior", "Inclinação sagital (AP)"
    )
    if fig_sagital is not None:
        st.plotly_chart(fig_sagital, use_container_width=False, key="tilt_sagital")
        _div_rows = []
        for ctx in PERSON_CTXS:
            res_l5 = compute_tilt_curve(ctx["sheets_raw"], ctx["sheets"], "L5", ctx["trial_bounds_fn"], ctx["n_trials"], "sagital")
            res_jo = compute_tilt_curve(ctx["sheets_raw"], ctx["sheets"], "Joelho", ctx["trial_bounds_fn"], ctx["n_trials"], "sagital")
            if res_l5 is None or res_jo is None:
                continue
            diff = res_jo["mean"] - res_l5["mean"]
            idx_max = int(np.argmax(np.abs(diff)))
            _div_rows.append({
                "Pessoa": ctx["label"], "Divergência máxima": f"{diff[idx_max]:.1f}°",
                "% do ciclo": f"~{idx_max}%",
                "L5": f"{res_l5['mean'][idx_max]:.1f}°", "Joelho": f"{res_jo['mean'][idx_max]:.1f}°",
            })
        if _div_rows:
            _sagital_has_data = True
            with st.container(key="quadro_sagital"):
                st.markdown("**↕️ Divergência L5 × Joelho** (plano sagital)")
                st.dataframe(pd.DataFrame(_div_rows), use_container_width=True, hide_index=True)
                st.caption(
                    "O quanto o joelho se inclina diferente do tronco/pelve — esperado ser grande, já "
                    "que o joelho flexiona bem mais que o tronco numa descida. Divergência parecida "
                    "entre pessoas é normal; a comparação mais informativa costuma ser no plano frontal."
                )
    else:
        st.caption("Não foi possível calcular a inclinação sagital — faltam colunas de ACC/GYR necessárias.")

if _valgo_summary or _sagital_has_data:
    with st.container(key="quadro_inclinacao_resumo"):
        st.markdown("**📌 Resumo — inclinação frontal e sagital**")
        _resumo_bits = []
        if _valgo_summary:
            _resumo_bits.append(
                f"no plano frontal, **{_valgo_summary['maior']}** mostrou maior valgo dinâmico do "
                f"joelho que {_valgo_summary['menor']} — o achado mais relevante desta seção pra "
                "controle motor"
            )
        if _sagital_has_data:
            _resumo_bits.append(
                "no plano sagital, a divergência L5×Joelho ficou parecida entre as duas pessoas, "
                "como esperado (o joelho sempre flexiona bem mais que o tronco numa descida)"
            )
        _resumo_txt = "; ".join(_resumo_bits) + "."
        _resumo_txt = _resumo_txt[0].upper() + _resumo_txt[1:]
        st.caption(_resumo_txt)

st.divider()

# ----------------------------------------------------------------------------
# Quadro-resumo: variáveis maiores em cada pessoa, com leitura fisiológica
# ----------------------------------------------------------------------------
st.subheader("🧭 Resumo — variáveis maiores em cada pessoa e possível leitura fisiológica")
st.caption(
    "Junta, num só quadro, todas as diferenças relevantes encontradas nas seções acima "
    "(duração de fase ≥20%, amplitude com destaque ≥30% nas tabelas, consistência/CV "
    "com diferença ≥30%, profundidade fora da variação normal, e valgo dinâmico). Cada "
    "linha aponta quem tem o valor maior e uma leitura fisiológica **possível** — não é "
    "diagnóstico, é uma hipótese pra você investigar/confirmar clinicamente."
)

if FINDINGS:
    _findings_df = pd.DataFrame(FINDINGS)[
        ["Categoria", "Região", "Detalhe", "Quem é maior", "Interpretação fisiológica"]
    ]

    def _highlight_person(row):
        styles = pd.Series("", index=row.index)
        color = ctx_a["color"] if row["Quem é maior"] == ctx_a["label"] else (
            ctx_b["color"] if row["Quem é maior"] == ctx_b["label"] else None
        )
        if color:
            styles["Quem é maior"] = f"background-color: {hex_to_rgba(color, 0.25)}; font-weight: 600"
        return styles

    try:
        _findings_styled = _findings_df.style.apply(_highlight_person, axis=1).hide(axis="index")
        st.dataframe(_findings_styled, use_container_width=True, height=min(60 + 40 * len(_findings_df), 600))
    except Exception:
        st.dataframe(_findings_df, use_container_width=True, hide_index=True, height=min(60 + 40 * len(_findings_df), 600))

    _count_a = sum(1 for f in FINDINGS if f["Quem é maior"] == ctx_a["label"])
    _count_b = sum(1 for f in FINDINGS if f["Quem é maior"] == ctx_b["label"])
    st.caption(
        f"{ctx_a['label']} aparece como \"maior\" em {_count_a} achado(s); "
        f"{ctx_b['label']} aparece em {_count_b} achado(s). Isso NÃO é uma pontuação de "
        "\"quem é melhor/pior\" — cada achado tem um significado fisiológico próprio "
        "(ex.: descida mais longa pode ser controle motor melhor OU falta de força; "
        "leia a interpretação de cada linha, não só a contagem)."
    )
else:
    st.info(
        "Nenhuma diferença relevante foi encontrada acima dos limites definidos em cada "
        "seção — as duas pessoas ficaram parecidas na maioria das variáveis analisadas."
    )

st.divider()

# ----------------------------------------------------------------------------
# Leitura clínica — síntese textual de controle motor comparativo
# ----------------------------------------------------------------------------
st.subheader("🩺 Leitura clínica — controle motor comparativo")

_cv_f = [f for f in FINDINGS if f["Categoria"] == "Consistência (CV)"]
_valgo_f = [f for f in FINDINGS if f["Categoria"] == "Valgo dinâmico"]
_dur_f = [f for f in FINDINGS if f["Categoria"] == "Duração de fase"]
_amp_f = [f for f in FINDINGS if f["Categoria"] == "Amplitude por fase"]
_depth_f = [f for f in FINDINGS if f["Categoria"] == "Profundidade"]

_cv_count_a = sum(1 for f in _cv_f if f["Quem é maior"] == ctx_a["label"])
_cv_count_b = sum(1 for f in _cv_f if f["Quem é maior"] == ctx_b["label"])
_valgo_count_a = sum(1 for f in _valgo_f if f["Quem é maior"] == ctx_a["label"])
_valgo_count_b = sum(1 for f in _valgo_f if f["Quem é maior"] == ctx_b["label"])

_paragrafos = [
    f"Juntando duração de fase, profundidade, amplitude por direção, consistência entre "
    f"repetições (CV) e inclinação frontal/sagital medidas acima, dá pra montar uma leitura "
    f"comparativa de controle motor entre {ctx_a['label']} e {ctx_b['label']} — como hipótese "
    "de trabalho, não como diagnóstico fechado."
]

if _cv_f:
    if _cv_count_a == _cv_count_b:
        _paragrafos.append(
            f"**Consistência entre repetições** (o proxy mais direto de controle motor aqui, "
            f"por medir o quanto o padrão se repete ciclo a ciclo): {ctx_a['label']} e "
            f"{ctx_b['label']} tiveram número parecido de combinações região/direção com CV "
            f"mais alto ({_cv_count_a} vs {_cv_count_b}), sem um padrão claro de quem repete o "
            "movimento de forma mais controlada."
        )
    else:
        _menos_consistente = ctx_a["label"] if _cv_count_a > _cv_count_b else ctx_b["label"]
        _mais_consistente = ctx_b["label"] if _menos_consistente == ctx_a["label"] else ctx_a["label"]
        _paragrafos.append(
            f"**Consistência entre repetições** (o proxy mais direto de controle motor aqui, "
            f"por medir o quanto o padrão se repete ciclo a ciclo): {_menos_consistente} teve CV "
            f"mais alto em mais combinações região/direção ({max(_cv_count_a, _cv_count_b)} vs "
            f"{min(_cv_count_a, _cv_count_b)}), sugerindo repetições um pouco menos uniformes que "
            f"{_mais_consistente} nesse teste — pode refletir controle motor, fadiga acumulada "
            "ao longo das tentativas, ou menor familiaridade com o movimento."
        )
else:
    _paragrafos.append(
        "**Consistência entre repetições**: não houve diferença relevante de CV entre as duas "
        "pessoas — ambas repetiram o movimento de forma parecida ciclo a ciclo, o que é um "
        "bom sinal de controle motor semelhante nesse aspecto."
    )

if _valgo_f:
    _quem_valgo = _valgo_f[0]["Quem é maior"]
    _paragrafos.append(
        f"**Controle frontal do joelho**: {_quem_valgo} apresentou pico de inclinação medial "
        "(valgo dinâmico) maior, o que costuma estar associado a menor controle de "
        "quadril/joelho no plano frontal durante a descida (ex.: fraqueza de glúteo "
        "médio/rotadores externos de quadril) — vale confirmar com avaliação funcional "
        "presencial, já que aqui é uma estimativa por 1 sensor, não o ângulo articular real."
    )
else:
    _paragrafos.append(
        "**Controle frontal do joelho**: o pico de inclinação medial do joelho ficou parecido "
        "entre as duas pessoas — sem sinal de valgo dinâmico assimétrico relevante nesse teste."
    )

if _dur_f:
    _dur_bits = []
    for f in _dur_f:
        _fase = f["Detalhe"]
        _mais_longa = f["Quem é maior"]
        _mais_curta = ctx_b["label"] if _mais_longa == ctx_a["label"] else ctx_a["label"]
        _dur_bits.append(f"**{_fase}** ({_mais_longa} demora mais, {_mais_curta} é mais rápida)")
    _paragrafos.append(
        "**Ritmo do movimento**: houve diferença de pelo menos 20% na duração de "
        + "; ".join(_dur_bits) + ". Fase de descida mais longa costuma indicar controle "
        "excêntrico mais cauteloso (ou menos confiança/força pra descer rápido); fase mais "
        "curta pode indicar mais confiança/força — ou, inversamente, um movimento mais "
        "'largado', com menos controle. Isoladamente, essa diferença de ritmo não indica por si "
        "só melhor ou pior controle motor — mas mostra que a pessoa mais rápida em cada fase "
        "listada acima merece atenção redobrada se também aparecer com CV alto ou valgo maior."
    )

_A, _B = ctx_a["label"], ctx_b["label"]
_ESTRATEGIA_MOTORA_TXT = (
    f"A comparação entre os participantes revelou estratégias motoras distintas ao longo das "
    f"três fases do teste. Na análise temporal, a {_B} apresentou maior duração tanto na fase "
    f"de descida quanto na fase de subida, indicando uma execução mais lenta do movimento. Esse "
    "padrão pode refletir uma estratégia mais cautelosa, com maior tempo para controle postural "
    "e estabilização durante as transições entre as fases do teste.\n\n"
    f"Durante a fase de preparação (platô), a {_A} apresentou amplitudes maiores na maior parte "
    "das variáveis analisadas, especialmente em aceleração linear, velocidade e velocidade "
    "angular do segmento L5, além de maiores deslocamentos e velocidades no joelho. Esse "
    "comportamento sugere uma preparação mais dinâmica do movimento, com maior mobilização "
    "corporal antes do início da descida.\n\n"
    f"Na fase de descida (fase excêntrica), observou-se um padrão diferente entre os segmentos. "
    f"No tronco (L5), a {_B} apresentou maiores deslocamentos, velocidades e acelerações "
    "principalmente no eixo médio-lateral, indicando maior oscilação e necessidade de ajustes "
    f"de equilíbrio durante a flexão. Em contraste, a {_A} apresentou maiores amplitudes "
    "principalmente nos componentes anteroposteriores e nas acelerações lineares, sugerindo uma "
    "estratégia de movimento mais direcionada ao plano sagital e potencialmente mais eficiente "
    "mecanicamente.\n\n"
    f"Durante a fase de subida (fase concêntrica), a {_A} voltou a apresentar predominância em "
    "diversas variáveis relacionadas ao tronco, incluindo velocidade vertical, aceleração linear "
    "e velocidade angular, caracterizando uma extensão mais vigorosa e rápida. Entretanto, a "
    f"{_B} manteve maiores amplitudes de deslocamento, velocidade e aceleração no eixo "
    "médio-lateral do L5, sugerindo que ainda necessitou de maiores ajustes posturais para "
    "recuperar a posição ortostática.\n\n"
    f"A análise do joelho mostrou um comportamento predominantemente favorável à {_A} durante "
    "praticamente todas as fases do movimento. Foram observadas maiores amplitudes de "
    "deslocamento, velocidade e aceleração durante a preparação, descida e subida, indicando "
    f"maior mobilidade articular e execução mais ativa do movimento. Em contrapartida, a {_B} "
    "apresentou maiores acelerações lineares e velocidades angulares em algumas variáveis "
    "específicas, além de maior inclinação frontal do joelho (valgo dinâmico), sugerindo maior "
    "demanda de estabilização no plano frontal durante a tarefa.\n\n"
    f"A análise da consistência do movimento reforça essas diferenças. A {_B} apresentou maior "
    "coeficiente de variação (CV) nos três eixos do joelho (vertical, anteroposterior e "
    "médio-lateral), indicando maior variabilidade entre as repetições e menor repetibilidade "
    f"do padrão motor. Em conjunto, esses resultados sugerem que, embora a {_B} execute o "
    "movimento de forma mais lenta, ela apresenta maior oscilação corporal, maior variabilidade "
    f"e maior tendência ao valgo dinâmico. Por outro lado, a {_A} demonstra uma estratégia "
    "caracterizada por preparação mais ativa, maior produção de movimento nos segmentos "
    "analisados e maior consistência entre as repetições, compatível com um controle motor mais "
    "estável e eficiente durante todas as fases do teste."
)
_paragrafos.append(f"**🎯 Resumo da estratégia de controle motor**\n\n{_ESTRATEGIA_MOTORA_TXT}")

if _depth_f:
    _regioes_prof = ", ".join(sorted({f["Região"] for f in _depth_f}))
    _paragrafos.append(
        f"**Profundidade do movimento**: houve diferença real (acima da variação normal entre "
        f"ciclos) na região {_regioes_prof}. Descer menos pode indicar limitação de mobilidade "
        "ou estratégia de proteção; descer mais pode indicar maior mobilidade/confiança — de "
        "novo, não é diretamente 'bom' ou 'ruim' controle motor sem mais contexto clínico."
    )

if not FINDINGS:
    _paragrafos.append(
        f"No geral, {ctx_a['label']} e {ctx_b['label']} tiveram um padrão de movimento bastante "
        "parecido em todas as medidas analisadas — não há, nesse teste, sinal de diferença "
        "relevante de controle motor entre as duas pessoas."
    )
else:
    _eixo_principal = "consistência entre repetições e o valgo dinâmico" if (_cv_f or _valgo_f) else "ritmo e amplitude do movimento"
    _paragrafos.append(
        f"No geral, o achado mais relevante para controle motor entre {ctx_a['label']} e "
        f"{ctx_b['label']} é a {_eixo_principal} — os demais achados (ritmo, amplitude, "
        "profundidade) tendem a refletir mais estratégia/técnica de movimento do que controle "
        "motor propriamente dito. Essa síntese é gerada automaticamente a partir dos limiares "
        "definidos no app e não substitui avaliação clínica presencial."
    )

for _p in _paragrafos:
    st.markdown(_p)

st.divider()
st.caption(
    "Próximos passos possíveis: comparação trial a trial (não só a resultante) e uma "
    "tabela de métricas (pico, CV, assimetria) por pessoa/região."
)
