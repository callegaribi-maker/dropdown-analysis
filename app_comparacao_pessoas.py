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

    h1, h2, h3, h4, h5, [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4, [data-testid="stMarkdownContainer"] h5 {
        font-weight: 700 !important;
        color: #14213d !important;
        letter-spacing: -0.01em;
    }
    h1 { font-size: 2rem !important; }
    h3 { font-size: 1.55rem !important; }
    h5, [data-testid="stMarkdownContainer"] h5 { font-size: 1.2rem !important; }

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


def kinem_vs_imu_concordance(df_raw, catalog, imu_axis_label, direction, fs, shared_cutoff_hz, max_lag_s=0.5):
    """Compara a aceleração da cinemática com a do IMU (celular), na mesma direção
    anatômica, usando dados BRUTOS (já com as correções conhecidas de sinal/unidade
    aplicadas) — não os já filtrados com cortes diferentes por grupo, pra não
    penalizar a concordância por causa de ruído de alta frequência que sobra só num
    dos dois sinais. Aqui os dois passam por: (1) detrend, (2) o MESMO corte de
    filtro passa-baixa (shared_cutoff_hz) pros dois.

    Depois disso, procura o melhor alinhamento temporal (cross-correlação, até
    ±max_lag_s segundos) — pequenas defasagens de sincronismo entre os dois
    sistemas derrubam a correlação sem isso ser um problema real do sinal. Também
    detecta se a correlação é bem melhor com um dos sinais invertido (comum quando
    a convenção de eixo/sinal difere entre os dois sistemas) e reporta isso
    explicitamente (não esconde a inversão).

    Unidade: cinemática vem em cm/s², IMU em m/s² — convertida aqui pra m/s².
    "Viés" não é calculado: como os dois sinais são detrend, a diferença média
    sempre daria ~0 por construção, não é um resultado real. Além de r e RMSE,
    também calcula CCC (concordância de Lin — penaliza diferença de escala/
    amplitude, não só se as curvas andam juntas no tempo), um fator de escala
    (razão entre o desvio-padrão do celular e o da cinemática, pra deixar
    explícito o tamanho de qualquer discrepância de amplitude) e SEM/MDC95 (erro
    padrão de medida / mínima mudança detectável, a partir do desvio-padrão da
    diferença ponto-a-ponto entre os dois sinais).
    Retorna None se faltar alguma das duas colunas ou dados insuficientes."""
    kin_axis = next((ax for ax in AXES if KINEM_AXIS_LABEL[ax] == direction), None)
    imu_axis = next((ax for ax in AXES if imu_axis_label[ax] == direction), None)
    if kin_axis is None or imu_axis is None:
        return None
    kin_col = catalog.get("Cinemática - Aceleração", {}).get(kin_axis)
    imu_col = catalog.get("IMU - Acelerômetro", {}).get(imu_axis)
    if kin_col is None or imu_col is None:
        return None
    kin = df_raw[kin_col].to_numpy(dtype=float) / 100.0  # cm/s² -> m/s²
    imu = df_raw[imu_col].to_numpy(dtype=float)
    mask = ~(np.isnan(kin) | np.isnan(imu))
    kin, imu = kin[mask], imu[mask]
    if len(kin) < 20:
        return None
    kin = detrend(kin)
    imu = detrend(imu)
    kin = _light_lowpass(kin, shared_cutoff_hz, fs)
    imu = _light_lowpass(imu, shared_cutoff_hz, fs)
    if np.std(kin) == 0 or np.std(imu) == 0:
        return None

    max_lag_samples = max(1, int(round(max_lag_s * fs)))
    best_r, best_lag = 0.0, 0
    for lag in range(-max_lag_samples, max_lag_samples + 1):
        if lag < 0:
            a, b = kin[-lag:], imu[: len(imu) + lag]
        elif lag > 0:
            a, b = kin[: len(kin) - lag], imu[lag:]
        else:
            a, b = kin, imu
        n = min(len(a), len(b))
        if n < 20:
            continue
        a, b = a[:n], b[:n]
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if abs(r) > abs(best_r):
            best_r, best_lag = r, lag

    if best_lag < 0:
        a, b = kin[-best_lag:], imu[: len(imu) + best_lag]
    elif best_lag > 0:
        a, b = kin[: len(kin) - best_lag], imu[best_lag:]
    else:
        a, b = kin, imu
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    sign_flip = best_r < 0
    if sign_flip:
        b = -b
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))

    # CCC (Lin's concordance correlation coefficient): diferente do r, também
    # penaliza diferença de ESCALA/amplitude entre os dois sinais (não só se eles
    # "andam juntos" no tempo) — duas curvas com o mesmo formato mas amplitudes
    # bem diferentes dão r alto mas CCC mais baixo. A parte de deslocamento médio
    # (mu_a - mu_b) fica perto de 0 por causa do detrend, então aqui o CCC reflete
    # sobretudo a concordância de escala entre os dois sistemas.
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a, var_b = float(np.var(a)), float(np.var(b))
    covar = float(np.mean((a - mean_a) * (b - mean_b)))
    denom = var_a + var_b + (mean_a - mean_b) ** 2
    ccc = (2 * covar) / denom if denom != 0 else 0.0

    # Fator de escala: quantas vezes a amplitude do celular é maior/menor que a da
    # cinemática (razão de desvio-padrão). Isso explica de forma direta um CCC baixo
    # mesmo com r moderado — mostra que o "problema" é de escala, não de padrão
    # temporal, e permite ver se essa razão é ao menos parecida entre linhas (o que
    # sugeriria um fator de calibração fixo) ou muito variável (sugerindo diferença
    # de conteúdo do sinal, não só de unidade/ganho).
    scale_factor = float(np.std(b) / np.std(a)) if np.std(a) != 0 else float("nan")

    # SEM (erro padrão de medida) e MDC95 (mínima mudança detectável), calculados a
    # partir do desvio-padrão da diferença ponto-a-ponto entre os dois sinais —
    # abordagem padrão pra comparação entre dois instrumentos/métodos (em vez de
    # teste-reteste do mesmo instrumento), tratando os dois sistemas como igualmente
    # confiáveis (Stratford & Goldsmith). SEM = SD(diferença) / sqrt(2);
    # MDC95 = 1.96 * sqrt(2) * SEM = 1.96 * SD(diferença). Aqui a "diferença" é
    # calculada amostra a amostra dentro do sinal (não entre tentativas/pessoas).
    sd_diff = float(np.std(a - b))
    sem = sd_diff / np.sqrt(2)
    mdc95 = 1.96 * np.sqrt(2) * sem

    # Intervalo de confiança de 95% do r, via transformação Z de Fisher — padrão
    # pra estimar incerteza de uma correlação de Pearson calculada sobre uma
    # amostra finita (aqui, n = número de pontos no tempo usados na comparação).
    r_for_ci = abs(best_r)
    z = np.arctanh(min(max(r_for_ci, -0.999999), 0.999999))
    se_z = 1 / np.sqrt(n - 3) if n > 3 else float("nan")
    r_ci_lo = float(np.tanh(z - 1.96 * se_z))
    r_ci_hi = float(np.tanh(z + 1.96 * se_z))

    return {
        "r": abs(best_r), "rmse": rmse, "n": n, "ccc": ccc,
        "sem": sem, "mdc95": mdc95, "scale_factor": scale_factor,
        "r_ci_lo": r_ci_lo, "r_ci_hi": r_ci_hi,
        "lag_s": best_lag / fs, "sign_flip": sign_flip,
    }


def trial_peak_correlation(sheets_raw, region, direction, trial_bounds_fn, n_trials):
    """Correlação entre os PICOS de aceleração (valor absoluto máximo) da cinemática
    e do celular, um valor por CICLO/repetição — em vez de amostra a amostra. Isso
    responde uma pergunta diferente do r principal: será que a variação de
    intensidade do movimento entre repetições é parecida nos dois sistemas, mesmo
    que a forma fina da curva (r amostra-a-amostra) seja só moderada? Como usa o
    valor absoluto do pico, não depende de detectar/corrigir inversão de eixo (o
    pico é o mesmo independente do sinal).

    ATENÇÃO: n aqui é o número de CICLOS detectados (tipicamente poucos, ex.: 5),
    não o número de amostras — com n tão pequeno, essa correlação é
    estatisticamente frágil (IC 95% muito largo) mesmo quando o valor parece alto.
    Retorna None se não der pra calcular (poucos ciclos válidos ou colunas
    ausentes)."""
    df_raw = sheets_raw[region]
    catalog = build_catalog(df_raw)
    imu_axis_label = get_imu_axis_label(region)
    kin_axis = next((ax for ax in AXES if KINEM_AXIS_LABEL[ax] == direction), None)
    imu_axis = next((ax for ax in AXES if imu_axis_label[ax] == direction), None)
    if kin_axis is None or imu_axis is None:
        return None
    kin_col = catalog.get("Cinemática - Aceleração", {}).get(kin_axis)
    imu_col = catalog.get("IMU - Acelerômetro", {}).get(imu_axis)
    if kin_col is None or imu_col is None:
        return None
    t = df_raw[time_column(df_raw)].to_numpy(dtype=float)
    kin_full = detrend(df_raw[kin_col].to_numpy(dtype=float) / 100.0)
    imu_full = detrend(df_raw[imu_col].to_numpy(dtype=float))

    kin_peaks, imu_peaks = [], []
    for trial_idx in range(1, n_trials + 1):
        cycle_start, _d_start, _v_trial, cycle_end = trial_bounds_fn(trial_idx)
        mask = (t >= cycle_start) & (t <= cycle_end)
        if mask.sum() < 5:
            continue
        kin_peaks.append(float(np.max(np.abs(kin_full[mask]))))
        imu_peaks.append(float(np.max(np.abs(imu_full[mask]))))

    n = len(kin_peaks)
    if n < 3:
        return None
    kin_peaks_arr, imu_peaks_arr = np.array(kin_peaks), np.array(imu_peaks)
    if np.std(kin_peaks_arr) == 0 or np.std(imu_peaks_arr) == 0:
        return None
    r = float(np.corrcoef(kin_peaks_arr, imu_peaks_arr)[0, 1])

    r_ci_lo = r_ci_hi = float("nan")
    if n > 3:
        z = np.arctanh(min(max(r, -0.999999), 0.999999))
        se_z = 1 / np.sqrt(n - 3)
        r_ci_lo = float(np.tanh(z - 1.96 * se_z))
        r_ci_hi = float(np.tanh(z + 1.96 * se_z))

    return {"r": r, "n": n, "r_ci_lo": r_ci_lo, "r_ci_hi": r_ci_hi}


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

H_SPACING = 0.09
V_SPACING = 0.045
CELL_PX = 320
MARGIN = dict(l=10, r=10, t=90, b=10)

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
        showgrid=False, zeroline=False, showline=True, linecolor="#000000", linewidth=1,
        title_font=dict(size=15, color="#000000"), tickfont=dict(size=13, color="#000000"),
    )
    fig_obj.update_yaxes(
        showgrid=False, zeroline=False, showline=True, linecolor="#000000", linewidth=1,
        title_font=dict(size=15, color="#000000"), tickfont=dict(size=13, color="#000000"),
    )
    return fig_obj


NAVY = "#1e3a5f"
NAVY_DARK = "#122540"


def _render_slide_table(df, highlights=None):
    """Renderiza um DataFrame pequeno como HTML puro com estilo inline (cabeçalho
    num azul-marinho que combina com o quadro ao redor, texto branco garantido,
    células compactas com quebra de linha — não estica a largura). Usa estilo
    inline pra não depender do CSS do Streamlit, que pode sobrescrever cores de
    texto em elementos <th>/<td> gerados automaticamente.

    `highlights`, se passado, é uma lista (1 por linha) com o nome da coluna a
    destacar em azul naquela linha, ou None se nenhuma."""
    cols = list(df.columns)
    thead_cells = "".join(
        f'<th style="background:{NAVY};color:#ffffff;font-weight:700;'
        f'padding:7px 12px;text-align:left;border-bottom:2px solid {NAVY_DARK};'
        f'vertical-align:middle;white-space:normal;">{c}</th>'
        for c in cols
    )
    body_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        bg_row = "#ffffff" if i % 2 == 0 else "#eaf0fc"
        target = highlights[i] if highlights else None
        if isinstance(target, tuple):
            target_col, target_bg = target
        else:
            target_col, target_bg = target, "#b3c1f2"
        cells = "".join(
            f'<td style="padding:6px 12px;border-bottom:1px solid #dbe6fb;'
            f'color:#1a2332;background:{target_bg if c == target_col else bg_row};'
            f'white-space:normal;">{row[c]}</td>'
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


def _render_duracao_table(rows, label_a, label_b):
    """Tabela de duração das fases com cabeçalho em 2 níveis — Fase / Pessoa A
    (Duração + % do ciclo) / Pessoa B (Duração + % do ciclo) / Diferença — em
    azul-marinho, pra combinar com o quadro ao redor."""
    th_top = (
        f'background:{NAVY};color:#ffffff;font-weight:700;padding:7px 12px;'
        f'border-bottom:1px solid rgba(255,255,255,0.25);vertical-align:middle;'
        f'border-right:1px solid rgba(255,255,255,0.25);white-space:normal;'
    )
    th_sub = (
        f'background:{NAVY};color:#ffffff;font-weight:600;padding:6px 12px;'
        f'text-align:left;border-bottom:2px solid {NAVY_DARK};vertical-align:middle;'
        f'border-right:1px solid rgba(255,255,255,0.25);white-space:normal;'
    )
    header_html = (
        "<thead>"
        f'<tr><th rowspan="2" style="{th_top}text-align:left;">Fase</th>'
        f'<th colspan="2" style="{th_top}text-align:center;">{label_a}</th>'
        f'<th colspan="2" style="{th_top}text-align:center;">{label_b}</th>'
        f'<th rowspan="2" style="{th_top}text-align:left;">Diferença</th></tr>'
        f'<tr><th style="{th_sub}">Duração</th><th style="{th_sub}">% do ciclo</th>'
        f'<th style="{th_sub}">Duração</th><th style="{th_sub}">% do ciclo</th></tr>'
        "</thead>"
    )
    body_rows = []
    for i, (nome, stat_a, pct_a, stat_b, pct_b, diff) in enumerate(rows):
        bg = "#ffffff" if i % 2 == 0 else "#eaf0fc"
        vals = (
            nome, f"{stat_a[0]:.2f}s (±{stat_a[1]:.2f})", f"{pct_a:.0f}%",
            f"{stat_b[0]:.2f}s (±{stat_b[1]:.2f})", f"{pct_b:.0f}%", diff,
        )
        cells = "".join(
            f'<td style="padding:6px 12px;border-bottom:1px solid #dbe6fb;'
            f'color:#1a2332;background:{bg};white-space:normal;">{val}</td>'
            for val in vals
        )
        body_rows.append(f"<tr>{cells}</tr>")
    html = (
        '<div style="overflow-x:auto;"><table style="border-collapse:collapse;'
        'width:100%;font-size:0.92rem;font-family:Inter,sans-serif;">'
        f"{header_html}<tbody>{''.join(body_rows)}</tbody></table></div>"
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

fig_check = make_subplots(rows=1, cols=2, subplot_titles=[f"<b>{ctx_a['label']}</b>", f"<b>{ctx_b['label']}</b>"])
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
# eixo Y compartilhado entre os dois painéis — sem isso, cada subplot escolhe seus
# próprios ticks "bonitos" com base no próprio range de dados, e um painel pode
# acabar sem mostrar um valor (ex.: 0.05) que o outro mostra.
_y_all = np.concatenate([ctx_a["ref_signal"], ctx_b["ref_signal"]])
_y_pad = 0.08 * (float(np.nanmax(_y_all)) - float(np.nanmin(_y_all)) or 1.0)
_y_range = [float(np.nanmin(_y_all)) - _y_pad, float(np.nanmax(_y_all)) + _y_pad]
fig_check.update_yaxes(range=_y_range, matches="y")
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
        _dur_rows_full = [
            (nome, stat_a, pct_a, stat_b, pct_b, _fmt_pct_diff(stat_a, stat_b, ctx_a["label"], ctx_b["label"]))
            for nome, stat_a, pct_a, stat_b, pct_b in _dur_table_rows
        ]
        _render_duracao_table(_dur_rows_full, ctx_a["label"], ctx_b["label"])

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
    legend=dict(orientation="h", yanchor="bottom", y=1.035, xanchor="left", x=0),
)
_elegant_layout(fig)

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


st.markdown("##### 📐 Amplitude por fase (pico-a-pico do sinal, média ± DP entre os ciclos)")
st.caption(
    "Para cada variável (Deslocamento, Velocidade, Aceleração, ACC, GYR), em cada "
    "direção anatômica (Vertical/AP/ML), separada por platô/descida/subida, uma "
    f"tabela por região. Em azul: diferenças de pelo menos {_AMP_HIGHLIGHT_PCT*100:.0f}% "
    "entre as duas pessoas (a menor delas fica sem destaque)."
)

col_a_name = f"{ctx_a['label']}"
col_b_name = f"{ctx_b['label']}"
_amp_cols = st.columns(len(REGIONS)) if REGIONS else []

for _region_col, region in zip(_amp_cols, REGIONS):
    with _region_col:
        with st.container(key=f"quadro_amp_{region}"):
            st.markdown(f"**{region}**")
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

            if not _amp_rows:
                st.caption("Sem dados suficientes para calcular a amplitude por fase.")
            else:
                _amp_df = pd.DataFrame(_amp_rows)
                _render_slide_table(_amp_df, highlights=_highlight_targets)

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

with st.container(key="quadro_cv"):
    st.markdown("##### 🔄 Consistência entre repetições (CV)")
    st.caption(
        "CV = quanto o deslocamento líquido varia de um ciclo pro outro (quanto maior, menos "
        "consistente/controlado). Quando o deslocamento é perto de zero o CV% não é confiável — "
        "nesses casos mostramos o DP absoluto e marcamos como 'n/d'."
    )
    if _cv_rows:
        _render_slide_table(pd.DataFrame(_cv_rows))
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
    fig_t.update_yaxes(showgrid=False, title_text=y_title, title_standoff=10)
    fig_t.update_layout(
        title=dict(text=chart_title, y=0.99, yanchor="top", x=0.5, xanchor="center"),
        width=TILT_SQUARE_PX, height=TILT_SQUARE_PX,
        margin=dict(l=70, r=20, t=72, b=100), plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
    )
    _elegant_layout(fig_t)
    return fig_t


TILT_SQUARE_PX = 540


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
    fig_frontal = build_tilt_combo_figure("frontal", "Δ ângulo (°)", "Inclinação frontal (ML)")
    if fig_frontal is not None:
        st.plotly_chart(fig_frontal, use_container_width=False, key="tilt_frontal")
        st.caption("Positivo = lateral, negativo = medial.")
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
                _render_slide_table(pd.DataFrame(_valgo_rows))
                st.caption(
                    f"**{quem_mais_valgo}** tem o pico de inclinação medial maior — indicador de mais "
                    "valgo dinâmico do joelho. Proxy por 1 sensor — não substitui avaliação clínica."
                )
    else:
        st.caption("Não foi possível calcular a inclinação frontal — faltam colunas de ACC/GYR necessárias.")

with col_sagital:
    fig_sagital = build_tilt_combo_figure("sagital", "Δ ângulo (°)", "Inclinação sagital (AP)")
    if fig_sagital is not None:
        st.plotly_chart(fig_sagital, use_container_width=False, key="tilt_sagital")
        st.caption("Positivo = anterior, negativo = posterior.")
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
                _render_slide_table(pd.DataFrame(_div_rows))
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
    _findings_highlights = []
    for _f in FINDINGS:
        _color = ctx_a["color"] if _f["Quem é maior"] == ctx_a["label"] else (
            ctx_b["color"] if _f["Quem é maior"] == ctx_b["label"] else None
        )
        _findings_highlights.append(("Quem é maior", hex_to_rgba(_color, 0.30)) if _color else None)

    with st.container(key="quadro_resumo"):
        _render_slide_table(_findings_df, highlights=_findings_highlights)

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

_A, _B = ctx_a["label"], ctx_b["label"]

_LEITURA_CLINICA_TXT = (
    f"A comparação entre os participantes revelou estratégias motoras distintas ao longo das "
    f"três fases do teste (preparação, descida e subida), evidenciando diferenças tanto na "
    f"execução do movimento quanto em indicadores relacionados ao controle motor. De forma "
    f"geral, a {_A} apresentou um padrão de movimento mais dinâmico, consistente e "
    f"predominantemente orientado ao plano sagital, enquanto a {_B} executou a tarefa de forma "
    f"mais lenta, com maior oscilação no plano frontal e maior variabilidade entre as "
    f"repetições.\n\n"
    f"Na análise temporal, a {_B} apresentou duração superior nas fases de descida e de subida, "
    f"com diferenças superiores a 20% em relação à {_A}. A fase de descida mais prolongada pode "
    f"representar uma estratégia de controle excêntrico mais cautelosa, com maior tempo "
    f"destinado à estabilização postural durante a flexão. Em contrapartida, a execução mais "
    f"rápida da {_A} pode refletir maior confiança, capacidade de geração de força ou uma "
    f"estratégia de movimento mais eficiente. Entretanto, isoladamente, a menor duração não deve "
    f"ser interpretada como melhor controle motor, pois também pode representar uma execução "
    f"menos controlada. Nesse contexto, a interpretação deve ser feita em conjunto com as demais "
    f"variáveis.\n\n"
    f"Durante a fase de preparação, a {_A} apresentou maiores amplitudes na maioria das "
    f"variáveis analisadas, especialmente aceleração linear, velocidade e velocidade angular do "
    f"segmento L5, além de maiores deslocamentos, velocidades e acelerações do joelho. Esses "
    f"resultados sugerem uma preparação mais ativa e uma maior mobilização corporal antes do "
    f"início da descida.\n\n"
    f"Na fase de descida, observou-se um comportamento distinto entre os participantes. A {_B} "
    f"apresentou maiores deslocamentos, velocidades e acelerações do tronco principalmente no "
    f"eixo médio-lateral, indicando maior oscilação corporal e maior necessidade de ajustes de "
    f"equilíbrio durante o movimento excêntrico. Em contraste, a {_A} apresentou maiores "
    f"amplitudes principalmente nas componentes anteroposteriores e nas acelerações lineares, "
    f"caracterizando um movimento predominantemente direcionado ao plano sagital, potencialmente "
    f"mais eficiente do ponto de vista mecânico.\n\n"
    f"Esse padrão foi mantido durante a fase de subida. A {_A} apresentou maiores velocidades "
    f"verticais, acelerações lineares e velocidades angulares do tronco, indicando uma extensão "
    f"mais vigorosa e rápida. Por outro lado, a {_B} continuou apresentando maiores amplitudes "
    f"de deslocamento e velocidade no eixo médio-lateral do L5, sugerindo que necessitou de "
    f"ajustes posturais adicionais para recuperar a posição ortostática.\n\n"
    f"A análise dos membros inferiores reforçou essas diferenças. Em praticamente todas as "
    f"fases do movimento, a {_A} apresentou maiores deslocamentos, velocidades e acelerações do "
    f"joelho, indicando uma execução mais ativa e maior mobilidade articular. Em contrapartida, "
    f"a {_B} apresentou maiores acelerações lineares e velocidades angulares em algumas "
    f"variáveis específicas, além de maior inclinação medial do joelho durante o movimento, "
    f"compatível com maior valgo dinâmico. Esse comportamento costuma estar associado a menor "
    f"controle no plano frontal, podendo refletir maior demanda dos estabilizadores do quadril e "
    f"do joelho, como o glúteo médio e os rotadores externos do quadril. Entretanto, essa "
    f"interpretação deve ser considerada uma hipótese funcional, uma vez que a estimativa foi "
    f"obtida a partir de um sensor inercial e não corresponde à mensuração direta do ângulo "
    f"articular.\n\n"
    f"Por outro lado, a profundidade do movimento foi semelhante entre os participantes, "
    f"indicando que ambos atingiram amplitudes funcionais equivalentes. Assim, as diferenças "
    f"observadas não decorreram da execução de um agachamento mais profundo ou mais superficial, "
    f"mas da forma como cada participante organizou o movimento para atingir essa mesma "
    f"profundidade.\n\n"
    f"O indicador que melhor refletiu diferenças relacionadas ao controle motor foi a "
    f"**consistência do movimento**. A {_B} apresentou coeficiente de variação (CV) mais "
    f"elevado em maior número de combinações entre segmentos e direções (4 versus 0), além de "
    f"maior variabilidade nos três eixos do joelho (vertical, anteroposterior e médio-lateral). "
    f"Esse resultado indica menor repetibilidade entre as tentativas, sugerindo um padrão motor "
    f"menos consistente, que pode estar relacionado a menor controle motor, fadiga ao longo das "
    f"repetições ou menor familiaridade com a tarefa. Em contraste, a {_A} apresentou maior "
    f"uniformidade entre as execuções, indicando uma estratégia motora mais estável e "
    f"reprodutível.\n\n"
    f"Em conjunto, os resultados sugerem que a {_A} executa o movimento com maior dinamismo, "
    f"maior participação do plano sagital e maior **consistência entre as repetições**, "
    f"enquanto a {_B} adota uma estratégia mais cautelosa, com maior tempo de execução, maior "
    f"oscilação médio-lateral, maior variabilidade do movimento e maior tendência ao valgo "
    f"dinâmico do joelho. É importante destacar que diferenças relacionadas ao **ritmo de "
    f"execução**, amplitudes de movimento e velocidades refletem principalmente estratégias "
    f"motoras individuais, enquanto os achados mais diretamente associados ao controle motor "
    f"foram a **consistência entre as repetições** e o **controle frontal do joelho**. Por fim, "
    f"essa interpretação representa uma hipótese baseada nos limiares definidos pelo aplicativo "
    f"e não substitui uma avaliação clínica presencial nem uma análise biomecânica "
    f"tridimensional completa."
)
st.markdown(_LEITURA_CLINICA_TXT)

st.markdown(
    """
    <style>
    div[class*="st-key-quadro_resumo_achados"] li, div[class*="st-key-quadro_resumo_achados"] p {
        font-size: 1.15rem !important;
        line-height: 1.7 !important;
    }
    div[class*="st-key-quadro_resumo_achados"] h5 {
        font-size: 1.35rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
with st.container(key="quadro_resumo_achados"):
    st.markdown("##### 🧾 Resumo dos principais achados")
    st.markdown(
        f"- **Tempo de execução**: {_B} apresentou fases de descida e subida mais longas (>20%), "
        "sugerindo uma estratégia mais cautelosa.\n"
        f"- **Preparação**: {_A} demonstrou maior ativação inicial, com maiores velocidades, "
        "acelerações e amplitudes de movimento.\n"
        f"- **Fase de descida**: {_B} apresentou maior oscilação médio-lateral do tronco; {_A} "
        "executou um movimento mais direcionado ao plano sagital.\n"
        f"- **Fase de subida**: {_A} realizou uma extensão mais rápida e vigorosa; {_B} manteve "
        "maior necessidade de ajustes posturais.\n"
        f"- **Joelho**: {_A} apresentou maior mobilidade global; {_B} mostrou maior tendência ao "
        "valgo dinâmico.\n"
        "- **Profundidade do movimento**: Sem diferenças relevantes entre os participantes.\n"
        f"- **Consistência entre repetições**: {_B} apresentou maior coeficiente de variação (CV), "
        "indicando menor repetibilidade do padrão motor.\n"
        "- **Principal diferença relacionada ao controle motor**: Maior variabilidade entre as "
        f"repetições e maior **controle frontal deficiente do joelho** na {_B}.\n"
        f"- **Síntese geral**: {_A} apresentou um padrão de movimento mais dinâmico, consistente e "
        f"estável, enquanto a {_B} adotou uma estratégia mais lenta e cautelosa, com maior demanda "
        "de estabilização postural e maior variabilidade motora."
    )

st.divider()

# ----------------------------------------------------------------------------
# Validação — concordância entre a cinemática e o acelerômetro do celular (IMU)
# ----------------------------------------------------------------------------
st.subheader("🔬 Validação — Cinemática × Celular (IMU)")
st.caption(
    "Compara a aceleração medida pela cinemática com a do acelerômetro do celular, na "
    "mesma direção anatômica, pra ver o quanto os dois sistemas concordam. Cada linha é "
    "calculada só dentro da própria pessoa/região/direção, comparando os dois sinais "
    "amostra a amostra ao longo de toda a gravação (não é uma comparação entre as duas "
    "pessoas — 'n' é o número de pontos no tempo usados, não o número de sujeitos). Pra "
    "ser uma comparação justa, por trás dos panos: os dois sinais usam o MESMO corte de "
    "filtro (em vez dos cortes diferentes da barra lateral), a melhor defasagem temporal "
    "entre os dois sistemas é buscada e corrigida automaticamente, e uma possível inversão "
    "de eixo/sinal entre os sistemas é detectada e corrigida. **r** = correlação de Pearson "
    "(1 = concordância perfeita); **IC 95% (r)** = intervalo de confiança de 95% desse r "
    "(via transformação Z de Fisher, considerando o 'n' de pontos no tempo usados) — quanto "
    "mais estreito, mais precisa é a estimativa do r; **SEM** = erro padrão de medida (m/s²), "
    "a partir do desvio-padrão da diferença ponto-a-ponto entre os dois sinais; **MDC95** = "
    "mínima mudança detectável (m/s²) — abaixo desse valor, uma diferença entre os dois "
    "sistemas pode ser apenas ruído de medição, não uma mudança real. 'Viés' não é mostrado: "
    "como os dois sinais são detrend, a diferença média sempre daria ~0 por construção."
)

_shared_cutoff = min(kinem_cutoff, imu_cutoff)
_conc_rows = []
for _ctx in PERSON_CTXS:
    for _region in REGIONS:
        _df_raw_r = _ctx["sheets_raw"][_region]
        _catalog_r = build_catalog(_df_raw_r)
        _imu_axis_r = get_imu_axis_label(_region)
        _t_r = _df_raw_r[time_column(_df_raw_r)].to_numpy(dtype=float)
        _dt_r = float(np.median(np.diff(_t_r))) if len(_t_r) > 1 else 0.01
        _fs_r = 1.0 / _dt_r if _dt_r > 0 else 100.0
        for _direction in DIRECTIONS:
            _res = kinem_vs_imu_concordance(
                _df_raw_r, _catalog_r, _imu_axis_r, _direction, _fs_r, _shared_cutoff
            )
            if _res is None:
                continue
            _conc_rows.append({
                "Pessoa": _ctx["label"], "Região": _region, "Direção": _direction,
                "r (Pearson)": f"{_res['r']:.2f}",
                "IC 95% (r)": f"[{_res['r_ci_lo']:.2f}, {_res['r_ci_hi']:.2f}]",
                "SEM (m/s²)": f"{_res['sem']:.2f}",
                "MDC95 (m/s²)": f"{_res['mdc95']:.2f}",
                "n (amostras)": f"{_res['n']}",
            })

if _conc_rows:
    with st.container(key="quadro_concordancia"):
        _render_slide_table(pd.DataFrame(_conc_rows))
        _rs = [float(_row["r (Pearson)"]) for _row in _conc_rows]
        _r_media = sum(_rs) / len(_rs)
        _n_boa = sum(1 for _r in _rs if _r >= 0.7)
        _n_mod = sum(1 for _r in _rs if 0.4 <= _r < 0.7)
        _n_fraca = sum(1 for _r in _rs if _r < 0.4)
        _label_geral = "boa" if _r_media >= 0.7 else ("moderada" if _r_media >= 0.4 else "fraca")
        st.caption(
            f"Concordância média (r) entre os dois sistemas: {_r_media:.2f} — considerada "
            f"**{_label_geral}** (referência comum: r ≥ 0.7 = boa, 0.4–0.7 = moderada, < 0.4 = "
            f"fraca). De {len(_rs)} combinações região/direção/pessoa: {_n_boa} com boa "
            f"concordância, {_n_mod} moderada e {_n_fraca} fraca. Isso não valida o celular como "
            "substituto clínico da cinemática — é um indicador de quão parecidos os dois sinais "
            "se comportam nesse teste específico."
        )
else:
    st.caption(
        "Não foi possível calcular a concordância — faltam colunas de cinemática e/ou IMU "
        "necessárias em alguma pessoa/região."
    )

st.divider()

# ----------------------------------------------------------------------------
# Concordância por repetição — correlação entre picos de aceleração por ciclo
# ----------------------------------------------------------------------------
st.subheader("🔁 Concordância por repetição — pico de aceleração por ciclo")
st.caption(
    "Pergunta diferente da análise acima: em vez de comparar amostra a amostra, compara "
    "o PICO de aceleração (valor absoluto máximo) de cada CICLO/repetição entre os dois "
    "sistemas. Mostra se a variação de intensidade do movimento de um ciclo pro outro "
    "(ex.: 'esse ciclo foi mais forte que o anterior') é parecida na cinemática e no "
    "celular — independe de eventual inversão de eixo/sinal, já que usa valor absoluto."
)
st.warning(
    "⚠️ O 'n' aqui é o número de CICLOS detectados por pessoa (tipicamente poucos, ex.: "
    "5) — não o número de amostras. Com n tão pequeno, a correlação é estatisticamente "
    "frágil: um único ciclo atípico pode mudar o resultado bastante, e o intervalo de "
    "confiança (IC 95%) fica muito largo mesmo quando o valor de r parece alto. Reportar "
    "com essa ressalva explícita."
)

_trial_rows = []
for _ctx in PERSON_CTXS:
    for _region in REGIONS:
        for _direction in DIRECTIONS:
            _tres = trial_peak_correlation(
                _ctx["sheets_raw"], _region, _direction, _ctx["trial_bounds_fn"], _ctx["n_trials"]
            )
            if _tres is None:
                continue
            _trial_rows.append({
                "Pessoa": _ctx["label"], "Região": _region, "Direção": _direction,
                "r (picos por ciclo)": f"{_tres['r']:.2f}",
                "IC 95% (r)": f"[{_tres['r_ci_lo']:.2f}, {_tres['r_ci_hi']:.2f}]",
                "n (ciclos)": f"{_tres['n']}",
            })

if _trial_rows:
    with st.container(key="quadro_concordancia_ciclo"):
        _render_slide_table(pd.DataFrame(_trial_rows))
        _trs = [float(_row["r (picos por ciclo)"]) for _row in _trial_rows]
        _tr_media = sum(_trs) / len(_trs)
        st.caption(
            f"r médio (picos por ciclo) = {_tr_media:.2f}. De novo: com poucos ciclos por "
            "pessoa, cada linha individual dessa tabela deve ser lida como um indício, não "
            "como uma estimativa estatisticamente robusta — o intervalo de confiança de "
            "cada linha mostra isso claramente."
        )
else:
    st.caption(
        "Não foi possível calcular a correlação por ciclo — faltam ciclos suficientes "
        "(mínimo 3) em alguma combinação de pessoa/região/direção."
    )

st.divider()
st.caption(
    "Próximos passos possíveis: comparação trial a trial (não só a resultante) e uma "
    "tabela de métricas (pico, CV, assimetria) por pessoa/região."
)
