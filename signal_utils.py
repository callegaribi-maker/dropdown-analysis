"""
signal_utils.py
────────────────
Funções puras de leitura, sincronização e processamento de sinais para o
Visualizador de Sinais (Y-Balance & Step-Down). Não dependem do Streamlit,
o que facilita testes unitários e reuso.
"""

from __future__ import annotations

import io
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy import signal as sp_signal

NONE_LABEL = "— nenhum —"


# ──────────────────────────────────────────────
# Normalização de texto / nomes de coluna
# ──────────────────────────────────────────────

def norm(s) -> str:
    """Minúsculas + remove acentos, para comparação robusta de nomes."""
    return "".join(
        c for c in unicodedata.normalize("NFD", str(s).lower())
        if unicodedata.category(c) != "Mn"
    )


def col_default(cols: list[str], keywords: list[str]) -> int:
    """Índice da primeira coluna cujo nome contém algum keyword (normalizado)."""
    normed_cols = [norm(c) for c in cols]
    for kw in keywords:
        kw_n = norm(kw)
        for i, cn in enumerate(normed_cols):
            if kw_n in cn:
                return i
    return 0


def best_match(names: list[str], *kw_sets: tuple[str, ...]) -> int:
    """
    Índice (1-based, deslocado por NONE_LABEL na frente da lista) do primeiro
    nome que contém todas as palavras-chave de algum kw_set, em ordem de
    prioridade dos kw_sets.
    """
    for kws in kw_sets:
        for i, n in enumerate(names):
            if all(k in n.lower() for k in kws):
                return i + 1
    return 0


# ──────────────────────────────────────────────
# Leitura de arquivos
# ──────────────────────────────────────────────

def try_numeric(series: pd.Series) -> pd.Series:
    """Converte série para numérico, aceitando vírgula decimal."""
    try:
        return pd.to_numeric(
            series.astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )
    except Exception:
        return pd.to_numeric(series, errors="coerce")


def load_file(uploaded_file) -> pd.DataFrame | None:
    """
    Tenta várias combinações de encoding/separador até conseguir ler um CSV/TXT
    com mais de uma coluna, convertendo colunas numéricas quando possível.
    """
    content = uploaded_file.read()
    uploaded_file.seek(0)

    for enc in ["utf-8-sig", "utf-8", "latin-1", "cp1252", "iso-8859-1"]:
        for sep in [";", ",", "\t", r"\s+"]:
            try:
                df = pd.read_csv(
                    io.BytesIO(content), sep=sep, engine="python",
                    encoding=enc, on_bad_lines="skip",
                )
            except Exception:
                continue
            if df.shape[1] <= 1:
                continue
            for col in df.columns:
                conv = try_numeric(df[col])
                if conv.notna().sum() > len(df) * 0.5:
                    df[col] = conv
            return df
    return None


def numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


# ──────────────────────────────────────────────
# Classificação / rotulagem de colunas
# ──────────────────────────────────────────────

_AXIS_EXCLUDE_TERMS = ("abs", "magnitude", "length", "norma", "mag", "len", "norm")
_AXIS_RE = re.compile(r'(?:^|[_\s\(])([xyz])(?:[_\s\)]|$)')
_SUFFIX_AXIS_RE = re.compile(r'[xyz]$')
_LENGTH_COL_RE = re.compile(r'\bl\(')


def is_xyz_col(col: str) -> bool:
    """True se a coluna representar um eixo X, Y ou Z (exclui abs/magnitude/etc.)."""
    cn = norm(col).lower()
    if any(term in cn for term in _AXIS_EXCLUDE_TERMS):
        return False
    return bool(_AXIS_RE.search(cn))


def axis_label(fname, col, kinem_ref, l5_acc, l5_gyr, knee_acc, knee_gyr) -> str:
    """
    Rótulo anatômico do eixo (ex.: 'Vertical', 'ML', 'AP').

    Celular:  X = Mediolateral, Y = Vertical, Z = Anteroposterior
    Kinem:    X = Mediolateral, Y = Anteroposterior, Z = Vertical
    """
    cn = norm(col)

    axis = None
    for ax in ("x", "y", "z"):
        if f"({ax})" in cn:
            axis = ax
            break
    if axis is None:
        for ax in ("z", "y", "x"):  # 'z' primeiro para não confundir com "kx"
            if cn.rstrip().endswith(ax):
                axis = ax
                break
    if axis is None:
        return ""

    is_l5_phone = fname in (l5_acc, l5_gyr)
    is_knee_phone = fname in (knee_acc, knee_gyr)
    is_kinem = fname == kinem_ref

    if is_l5_phone:
        mapping = {"x": "ML", "y": "Vertical", "z": "AP"}
    elif is_knee_phone:
        mapping = {"x": "AP", "y": "Vertical", "z": "ML"}
    elif is_kinem:
        mapping = {"x": "ML", "y": "AP", "z": "Vertical"}
    else:
        return ""

    return mapping.get(axis, "")


def display_col_name(fname, col, kinem_ref, l5_acc, l5_gyr, knee_acc, knee_gyr) -> str:
    """Nome original da coluna + rótulo anatômico entre parênteses, quando houver."""
    lbl = axis_label(fname, col, kinem_ref, l5_acc, l5_gyr, knee_acc, knee_gyr)
    return f"{col}  ({lbl})" if lbl else col


def classify_trace(fname, col, kinem_ref, l5_acc, l5_gyr, knee_acc, knee_gyr) -> str:
    """Classifica um traço como 'l5', 'joelho' ou 'outro'."""
    if fname in (l5_acc, l5_gyr):
        return "l5"
    if fname in (knee_acc, knee_gyr):
        return "joelho"
    if fname == kinem_ref:
        cn = norm(col)
        if "l5" in cn or "l 5" in cn:
            return "l5"
        if any(k in cn for k in ("condilo", "joelho", "knee", "patela")):
            return "joelho"
    return "outro"


def kinem_cols_for_body(df: pd.DataFrame, *body_keywords: str) -> list[str]:
    """
    Colunas do Kinem pertencentes a uma região anatômica.

    Inclui colunas cujo nome contém algum body_keyword E (termina em X/Y/Z
    ou contém (X)/(Y)/(Z)). Exclui comprimento ("l("), valores absolutos e
    métricas 2D.
    """
    result = []
    for col in df.columns:
        cn = norm(col).lower().strip()
        if not any(kw in cn for kw in body_keywords):
            continue
        if "abs" in cn or "length" in cn or "#2d" in cn or _LENGTH_COL_RE.search(cn):
            continue
        has_paren_axis = any(f"({ax})" in cn for ax in ("x", "y", "z"))
        has_suffix_axis = bool(_SUFFIX_AXIS_RE.search(cn))
        if has_paren_axis or has_suffix_axis:
            result.append(col)
    return result


# ──────────────────────────────────────────────
# Eixo de tempo / reamostragem
# ──────────────────────────────────────────────

_TIME_MS_NAMES = {"tempoms", "tempo_ms", "time_ms", "timestamp_ms"}
_TIME_S_NAMES = {"time", "tempo", "t", "timestamp", "tempo (s)", "time (s)"}


def detect_time_axis(df: pd.DataFrame):
    """
    Detecta a coluna de tempo. Retorna (tempo_em_segundos, nome_coluna)
    ou (None, None) se nenhuma coluna reconhecida for encontrada.
    """
    for col in df.columns:
        cl = str(col).lower().strip()
        if cl in _TIME_MS_NAMES:
            return df[col].values.astype(float) / 1000.0, col
        if cl in _TIME_S_NAMES:
            return df[col].values.astype(float), col
    return None, None


def resample_to_regular(df: pd.DataFrame, fs_target: float):
    """
    Reamostra df para uma grade regular em fs_target Hz, usando o eixo de
    tempo real detectado. Retorna (df_reamostrado, fs_original, descrição).
    """
    t, time_col = detect_time_axis(df)
    if t is None:
        return df, None, "sem coluna de tempo (não reamostrado)"

    data_cols = [c for c in df.columns if c != time_col]
    t_norm = t - t[0]
    duration = t_norm[-1]
    fs_orig = (len(t) - 1) / duration if duration > 0 else fs_target

    n_target = max(2, int(round(duration * fs_target)))
    t_target = np.linspace(0, duration, n_target)

    result = {}
    for col in data_cols:
        y = df[col].values
        if np.issubdtype(np.array(y).dtype, np.number):
            y = np.where(np.isnan(y.astype(float)), 0.0, y.astype(float))
            f_interp = interpolate.interp1d(
                t_norm, y, kind="linear", bounds_error=False, fill_value="extrapolate",
            )
            result[col] = f_interp(t_target)

    return pd.DataFrame(result), fs_orig, f"~{fs_orig:.0f} Hz → {fs_target} Hz"


# ──────────────────────────────────────────────
# Filtros
# ──────────────────────────────────────────────

def apply_detrend(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in numeric_cols(df):
        result[col] = sp_signal.detrend(df[col].fillna(0).values)
    return result


def apply_lowpass(df: pd.DataFrame, fs: float, cutoff_hz: float, order: int = 4) -> pd.DataFrame:
    result = df.copy()
    nyq = fs / 2.0
    if cutoff_hz >= nyq:
        return result
    sos = sp_signal.butter(order, cutoff_hz / nyq, btype="low", output="sos")
    for col in numeric_cols(df):
        y = df[col].fillna(0).values
        result[col] = sp_signal.sosfiltfilt(sos, y)
    return result


# ──────────────────────────────────────────────
# Detecção de pico / sincronização
# ──────────────────────────────────────────────

def _impact_envelope(v, fs: float = 100.0) -> np.ndarray:
    """Envelope de impacto: valor absoluto do sinal filtrado passa-alta."""
    v = np.asarray(v, dtype=float)
    if len(v) < 12:
        return np.abs(v - np.mean(v))
    nyq = fs / 2.0
    cutoff = min(1.0, nyq * 0.95)
    sos = sp_signal.butter(2, cutoff / nyq, btype="high", output="sos")
    return np.abs(sp_signal.sosfiltfilt(sos, v))


def find_highest_peak(series: pd.Series, search_end: int, fs: float = 100.0) -> int:
    """Índice do pico de maior amplitude no envelope de impacto, dentro da janela."""
    raw = try_numeric(series).fillna(0).values[:search_end].astype(float)
    if len(raw) == 0:
        return 0
    vals = _impact_envelope(raw, fs)
    max_val = vals.max()
    if max_val == 0:
        return int(np.argmax(vals))
    peaks, _ = sp_signal.find_peaks(vals, prominence=max_val * 0.30)
    if len(peaks) == 0:
        return int(np.argmax(vals))
    return int(peaks[np.argmax(vals[peaks])])


def _local_corr(kinem_vals, phone_vals, kinem_peak: int, phone_peak: int, fs: float) -> float:
    """Correlação local (±1s) entre os envelopes de impacto ao redor de dois picos candidatos."""
    win = int(fs)
    ks, ke = max(0, kinem_peak - win), min(len(kinem_vals), kinem_peak + win)
    ps, pe = max(0, phone_peak - win), min(len(phone_vals), phone_peak + win)
    k_seg = _impact_envelope(kinem_vals[ks:ke], fs)
    p_seg = _impact_envelope(phone_vals[ps:pe], fs)
    n = min(len(k_seg), len(p_seg))
    if n < 4:
        return 0.0
    k_seg, p_seg = k_seg[:n], p_seg[:n]
    if k_seg.std() == 0 or p_seg.std() == 0:
        return 0.0
    return float(abs(np.corrcoef(k_seg, p_seg)[0, 1]))


def find_sync_xcorr(kinem_ser, phone_ser, kinem_peak: int, search_end: int, fs: float) -> int:
    """
    Estima o índice de pico do sinal do celular que melhor corresponde ao pico
    do Kinem, combinando detecção de pico simples e correlação cruzada;
    escolhe o candidato com maior correlação local.
    """
    k_vals = try_numeric(kinem_ser).fillna(0).values.astype(float)
    p_vals = try_numeric(phone_ser).fillna(0).values[:search_end].astype(float)

    p_simple = find_highest_peak(pd.Series(p_vals), len(p_vals), fs)

    half_tpl = int(2 * fs)
    k_start = max(0, kinem_peak - half_tpl)
    k_end = min(len(k_vals), kinem_peak + half_tpl)
    k_seg = k_vals[k_start:k_end]

    p_xcorr = None
    if len(p_vals) >= len(k_seg) + 1 and len(k_seg) >= 4:
        ref_env = _impact_envelope(k_seg, fs)
        phone_env = _impact_envelope(p_vals, fs)
        corr = np.correlate(phone_env, ref_env, mode="valid")
        lag = int(np.argmax(corr))
        candidate = lag + (kinem_peak - k_start)
        if 0 <= candidate < search_end:
            p_xcorr = candidate

    if p_xcorr is None:
        return p_simple

    c_simple = _local_corr(k_vals, p_vals, kinem_peak, p_simple, fs)
    c_xcorr = _local_corr(k_vals, p_vals, kinem_peak, p_xcorr, fs)
    return p_xcorr if c_xcorr > c_simple else p_simple


# ──────────────────────────────────────────────
# Alinhamento entre arquivos
# ──────────────────────────────────────────────

def get_aligned_data(files_data: dict, offsets: dict, peak_ref: int, ref_file: str | None = None):
    """
    Alinha todos os arquivos usando os offsets calculados, recortando para a
    janela comum. ref_file define o fim da janela (comprimento de referência);
    arquivos mais curtos que a janela são preenchidos com NaN.

    Retorna (dict_alinhado, eixo_x_em_amostras, mensagem_info) ou
    (None, None, mensagem_erro) se não houver sobreposição.
    """
    common_start = int(max(offsets.get(f, 0) for f in files_data))

    if ref_file and ref_file in files_data:
        common_end = int(offsets.get(ref_file, 0) + len(files_data[ref_file]))
    else:
        common_end = int(min(offsets.get(f, 0) + len(df) for f, df in files_data.items()))

    if common_start >= common_end:
        return None, None, "Sem sobreposição após sincronização."

    n = common_end - common_start
    aligned = {}
    short_files = []

    for fname, df in files_data.items():
        s = offsets.get(fname, 0)
        i_start, i_end = int(common_start - s), int(common_end - s)
        a_start, a_end = max(0, i_start), min(len(df), i_end)

        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        chunk = df.iloc[a_start:a_end][num_cols].reset_index(drop=True)

        pad_before = a_start - i_start
        pad_after = n - pad_before - len(chunk)

        if pad_before > 0 or pad_after > 0:
            short_files.append(f"{fname} (faltam {max(0, pad_after)} amostras no fim)")
            rows = {
                col: np.concatenate([
                    np.full(pad_before, np.nan),
                    chunk[col].values,
                    np.full(max(0, pad_after), np.nan),
                ])
                for col in num_cols
            }
            aligned[fname] = pd.DataFrame(rows)
        else:
            aligned[fname] = chunk

    peak_in_window = int(peak_ref - common_start)
    x_axis = np.arange(n) - peak_in_window
    info = f"Janela: **{n} amostras** ({n/100:.1f} s) | pico em **x = 0**"
    if short_files:
        info += f"  ⚠️ arquivos mais curtos que o Kinem: {', '.join(short_files)}"
    return aligned, x_axis, info


# ──────────────────────────────────────────────
# Ângulo do joelho
# ──────────────────────────────────────────────

# Mapeamento de eixo do celular → rótulo anatômico, por "papel" do sensor.
# 'l5': sensor na lombar. 'limb': sensor em segmento de membro (coxa, tornozelo).
PHONE_AXIS_ROLES = {
    "l5": {"x": "ML", "y": "Vertical", "z": "AP"},
    "limb": {"x": "AP", "y": "Vertical", "z": "ML"},
}

# Para cada plano anatômico: quais dois eixos do acelerômetro formam o plano
# (referência de gravidade) e qual eixo do giroscópio gira nesse plano.
# Sagital (flexão/extensão) gira em torno do eixo ML; frontal (valgo/varo)
# gira em torno do eixo AP. O plano transverso (rotação interna/externa) NÃO
# tem componente de gravidade própria — o acelerômetro não enxerga rotação em
# torno do eixo vertical — por isso não entra nesse esquema (ver
# knee_rotation_from_phone, que usa só giroscópio, sem correção).
_PHONE_PLANE_CONFIG = {
    "sagittal": {"accel_a": "AP", "accel_b": "Vertical", "gyro_axis": "ML"},
    "frontal": {"accel_a": "ML", "accel_b": "Vertical", "gyro_axis": "AP"},
}


def phone_axis_col(df: pd.DataFrame, role: str, target_label: str) -> str | None:
    """
    Nome da coluna X/Y/Z do celular que corresponde a um rótulo anatômico
    ('Vertical', 'AP' ou 'ML'), dado o papel do sensor ('l5' ou 'limb').
    """
    mapping = PHONE_AXIS_ROLES.get(role, {})
    for raw_axis, lbl in mapping.items():
        if lbl != target_label:
            continue
        for col in df.columns:
            if norm(col) == raw_axis:
                return col
    return None


def complementary_angle(acc_df: pd.DataFrame, gyro_df: pd.DataFrame, fs: float,
                         role: str = "limb", plane: str = "sagittal",
                         alpha: float = 0.98) -> np.ndarray | None:
    """
    Ângulo de inclinação de um segmento (coxa ou tornozelo) via filtro
    complementar: fusão do ângulo estimado pelo acelerômetro (atan2 entre os
    dois componentes do plano escolhido, referência de gravidade) com a
    integração do giroscópio no eixo perpendicular a esse plano.

    plane: 'sagittal' (flexão/extensão, eixo de giro ML) ou
    'frontal' (valgo/varo do segmento, eixo de giro AP).

    Retorna array de ângulo em graus, ou None se as colunas necessárias não
    forem encontradas. É um ângulo relativo (não calibrado clinicamente),
    útil para comparar o formato do movimento entre coxa e tornozelo.
    """
    cfg = _PHONE_PLANE_CONFIG[plane]
    a_col = phone_axis_col(acc_df, role, cfg["accel_a"])
    b_col = phone_axis_col(acc_df, role, cfg["accel_b"])
    gyro_col = phone_axis_col(gyro_df, role, cfg["gyro_axis"])
    if a_col is None or b_col is None or gyro_col is None:
        return None

    a = try_numeric(acc_df[a_col]).fillna(0).values.astype(float)
    b = try_numeric(acc_df[b_col]).fillna(0).values.astype(float)
    gyro = try_numeric(gyro_df[gyro_col]).fillna(0).values.astype(float)

    n = min(len(a), len(b), len(gyro))
    if n == 0:
        return None
    a, b, gyro = a[:n], b[:n], gyro[:n]

    angle_acc = np.degrees(np.arctan2(a, b))
    gyro_dps = np.degrees(gyro)  # assume giroscópio em rad/s (padrão de smartphones)

    dt = 1.0 / fs
    theta = np.empty(n)
    theta[0] = angle_acc[0]
    for i in range(1, n):
        gyro_estimate = theta[i - 1] + gyro_dps[i] * dt
        theta[i] = alpha * gyro_estimate + (1.0 - alpha) * angle_acc[i]
    return theta


def knee_angle_from_phone(thigh_acc: pd.DataFrame, thigh_gyro: pd.DataFrame,
                          shank_acc: pd.DataFrame, shank_gyro: pd.DataFrame,
                          fs: float, alpha: float = 0.98) -> np.ndarray | None:
    """
    Ângulo relativo do joelho estimado pelos celulares no plano sagital
    (flexão/extensão): diferença entre o ângulo da coxa e o do tornozelo,
    cada um calculado por filtro complementar (fusão acelerômetro + giroscópio).
    """
    return knee_angle_from_phone_plane(thigh_acc, thigh_gyro, shank_acc, shank_gyro, fs,
                                       plane="sagittal", alpha=alpha)


def knee_angle_from_phone_plane(thigh_acc: pd.DataFrame, thigh_gyro: pd.DataFrame,
                                shank_acc: pd.DataFrame, shank_gyro: pd.DataFrame,
                                fs: float, plane: str = "sagittal",
                                alpha: float = 0.98) -> np.ndarray | None:
    """
    Ângulo relativo do joelho estimado pelos celulares num plano específico
    ('sagittal' ou 'frontal'): diferença entre o ângulo da coxa e o do
    tornozelo nesse plano, cada um via filtro complementar. Sinal preservado
    (pode ficar negativo) — ver knee_angle_direction_note() para orientação
    de qual lado é qual.
    """
    thigh_angle = complementary_angle(thigh_acc, thigh_gyro, fs, role="limb", plane=plane, alpha=alpha)
    shank_angle = complementary_angle(shank_acc, shank_gyro, fs, role="limb", plane=plane, alpha=alpha)
    if thigh_angle is None or shank_angle is None:
        return None
    n = min(len(thigh_angle), len(shank_angle))
    return thigh_angle[:n] - shank_angle[:n]


def knee_rotation_from_phone(thigh_gyro: pd.DataFrame, shank_gyro: pd.DataFrame,
                             fs: float, role: str = "limb") -> np.ndarray | None:
    """
    Estimativa (pouco confiável) da rotação interna/externa do joelho pelos
    celulares: integração pura do giroscópio no eixo vertical de cada
    segmento, sem nenhuma correção do acelerômetro (a gravidade não muda com
    rotação em torno do eixo vertical, então não há como corrigir a deriva).
    Tende a "escorregar" cada vez mais quanto mais longo o trecho analisado —
    use com cautela, principalmente fora de uma janela curta ao redor do pico.
    """
    thigh_col = phone_axis_col(thigh_gyro, role, "Vertical")
    shank_col = phone_axis_col(shank_gyro, role, "Vertical")
    if thigh_col is None or shank_col is None:
        return None

    thigh_gyro_vals = try_numeric(thigh_gyro[thigh_col]).fillna(0).values.astype(float)
    shank_gyro_vals = try_numeric(shank_gyro[shank_col]).fillna(0).values.astype(float)
    n = min(len(thigh_gyro_vals), len(shank_gyro_vals))
    if n == 0:
        return None

    dt = 1.0 / fs
    thigh_yaw = np.cumsum(np.degrees(thigh_gyro_vals[:n])) * dt
    shank_yaw = np.cumsum(np.degrees(shank_gyro_vals[:n])) * dt
    return thigh_yaw - shank_yaw


_POS_COL_EXCLUDE = ("v(", "a(", "length", "#2d")


def position_xyz_cols(df: pd.DataFrame, *body_keywords: str) -> dict:
    """
    Colunas de posição (X, Y, Z) — sem v(), a(), Length nem #2D — de uma
    região anatômica do Kinem. Retorna {'X': col, 'Y': col, 'Z': col}
    (chaves ausentes se a coluna correspondente não for encontrada).
    """
    result: dict = {}
    for col in df.columns:
        cn = norm(col).lower().strip()
        if not any(kw in cn for kw in body_keywords):
            continue
        if any(tok in cn for tok in _POS_COL_EXCLUDE):
            continue
        if cn.endswith("x"):
            result.setdefault("X", col)
        elif cn.endswith("y"):
            result.setdefault("Y", col)
        elif cn.endswith("z"):
            result.setdefault("Z", col)
    return result


def knee_angle_from_kinem(df: pd.DataFrame, hip_keywords: tuple, knee_keywords: tuple,
                          ankle_keywords: tuple) -> np.ndarray | None:
    """
    Ângulo de flexão do joelho "ótico" (3D completo), calculado a partir das
    posições 3D do Kinem: vetor coxa (quadril→joelho) e vetor perna
    (joelho→tornozelo). 0° = perna estendida (vetores colineares); aumenta
    com a flexão. Mistura qualquer componente fora do plano sagital (valgo/
    varo, rotação) — para isolar só a flexão/extensão, use
    knee_angle_from_kinem_plane(..., plane="sagittal").
    """
    thigh_vec, shank_vec = _kinem_segment_vectors(df, hip_keywords, knee_keywords, ankle_keywords)
    if thigh_vec is None:
        return None
    return _vector_angle_deg(thigh_vec, shank_vec)


# No Kinem: X = Mediolateral, Y = Anteroposterior, Z = Vertical (ver axis_label).
# Cada plano anatômico usa duas dessas componentes, ignorando a terceira.
_KINEM_PLANE_AXES = {
    "sagittal": (1, 2),    # AP × Vertical — flexão/extensão (o que o celular mede)
    "frontal": (0, 2),     # ML × Vertical — valgo/varo
    "transverse": (0, 1),  # ML × AP — rotação interna/externa
}


def _kinem_segment_vectors(df: pd.DataFrame, hip_keywords: tuple, knee_keywords: tuple,
                           ankle_keywords: tuple):
    """Vetores 3D coxa (quadril→joelho) e perna (joelho→tornozelo) a partir do Kinem."""
    hip = position_xyz_cols(df, *hip_keywords)
    knee = position_xyz_cols(df, *knee_keywords)
    ankle = position_xyz_cols(df, *ankle_keywords)
    if not all(k in hip for k in "XYZ") or not all(k in knee for k in "XYZ") or not all(k in ankle for k in "XYZ"):
        return None, None

    hip_pos = np.column_stack([try_numeric(df[hip[a]]).values for a in "XYZ"]).astype(float)
    knee_pos = np.column_stack([try_numeric(df[knee[a]]).values for a in "XYZ"]).astype(float)
    ankle_pos = np.column_stack([try_numeric(df[ankle[a]]).values for a in "XYZ"]).astype(float)

    return knee_pos - hip_pos, ankle_pos - knee_pos


def _vector_angle_deg(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Ângulo (graus, 0-180) entre dois conjuntos de vetores por amostra."""
    dot = np.sum(v1 * v2, axis=1)
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    denom = norm1 * norm2
    cos_angle = np.divide(dot, denom, out=np.zeros_like(dot), where=denom != 0)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def _signed_planar_angle_deg(v1_2d: np.ndarray, v2_2d: np.ndarray) -> np.ndarray:
    """
    Ângulo assinado (-180° a 180°) entre dois vetores 2D por amostra, via
    atan2(produto vetorial, produto escalar). O sinal indica de que lado v2
    está em relação a v1 no plano — permite diferenciar valgo de varo, ou
    rotação interna de externa, em vez de só a magnitude.
    """
    cross = v1_2d[:, 0] * v2_2d[:, 1] - v1_2d[:, 1] * v2_2d[:, 0]
    dot = np.sum(v1_2d * v2_2d, axis=1)
    return np.degrees(np.arctan2(cross, dot))


def knee_angle_from_kinem_plane(df: pd.DataFrame, hip_keywords: tuple, knee_keywords: tuple,
                                ankle_keywords: tuple, plane: str = "sagittal",
                                signed: bool = False) -> np.ndarray | None:
    """
    Ângulo do joelho do Kinem projetado num único plano anatômico:
    'sagittal' (flexão/extensão — comparável ao ângulo do celular),
    'frontal' (valgo/varo) ou 'transverse' (rotação interna/externa).

    Por padrão (signed=False) retorna a magnitude (0° = segmentos colineares
    nesse plano, aumenta conforme o ângulo abre — bom pra sagital, onde só a
    quantidade de flexão importa). Com signed=True retorna um ângulo com
    sinal (-180° a 180°), preservando de que lado o desvio ocorre — use para
    frontal/transverso, onde a direção (valgo x varo, interna x externa)
    importa. Ver knee_angle_direction_note() para orientação de qual sinal
    corresponde a qual lado.
    """
    thigh_vec, shank_vec = _kinem_segment_vectors(df, hip_keywords, knee_keywords, ankle_keywords)
    if thigh_vec is None:
        return None
    i, j = _KINEM_PLANE_AXES[plane]
    v1, v2 = thigh_vec[:, [i, j]], shank_vec[:, [i, j]]
    if signed:
        return _signed_planar_angle_deg(v1, v2)
    return _vector_angle_deg(v1, v2)


def knee_angle_direction_note(plane: str) -> str:
    """
    Texto explicando, em termos matemáticos verificáveis, o que significa o
    sinal (positivo/negativo) de um ângulo de plano frontal ou transverso.
    A correspondência exata com valgo/varo ou rotação interna/externa
    depende da orientação dos eixos configurada no Kinem e de como os
    celulares foram fixados — não é possível garantir isso só a partir dos
    dados. Recomenda-se confirmar empiricamente: peça pro avaliado fazer um
    movimento conhecido (ex.: joelho pra dentro/pra fora de propósito) e veja
    pra que lado a curva se move.
    """
    if plane == "frontal":
        return ("Frontal: sinal positivo/negativo indica desvio medial (valgo) ou lateral (varo) "
                "do joelho — qual é qual depende da orientação dos eixos no seu setup. "
                "Confirme fazendo o avaliado desviar o joelho de propósito pra um lado conhecido.")
    if plane == "transverse":
        return ("Transverso: sinal positivo/negativo indica rotação interna ou externa da perna — "
                "qual é qual depende da orientação dos eixos no seu setup, e esse sinal em especial "
                "tende a \"derivar\" com o tempo (menos confiável). Confirme fazendo o avaliado girar "
                "a perna de propósito pra um lado conhecido.")
    return ""


def zero_reference_angle(series: np.ndarray | None, x_axis: np.ndarray,
                         baseline_start: float, baseline_end: float) -> np.ndarray | None:
    """
    Ajusta um sinal de ângulo para que a média numa janela de referência
    (em segundos relativos ao pico) vire 0° — útil pra fazer 0° = extensão
    completa (postura inicial, antes do movimento) e o resto do sinal subir
    conforme a flexão aumenta.
    """
    if series is None:
        return None
    n = min(len(series), len(x_axis))
    mask = (x_axis[:n] >= baseline_start) & (x_axis[:n] <= baseline_end)
    if not np.any(mask):
        return series
    baseline = np.nanmean(series[:n][mask])
    if np.isnan(baseline):
        return series
    return series - baseline


# ──────────────────────────────────────────────
# Exportação
# ──────────────────────────────────────────────

def build_export_sheet(aligned, kinem_ref, acc_file, gyr_file, kinem_keywords, t,
                       none_label: str = NONE_LABEL) -> pd.DataFrame:
    """Monta o DataFrame de uma aba do Excel (L5 ou Joelho): tempo + Kinem + ACC + GYR."""
    dfs = [pd.DataFrame({"Tempo (s)": t})]

    kdf = aligned.get(kinem_ref, pd.DataFrame())
    k_cols = kinem_cols_for_body(kdf, *kinem_keywords)
    if k_cols:
        dfs.append(kdf[k_cols].reset_index(drop=True))

    if acc_file and acc_file != none_label and acc_file in aligned:
        adf = aligned[acc_file]
        cols = [c for c in adf.columns if is_xyz_col(c)]
        if cols:
            dfs.append(adf[cols].add_prefix("ACC_").reset_index(drop=True))

    if gyr_file and gyr_file != none_label and gyr_file in aligned:
        gdf = aligned[gyr_file]
        cols = [c for c in gdf.columns if is_xyz_col(c)]
        if cols:
            dfs.append(gdf[cols].add_prefix("GYR_").reset_index(drop=True))

    result = pd.concat(dfs, axis=1)
    return result.iloc[:len(t)]
