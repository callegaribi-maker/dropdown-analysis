"""
app.py
──────
Visualizador de Sinais — Y-Balance & Step-Down

Carrega arquivos de Kinem (câmera) e de celulares (ACC/GYR) posicionados na
L5, na coxa e no tornozelo, sincroniza-os pelo pico de impacto do salto/step,
e calcula o ângulo do joelho (celular vs. Kinem, decomposto por plano
anatômico) com calibração automática de amplitude, direto a partir dos
dados sincronizados — sem etapas de processamento manual.
"""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from signal_utils import (
    NONE_LABEL,
    apply_time_shift,
    auto_calibration_window,
    best_match,
    build_export_sheet,
    col_default,
    compute_derivative,
    compute_path_length,
    compute_mean,
    compute_peak,
    compute_rms,
    compute_rom,
    detect_time_axis,
    detect_first_drop,
    detect_trial_windows,
    estimate_time_lag_from_peaks,
    find_highest_peak,
    find_sync_xcorr,
    find_stable_plateau,
    fit_scale_gain,
    get_aligned_data,
    hip_angle_from_phone_plane,
    knee_angle_direction_note,
    knee_angle_from_kinem,
    knee_angle_from_kinem_plane,
    knee_angle_gyro_relative_integration,
    knee_angle_frontal_functional,
    linear_detrend_between,
    calibrate_two_point,
    gyro_relative_velocity,
    angular_velocity_from_angle,
    estimate_gyro_lag,
    apply_lag_to_series,
    knee_angle_from_phone,
    knee_angle_from_phone_plane,
    load_file,
    lowpass_array,
    numeric_cols,
    normalize_trial_curve,
    norm,
    phone_axis_col,
    position_xyz_cols,
    resample_to_regular,
    segment_trial_phases,
    time_to_peak,
    try_numeric,
    zero_reference_angle,
)

st.set_page_config(page_title="Lateral Step-Down test data processing", layout="wide")
st.title("📊 Lateral Step-Down test data processing")

NONE = NONE_LABEL

# Definição dos 3 grupos anatômicos: (chave, rótulo, cor, keywords p/ auto-match
# de arquivo de celular, keywords p/ colunas do Kinem)
GROUPS = {
    "l5": dict(label="L5", emoji="🟢", kinem_kw=("l5", "l 5"),
               file_kw=(("acel", "l5"), ("acc", "l5"))),
    "coxa": dict(label="Coxa", emoji="🟠", kinem_kw=("trocanter",),
                 file_kw=(("acel", "coxa"), ("acc", "coxa"), ("acel", "quadril"), ("acc", "quadril"))),
    "tornozelo": dict(label="Tornozelo", emoji="🔵", kinem_kw=("torn", "maleolo"),
                      file_kw=(("acel", "tornozelo"), ("acc", "tornozelo"), ("acel", "ankle"), ("acc", "ankle"), ("acel", "perna"), ("acc", "perna"))),
}
GYR_FILE_KW = {
    "l5": (("gyro", "l5"), ("gyr", "l5")),
    "coxa": (("gyro", "coxa"), ("gyr", "coxa"), ("gyro", "quadril"), ("gyr", "quadril")),
    "tornozelo": (("gyro", "tornozelo"), ("gyr", "tornozelo"), ("gyro", "ankle"), ("gyr", "ankle"), ("gyro", "perna"), ("gyr", "perna")),
}

DEFAULT_SESSION_STATE = {
    "files_data": {},
    "raw_synced": {},
    "offsets": {},
    "peak_ref": None,
    "target_fs": 100,
    "fs_info": {},
    "show_preview": False,
    "synced": False,
    "synced_kinem_cols": {},
    "ignorar_antes_seg": 0.0,
}
for key, default in DEFAULT_SESSION_STATE.items():
    st.session_state.setdefault(key, default)


# ══════════════════════════════════════════════
# 1 · Upload de arquivos
# ══════════════════════════════════════════════
with st.sidebar:
    st.header("1 · Carregar Arquivos")
    uploaded = st.file_uploader(
        "CSV ou TXT (até 7 arquivos: Kinem + ACC/GYR de L5, Coxa e Tornozelo)",
        type=["csv", "txt"], accept_multiple_files=True,
    )
    if uploaded:
        loaded, errors = {}, []
        for f in uploaded:
            df = load_file(f)
            if df is not None:
                loaded[f.name] = df
            else:
                errors.append(f.name)

        if set(loaded.keys()) != set(st.session_state.files_data.keys()):
            st.session_state.files_data = loaded
            st.session_state.offsets = {}
            st.session_state.fs_info = {}

        if errors:
            st.error(f"Não carregou: {', '.join(errors)}")
        st.success(f"{len(loaded)} arquivo(s) ✔")

files_data = st.session_state.files_data
if not files_data:
    st.info("👈 Carregue os arquivos na barra lateral para começar.")
    st.stop()

file_names = list(files_data.keys())


# ══════════════════════════════════════════════
# 2 · Kinem (referência)
# ══════════════════════════════════════════════
with st.sidebar:
    st.header("2 · Kinem (referência)")
    kinem_idx = next((i for i, n in enumerate(file_names) if "kinem" in n.lower()), 0)
    kinem_ref = st.selectbox("Arquivo Kinem", file_names, index=kinem_idx)
    kinem_num = numeric_cols(files_data[kinem_ref])

    st.caption("As três colunas vêm do mesmo arquivo — cada pico ocorre na mesma amostra do Kinem.")
    st.caption("⚠️ No Kinem: Vertical = Z, AP = Y, ML = X. Selecione a coluna Z (a) vertical de cada marcador.")

    kinem_sync_cols = {}
    kinem_sync_cols["l5"] = st.selectbox(
        "Coluna L5 vertical (referência sync)", kinem_num,
        index=col_default(kinem_num, ["l 5 a(z)", "l5 a(z)", "l5a(z)", "l 5 z", "l5_az", "l5"]),
        key="kinem_col_l5",
    )
    kinem_sync_cols["coxa"] = st.selectbox(
        "Coluna Coxa (Trocânter) vertical (referência sync)", kinem_num,
        index=col_default(kinem_num, [
            "trocanter maior dir. a(z)", "trocanter a(z)", "trocanter dir. a(z)",
            "trocanter maior dir.", "trocanter",
        ]),
        key="kinem_col_coxa",
    )
    kinem_sync_cols["tornozelo"] = st.selectbox(
        "Coluna Tornozelo vertical (referência sync)", kinem_num,
        index=col_default(kinem_num, [
            "osso externo do torn. dir. a(z)", "osso externo do torn. a(z)",
            "maleolo lateral dir. a(z)", "maleolo dir. a(z)", "maleolo a(z)",
            "torn. dir. a(z)", "osso externo do torn.", "maleolo lateral dir.",
            "tornozelo", "torn", "maleolo",
        ]),
        key="kinem_col_tornozelo",
    )

others = [n for n in file_names if n != kinem_ref]

# ══════════════════════════════════════════════
# 3/4/5 · Grupos de celular (L5, Coxa, Tornozelo)
# ══════════════════════════════════════════════
phone_files = {}   # group_key -> {"acc": fname|NONE, "acc_col": colname|None, "gyr": fname|NONE}
section_titles = {"l5": "3 · Grupo L5 (celular)", "coxa": "4 · Grupo Coxa (celular)",
                   "tornozelo": "5 · Grupo Tornozelo (celular)"}

with st.sidebar:
    for gkey, gdef in GROUPS.items():
        st.header(section_titles[gkey])
        if gkey == "l5":
            st.caption("ACC e GYR já saem sincronizados entre si pelo celular.")

        acc = st.selectbox(
            f"ACC {gdef['label']}", [NONE] + others,
            index=best_match(others, *gdef["file_kw"], exclude=() if gkey == "l5" else ("l5", "l 5")), key=f"{gkey}_acc",
        )
        acc_col = None
        if acc != NONE:
            num = numeric_cols(files_data[acc])
            acc_col = st.selectbox(
                f"Coluna Y do ACC {gdef['label']}", num,
                index=col_default(num, ["y"]), key=f"{gkey}_acc_col",
            )
        gyr = st.selectbox(
            f"GYR {gdef['label']}  ← offset = ACC", [NONE] + others,
            index=best_match(others, *GYR_FILE_KW[gkey], exclude=() if gkey == "l5" else ("l5", "l 5")), key=f"{gkey}_gyr",
        )
        phone_files[gkey] = {"acc": acc, "acc_col": acc_col, "gyr": gyr}

# ══════════════════════════════════════════════
# Configurações avançadas de sincronização
# ══════════════════════════════════════════════
with st.sidebar:
    with st.expander("⚙️ Configurações avançadas de sincronização", expanded=False):
        fs_target = st.number_input(
            "Frequência alvo após reamostragem (Hz)",
            min_value=1, max_value=10000, value=100, step=10,
            help="Todos os arquivos serão reamostrados para esta frequência comum.",
        )
        cf_alpha = st.slider(
            "Filtro complementar (ângulo do joelho) — peso do giroscópio", 0.05, 0.999,
            value=0.995, step=0.005,
            help="Mais próximo de 1 = confia mais no giroscópio (menos deriva do acelerômetro). 0,995 tende a captar melhor a amplitude do plano frontal (valgo/varo) sem prejudicar o sagital — testado empiricamente. Valores baixos (perto de 0.05) confiam quase só no acelerômetro.",
        )


# ══════════════════════════════════════════════
# Botões: Preview + Sincronizar
# ══════════════════════════════════════════════
btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1.6, 1, 1, 1.6])

with btn_col1:
    if st.button("👁 Preview sinais brutos", use_container_width=True):
        st.session_state.show_preview = not st.session_state.show_preview

with btn_col2:
    janela_seg = st.number_input(
        "Pico nos primeiros (s)", min_value=0.1, max_value=300.0, value=16.0, step=0.5,
        help="Janela de busca do pico de sincronização.",
    )

with btn_col3:
    ignorar_antes_seg = st.number_input(
        "Ignorar picos antes de (s)", min_value=0.0, max_value=300.0, value=0.0, step=0.5,
        help="Se houver um movimento preparatório antes do evento que você quer sincronizar (um pico de aceleração maior mas que não é o que importa), aumente esse valor pra pular ele e sincronizar no próximo pico.",
    )

with btn_col4:
    sincronizar = st.button("🔗 Sincronizar", type="primary", use_container_width=True)


def _sync_phone_group(kinem_col, peak_kinem, acc_file, acc_col, gyr_file,
                       raw_synced, fs, janela_samp, none_label=NONE):
    """Sincroniza um par ACC/GYR de celular contra uma coluna vertical do Kinem.

    Retorna (offsets_parciais, mensagem|None) — GYR herda o offset do ACC.
    """
    offsets = {}
    if acc_file == none_label or not acc_col:
        return offsets, None
    if acc_col not in raw_synced.get(acc_file, pd.DataFrame()).columns:
        return offsets, None

    p = find_sync_xcorr(raw_synced[kinem_ref][kinem_col], raw_synced[acc_file][acc_col],
                         peak_kinem, janela_samp, fs)
    offsets[acc_file] = peak_kinem - p
    msg = f"pico @ {p} ({p/fs:.2f} s) → offset {peak_kinem-p:+d}"
    if gyr_file != none_label:
        offsets[gyr_file] = peak_kinem - p
    return offsets, msg


def _find_secondary_peak(raw_synced, kinem_col, peak_l5, fs, win_seconds=1.0):
    """Pico do Kinem de um grupo secundário (coxa/tornozelo), buscado numa
    janela de ±win_seconds ao redor do pico de referência do L5."""
    win = int(win_seconds * fs)
    s = try_numeric(raw_synced[kinem_ref][kinem_col])
    k_start, k_end = max(0, peak_l5 - win), min(len(s), peak_l5 + win)
    return find_highest_peak(s.iloc[k_start:k_end].reset_index(drop=True), k_end - k_start, fs) + k_start


if sincronizar:
    with st.spinner("Reamostrando e detectando pico…"):
        raw_synced, fs_info, msgs_pre = {}, {}, []
        for fname, df in files_data.items():
            r, fs_orig, desc = resample_to_regular(df, fs_target)
            raw_synced[fname] = r
            fs_info[fname] = fs_orig
            msgs_pre.append(f"**{fname[:35]}**: {desc}")

        st.session_state.raw_synced = raw_synced
        st.session_state.target_fs = fs_target
        st.session_state.fs_info = fs_info

        janela_samp = int(janela_seg * fs_target)
        ignorar_samp = int(ignorar_antes_seg * fs_target)
        offsets = {kinem_ref: 0}
        msgs_sync = []

        peak_l5 = find_highest_peak(
            try_numeric(raw_synced[kinem_ref][kinem_sync_cols["l5"]]), janela_samp, fs_target,
            search_start=ignorar_samp,
        )
        st.session_state.ignorar_antes_seg = ignorar_antes_seg
        st.session_state.peak_ref = peak_l5
        st.session_state.synced = True
        st.session_state.show_preview = False
        msgs_sync.append(f"**Kinem L5** — pico @ {peak_l5} ({peak_l5/fs_target:.2f} s) → x=0")
        if ignorar_antes_seg > 0:
            msgs_sync.append(f"⏭️ Ignorados picos antes de {ignorar_antes_seg:.1f}s")

        group_peaks = {"l5": peak_l5}
        for gkey in ("coxa", "tornozelo"):
            pk = _find_secondary_peak(raw_synced, kinem_sync_cols[gkey], peak_l5, fs_target)
            group_peaks[gkey] = pk
            msgs_sync.append(
                f"**Kinem {GROUPS[gkey]['label']}** — pico @ {pk} ({pk/fs_target:.2f} s) "
                f"→ Δ {(pk-peak_l5)/fs_target:+.3f} s"
            )

        for gkey, gdef in GROUPS.items():
            pf = phone_files[gkey]
            g_offs, g_msg = _sync_phone_group(
                kinem_sync_cols[gkey], group_peaks[gkey], pf["acc"], pf["acc_col"], pf["gyr"],
                raw_synced, fs_target, janela_samp,
            )
            offsets.update(g_offs)
            if g_msg:
                msgs_sync.append(f"**{gdef['label']} ACC** — {g_msg}")
                if pf["gyr"] != NONE and pf["gyr"] in g_offs:
                    msgs_sync.append(f"**{gdef['label']} GYR** — offset {g_offs[pf['gyr']]:+d} (= ACC {gdef['label']})")

        for fname in file_names:
            offsets.setdefault(fname, 0)
        st.session_state.offsets = offsets
        st.session_state.synced_kinem_cols = dict(kinem_sync_cols)

        with st.expander("📋 Detalhes da sincronização", expanded=False):
            st.markdown("**Frequências detectadas:**")
            for m in msgs_pre:
                st.write(m)
            st.markdown("**Offsets calculados:**")
            for m in msgs_sync:
                st.write(m)


# ══════════════════════════════════════════════
# Auto-resync quando alguma coluna de referência muda
# ══════════════════════════════════════════════
if st.session_state.synced and st.session_state.raw_synced and st.session_state.peak_ref is not None:
    prev_cols = st.session_state.synced_kinem_cols
    changed = any(prev_cols.get(k) != kinem_sync_cols[k] for k in kinem_sync_cols)

    if changed:
        raws = st.session_state.raw_synced
        tfs = st.session_state.target_fs or 100
        jsamp = int(janela_seg * tfs)
        offs = dict(st.session_state.offsets)

        if prev_cols.get("l5") != kinem_sync_cols["l5"] and kinem_sync_cols["l5"] in raws.get(kinem_ref, pd.DataFrame()).columns:
            ignorar_samp = int((st.session_state.ignorar_antes_seg or 0.0) * tfs)
            pk_l5 = find_highest_peak(try_numeric(raws[kinem_ref][kinem_sync_cols["l5"]]), jsamp, tfs, search_start=ignorar_samp)
            st.session_state.peak_ref = pk_l5
            offs[kinem_ref] = 0
            pf = phone_files["l5"]
            g_offs, _ = _sync_phone_group(
                kinem_sync_cols["l5"], pk_l5, pf["acc"], pf["acc_col"], pf["gyr"], raws, tfs, jsamp,
            )
            offs.update(g_offs)

        pk_l5 = st.session_state.peak_ref
        for gkey in ("coxa", "tornozelo"):
            if kinem_sync_cols[gkey] not in raws.get(kinem_ref, pd.DataFrame()).columns:
                continue
            pk_g = _find_secondary_peak(raws, kinem_sync_cols[gkey], pk_l5, tfs)
            pf = phone_files[gkey]
            g_offs, _ = _sync_phone_group(
                kinem_sync_cols[gkey], pk_g, pf["acc"], pf["acc_col"], pf["gyr"], raws, tfs, jsamp,
            )
            offs.update(g_offs)

        st.session_state.offsets = offs
        st.session_state.synced_kinem_cols = dict(kinem_sync_cols)


# ══════════════════════════════════════════════
# Preview bruto
# ══════════════════════════════════════════════
if st.session_state.show_preview:
    st.subheader("👁 Sinais brutos — sem pré-processamento")

    sync_cols = [(kinem_ref, kinem_sync_cols["l5"])]
    for gkey in ("coxa", "tornozelo"):
        col = kinem_sync_cols[gkey]
        if col and col != kinem_sync_cols["l5"]:
            sync_cols.append((kinem_ref, col))
    for gkey, pf in phone_files.items():
        if pf["acc"] != NONE and pf["acc_col"]:
            sync_cols.append((pf["acc"], pf["acc_col"]))

    pc1, pc2 = st.columns(2)
    with pc1:
        prev_t_start = st.number_input("Ver a partir de (s)", min_value=0.0, value=0.0, step=1.0, key="prev_start")
    with pc2:
        prev_t_end = st.number_input("Até (s)  — 0 = fim do sinal", min_value=0.0, value=0.0, step=1.0, key="prev_end")

    n_prev = len(sync_cols)
    fig_p = make_subplots(
        rows=n_prev, cols=1, shared_xaxes=False,
        subplot_titles=[f"{fn} · {c}" for fn, c in sync_cols], vertical_spacing=0.06,
    )
    for row, (fname, col) in enumerate(sync_cols, start=1):
        t, _ = detect_time_axis(files_data[fname])
        x = t - t[0] if t is not None else np.arange(len(files_data[fname]))
        y = try_numeric(files_data[fname][col])
        mask = x >= prev_t_start
        if prev_t_end > prev_t_start:
            mask &= x <= prev_t_end
        fig_p.add_trace(go.Scatter(x=x[mask], y=y[mask], mode="lines", showlegend=False), row=row, col=1)

    fig_p.update_layout(
        height=240 * n_prev, template="plotly_white",
        title="Colunas de sync — tempo original de cada arquivo", hovermode="x unified",
    )
    st.plotly_chart(fig_p, use_container_width=True)
    st.divider()


# ══════════════════════════════════════════════
# Verificação de alinhamento
# ══════════════════════════════════════════════
def render_alignment_check(title, kinem_col, phone_file, phone_col, label_k, label_p,
                            vraw, vx, vfs, raw_sync_x=0.0):
    check = []
    df_k = vraw.get(kinem_ref, pd.DataFrame())
    if kinem_col in df_k.columns:
        check.append((df_k, kinem_col, label_k))
    if phone_file != NONE and phone_col and phone_file in vraw:
        df_p = vraw[phone_file]
        if phone_col in df_p.columns:
            check.append((df_p, phone_col, label_p))
    if len(check) < 2:
        return

    with st.expander(f"🔍 Verificação — alinhamento {title}", expanded=True):
        colors_v = ["blue", "red"]
        series, caps = [], []
        for df_s, col, lbl in check:
            s = try_numeric(df_s[col]).fillna(0).values.astype(float)
            pk = np.nanmax(np.abs(s))
            series.append((s / pk if pk > 0 else s, lbl))
            caps.append(f"`{col}`")

        cap = "  |  ".join(
            f"{'🔵' if i == 0 else '🔴'} **{series[i][1]}**: {caps[i]}" for i in range(len(series))
        )
        st.caption(cap + f"  ·  reamostrado a {vfs:.0f} Hz  ·  normalizado pelo pico  ·  sem filtro passa-baixa")

        x_view_lo, x_view_hi = raw_sync_x - 2, raw_sync_x + 2
        mask_2 = (vx >= x_view_lo) & (vx <= x_view_hi)
        all_vals = np.concatenate([s[mask_2] for s, _ in series if len(s) == len(vx)])
        all_vals = all_vals[~np.isnan(all_vals)]
        y_lo, y_hi = (float(np.nanmin(all_vals)) - 0.5, float(np.nanmax(all_vals)) + 0.5) if len(all_vals) else (-1.5, 1.5)

        fig_v = go.Figure()
        for i, (s_n, lbl) in enumerate(series):
            fig_v.add_trace(go.Scatter(
                x=vx, y=s_n, mode="lines", line=dict(color=colors_v[i], width=2), name=lbl, opacity=0.85,
            ))
        if len(series) == 2:
            diff = series[0][0] - series[1][0]
            fig_v.add_trace(go.Scatter(
                x=vx, y=diff, mode="lines", line=dict(color="gray", width=1, dash="dot"), name="Diferença",
            ))
        fig_v.add_vline(x=0, line_dash="dash", line_color="black",
                         annotation_text="pico flexão", annotation_position="top right")
        if abs(raw_sync_x) > 1e-9:
            fig_v.add_vline(x=raw_sync_x, line_dash="dot", line_color="purple",
                             annotation_text="sinc. bruta", annotation_position="bottom right")
        fig_v.update_layout(
            title=title,
            xaxis=dict(title="Tempo (s)  —  0 = pico de flexão do joelho", range=[x_view_lo, x_view_hi]),
            yaxis=dict(title="Amplitude norm.", range=[y_lo, y_hi]),
            hovermode="x unified", template="plotly_white", height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(t=50, b=50, l=60, r=20),
        )
        st.plotly_chart(fig_v, use_container_width=True, key=f"verif_{title}_{kinem_col}_{phone_col}")


if st.session_state.synced and st.session_state.raw_synced and st.session_state.peak_ref is not None:
    pfs = st.session_state.target_fs or 100
    aligned_data, x_samp, align_msg = get_aligned_data(
        st.session_state.raw_synced, st.session_state.offsets, st.session_state.peak_ref, ref_file=kinem_ref,
    )
    if aligned_data is None:
        st.error(align_msg)
        st.stop()
    x_axis = x_samp / pfs
    x_min_data, x_max_data = float(x_axis.min()), float(x_axis.max())
    kdf = aligned_data.get(kinem_ref, pd.DataFrame())

    # ── Recentraliza x=0 no pico de flexão do joelho (Kinem sagital), não no
    # pico de aceleração vertical do L5. O pico de aceleração é usado só pra
    # alinhar os arquivos no tempo (isso não muda); mas como referência de
    # 0" na tela, o pico de flexão é mais direto e sem ambiguidade — evita
    # ter que adivinhar qual pico de aceleração é "o certo" quando há mais
    # de um candidato (ex.: um movimento preparatório antes do teste).
    #
    kinem_angle_kw = (GROUPS["coxa"]["kinem_kw"], ("condilo",), GROUPS["tornozelo"]["kinem_kw"])
    angle_peak_time = 0.0
    if not kdf.empty:
        angle_kinem_sagital_prelim = knee_angle_from_kinem_plane(kdf, *kinem_angle_kw, plane="sagittal")
        if angle_kinem_sagital_prelim is not None:
            n_prelim = min(len(angle_kinem_sagital_prelim), len(x_axis))
            valid_prelim = ~np.isnan(angle_kinem_sagital_prelim[:n_prelim])
            if np.any(valid_prelim):
                masked_vals = np.where(valid_prelim, angle_kinem_sagital_prelim[:n_prelim], -np.inf)
                peak_idx_prelim = int(np.nanargmax(masked_vals))
                angle_peak_time = float(x_axis[:n_prelim][peak_idx_prelim])
                if abs(angle_peak_time) > 1e-9:
                    x_axis = x_axis - angle_peak_time
                    x_min_data, x_max_data = float(x_axis.min()), float(x_axis.max())
                    st.caption(f"↕️ Referência 0s recentralizada no pico de flexão do joelho (estava a {angle_peak_time:+.2f}s do pico de aceleração usado pra sincronizar os arquivos).")

    verif_cols = st.columns(len(GROUPS))
    for col, (gkey, gdef) in zip(verif_cols, GROUPS.items()):
        with col:
            pf = phone_files[gkey]
            render_alignment_check(
                gdef["label"], kinem_sync_cols[gkey], pf["acc"], pf["acc_col"],
                f"Kinem {gdef['label']}", f"ACC {gdef['label']}", aligned_data, x_axis, pfs,
                raw_sync_x=-angle_peak_time,
            )

    st.divider()

    view_start, view_end = float(x_min_data), float(x_max_data)

    st.divider()

    # ══════════════════════════════════════════
    # Ângulo do joelho (celular vs. Kinem)
    # ══════════════════════════════════════════
    st.subheader("🦵 Ângulo do joelho")

    # O ângulo é calculado a partir dos dados sincronizados BRUTOS (reamostrados,
    # sem detrend/filtro): o detrend distorce a posição 3D real do Kinem e
    # remove o componente de gravidade que o acelerômetro precisa para
    # estimar a inclinação do segmento. Este app não aplica nenhum outro
    # processamento além da sincronização.
    aligned_raw, kdf_raw = aligned_data, kdf

    pf_coxa, pf_torn = phone_files["coxa"], phone_files["tornozelo"]
    phone_ready = bool(
        aligned_raw and all(pf_coxa[k] != NONE for k in ("acc", "gyr")) and all(pf_torn[k] != NONE for k in ("acc", "gyr"))
        and all(f in aligned_raw for f in [pf_coxa["acc"], pf_coxa["gyr"], pf_torn["acc"], pf_torn["gyr"]])
    )

    pf_l5 = phone_files["l5"]
    hip_phone_ready = bool(
        aligned_raw and all(pf_l5[k] != NONE for k in ("acc", "gyr")) and all(pf_coxa[k] != NONE for k in ("acc", "gyr"))
        and all(f in aligned_raw for f in [pf_l5["acc"], pf_l5["gyr"], pf_coxa["acc"], pf_coxa["gyr"]])
    )

    angle_phone_sagital = angle_phone_frontal = None
    if phone_ready:
        angle_phone_sagital = knee_angle_from_phone_plane(
            aligned_raw[pf_coxa["acc"]], aligned_raw[pf_coxa["gyr"]],
            aligned_raw[pf_torn["acc"]], aligned_raw[pf_torn["gyr"]],
            pfs, plane="sagittal", alpha=cf_alpha,
        )

    angle_hip_phone_sagital = angle_hip_phone_frontal = None
    if hip_phone_ready:
        angle_hip_phone_sagital = hip_angle_from_phone_plane(
            aligned_raw[pf_l5["acc"]], aligned_raw[pf_l5["gyr"]],
            aligned_raw[pf_coxa["acc"]], aligned_raw[pf_coxa["gyr"]],
            pfs, plane="sagittal", alpha=cf_alpha,
        )
        angle_hip_phone_frontal = hip_angle_from_phone_plane(
            aligned_raw[pf_l5["acc"]], aligned_raw[pf_l5["gyr"]],
            aligned_raw[pf_coxa["acc"]], aligned_raw[pf_coxa["gyr"]],
            pfs, plane="frontal", alpha=cf_alpha,
        )

    hip_angle_kw = (GROUPS["l5"]["kinem_kw"], GROUPS["coxa"]["kinem_kw"], ("condilo",))
    # Ambos os ângulos brutos (Kinem e celular) já vêm no MESMO sentido pra
    # essa combinação de segmentos (testado com dados reais) — sem precisar
    # de inversão nenhuma, ao contrário do que a versão anterior assumia.
    angle_hip_kinem_sagital = knee_angle_from_kinem_plane(
        kdf_raw, *hip_angle_kw, plane="sagittal",
    ) if not kdf_raw.empty else None
    angle_hip_kinem_frontal = knee_angle_from_kinem_plane(
        kdf_raw, *hip_angle_kw, plane="frontal", signed=True,
    ) if not kdf_raw.empty else None

    # Zera os dois no mesmo ponto (início da gravação) — sem isso, cada
    # fonte fica na sua própria escala bruta (geometria do Kinem vs.
    # convenção do filtro complementar), difícil de comparar visualmente.
    # Só offset (soma/subtrai), sem ganho — não muda a amplitude de
    # nenhum dos dois, só onde "zero" fica.
    angle_hip_kinem_sagital = zero_reference_angle(angle_hip_kinem_sagital, x_axis, x_min_data, x_min_data + 0.5)
    angle_hip_phone_sagital = zero_reference_angle(angle_hip_phone_sagital, x_axis, x_min_data, x_min_data + 0.5)

    kinem_angle_kw = (GROUPS["coxa"]["kinem_kw"], ("condilo",), GROUPS["tornozelo"]["kinem_kw"])
    angle_kinem_sagital = knee_angle_from_kinem_plane(
        kdf_raw, *kinem_angle_kw, plane="sagittal",
    ) if not kdf_raw.empty else None

    angle_kinem_3d = angle_kinem_frontal = None
    angle_phone_frontal = None
    if not kdf_raw.empty:
        angle_kinem_3d = knee_angle_from_kinem(kdf_raw, *kinem_angle_kw)
        angle_kinem_frontal = knee_angle_from_kinem_plane(kdf_raw, *kinem_angle_kw, plane="frontal", signed=True)
    if phone_ready:
        angle_phone_frontal = knee_angle_from_phone_plane(
            aligned_raw[pf_coxa["acc"]], aligned_raw[pf_coxa["gyr"]],
            aligned_raw[pf_torn["acc"]], aligned_raw[pf_torn["gyr"]],
            pfs, plane="frontal", alpha=cf_alpha,
        )

    # ── Velocidade angular (derivada dos ângulos já corrigidos/calibrados) —
    # métrica de qualidade de movimento: quão rápido o joelho flexiona/desvia,
    # não só o quanto. Filtra o ângulo (passa-baixa 10Hz) ANTES de derivar —
    # testamos com dados reais: sem isso o jerk fica dominado por ruído
    # amplificado pela derivação; no próprio ângulo (ADM/pico/forma) o
    # filtro não muda quase nada, então só filtramos aqui, não no ângulo
    # principal exibido nos gráficos. ──
    angle_kinem_sagital_sm = lowpass_array(angle_kinem_sagital, pfs, 10.0)
    angle_phone_sagital_sm = lowpass_array(angle_phone_sagital, pfs, 10.0)
    angle_kinem_frontal_sm = lowpass_array(angle_kinem_frontal, pfs, 10.0)
    angle_phone_frontal_sm = lowpass_array(angle_phone_frontal, pfs, 10.0)
    vel_kinem_sagital = compute_derivative(angle_kinem_sagital_sm, x_axis)
    vel_phone_sagital = compute_derivative(angle_phone_sagital_sm, x_axis)
    vel_kinem_frontal = compute_derivative(angle_kinem_frontal_sm, x_axis)
    vel_phone_frontal = compute_derivative(angle_phone_frontal_sm, x_axis)

    # ── Jerk (derivada da velocidade, já suavizada acima) — mede suavidade
    # do movimento; não é exibido como curva (fica ruidoso demais, mesmo
    # filtrado), só como RMS por trial na tabela — quanto maior, mais
    # "trêmulo"/irregular. ──
    jerk_kinem_sagital = compute_derivative(vel_kinem_sagital, x_axis)
    jerk_phone_sagital = compute_derivative(vel_phone_sagital, x_axis)

    # ── Detecta trials e segmenta cada um em preparação/descida/subida,
    # usando o deslocamento vertical do L5 (mesma lógica usada no Y-Balance
    # pra marcar as fases do movimento) ──
    pos_l5_cols = position_xyz_cols(kdf_raw, "l5", "l 5") if not kdf_raw.empty else {}
    l5_vertical = None
    l5_lateral = None
    if all(k in pos_l5_cols for k in "XYZ"):
        l5_vertical = try_numeric(kdf_raw[pos_l5_cols["Z"]]).values.astype(float)
        l5_lateral = try_numeric(kdf_raw[pos_l5_cols["X"]]).values.astype(float)

    # ── Ignora um trecho inicial diferente (ex.: calibração, com o L5 num
    # nível diferente do repouso normal) antes de procurar os trials —
    # detecta automaticamente onde o L5 se estabiliza de volta perto do
    # nível de repouso, e só busca trials a partir de alguns segundos antes
    # disso. Sempre visível (não fica atrás de nenhuma caixinha). ──
    st.subheader("🔎 A partir de quando procurar os trials")
    drop_time = detect_first_drop(l5_vertical, x_axis)
    sugestao = float(drop_time - 5.0) if drop_time is not None else float(x_min_data)
    if drop_time is not None:
        st.caption(f"📉 Detecção automática: L5 se estabiliza em t={drop_time:+.2f}s. Sugestão abaixo já vem com 5s de margem — ajuste livremente se não bater com o que você vê no gráfico do L5, logo abaixo.")
    else:
        st.caption("⚠️ Não detectei uma estabilização clara — ajuste o valor manualmente olhando o gráfico do L5, logo abaixo.")
    trial_search_start = st.number_input(
        "Início da busca de trials (s, relativo ao pico de flexão)", value=sugestao, step=0.5, key="trial_search_start",
        help="Trials só são procurados a partir deste instante em diante — digite o valor que quiser, olhando onde o teste realmente começa no gráfico do L5 abaixo.",
    )

    trials = detect_trial_windows(angle_kinem_sagital, x_axis, search_start=trial_search_start)

    st.subheader("🎚️ Sensibilidade da segmentação")
    onset_frac = st.slider(
        "Sensibilidade do início do movimento (segmentação)", 0.005, 0.40, value=0.15, step=0.005,
        key="onset_frac",
        help="Fração do deslocamento total do L5 usada pra marcar onde a descida realmente começa. "
             "Baixo = mais sensível (marca o início mais cedo, mas pode pegar ruído). Alto = mais "
             "conservador (só marca quando o movimento já está bem claro). Ajuste e confira no gráfico abaixo.",
    )
    ignorar_primeiro_ciclo = st.checkbox(
        "Ignorar o primeiro ciclo detectado (ex.: movimento preparatório antes do teste, não uma repetição de verdade)",
        value=False, key="ignorar_primeiro_ciclo",
        help="Descarta o primeiro trial detectado por completo — sem sombra, sem bolinha, sem entrar nas tabelas/análise. Use se o primeiro pico não for uma repetição real do teste.",
    )
    trial_phases = []
    if trials and l5_vertical is not None:
        for t_start, t_end in trials:
            phases = segment_trial_phases(l5_vertical, x_axis, t_start, t_end, onset_frac=onset_frac, max_lookback_before=5.0)
            trial_phases.append(phases)

        if ignorar_primeiro_ciclo and len(trials) > 1:
            trials = trials[1:]
            trial_phases = trial_phases[1:]

        # Emenda: a preparação de cada trial passa a começar exatamente onde a
        # subida do trial anterior terminou (em vez do limite arbitrário da
        # janela do trial) — elimina a lacuna sem fase entre um ciclo e outro.
        for i in range(1, len(trial_phases)):
            if trial_phases[i] and trial_phases[i - 1]:
                prev_return = trial_phases[i - 1]["subida"][1]
                this_onset = trial_phases[i]["preparacao"][1]
                trial_phases[i]["preparacao"] = (prev_return, this_onset)

        # O primeiro trial (o que sobrar depois do descarte acima, se marcado)
        # não tem um trial anterior pra emendar — a janela dele geralmente
        # pega sobra de antes da gravação/movimento real começar. Em vez
        # disso, usa a duração MÉDIA das preparações dos outros trials,
        # posicionada logo antes do início da descida; tudo antes disso fica
        # sem fase nenhuma (sem sombra, sem bolinha, sem entrar na análise).
        if trial_phases and trial_phases[0]:
            outras_duracoes = [
                tp["preparacao"][1] - tp["preparacao"][0]
                for tp in trial_phases[1:] if tp
            ]
            if outras_duracoes:
                dur_media = float(np.mean(outras_duracoes))
                onset0 = trial_phases[0]["preparacao"][1]
                trial_phases[0]["preparacao"] = (onset0 - dur_media, onset0)
    else:
        trial_phases = [None] * len(trials)

    # --- Validação da segmentação: deslocamento vertical do L5 ---
    st.subheader("📐 Deslocamento vertical do L5 (validação da segmentação)")
    if l5_vertical is None:
        st.info("Não encontrei as colunas de posição X/Y/Z do L5 no Kinem — não dá pra segmentar por deslocamento vertical.")
    else:
        n_l5 = min(len(l5_vertical), len(x_axis))
        mask_l5 = (x_axis[:n_l5] >= view_start) & (x_axis[:n_l5] <= view_end)
        fig_l5v = go.Figure()
        fig_l5v.add_trace(go.Scatter(
            x=x_axis[:n_l5][mask_l5], y=l5_vertical[:n_l5][mask_l5], mode="lines",
            line=dict(color="black", width=1.5), name="L5 — posição vertical (Kinem)",
        ))
        def y_at(x_target):
            idx = int(np.argmin(np.abs(x_axis[:n_l5] - x_target)))
            return l5_vertical[idx]

        marker_x, marker_y, marker_color = [], [], []
        for phases in trial_phases:
            if not phases:
                continue
            d_start, d_end = phases["descida"]
            s_start, s_end = phases["subida"]
            p_start, p_end = phases["preparacao"]
            if p_end > p_start:
                fig_l5v.add_vrect(x0=p_start, x1=p_end, fillcolor="lightgray", opacity=0.30, line_width=0)
            if d_end > d_start:
                fig_l5v.add_vrect(x0=d_start, x1=d_end, fillcolor="orange", opacity=0.15, line_width=0)
            if s_end > s_start:
                fig_l5v.add_vrect(x0=s_start, x1=s_end, fillcolor="steelblue", opacity=0.15, line_width=0)
            # bolinhas exatamente nas transições de cor: início da descida,
            # ponto mais baixo (descida→subida) e fim da subida
            marker_x += [d_start, d_end, s_end]
            marker_y += [y_at(d_start), y_at(d_end), y_at(s_end)]
            marker_color += ["orange", "black", "steelblue"]
        if marker_x:
            fig_l5v.add_trace(go.Scatter(
                x=marker_x, y=marker_y, mode="markers",
                marker=dict(color=marker_color, size=9, line=dict(color="black", width=1)),
                name="Transições de fase", showlegend=False,
            ))
        fig_l5v.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="pico flexão")
        fig_l5v.update_layout(
            xaxis=dict(title="Tempo (s)  —  0 = pico de flexão do joelho", range=[view_start, view_end]),
            yaxis_title="Posição vertical L5", height=340, template="plotly_white", hovermode="x unified",
            margin=dict(t=30, b=40),
        )
        st.plotly_chart(fig_l5v, use_container_width=True)
        st.caption(
            "Cinza = preparação · laranja = descida · azul = subida. Bolinhas laranja = início da descida · "
            "bolinha preta = ponto mais baixo (transição descida→subida) · bolinha azul = fim da subida."
        )
        st.divider()

    nota_clinica = None
    if angle_phone_sagital is None and angle_kinem_sagital is None:
        st.info("Selecione ACC + GYR de Coxa e Tornozelo (celular) e/ou confirme as colunas do Kinem para calcular o ângulo do joelho.")
    else:
        mask_ang = (x_axis >= view_start) & (x_axis <= view_end)

        def add_angle_trace(fig, series, color, name, dash=None):
            if series is None:
                return
            n = min(len(series), len(x_axis))
            y = series[:n]
            m = mask_ang[:n]
            fig.add_trace(go.Scatter(
                x=x_axis[:n][m], y=y[m], mode="lines",
                line=dict(color=color, width=2, dash=dash), name=name,
            ))

        def add_phase_shading(fig):
            """Sombreia preparação (cinza), descida (laranja) e subida (azul) de cada trial — mesmas cores do gráfico de validação do L5."""
            for phases in trial_phases:
                if not phases:
                    continue
                p_start, p_end = phases["preparacao"]
                d_start, d_end = phases["descida"]
                s_start, s_end = phases["subida"]
                if p_end > p_start:
                    fig.add_vrect(x0=p_start, x1=p_end, fillcolor="lightgray", opacity=0.20, line_width=0)
                if d_end > d_start:
                    fig.add_vrect(x0=d_start, x1=d_end, fillcolor="orange", opacity=0.10, line_width=0)
                if s_end > s_start:
                    fig.add_vrect(x0=s_start, x1=s_end, fillcolor="steelblue", opacity=0.10, line_width=0)

        def add_phase_markers(fig, series):
            """Bolinhas nas transições de fase (início da descida, ponto mais baixo, fim da subida), na curva 'series' desse gráfico."""
            if series is None:
                return
            n_s = min(len(series), len(x_axis))
            mx, my, mc = [], [], []
            for phases in trial_phases:
                if not phases:
                    continue
                d_start, d_end = phases["descida"]
                s_start, s_end = phases["subida"]
                for xt, color in [(d_start, "orange"), (d_end, "black"), (s_end, "steelblue")]:
                    idx = int(np.argmin(np.abs(x_axis[:n_s] - xt)))
                    mx.append(xt)
                    my.append(series[idx])
                    mc.append(color)
            if mx:
                fig.add_trace(go.Scatter(
                    x=mx, y=my, mode="markers",
                    marker=dict(color=mc, size=8, line=dict(color="black", width=1)),
                    name="Transições de fase", showlegend=False,
                ))

        def add_l5_overlay(fig):
            """Sobrepõe o deslocamento vertical do L5 num eixo Y secundário (direita, em cm)."""
            if l5_vertical is None:
                return
            n = min(len(l5_vertical), len(x_axis))
            y_cm = l5_vertical[:n] * 100
            m = mask_ang[:n]
            fig.add_trace(go.Scatter(
                x=x_axis[:n][m], y=y_cm[m], mode="lines",
                line=dict(color="rgba(0,0,0,0.35)", width=1.5, dash="dot"),
                name="L5 vertical (cm)", yaxis="y2",
            ))
            fig.update_layout(yaxis2=dict(title="L5 vertical (cm)", overlaying="y", side="right", showgrid=False))

        def render_sagital_chart(k_series, p_series, titulo_extra="", highlight_window=None):
            fig = go.Figure()
            add_phase_shading(fig)
            if highlight_window is not None:
                fig.add_vrect(
                    x0=highlight_window[0], x1=highlight_window[1],
                    fillcolor="rgba(255, 99, 71, 0.18)", line_width=0,
                    annotation_text="janela de calibração", annotation_position="bottom left",
                )
            add_angle_trace(fig, k_series, "blue", "Kinem — sagital")
            add_angle_trace(fig, p_series, "red", "Celular — sagital")
            add_angle_trace(fig, angle_kinem_3d, "gray", "Kinem — 3D total", dash="dot")
            add_l5_overlay(fig)
            add_phase_markers(fig, k_series)
            fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="pico flexão")
            fig.update_layout(
                xaxis=dict(title="Tempo (s)  —  0 = pico de flexão do joelho", range=[view_start, view_end]),
                yaxis_title="Ângulo (°)  —  ↑ flexão · ↓ extensão", height=380, template="plotly_white", hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), margin=dict(t=30, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- Plano sagital (flexão/extensão) — bruto, sem calibração ---
    # ── Métodos alternativos (opcional): integração direta da velocidade
    # angular relativa (celular, sem acelerômetro) + calibração de 2 pontos
    # pro sagital; calibração funcional por PCA (corrige cross-talk) pro
    # frontal do Kinem, com o celular também via integração direta. Testado
    # com dados reais: segue melhor a FORMA do Kinem no sagital, e reduz
    # bastante o vazamento do sagital pro frontal (cross-talk) no Kinem.
    #
    # As janelas padrão vêm dos limites do Trial 1 já detectado (não de
    # x_min_data cru) — sem isso, se as janelas caírem fora de dados reais,
    # a calibração falha silenciosamente e sobra o ângulo bruto (que sem
    # calibração pode chegar a milhares de graus, por deriva pura da
    # integração) — isso já aconteceu e o aviso abaixo evita repetir.
    st.subheader("🦵 Calibração do ângulo (goniômetro)")
    # Detecta automaticamente o platô de calibração (maior trecho estável
    # do ângulo sagital do Kinem) — sem precisar que você ache os
    # instantes na mão. 'Postura neutra' é sempre o começo da gravação.
    plato_detectado = find_stable_plateau(angle_kinem_sagital, x_axis, search_start=x_min_data, min_duration=3.0, tol_frac=0.08)
    t_neutro_ini_sugerido = float(x_min_data)
    t_neutro_fim_sugerido = float(x_min_data) + 0.5
    if plato_detectado is not None:
        t_calib_ini_sugerido, t_calib_fim_sugerido = plato_detectado
        st.caption(f"🎯 Platô de calibração detectado automaticamente: {t_calib_ini_sugerido:+.2f}s a {t_calib_fim_sugerido:+.2f}s — a região colorida no gráfico abaixo mostra esse trecho. Se não bater com o platô real, abra 'Ajustar detecção manualmente' logo abaixo.")
    elif trials:
        t_calib_ini_sugerido, t_calib_fim_sugerido = trials[0]
        st.caption("⚠️ Não consegui detectar um platô estável automaticamente — usando os limites do Trial 1 como sugestão inicial. Confira a região colorida no gráfico abaixo; se não bater, ajuste manualmente logo abaixo.")
    else:
        t_calib_ini_sugerido, t_calib_fim_sugerido = float(x_min_data), float(x_min_data) + 5.0
        st.caption("⚠️ Não consegui detectar um platô estável nem um Trial 1 — abra 'Ajustar detecção manualmente' abaixo e informe a janela na mão.")

    angulo_calib_sagital_alt = st.number_input(
        "Ângulo de referência conhecido (°, medido com goniômetro no platô)", value=90.0, step=1.0, key="ma_angulo_calib_sagital",
    )

    with st.expander("⚙️ Ajustar detecção manualmente (só se tiver erro)", expanded=False):
        usar_metodo_alt = not st.checkbox(
            "Desligar esse método e usar o filtro complementar antigo (não recomendado)",
            value=False, key="ma_desligar_metodo_alt",
        )
        ma1, ma2, ma3, ma4 = st.columns(4)
        t_neutro_ini = ma1.number_input("Início postura neutra (s)", value=t_neutro_ini_sugerido, step=0.5, key="ma_t_neutro_ini")
        t_neutro_fim = ma2.number_input("Fim postura neutra (s)", value=t_neutro_fim_sugerido, step=0.5, key="ma_t_neutro_fim")
        t_calib_ini = ma3.number_input("Início janela de calibração (s)", value=float(t_calib_ini_sugerido), step=0.5, key="ma_t_calib_ini")
        t_calib_fim = ma4.number_input("Fim janela de calibração (s)", value=float(t_calib_fim_sugerido), step=0.5, key="ma_t_calib_fim")
        st.caption(
            "'Postura neutra' = trecho parado em pé (referência de 0°). 'Janela de calibração' precisa cobrir um "
            "movimento de flexão claro (ex.: do início até o platô ou pico) — os valores acima já vêm sugeridos "
            "automaticamente, mas ajuste livremente olhando o gráfico bruto abaixo se não parecer certo."
        )
        eixo_sagital_gyro = st.selectbox("Eixo bruto do giroscópio p/ sagital", ["Z", "Y", "X"], index=0, key="ma_eixo_sagital")
        eixo_frontal_gyro = st.selectbox("Eixo bruto do giroscópio p/ frontal", ["X", "Y", "Z"], index=0, key="ma_eixo_frontal")

        usar_sync_giro = st.checkbox(
            "Refinar sincronização comparando velocidade angular (giroscópio vs. cinemática)",
            value=True, key="ma_usar_sync_giro",
            help="Deriva o ângulo do Kinem pra virar velocidade angular, e testa uma faixa de atrasos no giroscópio até achar o que maximiza a correlação entre os dois, só dentro da janela de calibração — evita 'forçar' a concordância usando os ciclos de teste. Refina a sincronização que o resto do app já fez, especificamente pra esse cálculo.",
        )
        lag_giro_sagital = None
        if usar_sync_giro and phone_ready:
            kinem_vel = angular_velocity_from_angle(angle_kinem_sagital, x_axis)
            gyro_vel = gyro_relative_velocity(aligned_raw[pf_coxa["gyr"]], aligned_raw[pf_torn["gyr"]], pfs, axis=eixo_sagital_gyro, sign=-1.0)
            lag_giro_sagital, corr_giro = estimate_gyro_lag(
                kinem_vel, gyro_vel, x_axis, t_calib_ini - 5.0, t_calib_fim + 2.0, max_lag=3.0, lag_step=0.01,
            )
            if lag_giro_sagital is not None:
                st.caption(f"⏱️ Sincronização por velocidade angular: atraso encontrado = {lag_giro_sagital:+.2f}s (correlação r={corr_giro:.3f}) — aplicado só no cálculo sagital do giroscópio, abaixo.")
            else:
                st.caption("⚠️ Não deu pra estimar o atraso por velocidade angular (dados insuficientes na janela de calibração) — usando a sincronização normal do app.")

    if usar_metodo_alt:
        def _checa_faixa(nome, serie):
            if serie is None:
                return
            amplitude = float(np.nanmax(serie) - np.nanmin(serie))
            if amplitude > 150:
                st.error(f"❌ {nome} ficou com amplitude de {amplitude:.0f}° — isso não é fisiologicamente plausível pra esse teste. A janela de calibração provavelmente está cortando no meio do platô/movimento (não cobrindo o trecho estável inteiro), ou o eixo do giroscópio está errado. Ajuste as janelas acima olhando o gráfico bruto abaixo — alargue a 'janela de calibração' até cobrir o trecho INTEIRO e estável do movimento de referência. Mantendo o método antigo pra essa curva por enquanto.")
                return False
            return True

        if phone_ready:
            angulo_sagital_bruto = knee_angle_gyro_relative_integration(
                aligned_raw[pf_coxa["gyr"]], aligned_raw[pf_torn["gyr"]], pfs, axis=eixo_sagital_gyro, lowpass_hz=5.0, sign=-1.0,
                x_axis=x_axis, lag_seconds=lag_giro_sagital,
            )
            if angulo_sagital_bruto is not None:
                angle_phone_sagital_alt, v0_sag, v1_sag = calibrate_two_point(
                    angulo_sagital_bruto, x_axis, t_neutro_ini, t_neutro_fim, t_calib_ini, t_calib_fim,
                    val0=0.0, val1=angulo_calib_sagital_alt,
                )
                if v0_sag is None or v1_sag is None:
                    st.error("❌ Não deu pra calibrar o sagital (alt.) — não há dados suficientes numa das duas janelas. Mantendo o método antigo.")
                else:
                    st.caption(f"📐 Celular sagital (alt.): postura neutra = {v0_sag:.1f}° bruto → 0°; janela de calibração = {v1_sag:.1f}° bruto → {angulo_calib_sagital_alt:.0f}°.")
                    if _checa_faixa("Celular sagital (alt.)", angle_phone_sagital_alt):
                        angle_phone_sagital = angle_phone_sagital_alt

            angulo_frontal_bruto = knee_angle_gyro_relative_integration(
                aligned_raw[pf_coxa["gyr"]], aligned_raw[pf_torn["gyr"]], pfs, axis=eixo_frontal_gyro, lowpass_hz=5.0, sign=1.0,
                x_axis=x_axis, lag_seconds=lag_giro_sagital,
            )
            if angulo_frontal_bruto is not None:
                n_af = min(len(angulo_frontal_bruto), len(x_axis))
                base_af = compute_mean(angulo_frontal_bruto[:n_af], x_axis[:n_af], t_neutro_ini, t_neutro_fim)
                if base_af is None:
                    st.error("❌ Não deu pra zerar o frontal do celular (alt.) na janela de postura neutra — não há dados ali. Mantendo o método antigo.")
                else:
                    angulo_frontal_bruto = angulo_frontal_bruto - base_af
                    if _checa_faixa("Celular frontal (alt.)", angulo_frontal_bruto):
                        angle_phone_frontal = angulo_frontal_bruto

        if not kdf_raw.empty:
            pos_troc_alt = position_xyz_cols(kdf_raw, "trocanter")
            pos_cond_alt = position_xyz_cols(kdf_raw, "condilo")
            pos_mal_alt = position_xyz_cols(kdf_raw, "maleolo")
            if all(k in pos_troc_alt for k in "XYZ") and all(k in pos_cond_alt for k in "XYZ") and all(k in pos_mal_alt for k in "XYZ"):
                quadril_alt = np.column_stack([try_numeric(kdf_raw[pos_troc_alt[a]]).values for a in "XYZ"])
                joelho_alt = np.column_stack([try_numeric(kdf_raw[pos_cond_alt[a]]).values for a in "XYZ"])
                tornozelo_alt = np.column_stack([try_numeric(kdf_raw[pos_mal_alt[a]]).values for a in "XYZ"])
                vetor_coxa_alt = quadril_alt - joelho_alt
                vetor_perna_alt = joelho_alt - tornozelo_alt  # invertido em relação ao sagital — convenção própria do frontal

                # O ajuste de cross-talk do frontal (PCA) precisa de uma
                # janela DIFERENTE da usada na calibração do sagital: uma
                # referência "neutra" que englobe o instante em que o
                # movimento começa a estabilizar no platô (não um trecho
                # limpo bem antes disso), e uma janela funcional BEM mais
                # larga (o platô inteiro + uma margem depois, cobrindo
                # parte da descida) — testado com dados reais: usar a
                # mesma janela do sagital aqui deixava um resíduo de
                # cross-talk bem maior (~10° em vez de ~1-3°). Calculado
                # automaticamente a partir do platô detectado, sem exigir
                # mais nenhum campo novo na tela.
                if plato_detectado is not None:
                    t_neutro_frontal_ini = plato_detectado[0] - 0.5
                    t_neutro_frontal_fim = plato_detectado[0] + 1.5
                    t_calib_frontal_ini = t_neutro_frontal_ini
                    t_calib_frontal_fim = plato_detectado[1] + 5.0
                else:
                    t_neutro_frontal_ini, t_neutro_frontal_fim = t_neutro_ini, t_neutro_fim
                    t_calib_frontal_ini, t_calib_frontal_fim = t_calib_ini, t_calib_fim

                angulo_frontal_kinem_alt, rotacao_func, variancia_func = knee_angle_frontal_functional(
                    vetor_coxa_alt, vetor_perna_alt, x_axis, t_neutro_frontal_ini, t_neutro_frontal_fim, t_calib_frontal_ini, t_calib_frontal_fim,
                )
                if angulo_frontal_kinem_alt is None:
                    st.error("❌ Não deu pra calcular o frontal funcional do Kinem — confira as janelas (precisam ter dados suficientes e um movimento de flexão claro na janela de calibração). Mantendo o método antigo.")
                else:
                    # Mostra o antes/depois da correção de cross-talk no platô
                    # (deveria ficar perto de 0°, já que é um movimento de
                    # flexão pura, sem valgo/varo intencional) — confirma
                    # visualmente que a correção funcionou.
                    platô_antes = compute_mean(angle_kinem_frontal, x_axis, t_calib_ini, t_calib_fim) if angle_kinem_frontal is not None else None
                    platô_depois = compute_mean(angulo_frontal_kinem_alt, x_axis, t_calib_ini, t_calib_fim)
                    if platô_antes is not None and platô_depois is not None:
                        st.caption(f"📐 Kinem frontal: cross-talk corrigido por calibração funcional — no platô, {platô_antes:+.1f}° (plano bruto da câmera) → {platô_depois:+.1f}° (plano funcional, eixo girado {rotacao_func:.1f}°, variância explicada {variancia_func*100:.0f}%). Deveria ficar perto de 0° (é um movimento de flexão pura).")
                    if _checa_faixa("Kinem frontal (alt.)", angulo_frontal_kinem_alt):
                        angle_kinem_frontal = angulo_frontal_kinem_alt

                        # Corrige a convenção de sinal do celular, se
                        # necessário: compara a correlação com o Kinem já
                        # corrigido dentro da janela de calibração e inverte
                        # o sinal do celular se estiver invertido.
                        if angle_phone_frontal is not None:
                            n_sc = min(len(angle_phone_frontal), len(angle_kinem_frontal), len(x_axis))
                            mask_sc = (x_axis[:n_sc] >= t_calib_ini) & (x_axis[:n_sc] <= t_calib_fim)
                            k_seg = angle_kinem_frontal[:n_sc][mask_sc]
                            p_seg = angle_phone_frontal[:n_sc][mask_sc]
                            valid_sc = ~np.isnan(k_seg) & ~np.isnan(p_seg)
                            if np.sum(valid_sc) > 10 and np.std(k_seg[valid_sc]) > 0 and np.std(p_seg[valid_sc]) > 0:
                                corr_sinal = float(np.corrcoef(k_seg[valid_sc], p_seg[valid_sc])[0, 1])
                                if np.isfinite(corr_sinal) and corr_sinal < 0:
                                    angle_phone_frontal = -angle_phone_frontal
                                    st.caption(f"🔄 Celular frontal: sinal invertido (correlação com o Kinem estava negativa, r={corr_sinal:.2f}) — mesma convenção de agora em diante (negativo = valgo, positivo = varo).")

                        # Novo zero bem antes do primeiro ciclo real de
                        # teste — reduz o efeito da deriva acumulada do
                        # giroscópio (o celular integra sem acelerômetro,
                        # então qualquer deriva desde o início da gravação
                        # se acumula; zerar de novo logo antes do teste
                        # deixa o zero mais preciso pros ciclos que
                        # realmente importam, mesmo que a gravação inteira
                        # tenha desviado um pouco).
                        if trials:
                            t_prezero_ini = trials[0][0]
                            t_prezero_fim = min(trials[0][0] + 1.0, trials[0][1])
                            base_k_prezero = compute_mean(angle_kinem_frontal, x_axis, t_prezero_ini, t_prezero_fim)
                            base_p_prezero = compute_mean(angle_phone_frontal, x_axis, t_prezero_ini, t_prezero_fim) if angle_phone_frontal is not None else None
                            if base_k_prezero is not None:
                                angle_kinem_frontal = angle_kinem_frontal - base_k_prezero
                            if base_p_prezero is not None:
                                angle_phone_frontal = angle_phone_frontal - base_p_prezero
                            if base_k_prezero is not None or base_p_prezero is not None:
                                st.caption(f"↕️ Novo zero definido bem antes do 1º ciclo real ({t_prezero_ini:+.1f}s a {t_prezero_fim:+.1f}s) — reduz o efeito de deriva acumulada até ali.")

                            # Corrige a deriva de integração do celular
                            # (não tem acelerômetro corrigindo o giroscópio
                            # continuamente, então a deriva se acumula ao
                            # longo dos testes) — puxa o início e o fim dos
                            # ciclos de teste de volta perto de zero,
                            # assumindo que a deriva é aproximadamente
                            # constante ao longo do tempo (comum nesse tipo
                            # de sensor). O Kinem não precisa disso (não
                            # integra nada, não deriva).
                            if angle_phone_frontal is not None and len(trials) > 1:
                                t_fim_testes = trials[-1][1]
                                antes_detrend = compute_mean(angle_phone_frontal, x_axis, t_fim_testes - 1.0, t_fim_testes)
                                angle_phone_frontal = linear_detrend_between(
                                    angle_phone_frontal, x_axis, t_prezero_ini, t_fim_testes, ref_window=1.0,
                                )
                                if antes_detrend is not None:
                                    st.caption(f"📉 Deriva do celular corrigida (correção linear): estava em {antes_detrend:+.1f}° no fim dos testes antes da correção, ajustado pra perto de 0°.")

    # ── Estabilidade de tronco: comparação Kinem × Celular via ACELERAÇÃO e
    # VELOCIDADE ANGULAR brutas (sem integrar nada) — testamos e essa é a
    # forma que dá uma relação consistente entre as duas fontes (ao
    # contrário da posição lateral integrada, que se mostrou pouco confiável). ──
    acc_ml_kinem_trunk = None
    if not kdf_raw.empty:
        acc_ml_kinem_cols = [c for c in kdf_raw.columns if norm(c).lower() == "l 5 a(x)"]
        if acc_ml_kinem_cols:
            acc_ml_kinem_trunk = try_numeric(kdf_raw[acc_ml_kinem_cols[0]]).values.astype(float)

    acc_ml_phone_trunk = None
    if pf_l5["acc"] != NONE and pf_l5["acc"] in aligned_raw:
        acc_ml_phone_trunk = try_numeric(
            aligned_raw[pf_l5["acc"]][phone_axis_col(aligned_raw[pf_l5["acc"]], "l5", "ML")]
        ).values.astype(float)

    trunk_angvel_kinem = None
    pos_troc_cols_trunk = position_xyz_cols(kdf_raw, "trocanter") if not kdf_raw.empty else {}
    if all(k in pos_l5_cols for k in "XYZ") and all(k in pos_troc_cols_trunk for k in "XYZ"):
        l5_pos_arr = np.column_stack([try_numeric(kdf_raw[pos_l5_cols[a]]).values for a in "XYZ"]).astype(float)
        troc_pos_arr = np.column_stack([try_numeric(kdf_raw[pos_troc_cols_trunk[a]]).values for a in "XYZ"]).astype(float)
        trunk_vec = troc_pos_arr - l5_pos_arr
        trunk_vec_frontal = trunk_vec[:, [0, 2]]
        vertical_ref = np.tile([0.0, -1.0], (len(trunk_vec_frontal), 1))
        cross = trunk_vec_frontal[:, 0] * vertical_ref[:, 1] - trunk_vec_frontal[:, 1] * vertical_ref[:, 0]
        dot = trunk_vec_frontal[:, 0] * vertical_ref[:, 0] + trunk_vec_frontal[:, 1] * vertical_ref[:, 1]
        trunk_lean_kinem_iso = np.degrees(np.arctan2(cross, dot))
        trunk_angvel_kinem = compute_derivative(trunk_lean_kinem_iso, x_axis)

    trunk_angvel_phone = None
    if pf_l5["gyr"] != NONE and pf_l5["gyr"] in aligned_raw:
        gyr_ap_raw = try_numeric(
            aligned_raw[pf_l5["gyr"]][phone_axis_col(aligned_raw[pf_l5["gyr"]], "l5", "AP")]
        ).values.astype(float)
        trunk_angvel_phone = np.degrees(gyr_ap_raw)

        st.markdown("**Sagital — flexão (↑) / extensão (↓) — bruto**")
        st.caption("Fundo cinza = preparação · laranja = descida · azul = subida · linha pontilhada cinza = deslocamento vertical do L5 (eixo direito, cm) · faixa vermelha = janela de calibração usada.")
        render_sagital_chart(angle_kinem_sagital, angle_phone_sagital, highlight_window=(t_calib_ini, t_calib_fim))

        if angle_kinem_sagital is None:
            st.caption("⚠️ Ângulo do Kinem não calculado — verifique se as colunas de posição X/Y/Z de Trocânter, Côndilo e Tornozelo estão presentes.")
        if angle_phone_sagital is None:
            st.caption("⚠️ Ângulo do celular não calculado — selecione ACC e GYR de Coxa e Tornozelo na barra lateral.")

        # --- Plano frontal ---
        if True:
            st.markdown("**Frontal — valgo (↑ ou ↓, ver nota) / varo (sentido oposto)**")
            fig_front = go.Figure()
            add_phase_shading(fig_front)
            fig_front.add_vrect(
                x0=t_calib_ini, x1=t_calib_fim,
                fillcolor="rgba(255, 99, 71, 0.18)", line_width=0,
                annotation_text="janela de calibração", annotation_position="bottom left",
            )
            add_angle_trace(fig_front, angle_kinem_frontal, "green", "Kinem — frontal")
            add_angle_trace(fig_front, angle_phone_frontal, "darkorange", "Celular — frontal")
            add_l5_overlay(fig_front)
            add_phase_markers(fig_front, angle_kinem_frontal)
            fig_front.add_hline(y=0, line_dash="dot", line_color="lightgray")
            fig_front.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="pico flexão")
            fig_front.update_layout(
                xaxis=dict(title="Tempo (s)  —  0 = pico de flexão do joelho", range=[view_start, view_end]),
                yaxis_title="Ângulo (°)  —  ↑ varo · ↓ valgo", height=340, template="plotly_white", hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), margin=dict(t=30, b=40),
            )
            st.plotly_chart(fig_front, use_container_width=True)
            st.caption(
                "ℹ️ Convenção adotada: **negativo = valgo, positivo = varo**. O sinal do Kinem é ajustado "
                "automaticamente pra manter essa convenção (assumindo valgo dinâmico como padrão predominante "
                "nesse tipo de teste — ver nota técnica no código se precisar desativar essa correção)."
            )

        # --- Ângulo do quadril (tronco/L5 vs coxa) ---
        st.divider()
        st.markdown("#### 🦴 Ângulo do quadril (tronco vs coxa)")
        st.caption("Mesma lógica do joelho, agora usando L5 (tronco) e Coxa — flexão/extensão de quadril.")
        if angle_hip_phone_sagital is None and angle_hip_kinem_sagital is None:
            st.info("Selecione ACC + GYR de L5 e Coxa (celular) e/ou confirme as colunas do Kinem para calcular o ângulo do quadril.")
        else:
            st.markdown("**Sagital — flexão (↑) / extensão (↓)**")
            fig_hip_sag = go.Figure()
            add_phase_shading(fig_hip_sag)
            add_angle_trace(fig_hip_sag, angle_hip_kinem_sagital, "teal", "Kinem — quadril sagital")
            add_angle_trace(fig_hip_sag, angle_hip_phone_sagital, "crimson", "Celular — quadril sagital")
            add_l5_overlay(fig_hip_sag)
            add_phase_markers(fig_hip_sag, angle_hip_kinem_sagital)
            fig_hip_sag.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="pico flexão")
            fig_hip_sag.update_layout(
                xaxis=dict(title="Tempo (s)  —  0 = pico de flexão do joelho", range=[view_start, view_end]),
                yaxis_title="Ângulo (°)  —  ↑ flexão · ↓ extensão", height=340, template="plotly_white", hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), margin=dict(t=30, b=40),
            )
            st.plotly_chart(fig_hip_sag, use_container_width=True)

        # --- Ver análise: trials sobrepostos (ciclo inteiro 0-1, com fases) por métrica ---
        st.divider()
        ver_analise_overlay = st.button("🔍 Ver análise", type="primary", use_container_width=True, key="btn_ver_analise_overlay")
        if ver_analise_overlay:
            st.session_state.mostrar_analise_overlay = True
        if st.session_state.get("mostrar_analise_overlay"):
            valid_trial_phases = [p for p in trial_phases if p]
            if not valid_trial_phases:
                st.info("Não há trials segmentados pra sobrepor.")
            else:
                st.markdown("#### 📊 Trials sobrepostos (ciclo inteiro normalizado: 0 = início da preparação, 1 = fim da subida)")
                st.caption(
                    f"{len(valid_trial_phases)} trials sobrepostos por métrica. Linhas finas = cada trial individual · "
                    "linha grossa = resultante (média). Fundo cinza/laranja/azul = preparação/descida/subida (posição média entre os trials)."
                )

                # Fração média (0-1) de onde cada fase termina, pra pintar o fundo
                prep_fracs, desc_fracs = [], []
                for phases in valid_trial_phases:
                    t0 = phases["preparacao"][0]
                    t_onset = phases["descida"][0]
                    t_bottom = phases["descida"][1]
                    t_fim = phases["subida"][1]
                    total = t_fim - t0
                    if total > 0:
                        prep_fracs.append((t_onset - t0) / total)
                        desc_fracs.append((t_bottom - t0) / total)
                avg_prep_frac = float(np.mean(prep_fracs)) if prep_fracs else 0.3
                avg_desc_frac = float(np.mean(desc_fracs)) if desc_fracs else 0.6

                def render_overlay_chart(title, kinem_series, phone_series, color_k, color_p, yaxis_title="Ângulo (graus)"):
                    fig = go.Figure()
                    fig.add_vrect(x0=0, x1=avg_prep_frac, fillcolor="lightgray", opacity=0.25, line_width=0)
                    fig.add_vrect(x0=avg_prep_frac, x1=avg_desc_frac, fillcolor="orange", opacity=0.12, line_width=0)
                    fig.add_vrect(x0=avg_desc_frac, x1=1, fillcolor="steelblue", opacity=0.12, line_width=0)

                    x_norm = np.linspace(0, 1, 101)
                    kinem_curves, phone_curves = [], []
                    for phases in valid_trial_phases:
                        t0, t1 = phases["preparacao"][0], phases["subida"][1]
                        yk = normalize_trial_curve(kinem_series, x_axis, t0, t1)
                        yp = normalize_trial_curve(phone_series, x_axis, t0, t1)
                        if yk is not None:
                            kinem_curves.append(yk)
                            fig.add_trace(go.Scatter(x=x_norm, y=yk, mode="lines", line=dict(color=color_k, width=1), opacity=0.30, showlegend=False))
                        if yp is not None:
                            phone_curves.append(yp)
                            fig.add_trace(go.Scatter(x=x_norm, y=yp, mode="lines", line=dict(color=color_p, width=1), opacity=0.30, showlegend=False))
                    if kinem_curves:
                        mean_k = np.nanmean(np.array(kinem_curves), axis=0)
                        fig.add_trace(go.Scatter(x=x_norm, y=mean_k, mode="lines", line=dict(color=color_k, width=3), name="Kinem"))
                    if phone_curves:
                        mean_p = np.nanmean(np.array(phone_curves), axis=0)
                        fig.add_trace(go.Scatter(x=x_norm, y=mean_p, mode="lines", line=dict(color=color_p, width=3), name="Celular"))
                    fig.update_layout(
                        title=dict(text=title, x=0.5, xanchor="center", y=0.97, yanchor="top", font=dict(size=13)),
                        xaxis_title="Ciclo normalizado (0 = início · 1 = fim)",
                        yaxis_title=yaxis_title,
                        height=460, width=420, template="plotly_white", hovermode="x unified",
                        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0.5, xanchor="center", font=dict(size=10)),
                        margin=dict(t=55, b=95, l=55, r=15),
                    )
                    st.plotly_chart(fig, use_container_width=False)

                def render_overlay_chart_single(title, series, color, yaxis_title="Posição vertical L5 (m)"):
                    fig = go.Figure()
                    fig.add_vrect(x0=0, x1=avg_prep_frac, fillcolor="lightgray", opacity=0.25, line_width=0)
                    fig.add_vrect(x0=avg_prep_frac, x1=avg_desc_frac, fillcolor="orange", opacity=0.12, line_width=0)
                    fig.add_vrect(x0=avg_desc_frac, x1=1, fillcolor="steelblue", opacity=0.12, line_width=0)
                    x_norm = np.linspace(0, 1, 101)
                    curves = []
                    for phases in valid_trial_phases:
                        t0, t1 = phases["preparacao"][0], phases["subida"][1]
                        y = normalize_trial_curve(series, x_axis, t0, t1)
                        if y is not None:
                            curves.append(y)
                            fig.add_trace(go.Scatter(x=x_norm, y=y, mode="lines", line=dict(color=color, width=1), opacity=0.30, showlegend=False))
                    if curves:
                        mean_y = np.nanmean(np.array(curves), axis=0)
                        fig.add_trace(go.Scatter(x=x_norm, y=mean_y, mode="lines", line=dict(color=color, width=3), name="Resultante"))
                    fig.update_layout(
                        title=dict(text=title, x=0.5, xanchor="center", y=0.97, yanchor="top", font=dict(size=13)),
                        xaxis_title="Ciclo normalizado (0 = início · 1 = fim)",
                        yaxis_title=yaxis_title,
                        height=460, width=420, template="plotly_white", hovermode="x unified",
                        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0.5, xanchor="center", font=dict(size=10)),
                        margin=dict(t=55, b=95, l=55, r=15),
                    )
                    st.plotly_chart(fig, use_container_width=False)

                oc1, oc2 = st.columns(2)
                with oc1:
                    render_overlay_chart("Joelho — Sagital", angle_kinem_sagital, angle_phone_sagital, "blue", "red")
                with oc2:
                    render_overlay_chart("Joelho — Frontal", angle_kinem_frontal, angle_phone_frontal, "green", "darkorange")
                oc3, oc4 = st.columns(2)
                with oc3:
                    render_overlay_chart("Quadril — Sagital", angle_hip_kinem_sagital, angle_hip_phone_sagital, "teal", "crimson")
                with oc4:
                    render_overlay_chart("Quadril — Frontal", angle_hip_kinem_frontal, angle_hip_phone_frontal, "darkcyan", "deeppink")
                oc5, oc6 = st.columns(2)
                with oc5:
                    render_overlay_chart("Vel. Angular — Joelho Sagital", vel_kinem_sagital, vel_phone_sagital, "blue", "red", yaxis_title="Velocidade (°/s)")
                with oc6:
                    render_overlay_chart("Vel. Angular — Joelho Frontal", vel_kinem_frontal, vel_phone_frontal, "green", "darkorange", yaxis_title="Velocidade (°/s)")
                oc7, oc8 = st.columns(2)
                with oc7:
                    render_overlay_chart_single("L5 — Deslocamento vertical", l5_vertical, "black", yaxis_title="Posição vertical (m)")
                with oc8:
                    render_overlay_chart_single("L5 — Deslocamento lateral", l5_lateral, "purple", yaxis_title="Posição lateral / ML (m)")
                oc9, oc10 = st.columns(2)
                with oc9:
                    render_overlay_chart("Tronco — Acel. Lateral (estabilidade)", acc_ml_kinem_trunk, acc_ml_phone_trunk, "black", "purple", yaxis_title="Aceleração (m/s²)")
                with oc10:
                    render_overlay_chart("Tronco — Vel. Angular (estabilidade)", trunk_angvel_kinem, trunk_angvel_phone, "black", "purple", yaxis_title="Velocidade (°/s)")

        # --- Avaliação clínica: nota + análise completa por trial ---
        st.divider()
        st.markdown("#### 🩺 Avaliação clínica")
        nota_clinica = st.radio(
            "Nota clínica do teste (avaliação visual)", ["1", "2", "3"],
            index=None, horizontal=True, key="nota_clinica",
            help="Classificação visual do teste, pra comparar depois com as métricas quantitativas do celular/Kinem (grau 1 = melhor, 3 = pior, ou a escala que você usa clinicamente).",
        )
        ver_analise = st.button("📋 Ver variáveis", type="primary", use_container_width=True, key="btn_ver_analise")

        if ver_analise:
            st.session_state.mostrar_analise_clinica = True
        if st.session_state.get("mostrar_analise_clinica"):
            if not trials:
                st.info("Não consegui detectar repetições — não dá pra montar a tabela de análise.")
            else:
                analise_rows = []
                for i, (t_start, t_end) in enumerate(trials, start=1):
                    phases = trial_phases[i - 1] if i - 1 < len(trial_phases) else None
                    d_start, d_end = phases["descida"] if phases else (t_start, t_start)
                    s_start, s_end = phases["subida"] if phases else (t_end, t_end)

                    t_flex_k, v_flex_k = time_to_peak(angle_kinem_sagital, x_axis, t_start, t_end)
                    t_flex_p, v_flex_p = time_to_peak(angle_phone_sagital, x_axis, t_start, t_end)
                    t_valgo_k, v_valgo_k = time_to_peak(angle_kinem_frontal, x_axis, t_start, t_end, signed=True)
                    t_valgo_p, v_valgo_p = time_to_peak(angle_phone_frontal, x_axis, t_start, t_end, signed=True)
                    diff_t_k = (t_valgo_k - t_flex_k) if (t_valgo_k is not None and t_flex_k is not None) else None
                    diff_t_p = (t_valgo_p - t_flex_p) if (t_valgo_p is not None and t_flex_p is not None) else None
                    razao_k = (abs(v_valgo_k) / v_flex_k) if (v_valgo_k is not None and v_flex_k not in (None, 0)) else None
                    razao_p = (abs(v_valgo_p) / v_flex_p) if (v_valgo_p is not None and v_flex_p not in (None, 0)) else None

                    rms_l5_lateral = compute_rms(l5_lateral, x_axis, t_start, t_end)
                    path_l5_lateral = compute_path_length(l5_lateral, x_axis, t_start, t_end)
                    rom_l5_lateral = compute_rom(l5_lateral, x_axis, t_start, t_end)
                    razao_path_rom = (path_l5_lateral / rom_l5_lateral) if (path_l5_lateral is not None and rom_l5_lateral not in (None, 0)) else None

                    jerk_rms_k = compute_rms(jerk_kinem_sagital, x_axis, t_start, t_end)
                    jerk_rms_p = compute_rms(jerk_phone_sagital, x_axis, t_start, t_end)

                    rms_accel_trunk_k = compute_rms(acc_ml_kinem_trunk, x_axis, d_start, s_end)
                    rms_accel_trunk_p = compute_rms(acc_ml_phone_trunk, x_axis, d_start, s_end)
                    razao_accel_trunk = (rms_accel_trunk_p / rms_accel_trunk_k) if (rms_accel_trunk_p is not None and rms_accel_trunk_k not in (None, 0)) else None

                    rms_angvel_trunk_k = compute_rms(trunk_angvel_kinem, x_axis, d_start, s_end)
                    rms_angvel_trunk_p = compute_rms(trunk_angvel_phone, x_axis, d_start, s_end)
                    razao_angvel_trunk = (rms_angvel_trunk_p / rms_angvel_trunk_k) if (rms_angvel_trunk_p is not None and rms_angvel_trunk_k not in (None, 0)) else None

                    prep_dur = (phases["preparacao"][1] - phases["preparacao"][0]) if phases else None
                    desc_dur = (phases["descida"][1] - phases["descida"][0]) if phases else None
                    sub_dur = (phases["subida"][1] - phases["subida"][0]) if phases else None

                    analise_rows.append({
                        "Trial": str(i),
                        "Nota clínica": nota_clinica if nota_clinica else "—",
                        "Duração Preparação (s)": prep_dur,
                        "Duração Descida (s)": desc_dur,
                        "Duração Subida (s)": sub_dur,
                        "ADM Joelho Sagital — Kinem": compute_rom(angle_kinem_sagital, x_axis, t_start, t_end),
                        "ADM Joelho Sagital — Celular": compute_rom(angle_phone_sagital, x_axis, t_start, t_end),
                        "Pico Flexão Joelho — Kinem": compute_peak(angle_kinem_sagital, x_axis, t_start, t_end),
                        "Pico Flexão Joelho — Celular": compute_peak(angle_phone_sagital, x_axis, t_start, t_end),
                        "Vel. Pico Flexão (°/s) — Kinem": compute_peak(vel_kinem_sagital, x_axis, t_start, t_end),
                        "Vel. Pico Flexão (°/s) — Celular": compute_peak(vel_phone_sagital, x_axis, t_start, t_end),
                        "ADM Joelho Frontal — Kinem": compute_rom(angle_kinem_frontal, x_axis, t_start, t_end),
                        "ADM Joelho Frontal — Celular": compute_rom(angle_phone_frontal, x_axis, t_start, t_end),
                        "Pico Valgo Joelho — Kinem": compute_peak(angle_kinem_frontal, x_axis, t_start, t_end, signed=True),
                        "Pico Valgo Joelho — Celular": compute_peak(angle_phone_frontal, x_axis, t_start, t_end, signed=True),
                        "Vel. Pico Valgo (°/s) — Kinem": compute_peak(vel_kinem_frontal, x_axis, t_start, t_end, signed=True),
                        "Vel. Pico Valgo (°/s) — Celular": compute_peak(vel_phone_frontal, x_axis, t_start, t_end, signed=True),
                        "Pico Valgo (Descida) — Kinem": compute_peak(angle_kinem_frontal, x_axis, d_start, d_end, signed=True),
                        "Pico Valgo (Descida) — Celular": compute_peak(angle_phone_frontal, x_axis, d_start, d_end, signed=True),
                        "Pico Valgo (Subida) — Kinem": compute_peak(angle_kinem_frontal, x_axis, s_start, s_end, signed=True),
                        "Pico Valgo (Subida) — Celular": compute_peak(angle_phone_frontal, x_axis, s_start, s_end, signed=True),
                        "Tempo até Pico Valgo − Flexão (s) — Kinem": diff_t_k,
                        "Tempo até Pico Valgo − Flexão (s) — Celular": diff_t_p,
                        "Razão |Valgo|/Flexão — Kinem": razao_k,
                        "Razão |Valgo|/Flexão — Celular": razao_p,
                        "ADM Quadril Sagital — Kinem": compute_rom(angle_hip_kinem_sagital, x_axis, t_start, t_end),
                        "ADM Quadril Sagital — Celular": compute_rom(angle_hip_phone_sagital, x_axis, t_start, t_end),
                        "ADM Quadril Frontal — Kinem": compute_rom(angle_hip_kinem_frontal, x_axis, t_start, t_end),
                        "ADM Quadril Frontal — Celular": compute_rom(angle_hip_phone_frontal, x_axis, t_start, t_end),
                        "Estabilidade Tronco — RMS lateral L5 (m)": rms_l5_lateral,
                        "Estabilidade Tronco — Razão caminho/deslocamento": razao_path_rom,
                        "Estabilidade Tronco — RMS Acel. Lateral (Kinem)": rms_accel_trunk_k,
                        "Estabilidade Tronco — RMS Acel. Lateral (Celular)": rms_accel_trunk_p,
                        "Estabilidade Tronco — Razão Acel. Celular/Kinem": razao_accel_trunk,
                        "Estabilidade Tronco — RMS Vel.Ang. (Kinem)": rms_angvel_trunk_k,
                        "Estabilidade Tronco — RMS Vel.Ang. (Celular)": rms_angvel_trunk_p,
                        "Estabilidade Tronco — Razão Vel.Ang. Celular/Kinem": razao_angvel_trunk,
                        "Suavidade (Jerk RMS) Joelho Sagital — Kinem": jerk_rms_k,
                        "Suavidade (Jerk RMS) Joelho Sagital — Celular": jerk_rms_p,
                    })
                analise_df = pd.DataFrame(analise_rows)

                resultante_analise = {"Trial": "Resultante (média)", "Nota clínica": nota_clinica if nota_clinica else "—"}
                desvio_analise = {"Trial": "Desvio padrão (variabilidade)", "Nota clínica": "—"}
                for col in analise_df.columns:
                    if col in ("Trial", "Nota clínica"):
                        continue
                    resultante_analise[col] = analise_df[col].mean()
                    desvio_analise[col] = analise_df[col].std()
                analise_df_full = pd.concat(
                    [analise_df, pd.DataFrame([resultante_analise]), pd.DataFrame([desvio_analise])],
                    ignore_index=True,
                )

                with st.container(border=True):
                    def fmt_for_col(col):
                        if "Razão" in col:
                            return "{:.2f}×"
                        if "Duração" in col:
                            return "{:.2f}s"
                        if "Tempo até" in col:
                            return "{:+.2f}s"
                        if "Jerk" in col:
                            return "{:.0f}°/s³"
                        if "RMS Acel" in col:
                            return "{:.3f}m/s²"
                        if "RMS Vel" in col or "Vel." in col:
                            return "{:.0f}°/s"
                        if "RMS lateral" in col:
                            return "{:.4f}m"
                        return "{:.1f}°"
                    fmt_cols = {c: fmt_for_col(c) for c in analise_df_full.columns if c not in ("Trial", "Nota clínica")}
                    st.dataframe(analise_df_full.style.format(fmt_cols), hide_index=True, use_container_width=True)

                st.caption(
                    "Pico = maior valor atingido no trial (não a variação total). Pico de valgo preserva o sinal "
                    "(positivo/negativo indicam o lado — ver nota do plano frontal)."
                )

                csv_bytes = analise_df_full.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "📥 Exportar análise (CSV)", csv_bytes,
                    file_name="analise_clinica_step_down.csv", mime="text/csv",
                    use_container_width=True,
                )
        # --- Fases do movimento por trial: duração já incluída na tabela "Ver variáveis" acima ---

    st.divider()


    # ══════════════════════════════════════════
    # Exportar Excel — apenas janela selecionada
    # ══════════════════════════════════════════
    st.subheader("📥 Exportar Excel")
    st.caption(f"Exporta todos os eixos X, Y, Z + ângulo do joelho • janela: **{view_start:+.1f} s → {view_end:+.1f} s** relativo ao pico")

    if st.button("Gerar arquivo Excel (L5 + Coxa + Tornozelo + Ângulo)", use_container_width=True):
        mask_exp = (x_axis >= view_start) & (x_axis <= view_end)
        win_idx = np.where(mask_exp)[0]

        if len(win_idx) == 0:
            st.error("Janela vazia — ajuste os limites de início/fim.")
        else:
            windowed = {fname: df.iloc[win_idx].reset_index(drop=True) for fname, df in aligned_data.items()}
            t_w = np.arange(len(win_idx)) / pfs

            sheets = {}
            for gkey, gdef in GROUPS.items():
                pf = phone_files[gkey]
                sheets[gdef["label"]] = build_export_sheet(
                    windowed, kinem_ref, pf["acc"], pf["gyr"], gdef["kinem_kw"], t_w,
                )

            df_angle = pd.DataFrame({"Tempo (s)": t_w})

            def add_angle_col(df_out, series, col_name):
                if series is None:
                    return
                if len(series) >= (win_idx.max() + 1):
                    df_out[col_name] = series[win_idx]
                else:
                    valid_idx = win_idx[win_idx < len(series)]
                    y = np.full(len(win_idx), np.nan)
                    y[:len(valid_idx)] = series[valid_idx]
                    df_out[col_name] = y

            add_angle_col(df_angle, angle_kinem_sagital, "Angulo_Kinem_Joelho_Sagital_graus")
            add_angle_col(df_angle, angle_phone_sagital, "Angulo_Celular_Joelho_Sagital_graus")
            add_angle_col(df_angle, angle_kinem_3d, "Angulo_Kinem_Joelho_3D_total_graus")
            add_angle_col(df_angle, angle_kinem_frontal, "Angulo_Kinem_Joelho_Frontal_graus")
            add_angle_col(df_angle, angle_phone_frontal, "Angulo_Celular_Joelho_Frontal_graus")
            add_angle_col(df_angle, angle_hip_kinem_sagital, "Angulo_Kinem_Quadril_Sagital_graus")
            add_angle_col(df_angle, angle_hip_phone_sagital, "Angulo_Celular_Quadril_Sagital_graus")
            add_angle_col(df_angle, angle_hip_kinem_frontal, "Angulo_Kinem_Quadril_Frontal_graus")
            add_angle_col(df_angle, angle_hip_phone_frontal, "Angulo_Celular_Quadril_Frontal_graus")

            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                for sheet_name, df_sheet in sheets.items():
                    df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
                df_angle.to_excel(writer, sheet_name="Angulo_Joelho", index=False)
            buf.seek(0)

            st.download_button(
                "⬇ Baixar sinais_sincronizados.xlsx", buf, file_name="sinais_sincronizados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    st.divider()

    # ══════════════════════════════════════════
    # Quadro-resumo do resultado do teste
    # ══════════════════════════════════════════
    valid_trials_summary = [(t, p) for t, p in zip(trials, trial_phases) if p]
    if valid_trials_summary:
        adm_sag_k, adm_sag_p = [], []
        peak_valgo_k, peak_valgo_p = [], []
        peak_valgo_desc_k, peak_valgo_desc_p = [], []
        peak_valgo_sub_k, peak_valgo_sub_p = [], []
        vel_pico_k, jerk_k_list = [], []
        rms_acc_k_list, rms_acc_p_list = [], []
        rms_acc_desc_k, rms_acc_desc_p, rms_acc_sub_k, rms_acc_sub_p = [], [], [], []
        rms_angvel_k_list, rms_angvel_p_list = [], []
        rms_angvel_desc_k, rms_angvel_desc_p, rms_angvel_sub_k, rms_angvel_sub_p = [], [], [], []

        def _add(lst, val):
            if val is not None:
                lst.append(val)

        for (t_start, t_end), phases in valid_trials_summary:
            d_start, d_end = phases["descida"]
            s_start, s_end = phases["subida"]

            _add(adm_sag_k, compute_rom(angle_kinem_sagital, x_axis, t_start, t_end))
            _add(adm_sag_p, compute_rom(angle_phone_sagital, x_axis, t_start, t_end))

            _add(peak_valgo_k, compute_peak(angle_kinem_frontal, x_axis, t_start, t_end, signed=True))
            _add(peak_valgo_p, compute_peak(angle_phone_frontal, x_axis, t_start, t_end, signed=True))
            _add(peak_valgo_desc_k, compute_peak(angle_kinem_frontal, x_axis, d_start, d_end, signed=True))
            _add(peak_valgo_desc_p, compute_peak(angle_phone_frontal, x_axis, d_start, d_end, signed=True))
            _add(peak_valgo_sub_k, compute_peak(angle_kinem_frontal, x_axis, s_start, s_end, signed=True))
            _add(peak_valgo_sub_p, compute_peak(angle_phone_frontal, x_axis, s_start, s_end, signed=True))

            _add(vel_pico_k, compute_peak(vel_kinem_sagital, x_axis, t_start, t_end))
            _add(jerk_k_list, compute_rms(jerk_kinem_sagital, x_axis, t_start, t_end))

            _add(rms_acc_k_list, compute_rms(acc_ml_kinem_trunk, x_axis, d_start, s_end))
            _add(rms_acc_p_list, compute_rms(acc_ml_phone_trunk, x_axis, d_start, s_end))
            _add(rms_acc_desc_k, compute_rms(acc_ml_kinem_trunk, x_axis, d_start, d_end))
            _add(rms_acc_desc_p, compute_rms(acc_ml_phone_trunk, x_axis, d_start, d_end))
            _add(rms_acc_sub_k, compute_rms(acc_ml_kinem_trunk, x_axis, s_start, s_end))
            _add(rms_acc_sub_p, compute_rms(acc_ml_phone_trunk, x_axis, s_start, s_end))

            _add(rms_angvel_k_list, compute_rms(trunk_angvel_kinem, x_axis, d_start, s_end))
            _add(rms_angvel_p_list, compute_rms(trunk_angvel_phone, x_axis, d_start, s_end))
            _add(rms_angvel_desc_k, compute_rms(trunk_angvel_kinem, x_axis, d_start, d_end))
            _add(rms_angvel_desc_p, compute_rms(trunk_angvel_phone, x_axis, d_start, d_end))
            _add(rms_angvel_sub_k, compute_rms(trunk_angvel_kinem, x_axis, s_start, s_end))
            _add(rms_angvel_sub_p, compute_rms(trunk_angvel_phone, x_axis, s_start, s_end))

        def _fmt(lst, suffix="", casas=1):
            return f"{np.mean(lst):.{casas}f}{suffix}" if lst else "—"

        def _razao(lst_p, lst_k):
            if lst_p and lst_k and np.mean(lst_k):
                return np.mean(lst_p) / np.mean(lst_k)
            return None

        razao_adm = _razao(adm_sag_p, adm_sag_k)
        razao_acc_medias = _razao(rms_acc_p_list, rms_acc_k_list)
        razao_angvel_medias = _razao(rms_angvel_p_list, rms_angvel_k_list)
        nota_txt = nota_clinica if nota_clinica else "não informada"

        # --- leitura interpretativa: joelho, celular x Kinem ---
        if razao_adm is None:
            leitura_adm = "não foi possível comparar (faltam dados de uma das fontes)."
        elif 0.85 <= razao_adm <= 1.15:
            leitura_adm = f"o celular captou uma amplitude **bem próxima** do Kinem (proporção {razao_adm:.2f}×)."
        elif razao_adm < 0.85:
            leitura_adm = f"o celular **subestimou** a amplitude em relação ao Kinem (proporção {razao_adm:.2f}× — capta {razao_adm*100:.0f}% do real)."
        else:
            leitura_adm = f"o celular **superestimou** a amplitude em relação ao Kinem (proporção {razao_adm:.2f}×)."

        # --- leitura interpretativa: joelho, descida vs subida ---
        media_desc_k = np.mean([abs(v) for v in peak_valgo_desc_k]) if peak_valgo_desc_k else None
        media_sub_k = np.mean([abs(v) for v in peak_valgo_sub_k]) if peak_valgo_sub_k else None
        if media_desc_k is not None and media_sub_k is not None:
            if media_desc_k > media_sub_k * 1.15:
                leitura_fase_joelho = f"o valgo predominou na **descida** (fase excêntrica): {media_desc_k:.1f}° vs {media_sub_k:.1f}° na subida — sugere menor controle ao absorver a descida."
            elif media_sub_k > media_desc_k * 1.15:
                leitura_fase_joelho = f"o valgo predominou na **subida** (fase concêntrica): {media_sub_k:.1f}° vs {media_desc_k:.1f}° na descida — sugere menor controle ao empurrar de volta."
            else:
                leitura_fase_joelho = f"o valgo ficou parecido nas duas fases (descida {media_desc_k:.1f}° · subida {media_sub_k:.1f}°) — sem predomínio claro de uma fase."
        else:
            leitura_fase_joelho = "não foi possível comparar as fases (dados insuficientes)."

        # --- leitura interpretativa: tronco, descida vs subida ---
        media_desc_acc_k = np.mean(rms_acc_desc_k) if rms_acc_desc_k else None
        media_sub_acc_k = np.mean(rms_acc_sub_k) if rms_acc_sub_k else None
        if media_desc_acc_k is not None and media_sub_acc_k is not None:
            if media_desc_acc_k > media_sub_acc_k * 1.15:
                leitura_fase_tronco = f"o tronco balançou mais na **descida** (RMS acel. {media_desc_acc_k:.2f} vs {media_sub_acc_k:.2f} na subida) — sugere menos estabilidade ao absorver o movimento."
            elif media_sub_acc_k > media_desc_acc_k * 1.15:
                leitura_fase_tronco = f"o tronco balançou mais na **subida** (RMS acel. {media_sub_acc_k:.2f} vs {media_desc_acc_k:.2f} na descida) — sugere menos estabilidade ao voltar à posição inicial."
            else:
                leitura_fase_tronco = f"a instabilidade de tronco ficou parecida nas duas fases (descida {media_desc_acc_k:.2f} · subida {media_sub_acc_k:.2f}, RMS acel. Kinem) — sem predomínio claro."
        else:
            leitura_fase_tronco = "não foi possível comparar as fases (dados insuficientes)."

        resumo_md = f"""
##### 📘 Resumo do resultado do teste

**{len(valid_trials_summary)} trials** analisados (repetições segmentadas com sucesso) · Nota clínica informada: **{nota_txt}**

---
**🦵 Joelho — Celular vs. Kinem**
- ADM de flexão (média): Kinem **{_fmt(adm_sag_k, '°')}** · Celular **{_fmt(adm_sag_p, '°')}** — {leitura_adm}
- Pico de valgo/varo (média, sinal = lado): Kinem **{_fmt(peak_valgo_k, '°')}** · Celular **{_fmt(peak_valgo_p, '°')}**
- Velocidade de pico na flexão: **{_fmt(vel_pico_k, '°/s', 0)}** · Suavidade (jerk RMS, Kinem): **{_fmt(jerk_k_list, '°/s³', 0)}** (quanto menor, mais suave)

**🦵 Joelho — o que aconteceu em cada fase**
- Valgo na descida: Kinem **{_fmt(peak_valgo_desc_k, '°')}** · Celular **{_fmt(peak_valgo_desc_p, '°')}**
- Valgo na subida: Kinem **{_fmt(peak_valgo_sub_k, '°')}** · Celular **{_fmt(peak_valgo_sub_p, '°')}**
- {leitura_fase_joelho}

---
**🧍 Coluna/Tronco — Celular vs. Kinem**
- RMS aceleração lateral: Kinem **{_fmt(rms_acc_k_list, '', 3)}** · Celular **{_fmt(rms_acc_p_list, '', 3)}** (razão {f'{razao_acc_medias:.2f}×' if razao_acc_medias else '—'})
- RMS velocidade angular: Kinem **{_fmt(rms_angvel_k_list, '°/s', 1)}** · Celular **{_fmt(rms_angvel_p_list, '°/s', 1)}** (razão {f'{razao_angvel_medias:.2f}×' if razao_angvel_medias else '—'})
- *A razão Celular/Kinem se manteve consistente entre trials nos testes que fizemos — é o número mais importante pra validar o sensor, não a escala absoluta.*

**🧍 Coluna/Tronco — o que aconteceu em cada fase**
- Aceleração lateral (RMS) na descida: Kinem **{_fmt(rms_acc_desc_k, '', 3)}** · Celular **{_fmt(rms_acc_desc_p, '', 3)}**
- Aceleração lateral (RMS) na subida: Kinem **{_fmt(rms_acc_sub_k, '', 3)}** · Celular **{_fmt(rms_acc_sub_p, '', 3)}**
- {leitura_fase_tronco}

*Resumo calculado automaticamente a partir dos trials segmentados — confira a tabela "Ver variáveis" para os valores por trial.*
"""
        st.info(resumo_md)
    else:
        st.info("📘 **Resumo do resultado do teste** — não há trials segmentados o suficiente pra gerar um resumo automático.")

    # ══════════════════════════════════════════
    # Quadro-detalhe do método de processamento e análise
    # ══════════════════════════════════════════
    with st.container(border=True):
        st.markdown("""
##### 🔬 Método de processamento e análise

1. **Sincronização bruta**: pico de aceleração vertical do L5 (Kinem) como referência inicial; correlação cruzada alinha Coxa, Tornozelo e os respectivos celulares a esse mesmo instante (±1s de busca por segmento).
2. **Recentralização (x=0)**: redefinida para o pico de flexão do joelho (Kinem), não o pico de aceleração — evita ambiguidade quando há um movimento preparatório antes do teste.
3. **Ângulo do joelho e do quadril**: Kinem via vetores 3D entre marcadores (ângulo = arco-cosseno do produto escalar, ou arco-tangente com sinal nos planos frontal/quadril); celular via filtro complementar (giroscópio integrado + correção pelo acelerômetro, peso do giroscópio α=0,995 — valor otimizado empiricamente).
4. **Correção de atraso**: desloca a curva do celular no tempo pra alinhar seu pico ao pico do Kinem, por plano — compensa o atraso mecânico de resposta do sensor (tecido mole/fixação da faixa).
5. **Calibração de amplitude**: fator de escala automático por plano (±1s ao redor do pico), ajustando a amplitude do celular à do Kinem **dessa gravação específica** — não é uma calibração permanente do sensor.
6. **Segmentação de fases**: preparação/descida/subida detectadas pelo deslocamento vertical do L5 (Kinem), com limiar de sensibilidade ajustável (fração do deslocamento total que marca início/fim do movimento).
7. **Velocidade e jerk**: derivadas numéricas do ângulo (`np.gradient`); ângulo filtrado (passa-baixa Butterworth, 10Hz) antes de derivar — testado com dados reais: reduz o ruído amplificado pela derivação sem alterar o ângulo em si.
8. **Estabilidade de tronco**: RMS da aceleração e da velocidade angular laterais **brutas** (sem integrar) do L5, comparando Kinem e celular — testado e validado como mais consistente entre trials do que tentar estimar deslocamento lateral via dupla integração (que se mostrou pouco confiável).
9. **Tempo até o pico / razão valgo-flexão**: instante e valor do maior desvio de cada curva dentro do trial, comparados entre planos e fontes.
10. **Variabilidade**: desvio padrão de cada métrica entre os trials detectados (linha "Desvio padrão" nas tabelas).

*Trials nas bordas (primeiro/último) podem ter métricas distorcidas — a janela deles inclui trecho antes do início ou depois do fim da gravação real.*
""")

