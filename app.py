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
    compute_peak,
    compute_rom,
    detect_time_axis,
    detect_trial_windows,
    estimate_time_lag_from_peaks,
    find_highest_peak,
    find_sync_xcorr,
    fit_scale_gain,
    get_aligned_data,
    hip_angle_from_phone_plane,
    knee_angle_direction_note,
    knee_angle_from_kinem,
    knee_angle_from_kinem_plane,
    knee_angle_from_phone,
    knee_angle_from_phone_plane,
    knee_rotation_from_phone,
    load_file,
    numeric_cols,
    resample_to_regular,
    try_numeric,
    zero_reference_angle,
)

st.set_page_config(page_title="Visualizador de Sinais", layout="wide")
st.title("📊 Visualizador de Sinais — Y-Balance & Step-Down")

NONE = NONE_LABEL

# Definição dos 3 grupos anatômicos: (chave, rótulo, cor, keywords p/ auto-match
# de arquivo de celular, keywords p/ colunas do Kinem)
GROUPS = {
    "l5": dict(label="L5", emoji="🟢", kinem_kw=("l5", "l 5"),
               file_kw=(("acel", "l5"), ("acc", "l5"))),
    "coxa": dict(label="Coxa", emoji="🟠", kinem_kw=("trocanter",),
                 file_kw=(("acel", "coxa"), ("acc", "coxa"), ("acel", "quadril"), ("acc", "quadril"))),
    "tornozelo": dict(label="Tornozelo", emoji="🔵", kinem_kw=("torn",),
                      file_kw=(("acel", "tornozelo"), ("acc", "tornozelo"), ("acel", "ankle"), ("acc", "ankle"))),
}
GYR_FILE_KW = {
    "l5": (("gyro", "l5"), ("gyr", "l5")),
    "coxa": (("gyro", "coxa"), ("gyr", "coxa"), ("gyro", "quadril"), ("gyr", "quadril")),
    "tornozelo": (("gyro", "tornozelo"), ("gyr", "tornozelo"), ("gyro", "ankle"), ("gyr", "ankle")),
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
            "torn. dir. a(z)", "osso externo do torn.", "tornozelo", "torn",
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
            index=best_match(others, *gdef["file_kw"]), key=f"{gkey}_acc",
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
            index=best_match(others, *GYR_FILE_KW[gkey]), key=f"{gkey}_gyr",
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
                            vraw, vx, vfs):
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

        mask_2 = (vx >= -2) & (vx <= 2)
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
                         annotation_text="salto", annotation_position="top right")
        fig_v.update_layout(
            title=f"{title} — normalizado pelo pico (sem filtro)",
            xaxis=dict(title="Tempo (s)  —  0 = pico do salto", range=[-2, 2]),
            yaxis=dict(title="Amplitude norm.", range=[y_lo, y_hi]),
            hovermode="x unified", template="plotly_white", height=400,
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
    kinem_angle_kw = (GROUPS["coxa"]["kinem_kw"], ("condilo",), GROUPS["tornozelo"]["kinem_kw"])
    if not kdf.empty:
        angle_kinem_sagital_prelim = knee_angle_from_kinem_plane(kdf, *kinem_angle_kw, plane="sagittal")
        if angle_kinem_sagital_prelim is not None:
            n_prelim = min(len(angle_kinem_sagital_prelim), len(x_axis))
            valid_prelim = ~np.isnan(angle_kinem_sagital_prelim[:n_prelim])
            if np.any(valid_prelim):
                peak_idx_prelim = np.nanargmax(angle_kinem_sagital_prelim[:n_prelim])
                angle_peak_time = float(x_axis[:n_prelim][peak_idx_prelim])
                if abs(angle_peak_time) > 1e-9:
                    x_axis = x_axis - angle_peak_time
                    x_min_data, x_max_data = float(x_axis.min()), float(x_axis.max())
                    st.caption(f"↕️ Referência 0s recentralizada no pico de flexão do joelho (estava a {angle_peak_time:+.2f}s do pico de aceleração usado pra sincronizar os arquivos).")

    for gkey, gdef in GROUPS.items():
        pf = phone_files[gkey]
        render_alignment_check(
            gdef["label"], kinem_sync_cols[gkey], pf["acc"], pf["acc_col"],
            f"Kinem {gdef['label']}", f"ACC {gdef['label']}", aligned_data, x_axis, pfs,
        )

    st.divider()

    # ══════════════════════════════════════════
    # Seleção de janela
    # ══════════════════════════════════════════
    st.subheader("🪟 Seleção de janela")
    wc1, wc2 = st.columns(2)
    with wc1:
        view_start = st.number_input(
            "Início (s) relativo ao pico", value=float(x_min_data), step=0.5, key="view_start",
        )
    with wc2:
        view_end = st.number_input(
            "Fim (s) relativo ao pico", value=float(x_max_data), step=0.5, key="view_end",
        )

    st.divider()

    # ══════════════════════════════════════════
    # Ângulo do joelho (celular vs. Kinem)
    # ══════════════════════════════════════════
    st.subheader("🦵 Ângulo do joelho")
    st.caption(
        "Sagital (flexão/extensão): sempre positivo, 0° = extensão completa, aumenta com a flexão — "
        "celular e Kinem usam a mesma definição, diretamente comparáveis. "
        "Frontal (valgo/varo) e Transverso (rotação) têm sinal (podem ficar negativos): o sinal indica "
        "o lado do desvio, mas qual sinal corresponde a qual lado clínico depende de como os sensores/"
        "marcadores foram orientados no seu setup — veja a nota abaixo de cada um."
    )

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

    angle_phone_sagital = angle_phone_frontal = angle_phone_transverse = None
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
    # Nota: invertido (×-1) porque a geometria vetorial do quadril (tronco→coxa)
    # tem convenção oposta à do joelho — o ângulo bruto DIMINUI quando o quadril
    # flexiona (vetores ficam mais paralelos), ao contrário do joelho, onde
    # aumenta. Invertendo, "sobe = flexiona" fica consistente nos dois
    # segmentos e bate com a convenção do celular (L5 − Coxa).
    angle_hip_kinem_sagital = knee_angle_from_kinem_plane(
        kdf_raw, *hip_angle_kw, plane="sagittal",
    ) if not kdf_raw.empty else None
    if angle_hip_kinem_sagital is not None:
        angle_hip_kinem_sagital = -angle_hip_kinem_sagital
    angle_hip_kinem_frontal = knee_angle_from_kinem_plane(
        kdf_raw, *hip_angle_kw, plane="frontal", signed=True,
    ) if not kdf_raw.empty else None
    if angle_hip_kinem_frontal is not None:
        angle_hip_kinem_frontal = -angle_hip_kinem_frontal

    kinem_angle_kw = (GROUPS["coxa"]["kinem_kw"], ("condilo",), GROUPS["tornozelo"]["kinem_kw"])
    angle_kinem_sagital = knee_angle_from_kinem_plane(
        kdf_raw, *kinem_angle_kw, plane="sagittal",
    ) if not kdf_raw.empty else None

    st.caption("Mostrando os 3 planos anatômicos: sagital, frontal (valgo/varo) e transverso (rotação) — celular e Kinem.")
    angle_kinem_3d = angle_kinem_frontal = angle_kinem_transverse = None
    angle_phone_frontal = angle_phone_transverse = None
    if not kdf_raw.empty:
        angle_kinem_3d = knee_angle_from_kinem(kdf_raw, *kinem_angle_kw)
        angle_kinem_frontal = knee_angle_from_kinem_plane(kdf_raw, *kinem_angle_kw, plane="frontal", signed=True)
        angle_kinem_transverse = knee_angle_from_kinem_plane(kdf_raw, *kinem_angle_kw, plane="transverse", signed=True)
    if phone_ready:
        angle_phone_frontal = knee_angle_from_phone_plane(
            aligned_raw[pf_coxa["acc"]], aligned_raw[pf_coxa["gyr"]],
            aligned_raw[pf_torn["acc"]], aligned_raw[pf_torn["gyr"]],
            pfs, plane="frontal", alpha=cf_alpha,
        )
        angle_phone_transverse = knee_rotation_from_phone(
            aligned_raw[pf_coxa["gyr"]], aligned_raw[pf_torn["gyr"]], pfs,
        )

    zero_baseline = st.checkbox(
        "Zerar no início (0° = extensão completa)", value=True, key="zero_baseline",
        help="Usa a média nos primeiros 0.5s da gravação como referência de 0° — o resto do sinal passa a mostrar o quanto flexionou/desviou a partir dessa postura.",
    )
    baseline_start, baseline_end = x_min_data, x_min_data + 0.5

    if zero_baseline:
        angle_kinem_sagital = zero_reference_angle(angle_kinem_sagital, x_axis, baseline_start, baseline_end)
        angle_phone_sagital = zero_reference_angle(angle_phone_sagital, x_axis, baseline_start, baseline_end)
        angle_kinem_3d = zero_reference_angle(angle_kinem_3d, x_axis, baseline_start, baseline_end)
        angle_kinem_frontal = zero_reference_angle(angle_kinem_frontal, x_axis, baseline_start, baseline_end)
        angle_kinem_transverse = zero_reference_angle(angle_kinem_transverse, x_axis, baseline_start, baseline_end)
        angle_phone_frontal = zero_reference_angle(angle_phone_frontal, x_axis, baseline_start, baseline_end)
        angle_phone_transverse = zero_reference_angle(angle_phone_transverse, x_axis, baseline_start, baseline_end)
        angle_hip_kinem_sagital = zero_reference_angle(angle_hip_kinem_sagital, x_axis, baseline_start, baseline_end)
        angle_hip_phone_sagital = zero_reference_angle(angle_hip_phone_sagital, x_axis, baseline_start, baseline_end)
        angle_hip_kinem_frontal = zero_reference_angle(angle_hip_kinem_frontal, x_axis, baseline_start, baseline_end)
        angle_hip_phone_frontal = zero_reference_angle(angle_hip_phone_frontal, x_axis, baseline_start, baseline_end)

    apply_corrections = st.radio(
        "Ângulo do celular:", ["Com correções (atraso + calibração de amplitude)", "Sem correções (estimativa bruta)"],
        index=0, horizontal=True, key="apply_corrections",
    ) == "Com correções (atraso + calibração de amplitude)"

    if apply_corrections:
        # ── Correção de atraso no tempo do celular (corrige atraso mecânico —
        # ex.: o sensor preso por faixa sobre tecido mole responde um instante
        # depois do movimento real do osso, medido pelo Kinem) ──
        # Detecta o atraso usando uma janela mais larga (±1.6s) que a de
        # calibração de amplitude, pra não cortar o pico do celular fora da
        # busca se o atraso for grande.
        lag_win_sag = auto_calibration_window(angle_kinem_sagital, x_axis, x_min_data, x_max_data, half_width=1.6)
        lag_win_front = auto_calibration_window(angle_kinem_frontal, x_axis, x_min_data, x_max_data, signed=True, half_width=1.6)
        lag_win_trans = auto_calibration_window(angle_kinem_transverse, x_axis, x_min_data, x_max_data, signed=True, half_width=1.6)
        lag_win_hip_sag = auto_calibration_window(angle_hip_kinem_sagital, x_axis, x_min_data, x_max_data, half_width=1.6)
        lag_win_hip_front = auto_calibration_window(angle_hip_kinem_frontal, x_axis, x_min_data, x_max_data, signed=True, half_width=1.6)

        lag_sagital = estimate_time_lag_from_peaks(angle_kinem_sagital, angle_phone_sagital, x_axis, *lag_win_sag)
        lag_frontal = estimate_time_lag_from_peaks(angle_kinem_frontal, angle_phone_frontal, x_axis, *lag_win_front, signed=True)
        lag_transverse = estimate_time_lag_from_peaks(angle_kinem_transverse, angle_phone_transverse, x_axis, *lag_win_trans, signed=True)
        lag_hip_sagital = estimate_time_lag_from_peaks(angle_hip_kinem_sagital, angle_hip_phone_sagital, x_axis, *lag_win_hip_sag)
        lag_hip_frontal = estimate_time_lag_from_peaks(angle_hip_kinem_frontal, angle_hip_phone_frontal, x_axis, *lag_win_hip_front, signed=True)

        angle_phone_sagital = apply_time_shift(angle_phone_sagital, pfs, lag_sagital)
        angle_phone_frontal = apply_time_shift(angle_phone_frontal, pfs, lag_frontal)
        angle_phone_transverse = apply_time_shift(angle_phone_transverse, pfs, lag_transverse)
        angle_hip_phone_sagital = apply_time_shift(angle_hip_phone_sagital, pfs, lag_hip_sagital)
        angle_hip_phone_frontal = apply_time_shift(angle_hip_phone_frontal, pfs, lag_hip_frontal)

        lag_msgs = []
        if lag_sagital is not None:
            lag_msgs.append(f"joelho sagital {lag_sagital:+.2f}s")
        if lag_frontal is not None:
            lag_msgs.append(f"joelho frontal {lag_frontal:+.2f}s")
        if lag_transverse is not None:
            lag_msgs.append(f"joelho transverso {lag_transverse:+.2f}s")
        if lag_hip_sagital is not None:
            lag_msgs.append(f"quadril sagital {lag_hip_sagital:+.2f}s")
        if lag_hip_frontal is not None:
            lag_msgs.append(f"quadril frontal {lag_hip_frontal:+.2f}s")
        if lag_msgs:
            st.caption(f"⏱️ Atraso do celular corrigido (adiantado no tempo): {', '.join(lag_msgs)} — positivo = celular estava atrasado em relação ao Kinem.")

        # ── Calibração de amplitude do celular (corrige desalinhamento de
        # montagem / artefato de tecido mole, que tende a atenuar o sinal do
        # celular por um fator ~constante em relação ao Kinem) ──
        # Cada plano usa sua PRÓPRIA janela automática (±1s ao redor do pico
        # daquele plano específico no Kinem), não uma janela única baseada no
        # sagital — frontal e transverso podem ter o pico em outro instante
        # (ex.: atraso mecânico do sensor no tecido mole), então ancorar todos
        # no pico sagital sub-otimizaria a calibração dos outros planos.
        cal_start_sag, cal_end_sag = auto_calibration_window(angle_kinem_sagital, x_axis, x_min_data, x_max_data)
        cal_start_front, cal_end_front = auto_calibration_window(angle_kinem_frontal, x_axis, x_min_data, x_max_data, signed=True)
        cal_start_trans, cal_end_trans = auto_calibration_window(angle_kinem_transverse, x_axis, x_min_data, x_max_data, signed=True)
        cal_start_hip_sag, cal_end_hip_sag = auto_calibration_window(angle_hip_kinem_sagital, x_axis, x_min_data, x_max_data)
        cal_start_hip_front, cal_end_hip_front = auto_calibration_window(angle_hip_kinem_frontal, x_axis, x_min_data, x_max_data, signed=True)

        st.caption(
            "A amplitude do celular é calibrada automaticamente pra bater com o Kinem "
            "(±1s ao redor do pico de cada plano, calibrado separadamente, já com o atraso corrigido). "
            "Corrige desalinhamento de montagem/tecido mole **dessa gravação específica** — não é uma "
            "calibração permanente do sensor."
        )

        gain_sagital = fit_scale_gain(angle_kinem_sagital, angle_phone_sagital, x_axis, cal_start_sag, cal_end_sag)
        if gain_sagital is not None and angle_phone_sagital is not None:
            angle_phone_sagital = angle_phone_sagital * gain_sagital

        gain_frontal = fit_scale_gain(angle_kinem_frontal, angle_phone_frontal, x_axis, cal_start_front, cal_end_front)
        if gain_frontal is not None and angle_phone_frontal is not None:
            angle_phone_frontal = angle_phone_frontal * gain_frontal

        gain_transverse = fit_scale_gain(angle_kinem_transverse, angle_phone_transverse, x_axis, cal_start_trans, cal_end_trans)
        if gain_transverse is not None and angle_phone_transverse is not None:
            angle_phone_transverse = angle_phone_transverse * gain_transverse

        gain_hip_sagital = fit_scale_gain(angle_hip_kinem_sagital, angle_hip_phone_sagital, x_axis, cal_start_hip_sag, cal_end_hip_sag)
        if gain_hip_sagital is not None and angle_hip_phone_sagital is not None:
            angle_hip_phone_sagital = angle_hip_phone_sagital * gain_hip_sagital

        gain_hip_frontal = fit_scale_gain(angle_hip_kinem_frontal, angle_hip_phone_frontal, x_axis, cal_start_hip_front, cal_end_hip_front)
        if gain_hip_frontal is not None and angle_hip_phone_frontal is not None:
            angle_hip_phone_frontal = angle_hip_phone_frontal * gain_hip_frontal

        gain_msgs = []
        if gain_sagital is not None:
            gain_msgs.append(f"joelho sagital ×{gain_sagital:.2f}")
        if gain_frontal is not None:
            gain_msgs.append(f"joelho frontal ×{gain_frontal:.2f}")
        if gain_transverse is not None:
            gain_msgs.append(f"joelho transverso ×{gain_transverse:.2f}")
        if gain_hip_sagital is not None:
            gain_msgs.append(f"quadril sagital ×{gain_hip_sagital:.2f}")
        if gain_hip_frontal is not None:
            gain_msgs.append(f"quadril frontal ×{gain_hip_frontal:.2f}")
        if gain_msgs:
            st.caption(f"📐 Fator de calibração aplicado ao celular: {', '.join(gain_msgs)} (não mexe no Kinem). O transverso continua sujeito a deriva — calibrar a amplitude não corrige isso.")
        else:
            st.caption("⚠️ Não deu pra calibrar — confira se há dados de ambas as fontes nessa janela.")
    else:
        st.caption("📴 Mostrando a estimativa bruta do celular, sem correção de atraso nem calibração de amplitude.")

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

        # --- Plano sagital (flexão/extensão) ---
        st.markdown("**Sagital — flexão (↑) / extensão (↓)**")
        fig_sag = go.Figure()
        add_angle_trace(fig_sag, angle_kinem_sagital, "blue", "Kinem — sagital")
        add_angle_trace(fig_sag, angle_phone_sagital, "red", "Celular — sagital")
        add_angle_trace(fig_sag, angle_kinem_3d, "gray", "Kinem — 3D total", dash="dot")
        fig_sag.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="salto")
        fig_sag.update_layout(
            xaxis=dict(title="Tempo (s)  —  0 = pico do salto", range=[view_start, view_end]),
            yaxis_title="Ângulo (graus)", height=380, template="plotly_white", hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), margin=dict(t=30, b=40),
        )
        st.plotly_chart(fig_sag, use_container_width=True)

        if angle_kinem_sagital is None:
            st.caption("⚠️ Ângulo do Kinem não calculado — verifique se as colunas de posição X/Y/Z de Trocânter, Côndilo e Tornozelo estão presentes.")
        if angle_phone_sagital is None:
            st.caption("⚠️ Ângulo do celular não calculado — selecione ACC e GYR de Coxa e Tornozelo na barra lateral.")

        # --- Planos frontal e transverso ---
        if True:
            st.markdown("**Frontal — valgo (↑ ou ↓, ver nota) / varo (sentido oposto)**")
            fig_front = go.Figure()
            add_angle_trace(fig_front, angle_kinem_frontal, "green", "Kinem — frontal")
            add_angle_trace(fig_front, angle_phone_frontal, "darkorange", "Celular — frontal")
            fig_front.add_hline(y=0, line_dash="dot", line_color="lightgray")
            fig_front.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="salto")
            fig_front.update_layout(
                xaxis=dict(title="Tempo (s)  —  0 = pico do salto", range=[view_start, view_end]),
                yaxis_title="Ângulo (graus)", height=340, template="plotly_white", hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), margin=dict(t=30, b=40),
            )
            st.plotly_chart(fig_front, use_container_width=True)
            st.caption("ℹ️ " + knee_angle_direction_note("frontal"))

            st.markdown("**Transverso — rotação interna (↑ ou ↓, ver nota) / externa (sentido oposto)**")
            fig_trans = go.Figure()
            add_angle_trace(fig_trans, angle_kinem_transverse, "purple", "Kinem — transverso")
            add_angle_trace(fig_trans, angle_phone_transverse, "brown", "Celular — transverso (deriva)")
            fig_trans.add_hline(y=0, line_dash="dot", line_color="lightgray")
            fig_trans.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="salto")
            fig_trans.update_layout(
                xaxis=dict(title="Tempo (s)  —  0 = pico do salto", range=[view_start, view_end]),
                yaxis_title="Ângulo (graus)", height=340, template="plotly_white", hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), margin=dict(t=30, b=40),
            )
            st.plotly_chart(fig_trans, use_container_width=True)
            st.caption("ℹ️ " + knee_angle_direction_note("transverse"))

        # --- Ângulo do quadril (tronco/L5 vs coxa) ---
        st.divider()
        st.markdown("#### 🦴 Ângulo do quadril (tronco vs coxa)")
        st.caption(
            "Mesma lógica do joelho, agora usando L5 (tronco) e Coxa. Sagital = flexão/extensão de quadril; "
            "frontal = inclinação lateral de tronco / adução-abdução do quadril (sinal com sentido, ver nota)."
        )
        if angle_hip_phone_sagital is None and angle_hip_kinem_sagital is None:
            st.info("Selecione ACC + GYR de L5 e Coxa (celular) e/ou confirme as colunas do Kinem para calcular o ângulo do quadril.")
        else:
            st.markdown("**Sagital — flexão (↑) / extensão (↓)**")
            fig_hip_sag = go.Figure()
            add_angle_trace(fig_hip_sag, angle_hip_kinem_sagital, "teal", "Kinem — quadril sagital")
            add_angle_trace(fig_hip_sag, angle_hip_phone_sagital, "crimson", "Celular — quadril sagital")
            fig_hip_sag.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="salto")
            fig_hip_sag.update_layout(
                xaxis=dict(title="Tempo (s)  —  0 = pico do salto", range=[view_start, view_end]),
                yaxis_title="Ângulo (graus)", height=340, template="plotly_white", hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), margin=dict(t=30, b=40),
            )
            st.plotly_chart(fig_hip_sag, use_container_width=True)

            st.markdown("**Frontal — inclinação lateral de tronco (↑ ou ↓, ver nota)**")
            fig_hip_front = go.Figure()
            add_angle_trace(fig_hip_front, angle_hip_kinem_frontal, "darkcyan", "Kinem — quadril frontal")
            add_angle_trace(fig_hip_front, angle_hip_phone_frontal, "deeppink", "Celular — quadril frontal")
            fig_hip_front.add_hline(y=0, line_dash="dot", line_color="lightgray")
            fig_hip_front.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="salto")
            fig_hip_front.update_layout(
                xaxis=dict(title="Tempo (s)  —  0 = pico do salto", range=[view_start, view_end]),
                yaxis_title="Ângulo (graus)", height=340, template="plotly_white", hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), margin=dict(t=30, b=40),
            )
            st.plotly_chart(fig_hip_front, use_container_width=True)
            st.caption("ℹ️ " + knee_angle_direction_note("frontal"))

        # --- ADM (Amplitude de Movimento) por trial ---
        st.markdown("#### 📏 ADM por trial — Joelho e Quadril (Sagital e Frontal)")
        trials = detect_trial_windows(angle_kinem_sagital, x_axis)
        if not trials:
            st.info("Não consegui detectar repetições individuais automaticamente (poucos picos claros no sinal do Kinem).")
        else:
            rom_rows = []
            for i, (t_start, t_end) in enumerate(trials, start=1):
                rom_rows.append({
                    "Trial": str(i),
                    "Kinem — Joelho Sagital": compute_rom(angle_kinem_sagital, x_axis, t_start, t_end),
                    "Celular — Joelho Sagital": compute_rom(angle_phone_sagital, x_axis, t_start, t_end),
                    "Kinem — Joelho Frontal": compute_rom(angle_kinem_frontal, x_axis, t_start, t_end),
                    "Celular — Joelho Frontal": compute_rom(angle_phone_frontal, x_axis, t_start, t_end),
                    "Kinem — Quadril Sagital": compute_rom(angle_hip_kinem_sagital, x_axis, t_start, t_end),
                    "Celular — Quadril Sagital": compute_rom(angle_hip_phone_sagital, x_axis, t_start, t_end),
                    "Kinem — Quadril Frontal": compute_rom(angle_hip_kinem_frontal, x_axis, t_start, t_end),
                    "Celular — Quadril Frontal": compute_rom(angle_hip_phone_frontal, x_axis, t_start, t_end),
                })
            rom_df = pd.DataFrame(rom_rows)

            resultante = {"Trial": "Resultante (média)"}
            for col in rom_df.columns:
                if col == "Trial":
                    continue
                resultante[col] = rom_df[col].mean()
            rom_df = pd.concat([rom_df, pd.DataFrame([resultante])], ignore_index=True)

            with st.container(border=True):
                st.dataframe(
                    rom_df.style.format({c: "{:.1f}°" for c in rom_df.columns if c != "Trial"}),
                    hide_index=True, use_container_width=True,
                )
            st.caption(f"{len(trials)} repetições detectadas automaticamente pelos picos do Kinem sagital. ADM = máximo − mínimo do ângulo dentro de cada trial.")

        # --- Avaliação clínica: nota + análise completa por trial ---
        st.divider()
        st.markdown("#### 🩺 Avaliação clínica")
        nota_clinica = st.radio(
            "Nota clínica do teste (avaliação visual)", ["1", "2", "3"],
            index=None, horizontal=True, key="nota_clinica",
            help="Classificação visual do teste, pra comparar depois com as métricas quantitativas do celular/Kinem (grau 1 = melhor, 3 = pior, ou a escala que você usa clinicamente).",
        )
        ver_analise = st.button("🔍 Ver análise", type="primary", use_container_width=True, key="btn_ver_analise")

        if ver_analise:
            st.session_state.mostrar_analise_clinica = True
        if st.session_state.get("mostrar_analise_clinica"):
            if not trials:
                st.info("Não consegui detectar repetições — não dá pra montar a tabela de análise.")
            else:
                analise_rows = []
                for i, (t_start, t_end) in enumerate(trials, start=1):
                    analise_rows.append({
                        "Trial": str(i),
                        "Nota clínica": nota_clinica if nota_clinica else "—",
                        "ADM Joelho Sagital — Kinem": compute_rom(angle_kinem_sagital, x_axis, t_start, t_end),
                        "ADM Joelho Sagital — Celular": compute_rom(angle_phone_sagital, x_axis, t_start, t_end),
                        "Pico Flexão Joelho — Kinem": compute_peak(angle_kinem_sagital, x_axis, t_start, t_end),
                        "Pico Flexão Joelho — Celular": compute_peak(angle_phone_sagital, x_axis, t_start, t_end),
                        "ADM Joelho Frontal — Kinem": compute_rom(angle_kinem_frontal, x_axis, t_start, t_end),
                        "ADM Joelho Frontal — Celular": compute_rom(angle_phone_frontal, x_axis, t_start, t_end),
                        "Pico Valgo Joelho — Kinem": compute_peak(angle_kinem_frontal, x_axis, t_start, t_end, signed=True),
                        "Pico Valgo Joelho — Celular": compute_peak(angle_phone_frontal, x_axis, t_start, t_end, signed=True),
                        "ADM Quadril Sagital — Kinem": compute_rom(angle_hip_kinem_sagital, x_axis, t_start, t_end),
                        "ADM Quadril Sagital — Celular": compute_rom(angle_hip_phone_sagital, x_axis, t_start, t_end),
                        "ADM Quadril Frontal — Kinem": compute_rom(angle_hip_kinem_frontal, x_axis, t_start, t_end),
                        "ADM Quadril Frontal — Celular": compute_rom(angle_hip_phone_frontal, x_axis, t_start, t_end),
                    })
                analise_df = pd.DataFrame(analise_rows)

                resultante_analise = {"Trial": "Resultante (média)", "Nota clínica": nota_clinica if nota_clinica else "—"}
                for col in analise_df.columns:
                    if col in ("Trial", "Nota clínica"):
                        continue
                    resultante_analise[col] = analise_df[col].mean()
                analise_df_full = pd.concat([analise_df, pd.DataFrame([resultante_analise])], ignore_index=True)

                with st.container(border=True):
                    fmt_cols = {c: "{:.1f}°" for c in analise_df_full.columns if c not in ("Trial", "Nota clínica")}
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

    st.divider()


    # ══════════════════════════════════════════
    # Check de qualidade
    # ══════════════════════════════════════════
    with st.expander("⚙️ Colunas para check de qualidade (1 por fonte)", expanded=False):
        st.caption("Escolha exatamente qual coluna usar de cada fonte. Os sinais serão plotados sobrepostos (z-score).")

        qa_kinem_keywords = {
            "l5": ["l 5 d(z)", "l5 d(z)", "l 5 v(z)", "l5 v(z)", "l 5 a(z)", "l5 a(z)", "l 5 z", "l5"],
            "coxa": ["trocanter maior dir. a(z)", "trocanter a(z)", "trocanter maior dir.", "trocanter"],
            "tornozelo": ["osso externo do torn. dir. a(z)", "osso externo do torn. a(z)", "osso externo do torn.", "torn"],
        }

        qa_kinem_cols, qa_phone_cols = {}, {}
        qa_cols_ui = st.columns(3)
        for ui_col, gkey in zip(qa_cols_ui, GROUPS):
            with ui_col:
                gdef = GROUPS[gkey]
                qa_kinem_cols[gkey] = st.selectbox(
                    f"{gdef['emoji']} Kinem — {gdef['label']}", kinem_num, key=f"qa_kinem_{gkey}",
                    index=col_default(kinem_num, qa_kinem_keywords[gkey]),
                )
                pf = phone_files[gkey]
                acc_num = numeric_cols(aligned_data.get(pf["acc"], pd.DataFrame())) if pf["acc"] != NONE else []
                gyr_num = numeric_cols(aligned_data.get(pf["gyr"], pd.DataFrame())) if pf["gyr"] != NONE else []
                qa_phone_cols[gkey] = {
                    "acc": st.selectbox(
                        f"{gdef['emoji']} ACC — {gdef['label']}", acc_num if acc_num else ["—"],
                        key=f"qa_acc_{gkey}", index=col_default(acc_num, ["z", "y", "x"]) if acc_num else 0,
                    ) if acc_num else None,
                    "gyr": st.selectbox(
                        f"{gdef['emoji']} GYR — {gdef['label']}", gyr_num if gyr_num else ["—"],
                        key=f"qa_gyr_{gkey}", index=col_default(gyr_num, ["z", "y", "x"]) if gyr_num else 0,
                    ) if gyr_num else None,
                }

    show_qa = st.checkbox("🔍 Checar qualidade dos dados", value=False)
    if show_qa:
        qa_xmin, qa_xmax = view_start, view_end
        mask_qa = (x_axis >= qa_xmin) & (x_axis <= qa_xmax)
        x_view = x_axis[mask_qa]

        def get_qa_entry(fname, col_name):
            df_q = aligned_data.get(fname) if (fname and fname != NONE) else None
            if df_q is None or col_name is None or col_name not in df_q.columns:
                return None
            y = try_numeric(df_q[col_name]).values[mask_qa].astype(float)
            if np.all(np.isnan(y)):
                return None
            return (float(np.nanstd(y)), f"{fname[:20]} · {col_name}", y)

        qa_cols_out = st.columns(3)
        for ui_col, gkey in zip(qa_cols_out, GROUPS):
            gdef = GROUPS[gkey]
            pf = phone_files[gkey]
            group_entries = [e for e in [
                get_qa_entry(kinem_ref, qa_kinem_cols[gkey]),
                get_qa_entry(pf["acc"] if pf["acc"] != NONE else "", qa_phone_cols[gkey]["acc"]),
                get_qa_entry(pf["gyr"] if pf["gyr"] != NONE else "", qa_phone_cols[gkey]["gyr"]),
            ] if e]
            with ui_col:
                st.markdown(f"#### {gdef['emoji']} {gdef['label']} — Kinem vs Celular")
                if not group_entries:
                    st.info("Nenhum sinal classificado neste grupo.")
                    continue
                fig_qa = go.Figure()
                for std_val, lbl, y_raw in group_entries:
                    mn, sd = np.nanmean(y_raw), np.nanstd(y_raw)
                    y_norm = (y_raw - mn) / sd if sd > 0 else y_raw - mn
                    fig_qa.add_trace(go.Scatter(
                        x=x_view, y=y_norm, mode="lines", name=f"{lbl}  (σ_orig={std_val:.3f})",
                    ))
                fig_qa.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="salto")
                fig_qa.update_layout(
                    xaxis=dict(title="Tempo (s)  —  0 = pico do salto", range=[qa_xmin, qa_xmax]),
                    yaxis_title="z-score", height=360, template="plotly_white", hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), margin=dict(t=30, b=40),
                )
                st.plotly_chart(fig_qa, use_container_width=True)

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
            add_angle_col(df_angle, angle_kinem_transverse, "Angulo_Kinem_Joelho_Transverso_graus")
            add_angle_col(df_angle, angle_phone_frontal, "Angulo_Celular_Joelho_Frontal_graus")
            add_angle_col(df_angle, angle_phone_transverse, "Angulo_Celular_Joelho_Transverso_graus")
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
