"""
app.py
──────
Visualizador de Sinais — Y-Balance & Step-Down

Carrega arquivos de Kinem (câmera) e de celulares (ACC/GYR) posicionados na
L5, na coxa e no tornozelo, sincroniza-os pelo pico de impacto do salto/step,
permite pré-processamento (detrend + filtro passa-baixa), visualização
automática de todos os eixos X/Y/Z, checagem de qualidade, estimativa do
ângulo do joelho (celular vs. Kinem) e exportação de uma janela selecionada
para Excel.
"""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from signal_utils import (
    NONE_LABEL,
    apply_detrend,
    apply_lowpass,
    best_match,
    build_export_sheet,
    col_default,
    detect_time_axis,
    find_highest_peak,
    find_sync_xcorr,
    fit_scale_gain,
    get_aligned_data,
    is_xyz_col,
    kinem_cols_for_body,
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
    "proc_data": {},
    "proc_data_nofilter": {},
    "offsets": {},
    "peak_ref": None,
    "target_fs": 100,
    "fs_info": {},
    "show_preview": False,
    "synced": False,
    "synced_kinem_cols": {},
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
            st.session_state.proc_data = {}
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
            "Filtro complementar (ângulo do joelho) — peso do giroscópio", 0.80, 0.999,
            value=0.98, step=0.005,
            help="Mais próximo de 1 = confia mais no giroscópio (menos deriva do acelerômetro).",
        )


# ══════════════════════════════════════════════
# Botões: Preview + Sincronizar
# ══════════════════════════════════════════════
btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 2])

with btn_col1:
    if st.button("👁 Preview sinais brutos", use_container_width=True):
        st.session_state.show_preview = not st.session_state.show_preview

with btn_col2:
    janela_seg = st.number_input(
        "Pico nos primeiros (s)", min_value=0.1, max_value=300.0, value=16.0, step=0.5,
        help="Janela de busca do pico de sincronização.",
    )

with btn_col3:
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
        st.session_state.proc_data = {}
        st.session_state.proc_data_nofilter = {}

        janela_samp = int(janela_seg * fs_target)
        offsets = {kinem_ref: 0}
        msgs_sync = []

        peak_l5 = find_highest_peak(
            try_numeric(raw_synced[kinem_ref][kinem_sync_cols["l5"]]), janela_samp, fs_target,
        )
        st.session_state.peak_ref = peak_l5
        st.session_state.synced = True
        st.session_state.show_preview = False
        msgs_sync.append(f"**Kinem L5** — pico @ {peak_l5} ({peak_l5/fs_target:.2f} s) → x=0")

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
            pk_l5 = find_highest_peak(try_numeric(raws[kinem_ref][kinem_sync_cols["l5"]]), jsamp, tfs)
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
        st.session_state.proc_data = {}  # força reprocessamento


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
    vfs = st.session_state.target_fs or 100
    vraw, vx_samp, _ = get_aligned_data(
        st.session_state.raw_synced, st.session_state.offsets, st.session_state.peak_ref, ref_file=kinem_ref,
    )
    if vraw is None:
        vraw = {f: df.copy() for f, df in st.session_state.raw_synced.items()}
        vx_samp = np.arange(max(len(d) for d in vraw.values())) - st.session_state.peak_ref
    vx = vx_samp / vfs

    for gkey, gdef in GROUPS.items():
        pf = phone_files[gkey]
        render_alignment_check(
            gdef["label"], kinem_sync_cols[gkey], pf["acc"], pf["acc_col"],
            f"Kinem {gdef['label']}", f"ACC {gdef['label']}", vraw, vx, vfs,
        )

    # ══════════════════════════════════════════
    # Processamento inline
    # ══════════════════════════════════════════
    st.divider()
    proc_done = bool(st.session_state.proc_data)
    with st.expander(
        "⚙️ Processamento  ✔ Aplicado" if proc_done else "⚙️ Processamento  ← Configure e processe aqui",
        expanded=not proc_done,
    ):
        pc1, pc2 = st.columns(2)
        with pc1:
            do_detrend = st.checkbox("Detrend (remover tendência linear)", value=True, key="do_detrend")
        with pc2:
            do_lowpass = st.checkbox("Filtro passa-baixa (Butterworth)", value=True, key="do_lowpass")

        if do_lowpass:
            fl1, fl2 = st.columns(2)
            with fl1:
                cutoff_hz = st.number_input(
                    "Frequência de corte (Hz)", min_value=0.1, max_value=float(fs_target // 2),
                    value=min(20.0, float(fs_target // 2 - 1)), step=0.5, key="cutoff_hz",
                )
            with fl2:
                filt_order = st.selectbox("Ordem do filtro", [2, 4, 6, 8], index=1, key="filt_order")
        else:
            cutoff_hz, filt_order = 20.0, 4

        if st.button("🔧 Processar", type="primary", use_container_width=True, key="btn_processar"):
            raw = st.session_state.raw_synced
            proc, proc_nofilter = {}, {}
            for fname, df in raw.items():
                r = df.copy()
                if do_detrend:
                    r = apply_detrend(r)
                proc_nofilter[fname] = r.copy()
                if do_lowpass:
                    r = apply_lowpass(r, fs_target, cutoff_hz, filt_order)
                proc[fname] = r
            st.session_state.proc_data = proc
            st.session_state.proc_data_nofilter = proc_nofilter
            st.rerun()


# ══════════════════════════════════════════════
# Auto-visualização — todos os eixos X, Y, Z
# ══════════════════════════════════════════════
if st.session_state.proc_data and st.session_state.synced:
    pfs = st.session_state.target_fs or 100

    aligned_data, x_samp, align_msg = get_aligned_data(
        st.session_state.proc_data, st.session_state.offsets, st.session_state.peak_ref, ref_file=kinem_ref,
    )
    if aligned_data is None:
        st.error(align_msg)
        st.stop()

    x_axis = x_samp / pfs
    x_min_data, x_max_data = float(x_axis.min()), float(x_axis.max())

    kdf = aligned_data.get(kinem_ref, pd.DataFrame())

    def get_phone_xyz(fname):
        if fname == NONE or fname not in aligned_data:
            return []
        return [c for c in aligned_data[fname].columns if is_xyz_col(c)]

    def make_auto_traces(gkey):
        gdef = GROUPS[gkey]
        pf = phone_files[gkey]
        k_cols = kinem_cols_for_body(kdf, *gdef["kinem_kw"])
        traces = [(kinem_ref, c, try_numeric(kdf[c])) for c in k_cols if c in kdf.columns]
        if pf["acc"] != NONE and pf["acc"] in aligned_data:
            for c in get_phone_xyz(pf["acc"]):
                traces.append((pf["acc"], c, try_numeric(aligned_data[pf["acc"]][c])))
        if pf["gyr"] != NONE and pf["gyr"] in aligned_data:
            for c in get_phone_xyz(pf["gyr"]):
                traces.append((pf["gyr"], c, try_numeric(aligned_data[pf["gyr"]][c])))
        return traces

    group_traces = {gkey: make_auto_traces(gkey) for gkey in GROUPS}

    st.divider()
    st.subheader("📊 Sinais sincronizados — todos os eixos X, Y, Z")
    st.caption(align_msg)

    def render_auto_charts(traces):
        for fname, col, y in traces:
            fig_i = go.Figure()
            fig_i.add_trace(go.Scatter(x=x_axis, y=y, mode="lines", line=dict(width=1.5), showlegend=False))
            fig_i.add_vline(x=0, line_dash="dash", line_color="gray",
                             annotation_text="salto", annotation_position="top right")
            fig_i.update_layout(
                title=dict(text=f"<b>{fname[:26]}</b> · {col}", font_size=12),
                xaxis=dict(title="Tempo (s)  —  0 = pico do salto", range=[x_min_data, x_max_data]),
                yaxis_title="", height=220,
                margin=dict(t=42, b=38, l=55, r=10), hovermode="x", template="plotly_white",
            )
            st.plotly_chart(fig_i, use_container_width=True)

    auto_cols = st.columns(3)
    for auto_col, gkey in zip(auto_cols, GROUPS):
        with auto_col:
            st.markdown(f"#### {GROUPS[gkey]['emoji']} {GROUPS[gkey]['label']}")
            render_auto_charts(group_traces[gkey])

    st.divider()

    # ══════════════════════════════════════════
    # Seleção de janela
    # ══════════════════════════════════════════
    st.subheader("🪟 Seleção de janela")
    wc1, wc2 = st.columns(2)
    with wc1:
        view_start = st.number_input(
            "Início (s) relativo ao pico", value=float(max(x_min_data, -2.0)), step=0.5, key="view_start",
        )
    with wc2:
        view_end = st.number_input(
            "Fim (s) relativo ao pico", value=float(min(x_max_data, 8.0)), step=0.5, key="view_end",
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

    # O ângulo é calculado a partir dos dados BRUTOS alinhados (reamostrados,
    # sem detrend/filtro): o detrend distorce a posição 3D real do Kinem e
    # remove o componente de gravidade que o acelerômetro precisa para
    # estimar a inclinação do segmento.
    aligned_raw, _, _ = get_aligned_data(
        st.session_state.raw_synced, st.session_state.offsets, st.session_state.peak_ref, ref_file=kinem_ref,
    )
    kdf_raw = aligned_raw.get(kinem_ref, pd.DataFrame()) if aligned_raw else pd.DataFrame()

    pf_coxa, pf_torn = phone_files["coxa"], phone_files["tornozelo"]
    phone_ready = bool(
        aligned_raw and all(pf_coxa[k] != NONE for k in ("acc", "gyr")) and all(pf_torn[k] != NONE for k in ("acc", "gyr"))
        and all(f in aligned_raw for f in [pf_coxa["acc"], pf_coxa["gyr"], pf_torn["acc"], pf_torn["gyr"]])
    )

    angle_phone_sagital = angle_phone_frontal = angle_phone_transverse = None
    if phone_ready:
        angle_phone_sagital = knee_angle_from_phone_plane(
            aligned_raw[pf_coxa["acc"]], aligned_raw[pf_coxa["gyr"]],
            aligned_raw[pf_torn["acc"]], aligned_raw[pf_torn["gyr"]],
            pfs, plane="sagittal", alpha=cf_alpha,
        )

    kinem_angle_kw = (GROUPS["coxa"]["kinem_kw"], ("condilo",), GROUPS["tornozelo"]["kinem_kw"])
    angle_kinem_sagital = knee_angle_from_kinem_plane(
        kdf_raw, *kinem_angle_kw, plane="sagittal",
    ) if not kdf_raw.empty else None

    show_planes_extra = st.checkbox(
        "Mostrar também Frontal (valgo/varo) e Transverso (rotação) — celular e Kinem",
        value=False, key="show_planes_extra",
        help="Frontal e transverso tendem a ser mais ruidosos: a coxa/perna ficam quase alinhadas com o eixo vertical, então a pequena parcela horizontal usada nesses planos é mais sensível a ruído. O transverso do celular também sofre deriva (sem correção do acelerômetro).",
    )
    angle_kinem_3d = angle_kinem_frontal = angle_kinem_transverse = None
    angle_phone_frontal = angle_phone_transverse = None
    if show_planes_extra:
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

    zc1, zc2, zc3 = st.columns([1.4, 1, 1])
    with zc1:
        zero_baseline = st.checkbox(
            "Zerar no início (0° = extensão completa)", value=True, key="zero_baseline",
            help="Usa a média numa janela no início do movimento como referência de 0° — o resto do sinal passa a mostrar o quanto flexionou/desviou a partir dessa postura.",
        )
    with zc2:
        baseline_start = st.number_input(
            "Referência de 0° — de (s)", value=float(view_start), step=0.1, key="baseline_start",
            disabled=not zero_baseline,
        )
    with zc3:
        baseline_end = st.number_input(
            "Referência de 0° — até (s)", value=float(view_start) + 0.5, step=0.1, key="baseline_end",
            disabled=not zero_baseline,
        )

    if zero_baseline:
        angle_kinem_sagital = zero_reference_angle(angle_kinem_sagital, x_axis, baseline_start, baseline_end)
        angle_phone_sagital = zero_reference_angle(angle_phone_sagital, x_axis, baseline_start, baseline_end)
        angle_kinem_3d = zero_reference_angle(angle_kinem_3d, x_axis, baseline_start, baseline_end)
        angle_kinem_frontal = zero_reference_angle(angle_kinem_frontal, x_axis, baseline_start, baseline_end)
        angle_kinem_transverse = zero_reference_angle(angle_kinem_transverse, x_axis, baseline_start, baseline_end)
        angle_phone_frontal = zero_reference_angle(angle_phone_frontal, x_axis, baseline_start, baseline_end)
        angle_phone_transverse = zero_reference_angle(angle_phone_transverse, x_axis, baseline_start, baseline_end)

    # ── Calibração de amplitude do celular (corrige desalinhamento de
    # montagem / artefato de tecido mole, que tende a atenuar o sinal do
    # celular por um fator ~constante em relação ao Kinem) ──
    # Janela padrão: ±1s ao redor do pico de flexão do Kinem, pra evitar que
    # a deriva de giroscópio fora do movimento principal distorça o cálculo.
    default_cal_start, default_cal_end = view_start, view_end
    if angle_kinem_sagital is not None:
        n_k = min(len(angle_kinem_sagital), len(x_axis))
        mask_view = (x_axis[:n_k] >= view_start) & (x_axis[:n_k] <= view_end)
        if np.any(mask_view):
            peak_idx = np.nanargmax(angle_kinem_sagital[:n_k][mask_view])
            peak_time = x_axis[:n_k][mask_view][peak_idx]
            default_cal_start = max(view_start, float(peak_time) - 1.0)
            default_cal_end = min(view_end, float(peak_time) + 1.0)

    cc1, cc2, cc3 = st.columns([1.4, 1, 1])
    with cc1:
        calibrate_amplitude = st.checkbox(
            "Calibrar amplitude do celular pra bater com o Kinem", value=True, key="calibrate_amplitude",
            help="Ajusta a escala do sinal do celular (sagital e frontal) por um fator fixo, calculado comparando a amplitude do movimento nessa janela. Não muda o formato da curva, só o quanto ela sobe/desce.",
        )
    with cc2:
        cal_start = st.number_input(
            "Calibrar usando de (s)", value=float(default_cal_start), step=0.1, key="cal_start",
            disabled=not calibrate_amplitude,
        )
    with cc3:
        cal_end = st.number_input(
            "Calibrar usando até (s)", value=float(default_cal_end), step=0.1, key="cal_end",
            disabled=not calibrate_amplitude,
        )

    gain_sagital = gain_frontal = None
    if calibrate_amplitude:
        gain_sagital = fit_scale_gain(angle_kinem_sagital, angle_phone_sagital, x_axis, cal_start, cal_end)
        if gain_sagital is not None and angle_phone_sagital is not None:
            angle_phone_sagital = angle_phone_sagital * gain_sagital

        if show_planes_extra:
            gain_frontal = fit_scale_gain(angle_kinem_frontal, angle_phone_frontal, x_axis, cal_start, cal_end)
            if gain_frontal is not None and angle_phone_frontal is not None:
                angle_phone_frontal = angle_phone_frontal * gain_frontal

        gain_msgs = []
        if gain_sagital is not None:
            gain_msgs.append(f"sagital ×{gain_sagital:.2f}")
        if gain_frontal is not None:
            gain_msgs.append(f"frontal ×{gain_frontal:.2f}")
        if gain_msgs:
            st.caption(f"📐 Calibração aplicada ao celular: {', '.join(gain_msgs)} (não mexe no Kinem, nem no transverso do celular — a deriva não é um problema de escala).")
        else:
            st.caption("⚠️ Não deu pra calibrar — confira se há dados de ambas as fontes nessa janela.")

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
        if show_planes_extra:
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

        # --- Planos frontal e transverso (opcionais) ---
        if show_planes_extra:
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

            add_angle_col(df_angle, angle_kinem_sagital, "Angulo_Kinem_Sagital_graus")
            add_angle_col(df_angle, angle_phone_sagital, "Angulo_Celular_Sagital_graus")
            add_angle_col(df_angle, angle_kinem_3d, "Angulo_Kinem_3D_total_graus")
            add_angle_col(df_angle, angle_kinem_frontal, "Angulo_Kinem_Frontal_graus")
            add_angle_col(df_angle, angle_kinem_transverse, "Angulo_Kinem_Transverso_graus")
            add_angle_col(df_angle, angle_phone_frontal, "Angulo_Celular_Frontal_graus")
            add_angle_col(df_angle, angle_phone_transverse, "Angulo_Celular_Transverso_graus")

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
