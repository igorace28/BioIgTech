import base64
import os
import tempfile
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st

# Configurazione della pagina
st.set_page_config(
    page_title="BioIgTech - Biomechanics Data Motion",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="auto",
)

# Design CSS personalizzato
st.markdown(
    """
    <style>
    .stApp {
        background-color: #110c1d;
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] {
        background-color: #1a132c;
        border-right: 1px solid #3b2b5c;
    }
    video {
        max-height: 280px !important;
        width: 100% !important;
        object-fit: contain;
        border-radius: 8px;
        border: 1px solid #5b37b7;
    }
    .stButton>button {
        background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%);
        color: white;
        border-radius: 6px;
        border: none;
        font-weight: 600;
        padding: 0.5rem 1rem;
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #a78bfa 0%, #7c3aed 100%);
        box-shadow: 0 6px 16px rgba(139, 92, 246, 0.5);
    }
    input[type="range"] {
        accent-color: #8b5cf6 !important;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #3b2b5c;
        border-radius: 8px;
        background-color: #161026;
    }
    div[data-testid="stMetricValue"] {
        color: #c4b5fd !important;
        font-weight: 700;
    }
    
    /* Header Centrato */
    .header-container {
        text-align: center;
        padding: 0.5rem 0 1rem 0;
    }
    .logo-img {
        width: 320px;
        max-width: 100%;
        border-radius: 8px;
        pointer-events: none;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0.8rem;
        margin-bottom: 1.2rem;
    }
    
    /* Footer Style */
    .custom-footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1.5rem 0;
        border-top: 1px solid #2e1d52;
        color: #94a3b8;
        font-size: 0.95rem;
    }
    .insta-link {
        color: #c4b5fd;
        text-decoration: none;
        font-weight: 600;
        transition: color 0.3s ease;
    }
    .insta-link:hover {
        color: #e879f9;
        text-decoration: underline;
    }
    </style>
""",
    unsafe_allow_html=True,
)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calcola_angolo(a, b, c):
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)
    vec1 = a - b
    vec2 = c - b
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    cosine_angle = np.dot(vec1, vec2) / (norm1 * norm2)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return float(angle)


# Lettura del logo in base64
logo_base64 = ""
for logo_name in ["logo.jpg", "logo.png"]:
    if os.path.exists(logo_name):
        with open(logo_name, "rb") as f:
            logo_base64 = base64.b64encode(f.read()).decode("utf-8")
        break

st.markdown(
    f"""
    <div class="header-container">
        <img src="data:image/jpeg;base64,{logo_base64}" class="logo-img">
        <div class="sub-title">Piattaforma professionale di Video Analisi Biomeccanica 2D (Piano Sagittale & Frontale)</div>
    </div>
""",
    unsafe_allow_html=True,
)

with st.expander("ℹ️ Guida Rapida ai Piani di Analisi e Soglie"):
    st.write(
        """
        - **Piano Sagittale (Profilo):** Valuta accuratamente i Range of Motion (ROM) e i picchi di cinematica articolare (ginocchio, anca, caviglia).
        - **Piano Frontale:** 
          - **KAR (Knee-to-Ankle Ratio):** Rapporto adimensionale tra la distanza orizzontale delle ginocchia e quella delle caviglie.
            - **0.85 – 1.15:** NEUTRO (Allineamento Corretto)
            - **< 0.85:** VALGO DINAMICO (rientro mediale del ginocchio)
            - **> 1.15:** VARO DINAMICO (deviazione laterale del ginocchio)
          - **Inclinazione Bacino:** < 15px = **STABILE (Equilibrato)**; ≥ 15px = **COMPENSO**.
        """
    )

# Sidebar
st.sidebar.header("🎛️ Pannello di Controllo")

piano_analisi = st.sidebar.selectbox(
    "📐 Piano di Analisi",
    [
        "Piano Sagittale (Profilo: ROM & Cinematica)",
        "Piano Frontale (Valgismo/Varismo Dinamico & Simmetrie)",
    ],
)

uploaded_file = st.sidebar.file_uploader(
    "Carica video (MP4, MOV)", type=["mp4", "mov", "avi"]
)

if uploaded_file is None:
    st.session_state.pop("analizzato", None)
    st.session_state.pop("pose_data", None)
    st.session_state.pop("output_path", None)

slow_factor = st.sidebar.slider(
    "Fattore Slow Motion", min_value=1.0, max_value=3.0, value=2.0, step=0.5
)

st.sidebar.subheader("🎯 Parametri di Tracciamento")

if "Sagittale" in piano_analisi:
    analizza_ginocchio = st.sidebar.checkbox("Cinematica Ginocchio", value=True)
    analizza_anca = st.sidebar.checkbox("Cinematica Anca", value=True)
    analizza_caviglia = st.sidebar.checkbox("Cinematica Caviglia", value=True)
    analizza_spalla = st.sidebar.checkbox("Cinematica Spalla", value=False)
    analizza_gomito = st.sidebar.checkbox("Cinematica Gomito", value=False)
else:
    analizza_kar = st.sidebar.checkbox(
        "Indice di Allineamento (KAR - Valgo/Varo/Neutro)", value=True
    )
    analizza_bacino = st.sidebar.checkbox(
        "Stabilità Bacino (Soglia < 15px)", value=True
    )

    with st.sidebar.expander("📖 Spiegazione Parametri Frontali"):
        st.markdown(
            """
        * **KAR (Knee-to-Ankle Ratio):** Rapporto tra la distanza orizzontale delle ginocchia e quella delle caviglie (Distanza Ginocchia / Distanza Caviglie).
          - **0.85 – 1.15:** **Neutro (Allineamento Equilibrato)**.
          - **< 0.85:** **Valgismo Dinamico** (le ginocchia cedono verso l'interno).
          - **> 1.15:** **Varismo Dinamico** (le ginocchia spingono verso l'esterno).
          - *Nota:* Essendo un rapporto adimensionale, è indipendente dalla distanza della fotocamera.
        * **Stabilità Bacino (< 15px):** Differenza di altezza in pixel tra l'anca destra e sinistra per verificare se il bacino rimane orizzontale ed equilibrato durante la discesa.
        """
        )

# Layout principale per i video
if uploaded_file is not None:
    col1, col2 = st.columns(2)

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())

    with col1:
        st.subheader("📹 Video Originale")
        st.video(tfile.name)

    if st.sidebar.button("🚀 Avvia Analisi Biomeccanica"):
        with st.spinner(
            "Elaborazione in corso... Calcolo indici cinematici e soglie."
        ):
            cap = cv2.VideoCapture(tfile.name)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            orig_fps = int(cap.get(cv2.CAP_PROP_FPS))
            if orig_fps == 0:
                orig_fps = 30

            new_fps = max(5, int(orig_fps / slow_factor))

            output_path = tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ).name
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            out = cv2.VideoWriter(output_path, fourcc, new_fps, (width, height))

            pose_data = []
            frame_idx = 0

            joint_spec = mp_drawing.DrawingSpec(
                color=(255, 0, 200), thickness=6, circle_radius=5
            )
            bone_spec = mp_drawing.DrawingSpec(color=(186, 85, 211), thickness=4)

            with mp_pose.Pose(
                min_detection_confidence=0.6, min_tracking_confidence=0.6
            ) as pose:
                while cap.isOpened():
                    success, image = cap.read()
                    if not success:
                        break

                    tempo_secondi = frame_idx / new_fps
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = pose.process(image_rgb)

                    row_data = {"tempo_s": round(tempo_secondi, 2)}
                    has_data = False

                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=joint_spec,
                            connection_drawing_spec=bone_spec,
                        )

                        lm = results.pose_landmarks.landmark

                        if "Sagittale" in piano_analisi:
                            vis_left = (
                                lm[mp_pose.PoseLandmark.LEFT_HIP].visibility
                                + lm[mp_pose.PoseLandmark.LEFT_KNEE].visibility
                            ) / 2
                            vis_right = (
                                lm[mp_pose.PoseLandmark.RIGHT_HIP].visibility
                                + lm[mp_pose.PoseLandmark.RIGHT_KNEE].visibility
                            ) / 2

                            side_prefix = (
                                "LEFT" if vis_left >= vis_right else "RIGHT"
                            )

                            spalla = [
                                lm[getattr(mp_pose.PoseLandmark, f"{side_prefix}_SHOULDER")].x * width,
                                lm[getattr(mp_pose.PoseLandmark, f"{side_prefix}_SHOULDER")].y * height,
                            ]
                            gomito = [
                                lm[getattr(mp_pose.PoseLandmark, f"{side_prefix}_ELBOW")].x * width,
                                lm[getattr(mp_pose.PoseLandmark, f"{side_prefix}_ELBOW")].y * height,
                            ]
                            polso = [
                                lm[getattr(mp_pose.PoseLandmark, f"{side_prefix}_WRIST")].x * width,
                                lm[getattr(mp_pose.PoseLandmark, f"{side_prefix}_WRIST")].y * height,
                            ]
                            anca = [
                                lm[getattr(mp_pose.PoseLandmark, f"{side_prefix}_HIP")].x * width,
                                lm[getattr(mp_pose.PoseLandmark, f"{side_prefix}_HIP")].y * height,
                            ]
                            ginocchio = [
                                lm[getattr(mp_pose.PoseLandmark, f"{side_prefix}_KNEE")].x * width,
                                lm[getattr(mp_pose.PoseLandmark, f"{side_prefix}_KNEE")].y * height,
                            ]
                            caviglia = [
                                lm[getattr(mp_pose.PoseLandmark, f"{side_prefix}_ANKLE")].x * width,
                                lm[getattr(mp_pose.PoseLandmark, f"{side_prefix}_ANKLE")].y * height,
                            ]
                            punta_piede = [
                                lm[getattr(mp_pose.PoseLandmark, f"{side_prefix}_FOOT_INDEX")].x * width,
                                lm[getattr(mp_pose.PoseLandmark, f"{side_prefix}_FOOT_INDEX")].y * height,
                            ]

                            if analizza_ginocchio:
                                ang = calcola_angolo(anca, ginocchio, caviglia)
                                ang_val = round(ang, 1)
                                row_data["ginocchio_deg"] = ang_val
                                cv2.putText(
                                    image,
                                    f"Gino: {ang_val} deg",
                                    (int(ginocchio[0]) + 15, int(ginocchio[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (255, 255, 0),
                                    2,
                                    cv2.LINE_AA,
                                )
                                has_data = True

                            if analizza_anca:
                                ang = calcola_angolo(spalla, anca, ginocchio)
                                ang_val = round(ang, 1)
                                row_data["anca_deg"] = ang_val
                                cv2.putText(
                                    image,
                                    f"Anca: {ang_val} deg",
                                    (int(anca[0]) + 15, int(anca[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (255, 255, 0),
                                    2,
                                    cv2.LINE_AA,
                                )
                                has_data = True

                            if analizza_caviglia:
                                ang = calcola_angolo(ginocchio, caviglia, punta_piede)
                                ang_val = round(ang, 1)
                                row_data["caviglia_deg"] = ang_val
                                cv2.putText(
                                    image,
                                    f"Cav.: {ang_val} deg",
                                    (int(caviglia[0]) + 15, int(caviglia[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (255, 255, 0),
                                    2,
                                    cv2.LINE_AA,
                                )
                                has_data = True

                            if analizza_spalla:
                                ang = calcola_angolo(anca, spalla, gomito)
                                ang_val = round(ang, 1)
                                row_data["spalla_deg"] = ang_val
                                cv2.putText(
                                    image,
                                    f"Spalla: {ang_val} deg",
                                    (int(spalla[0]) + 15, int(spalla[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (255, 255, 0),
                                    2,
                                    cv2.LINE_AA,
                                )
                                has_data = True

                            if analizza_gomito:
                                ang = calcola_angolo(spalla, gomito, polso)
                                ang_val = round(ang, 1)
                                row_data["gomito_deg"] = ang_val
                                cv2.putText(
                                    image,
                                    f"Gomito: {ang_val} deg",
                                    (int(gomito[0]) + 15, int(gomito[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (255, 255, 0),
                                    2,
                                    cv2.LINE_AA,
                                )
                                has_data = True

                        else:
                            # PIANO FRONTALE
                            anca_sx = [
                                lm[mp_pose.PoseLandmark.LEFT_HIP].x * width,
                                lm[mp_pose.PoseLandmark.LEFT_HIP].y * height,
                            ]
                            anca_dx = [
                                lm[mp_pose.PoseLandmark.RIGHT_HIP].x * width,
                                lm[mp_pose.PoseLandmark.RIGHT_HIP].y * height,
                            ]
                            ginocchio_sx = [
                                lm[mp_pose.PoseLandmark.LEFT_KNEE].x * width,
                                lm[mp_pose.PoseLandmark.LEFT_KNEE].y * height,
                            ]
                            ginocchio_dx = [
                                lm[mp_pose.PoseLandmark.RIGHT_KNEE].x * width,
                                lm[mp_pose.PoseLandmark.RIGHT_KNEE].y * height,
                            ]
                            caviglia_sx = [
                                lm[mp_pose.PoseLandmark.LEFT_ANKLE].x * width,
                                lm[mp_pose.PoseLandmark.LEFT_ANKLE].y * height,
                            ]
                            caviglia_dx = [
                                lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x * width,
                                lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y * height,
                            ]

                            dist_ginocchia = abs(ginocchio_sx[0] - ginocchio_dx[0])
                            dist_caviglie = abs(caviglia_sx[0] - caviglia_dx[0])

                            if analizza_kar:
                                kar = (
                                    dist_ginocchia / dist_caviglie
                                    if dist_caviglie > 0
                                    else 1.0
                                )
                                kar_val = round(kar, 2)
                                row_data["KAR_rapporto"] = kar_val

                                if kar_val < 0.85:
                                    stato_allineamento = "VALGO (Fuori Soglia)"
                                    colore_testo = (0, 0, 255)
                                elif kar_val > 1.15:
                                    stato_allineamento = "VARO (Fuori Soglia)"
                                    colore_testo = (255, 0, 0)
                                else:
                                    stato_allineamento = "EQUILIBRATO (OK)"
                                    colore_testo = (0, 255, 0)

                                cv2.putText(
                                    image,
                                    f"KAR: {kar_val:.2f} [{stato_allineamento}]",
                                    (
                                        int(ginocchio_sx[0]) + 15,
                                        int(ginocchio_sx[1]),
                                    ),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    colore_testo,
                                    2,
                                    cv2.LINE_AA,
                                )
                                has_data = True

                            if analizza_bacino:
                                diff_bacino = abs(anca_sx[1] - anca_dx[1])
                                bacino_val = round(diff_bacino, 1)
                                row_data["inclinazione_bacino_px"] = bacino_val

                                stato_bacino = (
                                    "STABILE (OK)"
                                    if bacino_val < 15
                                    else "COMPENSO"
                                )
                                colore_bacino = (
                                    (0, 255, 0)
                                    if bacino_val < 15
                                    else (0, 165, 255)
                                )

                                centro_bacino_x = int((anca_sx[0] + anca_dx[0]) / 2)
                                centro_bacino_y = int((anca_sx[1] + anca_dx[1]) / 2)

                                cv2.putText(
                                    image,
                                    f"Bacino: {bacino_val}px [{stato_bacino}]",
                                    (
                                        centro_bacino_x - 65,
                                        centro_bacino_y - 12,
                                    ),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    colore_bacino,
                                    2,
                                    cv2.LINE_AA,
                                )
                                has_data = True

                        cv2.putText(
                            image,
                            f"Tempo: {tempo_secondi:.2f}s",
                            (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                    if has_data:
                        pose_data.append(row_data)

                    out.write(image)
                    frame_idx += 1

                cap.release()
                out.release()

            st.session_state["output_path"] = output_path
            st.session_state["pose_data"] = pose_data
            st.session_state["analizzato"] = True

    # Rendering del video elaborato e delle metriche
    if st.session_state.get("analizzato", False):
        with col2:
            st.subheader("✨ Video Elaborato")
            st.video(st.session_state["output_path"], format="video/mp4")

        df_results = pd.DataFrame(st.session_state["pose_data"])
        cols_to_smooth = [c for c in df_results.columns if c != "tempo_s"]

        st.markdown("---")
        st.subheader("📊 Indici e Picchi di Movimento")

        metric_cols = st.columns(len(cols_to_smooth))
        for i, col in enumerate(cols_to_smooth):
            val_min = df_results[col].min()
            val_max = df_results[col].max()
            t_min = df_results.loc[df_results[col].idxmin(), "tempo_s"]
            with metric_cols[i]:
                st.metric(
                    label=f"{col.replace('_deg','').replace('_px','').replace('_rapporto','').upper()}",
                    value=f"{val_min}",
                    delta=f"Max: {val_max} (@ {t_min}s)",
                    delta_color="off",
                )

        st.markdown("---")
        st.subheader("📈 Grafico Dinamico")
        linee_selezionate = st.multiselect(
            "Seleziona i parametri da mostrare:",
            options=cols_to_smooth,
            default=cols_to_smooth,
        )

        if linee_selezionate:
            st.line_chart(df_results.set_index("tempo_s")[linee_selezionate])

        st.markdown("---")
        st.subheader("📋 Tabella Dati Cinematici Temporali")
        st.dataframe(df_results, height=200, use_container_width=True)

        # Generazione CSV ottimizzato per Excel Italiano (separatore ';' e virgola nei decimali)
        csv_data = df_results.to_csv(index=False, sep=";", decimal=",").encode("utf-8")
        st.download_button(
            label="📥 Scarica Report Dati in CSV (Excel)",
            data=csv_data,
            file_name="report_biomeccanico_tesi.csv",
            mime="text/csv",
        )

# Footer con Instagram Link e Crediti
st.markdown(
    """
    <div class="custom-footer">
        BioIgTech • Progetto di Tesi in Scienze Motorie | Sviluppato da <strong>Igor Aceto</strong><br>
        📸 Instagram: <a class="insta-link" href="https://www.instagram.com/igor_ace28/" target="_blank">@igor_ace28</a>
    </div>
""",
    unsafe_allow_html=True,
)
