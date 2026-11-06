import base64
import hashlib
import os
import shutil
import subprocess
import tempfile

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# CONFIGURAZIONE E FUNZIONI DI SUPPORTO
# ============================================================

st.set_page_config(
    page_title="BioIgTech - Biomechanics Data Motion",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="auto",
)


def crea_file_temporaneo(suffisso):
    fd, percorso = tempfile.mkstemp(suffix=suffisso)
    os.close(fd)
    return percorso


def elimina_file(percorso):
    if percorso and os.path.exists(percorso):
        try:
            os.remove(percorso)
        except OSError:
            pass


def converti_in_mp4_browser(input_path, output_path):
    """Converte il video in MP4 H.264 / AAC adatto alla riproduzione web."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg non è installato sul server. "
            "Verifica che packages.txt contenga la riga ffmpeg."
        )

    comando = [
        ffmpeg, "-y", "-i", input_path,
        "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "128k",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        output_path,
    ]
    risultato = subprocess.run(
        comando, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if risultato.returncode != 0 or not os.path.exists(output_path):
        dettaglio = risultato.stderr[-1500:] if risultato.stderr else ""
        raise RuntimeError(f"Conversione video non riuscita. {dettaglio}")



def mostra_video_browser(video_bytes, altezza=420):
    """Mostra il video con il player HTML nativo, evitando l'endpoint media di Streamlit."""
    video_base64 = base64.b64encode(video_bytes).decode("ascii")
    html = f"""
    <video controls playsinline preload="metadata"
           style="width:100%; max-height:{altezza}px; background:#111; border-radius:12px;">
      <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
      Il browser non riesce a riprodurre questo video.
    </video>
    """
    components.html(html, height=altezza + 20, scrolling=False)


def calcola_angolo(a, b, c):
    """Calcola l'angolo in gradi con vertice nel punto B."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    coseno = np.dot(v1, v2) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(coseno, -1.0, 1.0))))


# Stato della sessione
for chiave, valore in {
    "analizzato": False,
    "pose_data": [],
    "output_path": None,
    "video_hash": None,
    "browser_video_path": None,
}.items():
    if chiave not in st.session_state:
        st.session_state[chiave] = valore


# ============================================================
# DESIGN CSS PERSONALIZZATO
# ============================================================
def crea_file_temporaneo(suffisso):
    """Crea un percorso temporaneo senza lasciare aperto il file."""
    fd, percorso = tempfile.mkstemp(suffix=suffisso)
    os.close(fd)
    return percorso


# Inizializzazione dello stato della sessione
if "analizzato" not in st.session_state:
    st.session_state["analizzato"] = False

if "pose_data" not in st.session_state:
    st.session_state["pose_data"] = []

if "output_path" not in st.session_state:
    st.session_state["output_path"] = None

if "video_hash" not in st.session_state:
    st.session_state["video_hash"] = None


# ============================================================
# DESIGN CSS PERSONALIZZATO
# ============================================================

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


# ============================================================
# INIZIALIZZAZIONE MEDIAPIPE
# ============================================================

try:
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

except AttributeError:
    import mediapipe.python.solutions as mp_solutions

    mp_drawing = mp_solutions.drawing_utils
    mp_pose = mp_solutions.pose


# ============================================================
# FUNZIONE CALCOLO ANGOLO
# ============================================================

def calcola_angolo(a, b, c):
    """
    Calcola l'angolo in gradi con vertice nel punto B.
    I punti sono coordinate bidimensionali.
    """

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

    angle = np.degrees(
        np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    )

    return float(angle)


# ============================================================
# LOGO E HEADER
# ============================================================

logo_base64 = ""

for logo_name in ["logo.jpg", "logo.png"]:
    if os.path.exists(logo_name):
        with open(logo_name, "rb") as file:
            logo_base64 = base64.b64encode(
                file.read()
            ).decode("utf-8")
        break

if logo_base64:
    st.markdown(
        f"""
        <div class="header-container">
            <img src="data:image/jpeg;base64,{logo_base64}"
                 class="logo-img">
            <div class="sub-title">
                Piattaforma professionale di Video Analisi
                Biomeccanica 2D
                (Piano Sagittale & Frontale)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="header-container">
            <h1>⚡ BioIgTech</h1>
            <div class="sub-title">
                Piattaforma professionale di Video Analisi
                Biomeccanica 2D
                (Piano Sagittale & Frontale)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# GUIDA RAPIDA
# ============================================================

with st.expander("ℹ️ Guida Rapida ai Piani di Analisi e Soglie"):

    st.write(
        """
        - **Piano Sagittale (Profilo):**
          Valuta i Range of Motion (ROM) e gli angoli articolari
          di ginocchio, anca e caviglia.

        - **Piano Frontale:**
          - **KAR (Knee-to-Ankle Ratio):**
            Rapporto tra la distanza orizzontale delle ginocchia
            e quella delle caviglie.
            - **0.85 – 1.15:** NEUTRO
            - **< 0.85:** VALGO DINAMICO
            - **> 1.15:** VARO DINAMICO

          - **Inclinazione Bacino:**
            Differenza verticale tra le due anche espressa in pixel.
            - **< 15 px:** STABILE
            - **≥ 15 px:** POSSIBILE COMPENSO

        **Nota:** gli indicatori sono misure video bidimensionali
        e dipendono anche da inquadratura, prospettiva e qualità
        del rilevamento.
        """
    )


# ============================================================
# SIDEBAR - PANNELLO DI CONTROLLO
# ============================================================

st.sidebar.header("🎛️ Pannello di Controllo")

piano_analisi = st.sidebar.selectbox(
    "📐 Piano di Analisi",
    [
        "Piano Sagittale (Profilo: ROM & Cinematica)",
        "Piano Frontale (Valgismo/Varismo Dinamico & Simmetrie)",
    ],
)

uploaded_file = st.sidebar.file_uploader(
    "Carica video (MP4, MOV, AVI)",
    type=["mp4", "mov", "avi"],
)

slow_factor = st.sidebar.slider(
    "Fattore Slow Motion",
    min_value=1.0,
    max_value=3.0,
    value=2.0,
    step=0.5,
)

st.sidebar.subheader("🎯 Parametri di Tracciamento")


if "Sagittale" in piano_analisi:

    analizza_ginocchio = st.sidebar.checkbox(
        "Cinematica Ginocchio",
        value=True,
    )

    analizza_anca = st.sidebar.checkbox(
        "Cinematica Anca",
        value=True,
    )

    analizza_caviglia = st.sidebar.checkbox(
        "Cinematica Caviglia",
        value=True,
    )

    analizza_spalla = st.sidebar.checkbox(
        "Cinematica Spalla",
        value=False,
    )

    analizza_gomito = st.sidebar.checkbox(
        "Cinematica Gomito",
        value=False,
    )

else:

    analizza_kar = st.sidebar.checkbox(
        "Indice di Allineamento (KAR - Valgo/Varo/Neutro)",
        value=True,
    )

    analizza_bacino = st.sidebar.checkbox(
        "Stabilità Bacino (Soglia < 15px)",
        value=True,
    )

    with st.sidebar.expander("📖 Spiegazione Parametri Frontali"):

        st.markdown(
            """
            **KAR (Knee-to-Ankle Ratio)**

            Rapporto tra la distanza orizzontale delle ginocchia
            e quella delle caviglie.

            - **0.85 – 1.15:** Neutro
            - **< 0.85:** Valgismo dinamico
            - **> 1.15:** Varismo dinamico

            **Stabilità Bacino**

            Differenza di altezza in pixel tra l'anca destra
            e quella sinistra.
            """
        )


# ============================================================
# GESTIONE CARICAMENTO VIDEO
# ============================================================

if uploaded_file is None:

    # Se non c'è un video caricato, eliminiamo i risultati precedenti.
    elimina_file(st.session_state.get("output_path"))
    elimina_file(st.session_state.get("browser_video_path"))

    st.session_state["browser_video_path"] = None
    st.session_state["output_path"] = None
    st.session_state["pose_data"] = []
    st.session_state["analizzato"] = False
    st.session_state["video_hash"] = None

    st.info("Carica un video dalla barra laterale per iniziare l'analisi.")


else:

    video_bytes = uploaded_file.getvalue()

    video_hash = hashlib.md5(video_bytes).hexdigest()

    # Se viene caricato un video diverso, azzeriamo i risultati precedenti.
    if video_hash != st.session_state["video_hash"]:

        elimina_file(st.session_state.get("output_path"))
        elimina_file(st.session_state.get("browser_video_path"))
        st.session_state["browser_video_path"] = None

        st.session_state["output_path"] = None
        st.session_state["pose_data"] = []
        st.session_state["analizzato"] = False
        st.session_state["video_hash"] = video_hash

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📹 Video Originale")
        try:
            percorso_browser = st.session_state.get("browser_video_path")
            if not percorso_browser or not os.path.exists(percorso_browser):
                percorso_browser = crea_file_temporaneo(".mp4")
                sorgente_temporanea = crea_file_temporaneo(
                    os.path.splitext(uploaded_file.name)[1].lower() or ".mp4"
                )
                try:
                    with open(sorgente_temporanea, "wb") as file:
                        file.write(video_bytes)
                    converti_in_mp4_browser(sorgente_temporanea, percorso_browser)
                    st.session_state["browser_video_path"] = percorso_browser
                    percorso_browser = None
                finally:
                    elimina_file(sorgente_temporanea)
                    elimina_file(percorso_browser)

            with open(st.session_state["browser_video_path"], "rb") as file:
                video_originale_browser = file.read()
            mostra_video_browser(video_originale_browser)
        except Exception as errore_video:
            st.warning(
                "Non è stato possibile preparare il video per la riproduzione: "
                f"{errore_video}"
            )

    avvia_analisi = st.sidebar.button(
        "🚀 Avvia Analisi Biomeccanica",
        use_container_width=True,
    )


    # ========================================================
    # ELABORAZIONE VIDEO
    # ========================================================

    if avvia_analisi:

        input_path = None
        output_path = None
        intermediate_path = None
        cap = None
        out = None

        try:

            with st.spinner(
                "Elaborazione in corso... "
                "Calcolo indici cinematici e soglie."
            ):

                # Salvataggio temporaneo del video originale.
                estensione = os.path.splitext(
                    uploaded_file.name
                )[1].lower()

                if estensione not in [".mp4", ".mov", ".avi"]:
                    estensione = ".mp4"

                input_path = crea_file_temporaneo(estensione)

                with open(input_path, "wb") as file:
                    file.write(video_bytes)

                # Apertura video.
                cap = cv2.VideoCapture(input_path)

                if not cap.isOpened():
                    raise RuntimeError(
                        "Impossibile aprire il video. "
                        "Prova con un file MP4 o MOV differente."
                    )

                width = int(
                    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                )

                height = int(
                    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                )

                orig_fps = cap.get(cv2.CAP_PROP_FPS)

                if not np.isfinite(orig_fps) or orig_fps <= 0:
                    orig_fps = 30.0

                if width <= 0 or height <= 0:
                    raise RuntimeError(
                        "Le dimensioni del video non sono valide."
                    )

                # Calcolo FPS del video rallentato.
                new_fps = max(
                    5.0,
                    float(orig_fps) / float(slow_factor),
                )

                # OpenCV scrive un AVI intermedio; FFmpeg lo converte poi
                # in MP4 H.264, più compatibile con i browser.
                intermediate_path = crea_file_temporaneo(".avi")
                output_path = crea_file_temporaneo(".mp4")

                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                out = cv2.VideoWriter(
                    intermediate_path,
                    fourcc,
                    new_fps,
                    (width, height),
                )

                if not out.isOpened():
                    raise RuntimeError(
                        "Non è stato possibile creare il video intermedio."
                    )

                pose_data = []
                frame_idx = 0

                joint_spec = mp_drawing.DrawingSpec(
                    color=(255, 0, 200),
                    thickness=6,
                    circle_radius=5,
                )

                bone_spec = mp_drawing.DrawingSpec(
                    color=(186, 85, 211),
                    thickness=4,
                )

                # Inizializzazione MediaPipe Pose.
                with mp_pose.Pose(
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.6,
                ) as pose:

                    while cap.isOpened():

                        success, image = cap.read()

                        if not success:
                            break

                        tempo_secondi = frame_idx / new_fps

                        image_rgb = cv2.cvtColor(
                            image,
                            cv2.COLOR_BGR2RGB,
                        )

                        results = pose.process(image_rgb)

                        row_data = {
                            "tempo_s": round(
                                tempo_secondi,
                                2,
                            )
                        }

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

                            # ====================================
                            # PIANO SAGITTALE
                            # ====================================

                            if "Sagittale" in piano_analisi:

                                vis_left = (
                                    lm[
                                        mp_pose.PoseLandmark.LEFT_HIP
                                    ].visibility
                                    + lm[
                                        mp_pose.PoseLandmark.LEFT_KNEE
                                    ].visibility
                                ) / 2

                                vis_right = (
                                    lm[
                                        mp_pose.PoseLandmark.RIGHT_HIP
                                    ].visibility
                                    + lm[
                                        mp_pose.PoseLandmark.RIGHT_KNEE
                                    ].visibility
                                ) / 2

                                side_prefix = (
                                    "LEFT"
                                    if vis_left >= vis_right
                                    else "RIGHT"
                                )

                                spalla = [
                                    lm[
                                        getattr(
                                            mp_pose.PoseLandmark,
                                            f"{side_prefix}_SHOULDER",
                                        )
                                    ].x * width,

                                    lm[
                                        getattr(
                                            mp_pose.PoseLandmark,
                                            f"{side_prefix}_SHOULDER",
                                        )
                                    ].y * height,
                                ]

                                gomito = [
                                    lm[
                                        getattr(
                                            mp_pose.PoseLandmark,
                                            f"{side_prefix}_ELBOW",
                                        )
                                    ].x * width,

                                    lm[
                                        getattr(
                                            mp_pose.PoseLandmark,
                                            f"{side_prefix}_ELBOW",
                                        )
                                    ].y * height,
                                ]

                                polso = [
                                    lm[
                                        getattr(
                                            mp_pose.PoseLandmark,
                                            f"{side_prefix}_WRIST",
                                        )
                                    ].x * width,

                                    lm[
                                        getattr(
                                            mp_pose.PoseLandmark,
                                            f"{side_prefix}_WRIST",
                                        )
                                    ].y * height,
                                ]

                                anca = [
                                    lm[
                                        getattr(
                                            mp_pose.PoseLandmark,
                                            f"{side_prefix}_HIP",
                                        )
                                    ].x * width,

                                    lm[
                                        getattr(
                                            mp_pose.PoseLandmark,
                                            f"{side_prefix}_HIP",
                                        )
                                    ].y * height,
                                ]

                                ginocchio = [
                                    lm[
                                        getattr(
                                            mp_pose.PoseLandmark,
                                            f"{side_prefix}_KNEE",
                                        )
                                    ].x * width,

                                    lm[
                                        getattr(
                                            mp_pose.PoseLandmark,
                                            f"{side_prefix}_KNEE",
                                        )
                                    ].y * height,
                                ]

                                caviglia = [
                                    lm[
                                        getattr(
                                            mp_pose.PoseLandmark,
                                            f"{side_prefix}_ANKLE",
                                        )
                                    ].x * width,

                                    lm[
                                        getattr(
                                            mp_pose.PoseLandmark,
                                            f"{side_prefix}_ANKLE",
                                        )
                                    ].y * height,
                                ]

                                punta_piede = [
                                    lm[
                                        getattr(
                                            mp_pose.PoseLandmark,
                                            f"{side_prefix}_FOOT_INDEX",
                                        )
                                    ].x * width,

                                    lm[
                                        getattr(
                                            mp_pose.PoseLandmark,
                                            f"{side_prefix}_FOOT_INDEX",
                                        )
                                    ].y * height,
                                ]

                                # Angolo ginocchio.
                                if analizza_ginocchio:

                                    ang = calcola_angolo(
                                        anca,
                                        ginocchio,
                                        caviglia,
                                    )

                                    ang_val = round(ang, 1)

                                    row_data[
                                        "ginocchio_deg"
                                    ] = ang_val

                                    cv2.putText(
                                        image,
                                        f"Gino: {ang_val} deg",
                                        (
                                            int(ginocchio[0]) + 15,
                                            int(ginocchio[1]),
                                        ),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.55,
                                        (255, 255, 0),
                                        2,
                                        cv2.LINE_AA,
                                    )

                                    has_data = True

                                # Angolo anca.
                                if analizza_anca:

                                    ang = calcola_angolo(
                                        spalla,
                                        anca,
                                        ginocchio,
                                    )

                                    ang_val = round(ang, 1)

                                    row_data[
                                        "anca_deg"
                                    ] = ang_val

                                    cv2.putText(
                                        image,
                                        f"Anca: {ang_val} deg",
                                        (
                                            int(anca[0]) + 15,
                                            int(anca[1]),
                                        ),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.55,
                                        (255, 255, 0),
                                        2,
                                        cv2.LINE_AA,
                                    )

                                    has_data = True

                                # Angolo caviglia.
                                if analizza_caviglia:

                                    ang = calcola_angolo(
                                        ginocchio,
                                        caviglia,
                                        punta_piede,
                                    )

                                    ang_val = round(ang, 1)

                                    row_data[
                                        "caviglia_deg"
                                    ] = ang_val

                                    cv2.putText(
                                        image,
                                        f"Cav.: {ang_val} deg",
                                        (
                                            int(caviglia[0]) + 15,
                                            int(caviglia[1]),
                                        ),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.55,
                                        (255, 255, 0),
                                        2,
                                        cv2.LINE_AA,
                                    )

                                    has_data = True

                                # Angolo spalla.
                                if analizza_spalla:

                                    ang = calcola_angolo(
                                        anca,
                                        spalla,
                                        gomito,
                                    )

                                    ang_val = round(ang, 1)

                                    row_data[
                                        "spalla_deg"
                                    ] = ang_val

                                    cv2.putText(
                                        image,
                                        f"Spalla: {ang_val} deg",
                                        (
                                            int(spalla[0]) + 15,
                                            int(spalla[1]),
                                        ),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.55,
                                        (255, 255, 0),
                                        2,
                                        cv2.LINE_AA,
                                    )

                                    has_data = True

                                # Angolo gomito.
                                if analizza_gomito:

                                    ang = calcola_angolo(
                                        spalla,
                                        gomito,
                                        polso,
                                    )

                                    ang_val = round(ang, 1)

                                    row_data[
                                        "gomito_deg"
                                    ] = ang_val

                                    cv2.putText(
                                        image,
                                        f"Gomito: {ang_val} deg",
                                        (
                                            int(gomito[0]) + 15,
                                            int(gomito[1]),
                                        ),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.55,
                                        (255, 255, 0),
                                        2,
                                        cv2.LINE_AA,
                                    )

                                    has_data = True

                            # ====================================
                            # PIANO FRONTALE
                            # ====================================

                            else:

                                anca_sx = [
                                    lm[
                                        mp_pose.PoseLandmark.LEFT_HIP
                                    ].x * width,

                                    lm[
                                        mp_pose.PoseLandmark.LEFT_HIP
                                    ].y * height,
                                ]

                                anca_dx = [
                                    lm[
                                        mp_pose.PoseLandmark.RIGHT_HIP
                                    ].x * width,

                                    lm[
                                        mp_pose.PoseLandmark.RIGHT_HIP
                                    ].y * height,
                                ]

                                ginocchio_sx = [
                                    lm[
                                        mp_pose.PoseLandmark.LEFT_KNEE
                                    ].x * width,

                                    lm[
                                        mp_pose.PoseLandmark.LEFT_KNEE
                                    ].y * height,
                                ]

                                ginocchio_dx = [
                                    lm[
                                        mp_pose.PoseLandmark.RIGHT_KNEE
                                    ].x * width,

                                    lm[
                                        mp_pose.PoseLandmark.RIGHT_KNEE
                                    ].y * height,
                                ]

                                caviglia_sx = [
                                    lm[
                                        mp_pose.PoseLandmark.LEFT_ANKLE
                                    ].x * width,

                                    lm[
                                        mp_pose.PoseLandmark.LEFT_ANKLE
                                    ].y * height,
                                ]

                                caviglia_dx = [
                                    lm[
                                        mp_pose.PoseLandmark.RIGHT_ANKLE
                                    ].x * width,

                                    lm[
                                        mp_pose.PoseLandmark.RIGHT_ANKLE
                                    ].y * height,
                                ]

                                dist_ginocchia = abs(
                                    ginocchio_sx[0]
                                    - ginocchio_dx[0]
                                )

                                dist_caviglie = abs(
                                    caviglia_sx[0]
                                    - caviglia_dx[0]
                                )

                                # KAR.
                                if analizza_kar:

                                    kar = (
                                        dist_ginocchia / dist_caviglie
                                        if dist_caviglie > 0
                                        else 1.0
                                    )

                                    kar_val = round(kar, 2)

                                    row_data[
                                        "KAR_rapporto"
                                    ] = kar_val

                                    if kar_val < 0.85:

                                        stato_allineamento = (
                                            "VALGO (Fuori Soglia)"
                                        )

                                        colore_testo = (0, 0, 255)

                                    elif kar_val > 1.15:

                                        stato_allineamento = (
                                            "VARO (Fuori Soglia)"
                                        )

                                        colore_testo = (255, 0, 0)

                                    else:

                                        stato_allineamento = (
                                            "EQUILIBRATO (OK)"
                                        )

                                        colore_testo = (0, 255, 0)

                                    cv2.putText(
                                        image,
                                        (
                                            f"KAR: {kar_val:.2f} "
                                            f"[{stato_allineamento}]"
                                        ),
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

                                # Inclinazione bacino.
                                if analizza_bacino:

                                    diff_bacino = abs(
                                        anca_sx[1] - anca_dx[1]
                                    )

                                    bacino_val = round(
                                        diff_bacino,
                                        1,
                                    )

                                    row_data[
                                        "inclinazione_bacino_px"
                                    ] = bacino_val

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

                                    centro_bacino_x = int(
                                        (
                                            anca_sx[0]
                                            + anca_dx[0]
                                        ) / 2
                                    )

                                    centro_bacino_y = int(
                                        (
                                            anca_sx[1]
                                            + anca_dx[1]
                                        ) / 2
                                    )

                                    cv2.putText(
                                        image,
                                        (
                                            f"Bacino: {bacino_val}px "
                                            f"[{stato_bacino}]"
                                        ),
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

                            # Tempo sul video.
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

                        # Salvataggio dati del frame.
                        if has_data:
                            pose_data.append(row_data)

                        # Scrittura frame elaborato.
                        out.write(image)

                        frame_idx += 1

                # Chiusura video.
                cap.release()
                cap = None

                out.release()
                out = None

                if frame_idx == 0:
                    raise RuntimeError(
                        "Il video non contiene frame leggibili."
                    )

                if (
                    not intermediate_path
                    or not os.path.exists(intermediate_path)
                    or os.path.getsize(intermediate_path) == 0
                ):
                    raise RuntimeError(
                        "Il video intermedio elaborato non è stato creato correttamente."
                    )

                converti_in_mp4_browser(intermediate_path, output_path)
                elimina_file(intermediate_path)
                intermediate_path = None

                if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                    raise RuntimeError(
                        "La conversione del video elaborato in MP4 non è riuscita."
                    )

                # Salvataggio risultati nella sessione.
                elimina_file(
                    st.session_state.get("output_path")
                )

                st.session_state["output_path"] = output_path
                st.session_state["pose_data"] = pose_data
                st.session_state["analizzato"] = True

                output_path = None

        except Exception as errore:

            st.error(
                f"Si è verificato un errore durante l'analisi: {errore}"
            )

            st.session_state["analizzato"] = False
            st.session_state["pose_data"] = []

        finally:

            if cap is not None:
                cap.release()

            if out is not None:
                out.release()

            elimina_file(input_path)
            elimina_file(intermediate_path)
            elimina_file(output_path)


    # ========================================================
    # VIDEO ELABORATO E RISULTATI
    # ========================================================

    if st.session_state.get("analizzato", False):

        percorso_output = st.session_state.get("output_path")

        if (
            percorso_output
            and os.path.exists(percorso_output)
            and os.path.getsize(percorso_output) > 0
        ):

            with col2:

                st.subheader("✨ Video Elaborato")

                with open(percorso_output, "rb") as file:
                    video_elaborato = file.read()

                mostra_video_browser(video_elaborato)

        else:

            st.error(
                "Il file del video elaborato non è disponibile."
            )

        df_results = pd.DataFrame(
            st.session_state["pose_data"]
        )

        st.markdown("---")
        st.subheader("📊 Indici e Picchi di Movimento")

        if df_results.empty:

            st.warning(
                "Nessun dato biomeccanico rilevato. "
                "Verifica che la persona sia visibile nel video "
                "e che i parametri di analisi siano selezionati."
            )

        else:

            cols_to_smooth = [
                col
                for col in df_results.columns
                if col != "tempo_s"
            ]

            if not cols_to_smooth:

                st.warning(
                    "Non è stato selezionato nessun parametro "
                    "da analizzare."
                )

            else:

                metric_cols = st.columns(
                    len(cols_to_smooth)
                )

                for i, col in enumerate(cols_to_smooth):

                    val_min = df_results[col].min()
                    val_max = df_results[col].max()

                    t_min = df_results.loc[
                        df_results[col].idxmin(),
                        "tempo_s",
                    ]

                    t_max = df_results.loc[
                        df_results[col].idxmax(),
                        "tempo_s",
                    ]

                    with metric_cols[i]:

                        st.metric(
                            label=(
                                col.replace("_deg", "")
                                .replace("_px", "")
                                .replace("_rapporto", "")
                                .upper()
                            ),
                            value=f"{val_min}",
                            delta=(
                                f"Max: {val_max} "
                                f"(@ {t_max}s)"
                            ),
                            delta_color="off",
                        )

                        st.caption(
                            f"Minimo rilevato a {t_min}s"
                        )

                # ============================================
                # GRAFICO DINAMICO
                # ============================================

                st.markdown("---")
                st.subheader("📈 Grafico Dinamico")

                linee_selezionate = st.multiselect(
                    "Seleziona i parametri da mostrare:",
                    options=cols_to_smooth,
                    default=cols_to_smooth,
                )

                if linee_selezionate:

                    st.line_chart(
                        df_results.set_index("tempo_s")[
                            linee_selezionate
                        ]
                    )

                else:

                    st.info(
                        "Seleziona almeno un parametro "
                        "per visualizzare il grafico."
                    )

                # ============================================
                # TABELLA DATI
                # ============================================

                st.markdown("---")
                st.subheader(
                    "📋 Tabella Dati Cinematici Temporali"
                )

                st.dataframe(
                    df_results,
                    height=200,
                    use_container_width=True,
                )

                # ============================================
                # ESPORTAZIONE CSV
                # ============================================

                csv_data = df_results.to_csv(
                    index=False,
                    sep=";",
                    decimal=",",
                ).encode("utf-8-sig")

                st.download_button(
                    label="📥 Scarica Report Dati in CSV (Excel)",
                    data=csv_data,
                    file_name="report_biomeccanico_tesi.csv",
                    mime="text/csv",
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="custom-footer">
        BioIgTech • Progetto di Tesi in Scienze Motorie |
        Sviluppato da <strong>Igor Aceto</strong><br>
        📸 Instagram:
        <a class="insta-link"
           href="https://www.instagram.com/igor_ace28/"
           target="_blank">
           @igor_ace28
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)
