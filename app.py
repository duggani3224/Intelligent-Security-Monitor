import streamlit as st
import cv2
import tempfile
import os
import subprocess
import imageio_ffmpeg
from ultralytics import YOLO


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Intelligent Security Monitor",
    page_icon="🎥",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🎥 Intelligent Security Monitor")

st.write(
    "Upload a video to detect and track people using YOLOv8."
)


# =========================================================
# LOAD YOLO MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Settings")

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)


# =========================================================
# VIDEO UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📁 Upload a video",
    type=["mp4", "avi", "mov", "mkv"]
)


# =========================================================
# START ANALYSIS
# =========================================================

if uploaded_file is not None:

    st.success("Video uploaded successfully!")

    if st.button("▶️ Start Analysis"):

        # -------------------------------------------------
        # Save uploaded video
        # -------------------------------------------------

        input_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_file.write(uploaded_file.read())
        input_file.close()

        input_path = input_file.name


        # -------------------------------------------------
        # Open video
        # -------------------------------------------------

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():

            st.error("❌ Could not open the uploaded video.")

            os.remove(input_path)

            st.stop()


        # -------------------------------------------------
        # Video information
        # -------------------------------------------------

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )


        # -------------------------------------------------
        # Create temporary AVI
        # -------------------------------------------------

        temp_avi = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".avi"
        )

        avi_path = temp_avi.name
        temp_avi.close()


        fourcc = cv2.VideoWriter_fourcc(
            *"MJPG"
        )

        out = cv2.VideoWriter(
            avi_path,
            fourcc,
            fps,
            (width, height)
        )


        # -------------------------------------------------
        # Progress
        # -------------------------------------------------

        progress = st.progress(0)

        status = st.empty()


        # -------------------------------------------------
        # Analytics
        # -------------------------------------------------

        frame_number = 0
        total_people = 0
        max_people = 0


        # =================================================
        # PROCESS VIDEO
        # =================================================

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1


            # YOLO detection and tracking

            results = model.track(
                frame,
                persist=True,
                classes=[0],
                conf=confidence,
                imgsz=640,
                verbose=False
            )


            # Draw boxes and tracking IDs

            annotated_frame = results[0].plot()


            # Keep original dimensions

            if (
                annotated_frame.shape[1] != width
                or annotated_frame.shape[0] != height
            ):

                annotated_frame = cv2.resize(
                    annotated_frame,
                    (width, height),
                    interpolation=cv2.INTER_LANCZOS4
                )


            # Count people

            boxes = results[0].boxes

            if boxes is not None:
                people_count = len(boxes)
            else:
                people_count = 0


            total_people += people_count

            if people_count > max_people:
                max_people = people_count


            # Save frame

            out.write(annotated_frame)


            # Progress

            if total_frames > 0:

                percentage = int(
                    (frame_number / total_frames) * 100
                )

                progress.progress(
                    min(percentage, 100)
                )


            status.write(
                f"Processing frame {frame_number} "
                f"of {total_frames}"
            )


        # =================================================
        # CLOSE VIDEO
        # =================================================

        cap.release()
        out.release()


        progress.progress(100)

        status.write(
            "Processing completed!"
        )


        # =================================================
        # CONVERT VIDEO
        # =================================================

        st.info(
            "Creating browser-compatible processed video..."
        )


        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        output_path = output_file.name
        output_file.close()


        # Get FFmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()


        # -------------------------------------------------
        # FFmpeg command
        # -------------------------------------------------

        ffmpeg_command = [
            ffmpeg_exe,

            "-y",

            "-i",
            avi_path,

            "-vf",
            "scale=1080:1920",

            "-c:v",
            "libx264",

            "-preset",
            "ultrafast",

            "-crf",
            "20",

            "-pix_fmt",
            "yuv420p",

            "-threads",
            "2",

            "-movflags",
            "+faststart",

            output_path
        ]


        # Run FFmpeg

        conversion = subprocess.run(
            ffmpeg_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )


        # =================================================
        # CHECK CONVERSION
        # =================================================

        if conversion.returncode != 0:

            st.error(
                "❌ Video conversion failed."
            )

            st.code(
                conversion.stderr
            )

            if os.path.exists(input_path):
                os.remove(input_path)

            if os.path.exists(avi_path):
                os.remove(avi_path)

            st.stop()


        # =================================================
        # SUCCESS
        # =================================================

        st.success(
            "✅ Video processing completed successfully!"
        )


        # =================================================
        # DISPLAY VIDEO
        # =================================================

        st.subheader(
            "🎥 Processed Video - Person Detection"
        )


        with open(
            output_path,
            "rb"
        ) as video_file:

            video_bytes = video_file.read()


        st.video(
        video_bytes,
        format="video/mp4",
        width=700
        )


        # =================================================
        # ANALYTICS
        # =================================================

        st.subheader("📊 Analytics")


        average_people = (

            total_people / frame_number

            if frame_number > 0

            else 0
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Maximum People",
                max_people
            )


        with col2:

            st.metric(
                "Average People",
                f"{average_people:.2f}"
            )


        with col3:

            st.metric(
                "Frames Processed",
                frame_number
            )


        # =================================================
        # DOWNLOAD
        # =================================================

        st.subheader("📥 Download")


        st.download_button(
            label="⬇️ Download Processed Video",
            data=video_bytes,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )


        # =================================================
        # CLEAN TEMP FILES
        # =================================================

        if os.path.exists(input_path):
            os.remove(input_path)

        if os.path.exists(avi_path):
            os.remove(avi_path)