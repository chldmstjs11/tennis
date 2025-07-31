
import streamlit as st
import cv2
import tempfile
import numpy as np
from utils.pose_estimation import extract_keypoints_from_image
from utils.gpt_analysis import analyze_pose_with_gpt

st.set_page_config(page_title="Tennis Pose Analyzer", layout="centered")
st.title("🎾 테니스 자세 분석기")

uploaded_file = st.file_uploader("테니스 영상 또는 사진을 업로드하세요", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file:
    if uploaded_file.type.startswith("video"):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        success, frame = cap.read()
        if success:
            st.image(frame, caption="영상에서 추출된 첫 프레임")
            keypoints = extract_keypoints_from_image(frame)
            analysis = analyze_pose_with_gpt(keypoints)
            st.markdown("## 분석 결과")
            st.write(analysis)
        cap.release()
    else:
        file_bytes = uploaded_file.read()
        np_image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        st.image(np_image, caption="업로드된 이미지")
        keypoints = extract_keypoints_from_image(np_image)
        analysis = analyze_pose_with_gpt(keypoints)
        st.markdown("## 분석 결과")
        st.write(analysis)
