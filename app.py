
import streamlit as st
import cv2
import tempfile
import numpy as np
import openai
import os
import base64

openai.api_key = os.getenv("OPENAI_API_KEY")

def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode("utf-8")

def analyze_with_gpt(frame):
    base64_image = encode_image_to_base64(frame)
    prompt = "이 이미지는 테니스 자세입니다. 이 자세를 분석해서 점수(100점 만점)를 주고, 무엇이 잘 되었고 어떤 점을 개선해야 하는지 알려주세요."

    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        max_tokens=500
    )
    return response['choices'][0]['message']['content']

st.set_page_config(page_title="Tennis Pose GPT Analyzer", layout="centered")
st.title("🎾 테니스 자세 GPT 분석기")

uploaded_file = st.file_uploader("사진 또는 영상 파일을 업로드하세요", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file:
    if uploaded_file.type.startswith("video"):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        success, frame = cap.read()
        if success:
            st.image(frame, caption="첫 번째 프레임")
            analysis = analyze_with_gpt(frame)
            st.write("## 분석 결과")
            st.write(analysis)
        cap.release()
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        st.image(frame, caption="업로드한 이미지")
        analysis = analyze_with_gpt(frame)
        st.write("## 분석 결과")
        st.write(analysis)
