import streamlit as st
import os
import cv2
import numpy as np
from ultralytics import YOLO
import requests
from tempfile import NamedTemporaryFile

# Load YOLO model
yolo_model = YOLO(r"C:\Users\91630\OneDrive\Desktop\sem4 AI project\yolov8_model.pt")  # Update path to your model

# API Tokens
IPINFO_TOKEN = "6cc277dc69aff5"
GEOAPIFY_API_KEY = "7b513e9863a0441296c99881c47d9aec"

# Detect accident in image
def detect_accident(image_path, conf_threshold=0.3):
    results = yolo_model.predict(image_path, conf=conf_threshold)

    accident_detected = False
    fire_detected = False

    for result in results:
        for box in result.boxes:
            label = yolo_model.names[int(box.cls[0])]  # Correct label
            confidence = box.conf[0]


           # print(f"Detected label: {label}, confidence: {confidence:.2f}")  # Optional: debug print

            if label == "Accident" and confidence > conf_threshold:
                accident_detected = True
            if label == "Fire" and confidence > conf_threshold:
                fire_detected = True
    return {
        "Accident": accident_detected,
        "Fire": fire_detected
    }

# Detect accident in video
def detect_accident_in_video(video_path, conf_threshold=0.3):
    cap = cv2.VideoCapture(video_path)
    accident_detected = False
    fire_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection on the frame
        results = yolo_model.predict(frame, conf=conf_threshold)

        for result in results:
            for box in result.boxes:
                label = yolo_model.names[int(box.cls[0])]
                confidence = box.conf[0]

                if label == "Accident" and confidence > conf_threshold:
                    accident_detected = True
                if label == "Fire" and confidence > conf_threshold:
                    fire_detected = True

    cap.release()
    cv2.destroyAllWindows()
    return {
        "Accident": accident_detected,
        "Fire": fire_detected
    }
   
# Get location
def get_current_location():
    try:
        response = requests.get(f"https://ipinfo.io/json?token={IPINFO_TOKEN}")
        data = response.json()
        loc = data['loc'].split(',')
        lat, lon = float(loc[0]), float(loc[1])
        return {
            "city": data.get('city', ''),
            "region": data.get('region', ''),
            "country": data.get('country', ''),
            "lat": lat,
            "lon": lon
        }
    except Exception as e:
        print("Location error:", e)
        return None

# Get hospital
def get_nearby_hospital(lat, lon):
    url = (
        f"https://api.geoapify.com/v2/places"
        f"?categories=healthcare.hospital"
        f"&filter=circle:{lon},{lat},5000"
        f"&bias=proximity:{lon},{lat}"
        f"&limit=1"
        f"&apiKey={GEOAPIFY_API_KEY}"
    )
    response = requests.get(url)
    hospitals = response.json().get("features", [])
    if hospitals:
        nearest = hospitals[0]["properties"]
        return {
            "name": nearest.get("name", "Unknown Hospital"),
            "address": nearest.get("formatted", "No address available")
        }
    else:
        return None

# Streamlit UI
st.set_page_config(page_title="Accident Detection System", layout="centered")

st.title("🚑 Accident Detection System")

uploaded_file = st.file_uploader("Upload Image or Video", type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'])
if uploaded_file:
    is_video = uploaded_file.name.lower().endswith(('.mp4', '.avi', '.mov'))

    with st.spinner('Detecting...'):
        if is_video:
            with NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                temp_video.write(uploaded_file.read())
                temp_video_path = temp_video.name

            result = detect_accident_in_video(temp_video_path)
            os.remove(temp_video_path)
        else:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            result = detect_accident(image)

    # Show detection results
    if result["Accident"] or result["Fire"]:
        if result["Accident"]:
            st.error("⚠️ Accident Detected!")
        if result["Fire"]:
            st.warning("🔥 Fire Detected!")

        # Show location only for accident or fire
        location = get_current_location()
        if location:
            st.markdown(f"**📍 Location:** {location['city']}, {location['region']}, {location['country']}")
            st.markdown(f"**🌐 Coordinates:** {location['lat']}, {location['lon']}")

            hospital = get_nearby_hospital(location['lat'], location['lon'])
            if hospital:
                st.markdown(f"**🏥 Nearest Hospital:** {hospital['name']}")
                st.markdown(f"**📌 Address:** {hospital['address']}")
            else:
                st.warning("No nearby hospitals found.")
        else:
            st.warning("Could not fetch your location.")
    else:
        st.success("✅ No Accident or Fire Detected.")
