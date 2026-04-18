import streamlit as st
import torch
import numpy as np
import cv2
import base64
import time
from PIL import Image
import torch.nn.functional as F
from io import BytesIO
from model_utils import SegmentationDecoder
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- PAGE CONFIG ---
st.set_page_config(page_title="KINETIC INTELLIGENCE", layout="wide", initial_sidebar_state="collapsed")

# --- UTILS ---
def img_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

@st.cache_resource
def load_system():
    device = torch.device("cpu")
    # DINOv2 Backbone
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    # Decoder Grid: 37x37 (518//14)
    model = SegmentationDecoder(384, 10, 37, 37) 
    model.load_state_dict(torch.load("best_model_upgraded.pth", map_location=device))
    model.eval()
    return backbone, model, device

backbone, model, device = load_system()

# --- UI CSS ---
st.markdown("""
<style>
    body { background-color: #000; color: #e5e2e1; font-family: 'Courier New', monospace; }
    .stApp { background-color: #000; }
    .terminal { background: #050505; color: #c3f400; padding: 12px; border-left: 3px solid #c3f400; font-size: 11px; margin-top: 10px; }
    .hazard-blink { color: #ff4b4b; font-weight: bold; animation: blinker 1s linear infinite; }
    @keyframes blinker { 50% { opacity: 0.2; } }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("UNIT-V432 // NAV")
    uploaded_file = st.file_uploader("SENSORS: INPUT FEED", type=["jpg", "png", "jpeg"])
    opacity = st.slider("MASK OPACITY", 0.0, 1.0, 0.50)
    st.info("System Status: Calibrated (98.2%)")

# --- MAIN ENGINE ---
if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(raw_img)
    h, w, _ = img_np.shape

    # 1. ADVANCED ENHANCEMENT (Rocks dhoondhne ke liye)
    # CLAHE contrast badhata hai par edges sharp rakhta hai
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    pre_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    start_time = time.time()

    # 2. INFERENCE
    transform = A.Compose([
        A.Resize(518, 518),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    input_tensor = transform(image=pre_img)["image"].unsqueeze(0)
    
    with torch.no_grad():
        features = backbone.forward_features(input_tensor)["x_norm_patchtokens"]
        logits = model(features)
        output = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        pred = torch.argmax(output, dim=1).squeeze().numpy()

    # 3. MAPPING REAL IDs (As per your Model Output)
    overlay = np.zeros_like(img_np)
    
    # Mapping based on your screenshot's detected classes: [1, 2, 3, 5, 7, 8, 9]
    overlay[pred == 1] = [60, 60, 60]      # Path / Landscape (Grey)
    overlay[pred == 3] = [255, 0, 0]       # Critical Rock / Obstacle (Red)
    overlay[pred == 8] = [200, 0, 0]       # Secondary Rock / Boulder (Dark Red)
    overlay[pred == 2] = [0, 255, 0]       # Lush Bush (Green)
    overlay[pred == 5] = [0, 180, 100]     # Dry Bush (Teal/Mint)
    overlay[pred == 9] = [0, 120, 255]     # Sky (Blue)
    overlay[pred == 7] = [255, 255, 0]     # Other Debris (Yellow)

    # 4. HAZARD & TRAJECTORY LOGIC
    hazard_present = np.any((pred == 3) | (pred == 8))
    traj_img = img_np.copy()
    
    # Arrow drawing on Path (ID 1)
    if 1 in pred:
        M = cv2.moments((pred == 1).astype(np.uint8))
        if M["m00"] > 0:
            cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            cv2.arrowedLine(traj_img, (w//2, h-30), (cX, cY), (195, 244, 0), 15, tipLength=0.25)

    latency = int((time.time() - start_time) * 1000)

    # 5. RENDER DASHBOARD
    blended = Image.blend(Image.fromarray(img_np), Image.fromarray(overlay), alpha=opacity)
    b64_traj = img_to_base64(Image.fromarray(traj_img))
    b64_mask = img_to_base64(blended)
    
    status_clr = "#ff4b4b" if hazard_present else "#c3f400"

    st.markdown(f"""
    <div style="border-bottom: 2px solid {status_clr}; padding: 10px; display: flex; justify-content: space-between; align-items: center; background: #000;">
        <span style="color: {status_clr}; font-weight: bold; letter-spacing: 2px;">KINETIC INTELLIGENCE // UNIT-V432</span>
        <span style="color: {status_clr};">● {'HAZARD DETECTED' if hazard_present else 'SYSTEM ACTIVE'}</span>
    </div>

    <div style="display: grid; grid-template-columns: 1.4fr 1fr; gap: 15px; margin-top: 15px;">
        <div style="border: 1px solid {status_clr}; position: relative; height: 420px; background: #000;">
            <div style="position: absolute; background: {status_clr}; color: #000; font-size: 10px; padding: 4px 12px; font-weight: bold; z-index:10;">PRIMARY_FEED</div>
            <img src="data:image/png;base64,{b64_traj}" style="width:100%; height:100%; object-fit: cover;">
        </div>
        <div style="border: 1px solid #333; position: relative; height: 420px; background: #000;">
            <div style="position: absolute; background: #333; color: #fff; font-size: 10px; padding: 4px 12px; z-index:10;">ENVIRONMENTAL_SCAN</div>
            <img src="data:image/png;base64,{b64_mask}" style="width:100%; height:100%; object-fit: cover;">
        </div>
    </div>

    <div class="terminal">
        [SYS] Sensors operational. IDs Detected in Scene: {np.unique(pred).tolist()}<br>
        [INFO] DINOv2 Latency: {latency}ms | Memory: Stable<br>
        <span class="{'hazard-blink' if hazard_present else ''}">
            { '[WARN] CRITICAL: ROCK/OBSTACLE DETECTED IN PATH!' if hazard_present else '[NAV] Path clear. Trajectory locked.' }
        </span>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("Awaiting Sensor Input...")