import streamlit as st, numpy as np, tempfile, os, glob, pandas as pd
from pathlib import Path
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from google.colab import files

#configuring streamlit app and giving it styling

st.set_page_config(page_title="Object Recognition AI 🤖", layout="wide")
background_url = "https://github.com/RahulBansal-24/Object-Recognition-AI/raw/main/background.png" #backround image hosted in github

st.title("Object Recognition AI 🤖")

st.markdown(
    f"""
    <style>
    .stApp {{
      background: url("{background_url}") no-repeat center center fixed;
      background-size: cover
    }}
    /* Sidebar styling */
    section[data-testid="stSidebar"]{{
      background: rgba(0,0,0,0.3);
      backdrop-filter: blur(10px);
      border-radius: 12px;
      padding: 10px;
    }}
    /*File uploader styling*/
    div[data-testid="stFileUploader"] {{
      background: rgba(0,0,0,0.3);
      backdrop-filter: blur(10px);
      border-radius: 12px;
      padding: 15px;
    }}
    /*DataFrame styling*/
    .stDataFrame {{
      background: rgba(255,255,255,0.85);
      border-radius: 12px;
      padding: 10px;
      box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }}
    /*Dark Text*/
    h1, h2, h3, h4, h5, h6, p, label, span, div {{
      color: #1a1a1a !important;
      font-weight: 500;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

#creating sidebar slider and image box, defining functions for loading model and saving uploaded file

uploaded_weights = st.sidebar.file_uploader("Upload custom model .pt weights (optional)", type=["pt"])
conf = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
img_size = st.sidebar.selectbox("Inference image size (px)", [320,416,640,1280], index=2)

@st.cache_resource
def load_model(weights_path="yolov8n.pt"):
  return YOLO(weights_path)

def save_uploaded_file(uploaded_file, suffix=""):
  suffix = suffix if suffix else Path(uploaded_file.name).suffix
  tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
  tf.write(uploaded_file.getbuffer())
  tf.flush()
  return tf.name

#defining a function for annotating results on image

def annotate_and_table(results, model):
  res = results[0]
  try:
    plotted = res.plot()
    annotated = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
  except Exception:
    annotated = res.orig_img if hasattr(res, "orig_img") else None

  detections = []
  try:
    boxes = res.boxes
    if boxes is not None and len(boxes) > 0:
      for c, cf, box in zip(boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.xyxy.cpu().numpy()):
        name = model.names[int(c)]
        detections.append({"class":name, "conf":  float(cf), "bbox": [float(x) for x in box]})
  except:
    detections = []

  return annotated, pd.DataFrame(detections)

#I/O and processing

weights_to_load = "yolov8n.pt"
if uploaded_weights:
  weights_to_load = save_uploaded_file(uploaded_weights, suffix=".pt")
  st.sidebar.success("Using uploaded weights")

model = load_model(weights_to_load)

mode = st.radio("Select input", ["Image upload", "Video upload"])

#Image upload case
if mode == "Image upload":
  uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
  if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input Image")
    results = model.predict(np.array(img), conf=conf, imgsz=img_size)
    annotated, df = annotate_and_table(results, model)
    if annotated is not None:
      annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
      st.image(annotated_rgb, caption="Annotated")
    if not df.empty:
      st.dataframe(df)
#Video upload case
elif mode == "Video upload":
  uploaded_vid = st.file_uploader("Upload video", type=["mp4","mov","avi","mkv"])
  if uploaded_vid:
    tmp = save_uploaded_file(uploaded_vid)
    st.video(tmp)
    project_dir = tempfile.mkdtemp()
    results = model.predict(source=tmp, conf=conf, imgsz=img_size, project=project_dir,name="run", save=True)
    try:
      out_dir = str(results[0].save_dir)
      vids = glob.glob(os.path.join(out_dir, "*"))
      vids = [v for v in vids if Path(v).suffix.lower() in [".mp4",".avi",".mov",".mkv"]]
      if vids:
        st.success("Annotated video")
        st.video(vids[0])
    except:
      st.warning("Could not display annotated video")
