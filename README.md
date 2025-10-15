# 🧠 Object Recognition AI using YOLOv8

An advanced **Object Recognition AI** built using the **YOLOv8 model** that detects and labels real-world objects in **images, videos, and live camera feeds**.  
Includes a clean and interactive **Streamlit UI** for easy use.  

---

## 🚀 Features
- 🖼️ **Image Detection:** Upload an image and view annotated results instantly  
- 🎥 **Video Detection:** Upload a video and get an annotated output video  
- 📸 **Live Camera Detection:** Capture using webcam directly (in Colab)  
- 🧠 **YOLOv8 Integration:** High-performance detection using Ultralytics YOLOv8  
- 🪄 **Custom Streamlit UI:** Polished interface with background and result display  
- 🎓 **Training Ready:** Includes COCO dataset sample (coco128.zip) and weight support  

---

## 🗂️ Project Structure

| File Name | Description |
|------------|-------------|
| `Object_Recognition_AI.ipynb` | Jupyter notebook (training + testing) |
| `streamlit_app.py` | Streamlit UI app |
| `coco128.zip` | Dataset used for model training/testing |
| `background.png` | Streamlit UI background |
| `yolov8n.pt` | Model weights |
| `requirements.txt` | Dependencies |
| `README.md` | Documentation |
| `LICENSE` | GNU License |
| `preview.png` | Sample video preview image |

---

## ⚙️ Installation & Usage

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/Object_Recognition_AI.git
cd Object_Recognition_AI

```

## 2️⃣ Install dependencies

```bash
pip install -r requirements.txt

```

## 3️⃣ Run the Streamlit app
```bash
streamlit run streamlit_app.py

```

## 4️⃣ — Explore the Notebook (Optional)

You can also open `Object_Recognition_AI.ipynb` in **Google Colab** or **Jupyter Notebook** to explore:

- 🏋️ Training the YOLO model  
- 🖼️ Image and video inference  
- 📊 Dataset usage and visualization


## 🧩 Model Used

This project uses **YOLOv8n (Nano)** — a lightweight yet powerful object detection model from **Ultralytics**.  
It provides excellent performance with minimal computational requirements, ideal for real-time use cases.

## 📸 Sample Output

You can find an example detection result video here:


🎥 **Click the image below to watch the demo video!**

[![Watch Result Video](https://raw.githubusercontent.com/RahulBansal-24/Object_Recognition_AI/main/preview.png)](https://drive.google.com/file/d/1NJKEbDCaT6twxUVRV3IiEYaYnMSg0N6r/view)




---

## 📄 License

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**.  
You are free to use, modify, and distribute this project under the same open-source license.

📜 See the [LICENSE](./LICENSE) file for full terms and conditions.

---

## 👨‍💻 Author

**Rahul Bansal**  
- 🌐 [GitHub](https://github.com/RahulBansal-24)  
- 💼 [LinkedIn](https://www.linkedin.com/in/rahulbansal24)

---

⭐ **If you found this project helpful, consider giving it a star!**  
Your support motivates me to create and share more open-source AI projects 🚀



