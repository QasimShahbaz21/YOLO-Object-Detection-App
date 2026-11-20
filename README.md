# 🚦 Traffic Signal and Road Sign Detector (YOLOv8 + Streamlit)

An interactive computer vision web application powered by **YOLOv8** and **Streamlit** to detect traffic signals (Red, Green, Yellow) and various road signs in images.

---

## 🌟 Features
- **Interactive Confidence Slider:** Real-time filtering of detections based on confidence score.
- **Side-by-Side Visualization:** Direct comparison of the original image and YOLOv8 annotated predictions.
- **Detailed Detection Summary:** Metrics displaying total count, unique classes detected, and highest confidence score.
- **Object Breakdown Table:** Tabular inspection of each detected bounding box $[x_1, y_1, x_2, y_2]$ and class probability.
- **Sample Images Included:** Test the detector with built-in samples or upload your own images.
- **Annotated Image Download:** Download the annotated image with bounding boxes with one click.

---

## 📊 Model Training & Performance
- **Architecture:** Ultralytics YOLOv8 Nano (`yolov8n.pt`)
- **Dataset:** 5,800+ labeled images across 39 classes (Traffic lights & road signs)
- **Epochs:** 20 epochs on NVIDIA GeForce RTX 4060 GPU
- **Key Metric (mAP@50):** **85.4%**
- **Precision:** **85.4%**
- **Recall:** **80.4%**

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd traffic_signals_app
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

---

## ☁️ Deployment on Streamlit Community Cloud
1. Push this folder to a GitHub repository.
2. Sign in to [share.streamlit.io](https://share.streamlit.io) using your GitHub account.
3. Select your repository, set the branch to `main`, and main file path to `app.py`.
4. Click **Deploy**!
