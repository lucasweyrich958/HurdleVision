# 🏃‍♂️💨 HurdleVision: AI-Powered Biomechanics

**Automated 110m Hurdle Analysis using Computer Vision & Deep Learning**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![YOLO11](https://img.shields.io/badge/Model-YOLO11-orange)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **"A blink of an eye decides Gold."**
> This project democratizes elite-level sports analytics by extracting biomechanical metrics from standard broadcast video, eliminating the need for expensive markers or 3D labs.

---

## 🧐 Overview

**HurdleVision** is a computer vision pipeline designed to analyze the 110m hurdles. Developed as a Master's Thesis for the CUNY MS in Data Science, it utilizes a fine-tuned **YOLO11** model and **BoT-SORT** tracking to isolate athletes and quantify their rhythm.

### 🎯 Key Capabilities
* **🎥 Markerless Tracking:** Works on standard 2D TV broadcast footage (no motion capture suits required!).
* **🧠 Intelligent Event Detection:** Uses signal processing (Savitzky-Golay filters) to detect hurdle clearances based on simultaneous peaks in vertical displacement and aspect ratio.
* **⏱️ High Precision:** Achieves a mean absolute error (MAE) of **0.03 seconds** for mid-race splits.
    * *Note:* At 30 FPS, 0.03s is **exactly one video frame**. The model has hit the theoretical precision limit of the hardware! 🚀

---

## 📂 Project Structure

```text
HurdleVision/
│
├── 📜 main.py                  # The captain of the ship (Run this!)
├── 🛠️ hurdle_analysis_pipeline.py # The engine room (Logic & Algorithms)
├── 🖼️ extract_frames.py        # Helper to grab training images from video
│
├── 🧠 models/
│   ├── yolo11s_tuned.pt        # Our fine-tuned athlete/hurdle detector
│   └── yolov11s-pose.pt        # (Optional) Pose estimation model
│
├── 📓 notebooks/
│   └── YOLO_Finetuning.ipynb   # The training laboratory (Google Colab)
│
├── 🎓 thesis/
│   ├── Thesis.pdf              # The full academic paper
│   └── Slides.pdf              # Presentation deck
│
└── 📁 data/
    └── (See: https://drive.google.com/drive/folders/1RLC32_Wnuek-K-1vM5jxnDtjZIuHiUe5?usp=sharing)