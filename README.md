# 🏠 Indoor Object Detection (YOLOv8)

This project is a lightweight computer vision object detection model built with **Python**, **Ultralytics YOLOv8**, and a custom **Roboflow dataset**. 

This model is trained to detect **20 everyday household and office objects** for fetch-and-retrieve tasks.

---

## 🎥 Demo Results

### Web UI (Gradio)
The project includes a built-in Web UI to easily test the model using your webcam or uploaded images!

![Web UI Demo](assets/images/web_ui_demo.jpg) *(Add your screenshot here)*

### Detection Results
The trained YOLO model accurately detects highly-requested objects, even in cluttered environments.

![Detection Result 1](assets/images/result_1.jpg) ![Detection Result 2](assets/images/result_2.jpg)

---

## 🎯 Target Objects (20 Classes)

The model is trained to detect the following items:
`smartphone`, `laptop`, `bottle`, `keys`, `watch`, `remote`, `charging_cable`, `book`, `glasses`, `cup`, `coffee_mug`, `pen`, `earbuds`, `earbuds_case`, `headphone`, `shoe`, `keyboard`, `umbrella`, `plant`, `container`.

---

## 🚀 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/aungmyopaing890/indoor_object_detection.git
cd indoor_object_detection
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🎮 How to Use

### 1. Launch the Interactive Web UI
The easiest way to test the model is using the built-in Gradio interface:
```bash
python web_ui.py
```
*This will open a local webpage where you can upload photos or use your webcam to see the detection in real-time!*

### 2. Run Detection on an Image
```bash
yolo detect predict model=weights/best.pt source=your_image.jpg conf=0.40 save=True
```

### 3. Run Detection on a Video
```bash
yolo detect predict model=weights/best.pt source=your_video.mp4 conf=0.40 save=True
```

---

## 🧠 Model Training (Kaggle GPUs)
This model was trained purely in the cloud using Kaggle's free Dual T4 GPUs. 

The training script used:
```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=416 batch=32 name=indoor_assistant
```

---

## 👨‍💻 Author
Created by **Aung Myo Paing**
* [GitHub Profile](https://github.com/aungmyopaing890)
* [LinkedIn](https://linkedin.com/in/aungmyopaing)

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
 

### ☁️ Train Your Own Model (Kaggle)
If you want to train this model yourself using a free cloud GPU, I have included the training notebook:
* `notebooks/Indoor_Object_Detection_Kaggle_Training.ipynb`

You can upload this directly to Kaggle, attach a YOLO dataset, and run all cells to get your own `best.pt`!
