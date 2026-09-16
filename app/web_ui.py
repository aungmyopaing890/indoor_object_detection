import gradio as gr
from ultralytics import YOLO
import cv2

# Load the trained model
try:
    model = YOLO("weights/best.pt")
except Exception as e:
    print("⚠️ Please place your trained 'best.pt' file in the 'weights/' folder!")
    model = None

def detect_objects(image, conf_threshold):
    if model is None:
        return image
    
    # Run YOLOv8 inference with dynamic confidence
    results = model.predict(source=image, conf=conf_threshold)
    
    # Render the bounding boxes
    annotated_image = results[0].plot()
    return annotated_image

# Build the Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏠 Indoor Vision Assistant")
    gr.Markdown("Upload an image or use your webcam to test the indoor object detection model!")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Input Image / Webcam")
            conf_slider = gr.Slider(minimum=0.01, maximum=1.0, value=0.25, step=0.01, label="Confidence Threshold")
            submit_btn = gr.Button("🔍 Detect Objects", variant="primary")
        with gr.Column():
            output_image = gr.Image(type="numpy", label="Detection Output")
            
    submit_btn.click(fn=detect_objects, inputs=[input_image, conf_slider], outputs=output_image)

if __name__ == "__main__":
    print("Launching Indoor Object Detection Web UI...")
    demo.launch(share=False)
