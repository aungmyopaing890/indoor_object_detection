import cv2
import os
from ultralytics import YOLO

class IndoorVisionPipeline:
    """
    Core pipeline for Indoor Object Detection.
    Handles model loading and inference for images, videos, and real-time streams.
    """
    def __init__(self, weights_path="weights/best.pt", conf_threshold=0.40):
        self.conf_threshold = conf_threshold
        try:
            self.model = YOLO(weights_path)
            print(f"✅ Pipeline initialized with weights: {weights_path}")
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            self.model = None

    def process_image(self, source_img):
        """Runs inference on a single image array and returns the annotated image."""
        if self.model is None:
            return source_img
        
        results = self.model.predict(source=source_img, conf=self.conf_threshold, verbose=False)
        return results[0].plot()

    def process_video(self, video_path, output_path):
        """Runs inference on a video file and saves the annotated video."""
        if self.model is None:
            return

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"🎥 Processing video: {video_path}")
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            annotated_frame = self.process_image(frame)
            out.write(annotated_frame)

        cap.release()
        out.release()
        print(f"✅ Video saved to: {output_path}")
