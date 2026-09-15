import argparse
import cv2
import os
from src.pipeline import IndoorVisionPipeline

def main():
    parser = argparse.ArgumentParser(description="Run Indoor Object Detection Pipeline")
    parser.add_argument("--source", type=str, required=True, help="Path to image, video, or '0' for webcam")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="Path to YOLO weights")
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold")
    
    args = parser.parse_args()
    pipeline = IndoorVisionPipeline(weights_path=args.weights, conf_threshold=args.conf)

    # Handle Webcam
    if args.source == '0':
        cap = cv2.VideoCapture(0)
        print("📷 Press 'q' to quit webcam stream.")
        while True:
            ret, frame = cap.read()
            if not ret: break
            annotated_frame = pipeline.process_image(frame)
            cv2.imshow("Indoor Object Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    
    # Handle Video
    elif args.source.endswith(('.mp4', '.avi', '.mov')):
        filename = os.path.basename(args.source)
        out_path = f"data/output/annotated_{filename}"
        pipeline.process_video(args.source, out_path)
    
    # Handle Image
    else:
        img = cv2.imread(args.source)
        if img is None:
            print(f"❌ Error: Could not read {args.source}")
            return
        annotated_img = pipeline.process_image(img)
        filename = os.path.basename(args.source)
        out_path = f"data/output/annotated_{filename}"
        cv2.imwrite(out_path, annotated_img)
        print(f"✅ Image saved to: {out_path}")

if __name__ == "__main__":
    main()
