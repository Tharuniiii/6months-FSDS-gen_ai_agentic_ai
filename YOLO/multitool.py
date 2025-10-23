import gradio as gr
import cv2
from ultralytics import YOLO, solutions
import tempfile
import os

def run_yolo(task, video_file):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        return "Error opening video file"

    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    tmp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    out = cv2.VideoWriter(tmp_output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    if task == "Exercise Tracking":
        # Set keypoints for different exercises if needed; here default is pushups
        gym = solutions.AIGym(
            show=False,  # Set False for Gradio
            kpts=[5, 7, 9],  
            model="yolo11n-pose.pt",
            line_width=4,
            verbose=False,
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = gym(frame)
            out.write(results.plot_im)

    else:
        # Choose model for other tasks
        model_file = "yolov8n-pose.pt" if task == "Pose Estimation" else "yolov8n.pt"
        model = YOLO(model_file)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            annotated_frame = results[0].plot()

            if task == "Object Counting":
                num_objects = len(results[0].boxes)
                cv2.putText(annotated_frame, f'Count: {num_objects}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            out.write(annotated_frame)

    cap.release()
    out.release()
    return tmp_output

tasks = ["Pose Estimation", "Customer Detection", "Object Counting", "Exercise Tracking"]

gr.Interface(
    fn=run_yolo,
    inputs=[gr.Dropdown(choices=tasks, label="Select Task"),
            gr.Video(label="Upload Video")],
    outputs=gr.Video(label="Processed Video")
).launch()
