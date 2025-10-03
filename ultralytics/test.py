from ultralytics import YOLO
import argparse
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='yolo11s-pose.pt', help='Path to the model file')
parser.add_argument('input', type=Path, help='Path to the input image or video file')
parser.add_argument('output', type=Path, help='Path to save the output results')
args = parser.parse_args()

model = YOLO(args.model)

if args.input.suffix.lower() in [".jpg", ".jpeg", "png"]:

    results = model(str(args.input), verbose=False)
    start = time.time()
    results = model(str(args.input), verbose=False)
    end = time.time()
    print(f"Inference time: {end - start:.6f} seconds")
    results[0].save(args.output)

elif args.input.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv"]:

    results = model(str(args.input), stream=True)
    for result in results:
        result.plot()
    