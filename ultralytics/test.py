from ultralytics import YOLO
import argparse
from pathlib import Path
import time
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='yolo11s-pose.pt', help='Path to the model file')
parser.add_argument('input', type=Path, help='Path to the input image or video file')
parser.add_argument('output', type=Path, help='Path to save the output results')
args = parser.parse_args()

model = YOLO(args.model)

if args.input.suffix.lower() in [".jpg", ".jpeg", "png"]:

    results = model.predict(str(args.input), verbose=False)
    start = time.time()
    results = model.predict(str(args.input), verbose=False)
    end = time.time()
    print(f"Inference time: {end - start:.6f} seconds")
    results[0].save(args.output)

elif args.input.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv"]:

    cap = cv2.VideoCapture(str(args.input))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))

    results = model.predict(str(args.input), stream=True, verbose=False)
    times = []
    for result in results:
        pt = sum([t for t in result.speed.values()]) / 1000  # sum of preprocess, inference, postprocess times in ms
        times.append(pt)
        img = result.plot()
        cv2.imshow(str(args.output), img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        out.write(img)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Average inference time: {sum(times[1:])/len(times[1:]):.6f} seconds")
