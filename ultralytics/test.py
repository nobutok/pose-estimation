from ultralytics import YOLO
import argparse
from pathlib import Path
import time
import cv2
import os
from tqdm import tqdm

BLUR_SIZE = (32, 32)
DEVICE = os.environ.get("TEST_DEVICE", "cuda:0")

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='yolo11s-pose.pt', help='Path to the model file')
parser.add_argument('input', type=Path, help='Path to the input image or video file')
parser.add_argument('output', type=Path, help='Path to save the output results')
parser.add_argument('--no-blur', action='store_true', help='Disable blurring effect')
args = parser.parse_args()

model = YOLO(args.model)

if args.input.suffix.lower() in [".jpg", ".jpeg", "png"]:

    img = cv2.imread(str(args.input))
    if args.no_blur:
        blurred = img
    else:
        blurred = cv2.blur(img, BLUR_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model.predict(img, verbose=False, device=DEVICE)
    start = time.time()
    result = model.predict(img, verbose=False, device=DEVICE)[0]
    end = time.time()
    print(f"Inference time: {end - start:.6f} seconds")
    img = result.plot(img=blurred)
    cv2.imwrite(str(args.output), img)

elif args.input.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv"]:

    cap = cv2.VideoCapture(str(args.input))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))

    progress = tqdm(total=total_frames)
    times = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        progress.update(1)
        start = time.time()
        result = model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), verbose=False, device=DEVICE)[0]
        end = time.time()
        #print(f"Inference time: {end - start:.6f} seconds")
        times.append(end - start)

        if args.no_blur:
            blurred = frame
        else:
            blurred = cv2.blur(frame, BLUR_SIZE)
        img = result.plot(img=blurred)
        cv2.imshow(str(args.output), img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        out.write(img)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Average inference time: {sum(times[1:])/len(times[1:]):.6f} seconds")
