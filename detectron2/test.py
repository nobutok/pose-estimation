import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
import cv2
import numpy as np
from pathlib import Path

import argparse
import time

BLUR_SIZE = (32, 32)

parser = argparse.ArgumentParser()
parser.add_argument("input", type=Path, help="Path to the input image")
parser.add_argument("output", type=Path, help="Path to save the output image")
parser.add_argument("--no-blur", action="store_true", help="Disable blurring effect")
args = parser.parse_args()

cfg = get_cfg()

config_file = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
#config_file = "COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"
cfg.merge_from_file(model_zoo.get_config_file(config_file))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
predictor = DefaultPredictor(cfg)

if args.input.suffix.lower() in [".jpg", ".jpeg", ".png"]:
    img = cv2.imread(str(args.input))

    outputs = predictor(img)
    start = time.time()
    outputs = predictor(img)
    end = time.time()
    print(f"Time: {end - start:.6f} sec")

    if not args.no_blur:
        img = cv2.blur(img, BLUR_SIZE)
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(str(args.output), out.get_image()[:, :, ::-1])

elif args.input.suffix.lower() in [".mp4", ".avi", ".mov"]:

    cap = cv2.VideoCapture(str(args.input))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    vis = VideoVisualizer(metadata, ColorMode.IMAGE)

    times = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        predictions = predictor(frame)
        end = time.time()

        times.append(end - start)
        print(f"Time: {end - start:.6f} sec")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not args.no_blur:
            frame = cv2.blur(frame, BLUR_SIZE)
        vis_frame = vis.draw_instance_predictions(frame, predictions["instances"].to("cpu"))
        frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
        out.write(frame)
        cv2.imshow(str(args.output), frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    cap.release()
    out.release()

    print(f"Average Time: {np.mean(times):.6f} sec")
