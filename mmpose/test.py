# Copyright (c) OpenMMLab. All rights reserved.
import logging
from argparse import ArgumentParser
import numpy as np
import cv2
import os
import yaml
import time
from pathlib import Path
from tqdm import tqdm

from mmpose.apis import inference_topdown, init_model
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline
from mmdet.apis import inference_detector, init_detector

BLUR_SIZE = (32, 32)
DEVICE = os.environ.get("TEST_DEVICE", "cuda:0")

parser = ArgumentParser()
parser.add_argument('input', help='Image file')
parser.add_argument('output', help='Path to output file')
parser.add_argument('--no-blur', action='store_true', help='Disable blurring effect')
parser.add_argument('-s', '--skip', type=int, default=0, help='Skip frames')
parser.add_argument('-n', '--nframes', type=int, default=0, help='Number of frames to process')
args = parser.parse_args()

class TopdownModel:
    __visualizer = None

    def __init__(self, config: Path = Path('config.yml')):

        with open(config) as fp:
            cfg = yaml.safe_load(fp)
        det_config = cfg.get('det_config')
        det_checkpoint = cfg.get('det_checkpoint')
        # build detector
        self.detector = init_detector(
            det_config, det_checkpoint, device=DEVICE)
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)

        pose_config = cfg.get('pose_config')
        pose_checkpoint = cfg.get('pose_checkpoint')
        # build pose estimator
        self.pose_estimator = init_model(
            pose_config,
            pose_checkpoint,
            device=DEVICE)

    def predict(self, img, confidence: float = 0.3, nms_thr: float = 0.3):
        det_result = inference_detector(self.detector, img)
        pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                    pred_instance.scores > confidence)]
        bboxes = bboxes[nms(bboxes, nms_thr), :4]

        pose_results = inference_topdown(self.pose_estimator, img, bboxes)
        data_samples = merge_data_samples(pose_results)

        return data_samples

    def init_visualizer(self,
                        radius: int = 3,
                        alpha: float = 0.8,
                        thickness: int = 1,
                        skeleton_style: str = 'mmpose'):
        self.pose_estimator.cfg.visualizer.radius = radius
        self.pose_estimator.cfg.visualizer.alpha = alpha
        self.pose_estimator.cfg.visualizer.line_width = thickness

        self.visualizer = VISUALIZERS.build(self.pose_estimator.cfg.visualizer)
        self.visualizer.set_dataset_meta(
            self.pose_estimator.dataset_meta,
            skeleton_style=skeleton_style)

    def draw_lines(self, img, results,
                   draw_gt: bool = False,
                   draw_bbox: bool = True,
                   show_kpt_idx: bool = False,
                   kpt_thr: float = 0.3):
        return self.visualizer.add_datasample(
            'result',
            img,
            data_sample=results,
            draw_gt=draw_gt,
            draw_bbox=draw_bbox,
            kpt_thr=kpt_thr,
            draw_heatmap=False,
            show_kpt_idx=show_kpt_idx)

def inference_image(model: TopdownModel, inputfile, outputfile):
    img = cv2.imread(inputfile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # inference a single image
    results = model.predict(img)
    start = time.time()
    results = model.predict(img)
    end = time.time()
    print(f'Inference time: {end - start:.6f} seconds')

    # show the results
    if not args.no_blur:
        img = cv2.blur(img, BLUR_SIZE)

    vis_frame = model.draw_lines(img, results)
    cv2.imwrite(outputfile, cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
    print(f'the output image has been saved at {outputfile}')

def inference_video(model: TopdownModel, inputfile, outputfile, **kwargs):
    cap = cv2.VideoCapture(inputfile)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(outputfile, fourcc, fps, (width, height))

    if args.nframes > 0:
        total_frames = args.nframes

    for _ in tqdm(range(args.skip), desc="Skipping frames"):
        ret, frame = cap.read()

    progress = tqdm(total=total_frames)
    times = []
    nframes = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        progress.update()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        start = time.time()
        results = model.predict(img)
        end = time.time()
        #print(f'Inference time: {end - start:.6f} seconds')
        times.append(end - start)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not args.no_blur:
            img = cv2.blur(img, BLUR_SIZE)
        vis_frame = model.draw_lines(img, results)
        frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
        cv2.imshow(outputfile, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        nframes += 1
        if nframes >= total_frames:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f'Average inference time: {sum(times[1:])/len(times[1:]):.6f} seconds')

if __name__ == '__main__':

    model = TopdownModel()
    model.init_visualizer()

    if args.input.endswith((".jpg", ".jpeg", ".png", ".bmp")):
        inference_image(model, args.input, args.output)
    elif args.input.endswith((".mp4", ".mov", ".avi", ".mkv")):
        inference_video(model, args.input, args.output)
