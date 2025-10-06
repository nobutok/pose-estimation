# Copyright (c) OpenMMLab. All rights reserved.
import logging
from argparse import ArgumentParser
import cv2

from mmcv.image import imread
from mmengine.logging import print_log

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
import yaml
import time

BLUR_SIZE = (32, 32)

parser = ArgumentParser()
parser.add_argument('input', help='Image file')
parser.add_argument('output', help='Path to output file')
parser.add_argument('--no-blur', action='store_true', help='Disable blurring effect')
args = parser.parse_args()

def inference_image(config, model, visualizer):
    # inference a single image
    batch_results = inference_topdown(model, args.input)
    start = time.time()
    batch_results = inference_topdown(model, args.input)
    end = time.time()
    print(f'Inference time: {end - start:.6f} seconds')
    results = merge_data_samples(batch_results)

    # show the results
    img = imread(args.input, channel_order='rgb')
    if not args.no_blur:
        img = cv2.blur(img, BLUR_SIZE)

    vis_frame = visualizer.add_datasample(
        'result',
        img,
        data_sample=results,
        draw_gt=False,
        draw_bbox=True,
        kpt_thr=config.get("kpt-thr"),
        draw_heatmap=False,
        show_kpt_idx=config.get("show-kpt-idx"),
        skeleton_style=config.get("skeleton-style"),
        show=False,
        out_file=config.get("out_file"))

    print_log(
        f'the output image has been saved at {args.output}',
        logger='current',
        level=logging.INFO)
    cv2.imwrite(args.output, cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))

def inference_video(config, model, visualizer):
    cap = cv2.VideoCapture(args.input)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    times = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        start = time.time()
        predictions = inference_topdown(model, img)
        end = time.time()
        print(f'Inference time: {end - start:.6f} seconds')
        times.append(end - start)
        results = merge_data_samples(predictions)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not args.no_blur:
            img = cv2.blur(img, BLUR_SIZE)
        vis_frame = visualizer.add_datasample(
            'result',
            img,
            data_sample=results,
            draw_gt=False,
            draw_bbox=True,
            kpt_thr=config.get("kpt-thr"),
            draw_heatmap=False,
            show_kpt_idx=config.get("show-kpt-idx"),
            skeleton_style=config.get("skeleton-style"),
            show=False)

        frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
        cv2.imshow(args.output, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f'Average inference time: {sum(times[1:])/len(times[1:]):.6f} seconds')

def main():
    with open('config.yml') as f:
        config = yaml.safe_load(f)

    model = init_model(
        config.get("config"),
        config.get("checkpoint"),
        device=config.get("device"))

    # init visualizer
    model.cfg.visualizer.radius = config.get("radius")
    model.cfg.visualizer.alpha = config.get("alpha")
    model.cfg.visualizer.line_width = config.get("thickness")

    
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(
        model.dataset_meta, skeleton_style=config.get("skeleton_style"))

    if args.input.endswith((".jpg", ".jpeg", ".png", ".bmp")):
        inference_image(config, model, visualizer)
    elif args.input.endswith((".mp4", ".mov", ".avi", ".mkv")):
        inference_video(config, model, visualizer)

if __name__ == '__main__':
    main()
