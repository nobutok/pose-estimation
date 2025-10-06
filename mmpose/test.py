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

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('input', help='Image file')
    parser.add_argument('output', help='Path to output file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    args = parser.parse_args()
    return args

def inference_image(config, model, visualizer, inputfile, outputfile):
    # inference a single image
    batch_results = inference_topdown(model, inputfile)
    start = time.time()
    batch_results = inference_topdown(model, inputfile)
    end = time.time()
    print(f'Inference time: {end - start:.6f} seconds')
    results = merge_data_samples(batch_results)

    # show the results
    img = imread(inputfile, channel_order='rgb')
    visualizer.add_datasample(
        'result',
        img,
        data_sample=results,
        draw_gt=False,
        draw_bbox=True,
        kpt_thr=config.get("kpt-thr"),
        draw_heatmap=False,
        show_kpt_idx=config.get("show-kpt-idx"),
        skeleton_style=config.get("skeleton-style"),
        show=config.get("show"),
        out_file=config.get("out_file"))

    if outputfile is not None:
        print_log(
            f'the output image has been saved at {outputfile}',
            logger='current',
            level=logging.INFO)

def inference_video(config, model, visualizer, inputfile, outputfile):
    cap = cv2.VideoCapture(inputfile)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(outputfile, fourcc, fps, (width, height))

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
        cv2.imshow(outputfile, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f'Average inference time: {sum(times[1:])/len(times[1:]):.6f} seconds')

def main():
    with open('config.yml') as f:
        config = yaml.safe_load(f)

    args = parse_args()
    config.update(vars(args))

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
        inference_image(config, model, visualizer, args.input, args.output)
    elif args.input.endswith((".mp4", ".mov", ".avi", ".mkv")):
        inference_video(config, model, visualizer, args.input, args.output)

if __name__ == '__main__':
    main()
