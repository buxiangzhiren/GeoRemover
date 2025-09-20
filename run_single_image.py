# Copyright (2025) Bytedance Ltd. and/or its affiliates 
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0

import os
import numpy as np
import torch
import cv2
import matplotlib.cm as cm
from PIL import Image
from video_depth_anything.video_depth import VideoDepthAnything

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitl'])
    parser.add_argument('--max_len', type=int, default=-1)
    parser.add_argument('--target_fps', type=int, default=-1)
    parser.add_argument('--fp32', action='store_true')
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--save_npz', action='store_true')
    parser.add_argument('--save_exr', action='store_true')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    video_depth_anything.load_state_dict(
        torch.load(f'./checkpoints/video_depth_anything_{args.encoder}.pth', map_location='cpu'),
        strict=True
    )
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    # your image input and output path
    input_path = ""
    output_path = ""


    img = cv2.imread(input_path)[:, :, ::-1]
    h, w = img.shape[:2]

    if max(h, w) > args.max_res:
        scale = args.max_res / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    frame_tensor = np.stack([img], axis=0)


    depths, _ = video_depth_anything.infer_video_depth(
        frame_tensor, 32, input_size=518, device=DEVICE, fp32=False
    )
    depth = depths[0]


    colormap = np.array(cm.get_cmap("inferno").colors)
    d_min, d_max = depth.min(), depth.max()
    depth_norm = ((depth - d_min) / (d_max - d_min + 1e-6) * 255).astype(np.uint8)
    depth_vis = (colormap[depth_norm] * 255).astype(np.uint8)

    Image.fromarray(depth_vis).save(output_path)
    print(f"Saved depth map to: {output_path}")
