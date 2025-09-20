# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
import argparse
import numpy as np
import os
import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitl'])
    parser.add_argument('--max_len', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--fp32', action='store_true', help='model infer with torch.float32, default is torch.float16')
    parser.add_argument('--grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--save_npz', action='store_true', help='save depths as npz')
    parser.add_argument('--save_exr', action='store_true', help='save depths as exr')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    video_depth_anything.load_state_dict(torch.load(f'./checkpoints/video_depth_anything_{args.encoder}.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()
    
    # place input dir and out dir here
    root_img_dir = "RORD/train/img"
    root_gt_dir = "RORD/train/gt"
    save_root_img_base = "RORD/val/img_depth"
    save_root_gt_base = "RORD/val/gt_depth"

    video_ids = sorted(os.listdir(root_img_dir))

    for video_id in tqdm.tqdm(video_ids):
        frame_dir = os.path.join(root_img_dir, video_id)

        frame_paths = sorted([
            os.path.join(frame_dir, fname) for fname in os.listdir(frame_dir)
            if fname.endswith(".jpg") or fname.endswith(".png")
        ])
        frames = [cv2.imread(p)[:, :, ::-1] for p in frame_paths]
        gt_path = frame_paths[0].replace("/img/", "/gt/")

        gt_img = cv2.imread(gt_path)[:, :, ::-1]  # BGR to RGB
        frames.append(gt_img) 

        resized_frames = []
        max_res = 1280
        for f in frames:
            h, w = f.shape[:2]
            if max(h, w) > max_res:
                scale = max_res / max(h, w)
                f = cv2.resize(f, (int(w * scale), int(h * scale)))
            resized_frames.append(f)

        resized_frames = np.stack(resized_frames, axis=0)

        depths, _ = video_depth_anything.infer_video_depth(
            resized_frames, 32, input_size=518, device=DEVICE, fp32=False
        )

        save_root_img = os.path.join(save_root_img_base, video_id)
        save_root_gt = os.path.join(save_root_gt_base, video_id)
        os.makedirs(save_root_img, exist_ok=True)
        os.makedirs(save_root_gt, exist_ok=True)

        colormap = np.array(cm.get_cmap("inferno").colors) 
        d_min, d_max = depths.min(), depths.max()
        for i, path in enumerate(frame_paths):
            fname = os.path.basename(path)

            depth = depths[i]
            depth_norm = ((depth - d_min) / (d_max - d_min + 1e-6) * 255).astype(np.uint8)
            depth_vis = (colormap[depth_norm] * 255).astype(np.uint8)  # shape: (H, W, 3), uint8

            img_path = os.path.join(save_root_img, fname)
            Image.fromarray(depth_vis).save(img_path)

            gt_depth = depths[-1]
            gt_norm = ((gt_depth - d_min) / (d_max - d_min + 1e-6) * 255).astype(np.uint8)
            gt_vis = (colormap[gt_norm] * 255).astype(np.uint8)

            gt_save_path = os.path.join(save_root_gt, fname)
            Image.fromarray(gt_vis).save(gt_save_path)

