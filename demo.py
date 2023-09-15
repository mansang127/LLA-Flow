import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from core import LLAFlow
from core.utils import flow_viz
from core.utils.utils import InputPadder

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    model = torch.nn.DataParallel(LLAFlow(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    filepath = os.path.join(args.path, 'results')
    if not os.path.exists(filepath):
        os.makedirs(filepath, exist_ok=True)
    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=24, test_mode=True)

            image1 =  padder.unpad(image1[0]).permute(1, 2, 0).cpu().numpy()
            # image2 =  padder.unpad(image2[0]).permute(1, 2, 0).cpu().numpy()
            flow_up = padder.unpad(flow_up[0]).permute(1, 2, 0).cpu().numpy()
            filename = os.path.join(filepath, imfile1.split('/')[-1])

            flow_up = flow_viz.flow_to_image(flow_up)
            img_flo = np.concatenate([image1, flow_up], axis=0)
            # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
            # cv2.waitKey()
            cv2.imwrite(filename, img_flo[:, :, [2, 1, 0]])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--gma', action='store_true', help='use gma module')
    args = parser.parse_args()

    demo(args)
