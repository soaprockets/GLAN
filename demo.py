import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F

import os

from collections import OrderedDict
from natsort import natsorted
from glob import glob
import cv2
import argparse
from model.GLAN import GLAN
import numpy as np
from PIL import Image
parser = argparse.ArgumentParser(description='Demo Image Restoration')
parser.add_argument('--input_dir', default='/data/Data/dataset/VV/*', type=str, help='Input images')
parser.add_argument('--result_dir', default='/data/Data/GLAN/enhanced/VV', type=str, help='Directory for results')
parser.add_argument('--weights',
                    default='/data/Data/GLAN/model_bestPSNR.pth', type=str,
                    help='Path to weights')

args = parser.parse_args()


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

inp_dir = args.input_dir
out_dir = args.result_dir

os.makedirs(out_dir, exist_ok=True)

files = natsorted(glob(os.path.join(inp_dir)))

if len(files) == 0:
    raise Exception(f"No files found at {inp_dir}")

# Load corresponding models architecture and weights
model = GLAN()
model.cuda()

load_checkpoint(model, args.weights)
# print(model)
model.eval()

print('restoring images......')

index=0
for file_ in files:
    img = Image.open(file_).convert('RGB')
    input_ = TF.to_tensor(img).unsqueeze(0).cuda()
    # input_ = TF.to_tensor(img).unsqueeze(0).cuda()
    with torch.no_grad():
        restored = model(input_)
    
    restored = restored.permute(0, 2, 3, 1).cuda().data.cpu().numpy()
    f = os.path.basename(file_)[:-4]
    
    cv2.imwrite(os.path.join(out_dir, f+'.png' ), cv2.cvtColor(np.squeeze(restored)*255,cv2.COLOR_RGB2BGR))
    index+=1
    print('%d/%d' % (index, len(files)))

print(f"Files saved at {out_dir}")
print('finish !')
