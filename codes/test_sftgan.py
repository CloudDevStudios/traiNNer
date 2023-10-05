import glob
import os.path

import cv2
import models.modules.architectures.sft_arch as sft
import numpy as np
import torch

import utils.util as util
from dataops.common import modcrop
from dataops.imresize import resize as imresize

# model_path = '../experiments/pretrained_models/sft_net_torch.pth' # torch version
model_path = '../experiments/pretrained_models/SFTGAN_bicx4_noBN_OST_bg.pth'  # pytorch training

test_img_folder_name = 'samples'  # image folder name
test_img_folder = f'../data/{test_img_folder_name}'
test_prob_path = f'../data/{test_img_folder_name}_segprob'
save_result_path = f'../data/{test_img_folder_name}_result'

# make dirs
util.mkdirs([save_result_path])

model = sft.SFT_Net_torch() if 'torch' in model_path else sft.SFT_Net()
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.cuda()

print('sftgan testing...')

for idx, path in enumerate(glob.glob(f'{test_img_folder}/*'), start=1):
    basename = os.path.basename(path)
    base = os.path.splitext(basename)[0]
    print(idx, base)
    # read image
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = modcrop(img, 8)
    img = img * 1.0 / 255
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    # matlab imresize
    img_LR = imresize(img, 1 / 4, antialiasing=True)
    img_LR = img_LR.unsqueeze(0)
    img_LR = img_LR.cuda()

    # read seg
    seg = torch.load(os.path.join(test_prob_path, f'{base}_bic.pth'))
    seg = seg.unsqueeze(0)
    # change probability
    # seg.fill_(0)
    # seg[:,5].fill_(1)

    seg = seg.cuda()

    output = model((img_LR, seg)).data
    output = util.tensor2img(output.squeeze())
    util.save_img(output, os.path.join(save_result_path, f'{base}_rlt.png'))
