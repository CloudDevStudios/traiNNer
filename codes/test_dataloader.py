import math

import numpy as np
import torch
import torchvision.utils

import dataops.common as util
from data import create_dataloader, create_dataset
from dataops.colors import ycbcr_to_rgb
from dataops.debug import tmp_vis

# for segmentation
# look_up table # RGB
lookup_table = torch.from_numpy(
    np.array([
        [153, 153, 153],  # 0, background
        [0, 255, 255],  # 1, sky
        [109, 158, 235],  # 2, water
        [183, 225, 205],  # 3, grass
        [153, 0, 255],  # 4, mountain
        [17, 85, 204],  # 5, building
        [106, 168, 79],  # 6, plant
        [224, 102, 102],  # 7, animal
        [255, 255, 255],  # 8/255, void
    ])).float()
lookup_table /= 255


def render(seg):
    _, argmax = torch.max(seg, 0)
    argmax = argmax.squeeze().byte()
    # color img
    im_h, im_w = argmax.size()
    color = torch.FloatTensor(3, im_h, im_w).fill_(0)  # black
    for k in range(8):
        mask = torch.eq(argmax, k)
        color.select(0, 0).masked_fill_(mask, lookup_table[k][0])  # R
        color.select(0, 1).masked_fill_(mask, lookup_table[k][1])  # G
        color.select(0, 2).masked_fill_(mask, lookup_table[k][2])  # B
    # void
    mask = torch.eq(argmax, 255)
    color.select(0, 0).masked_fill_(mask, lookup_table[8][0])  # R
    color.select(0, 1).masked_fill_(mask, lookup_table[8][1])  # G
    color.select(0, 2).masked_fill_(mask, lookup_table[8][2])  # B
    return color


def feed_data(data, need_HR=True, scale=4, device='cpu'):
    # data
    if len(data['LR'].size()) == 4:
        # for networks that work with 3 channel images
        b, n_frames, h_lr, w_lr = data['LR'].size()
        LR = data['LR'].view(b, -1, 1, h_lr, w_lr)  # b, t, c, h, w
    elif len(data['LR'].size()) == 5:
        _, n_frames, _, _, _ = data['LR'].size()
        LR = data['LR']  # b, t, c, h, w

    idx_center = (n_frames - 1) // 2
    n_frames = n_frames

    # LR images
    var_L = LR.to(device)

    # bicubic upscaled LR and center HR
    if isinstance(data['HR_center'], torch.Tensor) and isinstance(data['LR_bicubic'], torch.Tensor):
        var_H_center = data['HR_center'].to(device)
        var_LR_bic = data['LR_bicubic'].to(device)

    if need_HR:  # train or val
        # HR images
        if len(data['HR'].size()) == 4:
            # b, _, h_hr, w_hr = data['HR'].size()
            HR = data['HR'].view(b, -1, 1, h_lr * scale, w_lr * scale)  # b, t, c, h, w
        elif len(data['HR'].size()) == 5:
            HR = data['HR']  # b, t, c, h, w #for networks that work with 3 channel images

        var_H = HR.to(device)
        # discriminator references
        input_ref = data.get('ref', data['HR'])
        if len(input_ref.size()) == 4:
            # b, _, h_hr, w_hr = input_ref.size()
            input_ref = input_ref.view(b, -1, 1, h_lr * scale, w_lr * scale)  # b, t, c, h, w
            var_ref = input_ref.to(device)
        elif len(input_ref.size()) == 5:
            var_ref = input_ref.to(device)

    return {'var_L': var_L, 'var_H': var_H, 'var_ref': var_ref, 'var_H_center': var_H_center, 'var_LR_bic': var_LR_bic}


def combine_colors(fake_H, var_LR_bic, device='cpu'):
    fake_H = fake_H[:, 0, :, :].to(device)  # tmp
    tmp_vis(ycbcr_to_rgb(var_LR_bic), True)
    fake_H_cb = var_LR_bic[:, 1, :, :].to(device)
    fake_H_cr = var_LR_bic[:, 2, :, :].to(device)
    centralSR = torch.stack((fake_H, fake_H_cb, fake_H_cr), -3)  # .squeeze(1)
    centralSR = ycbcr_to_rgb(centralSR)
    tmp_vis(centralSR, True)


def main():
    '''
    opt['name'] = 'DIV2K800'
    opt['dataroot_HR'] = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub.lmdb'
    opt['dataroot_LR'] = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub_bicLRx4.lmdb'

    # opt['dataroot_HR'] = '/mnt/SSD/xtwang/BasicSR_datasets/OST/train/img'
    # opt['dataroot_HR_bg'] = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub'
    # opt['dataroot_LR'] = None
    '''

    opt = {
        'name': 'testA',
        'dataroot_HR': "../training/pbr/",
        'dataroot_LR': None,
        'lr_downscale': True,
        'lr_downscale_types': [777],
        'dataroot_kernels': "../training/kernels/results/",
        'subset_file': None,
        'mode': 'LRHRPBR',
        'phase': 'train',
        'use_shuffle': True,
        'n_workers': 0,
        'batch_size': 1,
        'HR_size': 256,
        'scale': 1,
        'use_flip': False,
        'use_rot': False,
        'color': 'RGB',
        'data_type': 'img',
        'lr_noise': True,
        'lr_noise_types': ["patches"],
    }
    opt['patch_noise']: True
    opt['noise_data'] = "../training/noise_patches/normal/"  # jpeg

    # opt['lr_noise2'] = True
    # opt['lr_noise_types2'] = ["gaussian", "JPEG", "poisson", "dither", "s&p", "speckle", "clean"]
    # opt['lr_noise_types2'] = ["gaussian", "JPEG"]

    # opt['hr_noise'] = True
    # opt['hr_noise_types'] = ["gaussian", "JPEG", "quantize", "poisson", "dither", "s&p", "speckle", "clean"]

    # opt['color_HR'] = "y"
    # opt['color_LR'] = "y"

    # opt['lr_fringes']: True 
    # opt['lr_fringes_chance']: 0.4

    save_samples = False

    train_set = create_dataset(opt)
    train_loader = create_dataloader(train_set, opt)
    nrow = int(math.sqrt(opt['batch_size']))

    if save_samples:
        util.mkdir('tmp')
        padding = 2 if opt['phase'] == 'train' else 0
    for n, data in enumerate(train_loader, start=1):
        # test dataloader time
        # if i == 1:
        #     start_time = time.time()
        # if i == 500:
        #     print(time.time() - start_time)
        #     break
        if n > 10:
            break
        print(n)

        if isinstance(data['LR'], torch.Tensor):
            LR = data['LR']
            # print(LR.shape)
            print(data['LR_path'])
            tmp_vis(LR, True)
        if isinstance(data['HR'], torch.Tensor):
            HR = data['HR']
            # print(HR.shape)
            print(data['HR_path'])
            tmp_vis(HR, True)

        if save_samples:
            torchvision.utils.save_image(
                LR, 'tmp/LR_{:03d}.png'.format(i), nrow=nrow, padding=padding, normalize=False)
            torchvision.utils.save_image(
                HR, 'tmp/HR_{:03d}.png'.format(i), nrow=nrow, padding=padding, normalize=False)

        if opt['mode'] == 'LRHRseg_bg':
            seg = data['seg']
            seg_color_list = []
            for j in range(seg.size(0)):
                _seg = seg[j, :, :, :]
                seg_color_list.append(render(_seg).unsqueeze_(0))

            seg_color_batch = torch.cat(seg_color_list, 0)
            if save_samples:
                torchvision.utils.save_image(
                    seg_color_batch, 'tmp/seg_{:03d}.png'.format(i), nrow=nrow, padding=2, normalize=False)

        if opt['mode'] == 'LRHRPBR':
            if isinstance(data['AO'], torch.Tensor):
                tmp_vis(data['AO'], True)
            if isinstance(data['GL'], torch.Tensor):
                tmp_vis(data['GL'], True)
            if isinstance(data['HE'], torch.Tensor):
                tmp_vis(data['HE'], True)
            if isinstance(data['ME'], torch.Tensor):
                tmp_vis(data['ME'], True)
            if isinstance(data['RE'], torch.Tensor):
                tmp_vis(data['RE'], True)
            if isinstance(data['RO'], torch.Tensor):
                tmp_vis(data['RO'], True)


if __name__ == '__main__':
    main()
