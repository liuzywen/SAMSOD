import cv2
from sam2_model.rgbT.model_rgbt_moase_sep_512_part1 import Model

import argparse
from datasets.get_data import test_dataset
import os
from tools import *
from loss.lscloss import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--img_size', type=int, default=512, help='training dataset size')

parser.add_argument('--res_checkpoint', type=str, default='', help='test from checkpoints')

parser.add_argument('--save_dir', default='./map', help='path where to save predicted maps')

parser.add_argument('--sam2_cfg', type=str, default='sam2_hiera_l.yaml',
                    help='train from checkpoints')
parser.add_argument('--sam2_checkpoint', type=str, default='/checkpoints/sam2_hiera_large.pt',
                    help='train from checkpoints')
opt = parser.parse_args()

model = Model(opt)
model.load_state_dict(torch.load(opt.res_checkpoint))
for param in model.parameters():
    param.requires_grad_(False)
model.cuda()
model.eval()

# root = 'VT821'
# root = 'VT1000'
root = 'VT5000'

test_loader = test_dataset(
    os.path.join('/datasets/RGBT', 'Test', root, 'RGB') + '/',
    os.path.join('/datasets/RGBT', 'Test', root, 'T') + '/',
    os.path.join('/datasets/RGBT', 'Test', root, 'GT') + '/',
    opt.img_size)
save_path = os.path.join(opt.save_dir, root)
if not os.path.exists(save_path):
    os.makedirs(save_path)

for i in range(test_loader.size):
    image, _, gt, any_modal, name = test_loader.load_data()
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)
    image = image.cuda()
    any_modal = any_modal.cuda()
    _, _, _,res = model(image, any_modal)

    res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    print('[', i + 1, ']', 'save img to: ', save_path + name)
    res = np.round(res * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_path, name), res)
print('Test Done!')
