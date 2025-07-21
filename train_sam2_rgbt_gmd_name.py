import argparse
import logging
import os
from datetime import datetime

from datasets.get_data import get_loader, test_dataset
from loss import smoothness
from loss.lscloss import *
from sam2_model.rgbT.model_rgbt_moase_sep_512_part1 import Model
from tools.tools import *
from optim.gmd_name import GMD
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=45, help='epoch number')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
parser.add_argument('--img_size', type=int, default=512, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=140, help='every n epochs decay learning rate')
parser.add_argument('--sm_loss_weight', type=float, default=0.3, help='weight for smoothness loss')
parser.add_argument('--freeze_image_encoder', type=bool, default=True, help='True or False')

parser.add_argument('--save_path', type=str, default='./log/',
                    help='the path to save models and logs')

parser.add_argument('--sam2_cfg', type=str, default='sam2_hiera_l.yaml',
                    help='train from checkpoints')
parser.add_argument('--sam2_checkpoint', type=str, default='/checkpoints/sam2_hiera_large.pt',
                    help='train from checkpoints')
opt = parser.parse_args()

# 定义日志文件路径
save_path = opt.save_path  # 假设日志文件保存在当前目录下的logs文件夹中
log_filename = 'log.txt'
full_log_path = os.path.join(save_path, log_filename)

# 确保日志目录存在
if not os.path.exists(full_log_path):
    os.makedirs(full_log_path)

try:
    # 配置日志系统
    logging.basicConfig(
        filename=full_log_path,
        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
        level=logging.INFO,
        filemode='a',
        datefmt='%Y-%m-%d %I:%M:%S %p'
    )
except Exception as e:
    # 如果配置日志时发生异常，记录异常信息
    print(f"配置日志时发生异常: {e}")
    logging.critical(f"配置日志时发生异常: {e}")

print('Learning Rate: {}'.format(opt.lr))
model = Model(opt)
model.setup()
model.cuda()
params_with_names = []
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
        params_with_names.append({'name': name, 'param': param})

optimizer = torch.optim.Adam([p['param'] for p in params_with_names], lr=0.001)

for group, param_with_name in zip(optimizer.param_groups, params_with_names):
    group['name'] = [p['name'] for p in params_with_names]


# writer = SummaryWriter(log_dir='grad_name_tee_RGB')
gmd = GMD(optimizer, reduction='mean')
train_loader = get_loader('/datasets/RGBT/Train/RGB/',
                          '/datasets/RGBT/Train/T',
                          '/datasets/RGBT/Train/GT',
                          batchsize=opt.batchsize, trainsize=opt.img_size)

test_loader = test_dataset(
    '/datasets/RGBT/Test/VT821/RGB/',
    '/datasets/RGBT/Test/VT821/T/',
    '/datasets/RGBT/Test/VT821/GT/',
    opt.img_size)

total_step = len(train_loader)

# torch.nn.BCELoss()函数要求输入的预测值是经过sigmoid函数处理后的概率值，真实标签为0或1。
# F.binary_cross_entropy_with_logits()函数则要求输入的预测值是未经过sigmoid函数处理的logits（即线性输出），真实标签同样为0或1。
CE = torch.nn.BCELoss()
smooth_loss = smoothness.smoothness_loss(size_average=True)
best_mae = 1
best_epoch = 0

loss_lsc = LocalSaliencyCoherence().cuda()
loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
loss_lsc_radius = 5
save_path = opt.save_path


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(train_loader, model, gmd, epoch):
    model.train()

    for i, pack in enumerate(train_loader, start=1):
        gmd.zero_grad()

        images_rgb, aug_image, images_td, aug_modal, gts = pack
        aug_image = aug_image.cuda()
        aug_modal = aug_modal.cuda()
        gts = gts.cuda()

        SAM_mask1, SAM_mask2, SAM_mask3, SAM_mask = model(aug_image, aug_modal)

        loss1 = structure_loss(SAM_mask1, gts)
        loss2 = structure_loss(SAM_mask2, gts)
        loss3 = structure_loss(SAM_mask3, gts)
        # z_yu = structure_loss(SAM_mask, gts)
        # y_yu = structure_loss(mask, gts)

        gmd.pc_backward([loss1, loss2, loss3], model)

        gmd.step()

        if i % 100 == 0 or i == total_step or i == 1:
            now_formatted = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss1: {:.4f}, loss2: {:.4f}, loss3: {:.4f} '.
                  format(now_formatted, epoch, opt.epoch, i, total_step, loss1.data, loss2.data, loss3.data))
            logging.info(
                '#TRAIN#: Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss1: {:.4f}, loss2: {:.4f}, loss3: {:.4f} '.
                format(epoch, opt.epoch, i, total_step, loss1.data, loss2.data, loss3.data))

        #  手动释放变量，防止显存/内存积压
        del loss1, loss2, loss3, SAM_mask1, SAM_mask2, SAM_mask3, SAM_mask
        torch.cuda.empty_cache()


def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    # 神经网络沿用batch normalization的值，并不使用drop out
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            # print(i)
            image, image_enhance, gt, any_modal, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            any_modal = any_modal.cuda()
            _, _, _, res = model(image, any_modal)
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        print('Epoch: {} MAE: {:.10f} ####  bestMAE: {:.10f} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))

        if epoch == 1:
            best_mae = mae
            best_epoch = epoch
            torch.save(model.state_dict(),
                       save_path + 'sam_rgbt' + str(mae) + '_%d' % epoch + '.pth')
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(),
                           save_path + 'sam_rgbt' + str(best_mae) + 'best' + '_%d' % epoch + '.pth')
                print('best epoch:{}'.format(epoch))

            if mae < 0.025:
                torch.save(model.state_dict(),
                           save_path + 'sam_rgbt' + str(mae) + 'best' + '_%d' % epoch + '.pth')
        logging.info('#TEST#:Epoch:{} MAE:{:.10f} bestEpoch:{} bestMAE:{:.10f}'.
                     format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start Training!")
    logging.info("训练rgbt")

    for epoch in range(1, opt.epoch + 1):
        train(train_loader, model, gmd, epoch)
        test(test_loader, model, epoch, save_path)
