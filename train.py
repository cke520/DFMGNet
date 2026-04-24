import os
import torch
import torch.nn.functional as F
import sys

sys.path.append('./models')
import numpy as np
from datetime import datetime
from models.DFMG import DFMG  # 确保这个导入指向了我们修改后的模型文件
from torchvision.utils import make_grid
from tools.data import get_loader, test_dataset
from tools.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt
import pytorch_iou
from torch.cuda.amp import GradScaler, autocast

# Set the device for training
if opt.gpu_id == '4':
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    print('USE GPU 4')
elif opt.gpu_id == '7':
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    print('USE GPU 7')
cudnn.benchmark = True

# Data paths
image_root = opt.rgb_root
gt_root = opt.gt_root
depth_root = opt.depth_root
edge_root = opt.edge_root
test_image_root = opt.test_rgb_root
test_gt_root = opt.test_gt_root
test_depth_root = opt.test_depth_root
save_path = opt.save_path

# Logging setup
logging.basicConfig(filename=save_path + 'RGBD.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO,
                    filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("DFMG-Train_4_pairs_Improved_PVT")  # 可以更新日志名

# Build the model
model = DFMG()
if opt.load is not None:
    model._load_pre(opt.load)
    print('load model from ', opt.load)

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
scaler = GradScaler()  # For mixed precision training

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Load data
print('load data...')
train_loader = get_loader(image_root, gt_root, depth_root, edge_root,
                          batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_image_root, test_gt_root, test_depth_root, opt.trainsize)
total_step = len(train_loader)

logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))

# Improved loss functions
CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average=True)
step = 0
writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 0
accum_steps = 2  # Gradient accumulation steps


# Improved training function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0

    try:
        for i, (images, gts, depth, edge) in enumerate(train_loader, start=1):
            images = images.cuda()
            gts = gts.cuda()
            depth = depth.repeat(1, 3, 1, 1).cuda()
            edge = edge.cuda()

            # Mixed precision training with special handling for FFT
            with autocast():
                # To avoid FFT issues in half precision, compute model forward in full precision
                with torch.cuda.amp.autocast(enabled=False):
                    p, pg, h = model(images.float(), depth.float())

                # Loss computation remains in mixed precision
                cam_loss = 0.7 * CE(p, gts) + 0.3 * IOU(pg, gts)
                edge_loss = CE(h, edge)
                consistency_loss = F.mse_loss(torch.sigmoid(p), torch.sigmoid(h))  # 建议对h也用sigmoid
                loss = cam_loss + 0.5 * edge_loss + 0.1 * consistency_loss

            # Gradient accumulation
            scaler.scale(loss / accum_steps).backward()

            if (i + 1) % accum_steps == 0:
                clip_gradient(optimizer, opt.clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 100 == 0 or i == total_step or i == 1:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}||sal_loss:{:4f} ||edge_loss:{:4f} ||consistency:{:4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                           optimizer.state_dict()['param_groups'][0]['lr'],
                           cam_loss.data, edge_loss.data, consistency_loss.data))
                logging.info(
                    '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}, sal_loss:{:4f} ||edge_loss:{:4f} ||consistency:{:4f}, mem_use:{:.0f}MB'.
                    format(epoch, opt.epoch, i, total_step,
                           optimizer.state_dict()['param_groups'][0]['lr'],
                           cam_loss.data, edge_loss.data, consistency_loss.data, memory_used))

                writer.add_scalar('Loss', loss.data, global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)

                # --- 核心修改: 将未定义的 's' 替换为模型的主要输出 'p' ---
                res = p[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('res', torch.tensor(res), step, dataformats='HW')

        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if epoch % 5 == 0:
            torch.save(model.state_dict(), save_path + 'DFMG_epoch_{}.pth'.format(epoch))

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'DFMG_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise


# Test function (unchanged)
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()

    # --- 核心修复：在每次测试循环开始前，手动重置数据加载器的内部索引 ---
    test_loader.index = 0
    # ---------------------------------------------------------------

    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            # 您的原始加载方式现在可以安全地工作了
            image, gt, depth, name, img_for_post = test_loader.load_data()

            # (可选但推荐) 增加一个安全检查，以防万一
            if image is None:
                print(f"Warning: test_loader.load_data() returned None at index {i}. Skipping this sample.")
                continue

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.repeat(1, 3, 1, 1).cuda()

            # 使用正确的模型输入和输出来进行测试
            res, resg, h = model(image, depth)

            # --- 修复警告: 将 F.upsample 替换为 F.interpolate ---
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            # ----------------------------------------------------

            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'DFMG_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


# Improved learning rate schedule
def cosine_lr(optimizer, base_lr, epoch, total_epochs, warmup_epochs=5):
    if epoch <= warmup_epochs:
        lr = base_lr * epoch / warmup_epochs
    else:
        lr = base_lr * 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch):
        cur_lr = cosine_lr(optimizer, opt.lr, epoch, opt.epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        test(test_loader, model, epoch, save_path)