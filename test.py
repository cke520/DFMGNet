import torch
import torch.nn.functional as F
import sys
import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm  # 新增进度条库
from models.DFMG import DFMG
from tools.data import test_dataset

sys.path.append('./models')

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path', type=str,
                    default='../Data/COD/test/',
                    help='test dataset path')
opt = parser.parse_args()

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.benchmark = True

# 模型初始化
model = DFMG().to(device)
model.load_state_dict(torch.load('./checkpoints/ckpt/DFMG_epoch_best.pth', map_location=device))
model.eval()

# 测试数据集配置
test_datasets = ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']
save_parent = './final/COD_result/'

for dataset in test_datasets:
    dataset_save_path = os.path.join(save_parent, dataset)
    os.makedirs(dataset_save_path, exist_ok=True)
    os.makedirs(os.path.join(dataset_save_path, 'edge'), exist_ok=True)

    # 路径配置
    data_paths = {
        'image': os.path.join(opt.test_path, dataset, 'RGB') + '/',
        'gt': os.path.join(opt.test_path, dataset, 'GT') + '/',
        'depth': os.path.join(opt.test_path, dataset, 'depth') + '/'
    }

    # 路径验证
    for path_type, path in data_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"关键路径缺失: {path}")

    test_loader = test_dataset(
        image_root=data_paths['image'],
        gt_root=data_paths['gt'],
        depth_root=data_paths['depth'],
        testsize=opt.testsize
    )

    # 添加带进度条的循环
    progress_bar = tqdm(
        total=test_loader.size,
        desc=f'Processing {dataset}',
        unit='img',
        ncols=100  # 调整进度条宽度
    )

    for _ in range(test_loader.size):
        try:
            # 数据加载
            image, gt, depth, name, _ = test_loader.load_data()

            # GT处理
            if not isinstance(gt, np.ndarray):
                gt = np.array(gt)
            gt = gt.astype(np.float32)
            if gt.ndim == 3 and gt.shape[2] == 3:
                gt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
            gt = gt.squeeze()

            # 设备转移
            image = image.to(device)
            depth = depth.repeat(1, 3, 1, 1).to(device)

            # 模型推理
            with torch.no_grad():
                res, resg, edge = model(image, depth)

            # 结果处理
            target_size = gt.shape[-2:] if isinstance(gt, np.ndarray) else (opt.testsize, opt.testsize)

            # 主输出
            res_output = F.interpolate(res, size=target_size, mode='bilinear')
            res_output = res_output.sigmoid().squeeze().cpu().numpy()
            res_output = (res_output - res_output.min()) / (res_output.max() - res_output.min() + 1e-8)
            cv2.imwrite(os.path.join(dataset_save_path, name), res_output * 255)

            # 边缘输出
            edge_output = F.interpolate(edge, size=target_size, mode='bilinear')
            edge_output = edge_output.squeeze().cpu().numpy()
            edge_output = (edge_output - edge_output.min()) / (edge_output.max() - edge_output.min() + 1e-8)
            cv2.imwrite(os.path.join(dataset_save_path, 'edge', name), edge_output * 255)

            # 更新进度条描述
            progress_bar.set_postfix({'Current': name})

        except Exception as e:
            error_msg = f"处理异常: {str(e)}"
            if 'name' in locals():
                error_msg = f"处理图像 {name} 时出错: {str(e)}"
                progress_bar.write(error_msg)  # 使用进度条自带的输出方法
            continue
        finally:
            progress_bar.update(1)  # 确保无论如何都更新进度

    progress_bar.close()
    print(f'{dataset} 处理完成')

print('所有数据集处理完毕')