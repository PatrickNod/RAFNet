import h5py
import time
import torch
import yaml
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from caac import HWViT
import scipy.io as sio
import math
import os
from thop import profile
from torchinfo import summary
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def learning_rate_function(x, lr_max):
    """学习率衰减函数"""
    return lr_max * math.exp((-1) * (math.log(2)) / 90 * x)


def validate(model, val_loader, criterion, device, ratio):
    """验证函数"""
    model.eval()
    val_loss = []
    with torch.no_grad():
        for batch_idx, (pan, gt, ms, lms) in enumerate(val_loader):
            # DataParallel 会自动分发数据，这里只需确保数据在主设备上即可
            pan, gt, ms, lms = pan.to(device), gt.to(device), ms.to(device), lms.to(device)
            output = model(pan=pan, ms=ms, lms=lms)
            loss = criterion(output, gt)
            val_loss.append(float(loss))

    avg_val_loss = sum(val_loss) / len(val_loss)
    return avg_val_loss


def split_test(model, device, ratio, size, pad, main_folder, name, path):
    """分块测试函数，用于大图推理"""
    file_path = path
    with h5py.File(file_path, 'r') as f:
        datasets = f.keys()
        for dataset_name in datasets:
            dataset = f[dataset_name]
            data = dataset[:]
            if dataset_name == "pan":
                test_pan = torch.from_numpy(np.array(data, dtype=np.float32)) / ratio
            elif dataset_name == "ms":
                test_ms = torch.from_numpy(np.array(data, dtype=np.float32)) / ratio
            elif dataset_name == "lms":
                test_lms = torch.from_numpy(np.array(data, dtype=np.float32)) / ratio

    image_num, C, h, w = test_ms.shape
    _, _, H, W = test_pan.shape
    cut_size = size  # must be divided by 4, we recommend 64
    ms_size = cut_size // 4
    pad = pad  # must be divided by 4
    edge_H = cut_size - (H - (H // cut_size) * cut_size)
    edge_W = cut_size - (W - (W // cut_size) * cut_size)

    new_folder_1 = os.path.join(main_folder, name)
    os.makedirs(new_folder_1, exist_ok=True)

    for k in range(image_num):
        with torch.no_grad():
            x1, x2, x3 = test_ms[k, :, :, :], test_pan[k, 0, :, :], test_lms[k, :, :, :]
            x1 = x1.cpu().unsqueeze(dim=0).float()
            x2 = x2.cpu().unsqueeze(dim=0).unsqueeze(dim=1).float()
            x3 = x3.cpu().unsqueeze(dim=0).float()

            x1_pad = torch.zeros(1, C, h + pad // 2 + edge_H // 4, w + pad // 2 + edge_W // 4)
            x2_pad = torch.zeros(1, 1, H + pad * 2 + edge_H, W + pad * 2 + edge_W)
            x3_pad = torch.zeros(1, C, H + pad * 2 + edge_H, W + pad * 2 + edge_W)
            x1 = torch.nn.functional.pad(x1, (pad // 4, pad // 4, pad // 4, pad // 4), 'reflect')
            x2 = torch.nn.functional.pad(x2, (pad, pad, pad, pad), 'reflect')
            x3 = torch.nn.functional.pad(x3, (pad, pad, pad, pad), 'reflect')

            x1_pad[:, :, :h + pad // 2, :w + pad // 2] = x1
            x2_pad[:, :, :H + pad * 2, :W + pad * 2] = x2
            x3_pad[:, :, :H + pad * 2, :W + pad * 2] = x3
            output = torch.zeros(1, C, H + edge_H, W + edge_W)

            scale_H = (H + edge_H) // cut_size
            scale_W = (W + edge_W) // cut_size
            for i in range(scale_H):
                for j in range(scale_W):
                    MS = x1_pad[:, :, i * ms_size: (i + 1) * ms_size + pad // 2,
                         j * ms_size: (j + 1) * ms_size + pad // 2].to(device)
                    PAN = x2_pad[:, :, i * cut_size: (i + 1) * cut_size + 2 * pad,
                          j * cut_size: (j + 1) * cut_size + 2 * pad].to(device)
                    LMS = x3_pad[:, :, i * cut_size: (i + 1) * cut_size + 2 * pad,
                          j * cut_size: (j + 1) * cut_size + 2 * pad].to(device)
                    sr = model(pan=PAN, ms=MS, lms=LMS)
                    sr = torch.clamp(sr, 0, 1)
                    output[:, :, i * cut_size: (i + 1) * cut_size, j * cut_size: (j + 1) * cut_size] = \
                        sr[:, :, pad: cut_size + pad, pad: cut_size + pad] * ratio
            output = output[:, :, :H, :W]
            output = torch.squeeze(output).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
            new_path = os.path.join(new_folder_1, f'output_mulExm_{k}.mat')
            sio.savemat(new_path, {f'sr': output})
        print(f"Processed image {k}")


def main():
    # 设置随机种子
    SEED = 1
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    start_time = time.time()
    
    # 设备检测与设置
    if torch.cuda.is_available():
        # 这里默认主设备是 cuda:0，DataParallel 会处理后续
        device = torch.device("cuda:0")
        gpu_count = torch.cuda.device_count()
        print(f"Using device: {device} with {gpu_count} GPUs")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")

    # ===================================================================
    # 续训配置区
    # 如果需要续训，请在这里填入权重文件的路径，例如："checkpoints/CAAC_epoch_50.pth"
    # 如果从头开始训练，请保持设为 None
    # ===================================================================
    resume_weight_path = "/root/CAACNet/checkpoints/CAAC_epoch_330.pth" 

    # 加载超参数
    with open("super_para.yml") as f:
        super_para_dict = yaml.safe_load(f)
        lr_max = super_para_dict["lr_max"]
        
        # 强制设置 Epochs 为 400
        epochs = 400 
        print(f"Training Epochs set to: {epochs} (Overriding yaml config)")
        
        batch_size = super_para_dict["batch_size"]
        ratio = super_para_dict['ratio']
        weight_decay = super_para_dict['weight_decay']
        pan_target_channel = super_para_dict['pan_target_channel']
        ms_target_channel = super_para_dict['ms_target_channel']
        head_channel = super_para_dict['head_channel']
        dropout = super_para_dict['dropout']

    # 加载训练数据
    print("Loading training data...")
    file_path = "/root/autodl-tmp/Dataset/WV3/WorldView3/training_wv3/train_wv3.h5"
    with h5py.File(file_path, 'r') as f:
        datasets = f.keys()
        for dataset_name in datasets:
            dataset = f[dataset_name]
            data = dataset[:]
            if dataset_name == "gt":
                gt = torch.from_numpy(np.array(data, dtype=np.float32))
            elif dataset_name == "pan":
                pan = torch.from_numpy(np.array(data, dtype=np.float32))
            elif dataset_name == "ms":
                ms = torch.from_numpy(np.array(data, dtype=np.float32))
            elif dataset_name == "lms":
                lms = torch.from_numpy(np.array(data, dtype=np.float32))
    print("Training data loaded successfully")

    # 加载验证数据
    print("Loading validation data...")
    val_file_path = "/root/autodl-tmp/Dataset/WV3/WorldView3/training_wv3/valid_wv3.h5"
    with h5py.File(val_file_path, 'r') as f:
        datasets = f.keys()
        for dataset_name in datasets:
            dataset = f[dataset_name]
            data = dataset[:]
            if dataset_name == "gt":
                val_gt = torch.from_numpy(np.array(data, dtype=np.float32))
            elif dataset_name == "pan":
                val_pan = torch.from_numpy(np.array(data, dtype=np.float32))
            elif dataset_name == "ms":
                val_ms = torch.from_numpy(np.array(data, dtype=np.float32))
            elif dataset_name == "lms":
                val_lms = torch.from_numpy(np.array(data, dtype=np.float32))
    print("Validation data loaded successfully")

    _, c, H, W = pan.shape
    _, C, h, w = ms.shape

    # 创建数据加载器
    # 注意：在 DataParallel 中，Batch Size 是全局的。
    # 比如 batch_size=32，4张卡，每张卡会处理 8 个样本。
    train_ds = TensorDataset(pan / ratio, gt / ratio, ms / ratio, lms / ratio)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    val_ds = TensorDataset(val_pan / ratio, val_gt / ratio, val_ms / ratio, val_lms / ratio)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # 创建模型
    print("Initializing CAAC model...")
    model = HWViT(
        L_up_channel=ms.shape[1],
        pan_channel=pan.shape[1],
        ms_target_channel=ms_target_channel,
        pan_target_channel=pan_target_channel,
        head_channel=head_channel,
        dropout=dropout
    )
    
    # 多卡并行包装
    model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Activating DataParallel on {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model_optimizer = optim.Adam(
        [{'params': (p for name, p in model.named_parameters() if 'bias' not in name), 'weight_decay': weight_decay},
         {'params': (p for name, p in model.named_parameters() if 'bias' in name)}],
        lr=lr_max
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {num_params:,}')

    # ===================================================================
    # 断点续训核心逻辑
    # ===================================================================
    start_epoch = 1
    best_val_loss = float('inf')
    
    if resume_weight_path and os.path.isfile(resume_weight_path):
        print(f"\n=> 正在加载预训练权重: '{resume_weight_path}'")
        checkpoint = torch.load(resume_weight_path, map_location=device)
        
        # 1. 加载模型权重 (处理了 DataParallel 保存时去除 module. 前缀的情况)
        if isinstance(model, nn.DataParallel):
            # 如果当前模型是 DataParallel 包装过的，但保存的权重没有 'module.' 前缀
            if not list(checkpoint['model'].keys())[0].startswith('module.'):
                model.module.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint['model'])
        else:
            # 如果当前模型是单卡，但保存的权重有 'module.' 前缀（在多卡上保存的）
            if list(checkpoint['model'].keys())[0].startswith('module.'):
                # 移除 'module.' 前缀
                new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(checkpoint['model'])
            
        # 2. 加载优化器状态
        model_optimizer.load_state_dict(checkpoint['model_optimizer'])
        
        # 3. 恢复 epoch 计数
        start_epoch = checkpoint['epoch'] + 1
        
        # 4. 恢复历史最优验证集 loss
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
            
        remaining_epochs = epochs - start_epoch + 1
        print(f"=> 权重加载成功！历史训练至第 {checkpoint['epoch']} 个 epoch。")
        print(f"=> 目标总 Epoch: {epochs}，剩余需训练 Epoch 数量: {max(0, remaining_epochs)}\n")
    elif resume_weight_path:
        print(f"\n=> 警告: 找不到权重文件 '{resume_weight_path}'，将从头开始训练。\n")

    # 定义损失函数
    criterion = nn.L1Loss(reduction='mean')

    # 训练历史记录
    train_loss_history = []
    val_loss_history = []

    # 创建保存目录
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print("Starting training CAAC with Multi-GPU Support...")
    print("=" * 60)

    torch.cuda.empty_cache()

    # 从 start_epoch 开始训练
    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()

        # 训练阶段
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{epochs} [Train]", ncols=100, leave=True)
        for batch_idx, (pan_batch, gt_batch, ms_batch, lms_batch) in enumerate(pbar):
            pan_batch = pan_batch.to(device)
            gt_batch = gt_batch.to(device)
            ms_batch = ms_batch.to(device)
            lms_batch = lms_batch.to(device)

            model_optimizer.zero_grad()
            output = model(pan=pan_batch, ms=ms_batch, lms=lms_batch)
            loss = criterion(output, gt_batch)
            loss.backward()
            model_optimizer.step()

            loss_val = float(loss)
            train_losses.append(loss_val)

            pbar.set_postfix({'loss': f'{loss_val:.6f}'})

        # 更新学习率
        current_lr = learning_rate_function(epoch, lr_max)
        for param_group in model_optimizer.param_groups:
            param_group['lr'] = current_lr

        # 计算平均训练损失
        avg_train_loss = sum(train_losses) / len(train_losses)
        train_loss_history.append(avg_train_loss)

        # 验证阶段
        avg_val_loss = validate(model, val_loader, criterion, device, ratio)
        val_loss_history.append(avg_val_loss)

        epoch_time = time.time() - epoch_start_time

        print(f"Epoch [{epoch:3d}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.2f}s")

        # 保存模型时的处理 (Unwrap DataParallel)
        # 这样保存的权重可以在单卡/CPU上直接加载，通用性更强
        if isinstance(model, nn.DataParallel):
            model_state_to_save = model.module.state_dict()
        else:
            model_state_to_save = model.state_dict()

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint_path = os.path.join("checkpoints", f'CAAC_best.pth')
            torch.save({
                'model': model_state_to_save,
                'model_optimizer': model_optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss
            }, best_checkpoint_path)
            print(f"  → Best model saved (Val Loss: {best_val_loss:.6f})")

        # 定期保存检查点
        if epoch % 10 == 0 or epoch == epochs:
            checkpoint_path = os.path.join("checkpoints", f'CAAC_epoch_{epoch}.pth')
            torch.save({
                'model': model_state_to_save,
                'model_optimizer': model_optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, checkpoint_path)
            print(f"  → Checkpoint saved at epoch {epoch}")

    end_time = time.time()
    total_training_time = end_time - start_time
    print("=" * 60)
    print(f"Training completed! Total time: {total_training_time:.2f}s ({total_training_time/60:.2f} min)")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("=" * 60)

    # 保存训练历史
    history_path = os.path.join("checkpoints", "training_history.yml")
    history_data = {
        "train_loss": [float(x) for x in train_loss_history],
        "val_loss": [float(x) for x in val_loss_history],
        "total_training_time": total_training_time,
        "best_val_loss": float(best_val_loss),
        "epochs": epochs
    }
    with open(history_path, 'w') as file:
        yaml.dump(history_data, file, default_flow_style=False)
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()