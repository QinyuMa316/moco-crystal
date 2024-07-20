import time
import pickle
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch import optim
import os

import matplotlib.pyplot as plt
from dataset_crystal_ssl_disk import CrystalDataLoader
# from _gcn_crystal_residual_1x_after_gcn_moco import GCN
from moco_builder import MoCo
from gatmodel_ssl import GATModel

def check_model_param(model):
    total_params = 0
    for name, parameter in moco_model.named_parameters():
        if parameter.requires_grad:
            param_size = parameter.numel()  # 获取参数的总数量
            total_params += param_size
            print(f"{name}: {param_size}")
    print(f"Total Trainable Params: {total_params}")

if __name__ == '__main__':


    folder_path = f'crystal_structures'
    batch_size = 32  # N: 256/8=32
    dataloader = CrystalDataLoader(folder_path=folder_path,
                                   batch_size=batch_size, num_workers=10,
                                   augmentation=True)
    # dataloader = CrystalDataLoader(structure_list=structures_list, valid_size=0.1,...)

    dataloader = dataloader.get_data_loaders()
    print(f'num of data for SSL: {len(dataloader) * batch_size}.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_encoder = GATModel(num_layers=10,dropout=0.1).to(device)
    moco_model = MoCo(base_encoder, dim=64, K=16384, m=0.999, T=0.07, mlp=False).to(device)  # K: 65536
    # K: 65536/8=8192
    # check_model_param(moco_model)

    # ----------------------------------------------------------
    checkpoint_folder = "checkpoints_crystal"
    os.makedirs(checkpoint_folder, exist_ok=True)

    checkpoint_filename_old = 'best_model_checkpoint_1e4_aug1.pth'

    learning_rate = 1e-4
    checkpoint_filename_new = 'best_model_checkpoint_1e4_aug1.pth'
    # ----------------------------------------------------------

    model_save_path_old = f'{checkpoint_folder}/{checkpoint_filename_old}'
    # 尝试加载模型
    try:
        moco_model.load_state_dict(torch.load(model_save_path_old))
        print("模型加载成功，继续训练...")
    except FileNotFoundError:
        print("未找到保存的模型，从头开始训练...")

    num_epochs = 5
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = Adam(moco_model.parameters(), lr=learning_rate, weight_decay=1e-3)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0)

    best_loss = float('inf')
    patience = 10
    patience_counter = 0

    print(f'Start training with MoCo v2!\nlearning rate:{learning_rate}')
    avg_train_losses = []

    for epoch in range(num_epochs):
        start_time = time.time()
        moco_model.train()
        total_train_loss = 0
        for data in tqdm(dataloader):
            data_q, data_k = data  # 假设train_loader返回的是成对的数据
            data_q = data_q.to(device)
            data_k = data_k.to(device)
            optimizer.zero_grad()
            logits, labels = moco_model(data_q, data_k)
            loss = loss_fn(logits, labels)

            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        # scheduler.step()
        avg_train_loss = total_train_loss / len(dataloader)
        avg_train_losses.append(avg_train_loss)

        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f'Epoch: {epoch}/{num_epochs}, train loss: {avg_train_loss:.4f}, '
              f'epoch duration: {epoch_duration / 60:.3f} min. ')

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            torch.save(moco_model.state_dict(), f'{checkpoint_folder}/{checkpoint_filename_new}')
            patience_counter = 0  # Reset patience counter after improvement
        else:
            patience_counter += 1
            print(f'patience counter: {patience_counter}')

        if patience_counter >= patience:
            print(f"No improvement in {patience} epochs, stopping training.")
            break

    plt.plot(avg_train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Train Loss')
    plt.title('Train Loss vs. Epochs')
    plt.legend()
    plt.savefig('train_loss_vs_epochs.png')  # 将曲线图保存为PNG文件
    plt.show()  # 显示曲线图
    '''
    aug1: atom_mask: 0.1
    
    
    '''
