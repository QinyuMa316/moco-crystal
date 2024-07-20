import time
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.nn import MSELoss

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from dataset_crystal_sl import CrystalDataLoader

from gatmodel1 import GATModel
import pickle
import warnings
warnings.filterwarnings("ignore")
def set_seed(seed=42):
    random.seed(seed)        # Python random module.
    np.random.seed(seed)     # Numpy module.
    torch.manual_seed(seed)  # PyTorch.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)         # Seeds the RNG for all devices (both CPU and CUDA)
        torch.cuda.manual_seed_all(seed)     # Deprecated: use torch.cuda.manual_seed() instead
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    set_seed(3407)
    #------------------------------------------------
    target = 'average_voltage'
    working_ions = ['Li', 'Na', 'Mg', 'Ca', 'K', 'Zn', 'Al']
    all_structures = []
    all_labels = []
    for working_ion in working_ions:
        filename = f'../data_all/{working_ion}_cathode_{target}.pkl'
        with open(filename, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
        structures = data['structures']
        labels = data[target]
        all_structures += structures
        all_labels += labels
    #------------------------------------------------

    batch_size = 16
    dataloader = CrystalDataLoader(structures=all_structures, labels=all_labels,
                                   valid_size=0.1, test_size=0.1,
                                   batch_size=batch_size, num_workers=5,
                                   augmentation=False)

    train_loader, valid_loader, test_loader = dataloader.get_data_loaders()

    print(f'num of data for training: {len(train_loader)*batch_size},\n'
          f'num of data for validation: {len(valid_loader)*batch_size},\n'
          f'num of data for testing: {len(test_loader)*batch_size}.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GATModel(num_layers=6, dropout=0.2).to(device)
    model_save_path_old = 'best_model_checkpoint_5e4.pth'
    try:
        model.load_state_dict(torch.load(model_save_path_old))
        print("模型加载成功，继续训练...")
    except FileNotFoundError:
        print("未找到保存的模型，从头开始训练...")
    learning_rate = 5e-4
    model_save_path_new = f'best_model_checkpoint_{target}.pth'
    num_epochs = 500
    '''
    without weight decay
    5e4: Test MAE: 0.4942 Test R^2: 0.7458
    5e5: Test MAE: 0.4470 Test R^2: 0.7829
    5e6: Test MAE: 0.4454 Test R^2: 0.7837
    with weight decay
    
    
    '''
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

    patience = 10
    patience_count = 0
    best_valid_loss = float('inf')

    print(f'start training!\nlearning rate:{learning_rate}')

    criterion = MSELoss()
    avg_val_losses = []
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_train_loss = 0
        for data_label in tqdm(train_loader):
            data, labels = data_label
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            output, features = model(data)
            train_loss = criterion(output, labels)
            total_train_loss += train_loss.item()
            train_loss.backward()
            optimizer.step()
        # scheduler.step()
        # current_lr = optimizer.param_groups[0]["lr"]
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for data_label in valid_loader:
                data, labels = data_label
                data, labels = data.to(device), labels.to(device)
                output, features = model(data)
                valid_loss = criterion(output, labels)
                total_valid_loss += valid_loss.item()

        avg_valid_loss = total_valid_loss / len(valid_loader)
        avg_val_losses.append(avg_valid_loss)
        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f'Epoch: {epoch}/{num_epochs}, '
              f'train loss: {avg_train_loss:.4f}, valid loss: {avg_valid_loss:.4f}, '
              f'epoch duration: {epoch_duration / 60:.3f} min. ')

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            patience_count = 0
            torch.save(model.state_dict(), model_save_path_new)
        else:
            patience_count += 1
            print(f'EarlyStopping counter: {patience_count}')
            if patience_count >= patience:
                learning_rate *= 0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
                patience_count = 0
                print(f"Learning rate updated to: {learning_rate}")
                # torch.save(model.state_dict(), model_save_path_new)
                if learning_rate <= 5e-8:
                    print("Early stopping")
                    break

    model.load_state_dict(torch.load(model_save_path_new))

    plt.plot(avg_val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average val Loss')
    plt.title('val Loss vs. Epochs')
    plt.legend()
    plt.savefig('val_loss_vs_epochs.png')  # 将曲线图保存为PNG文件
    plt.show()  # 显示曲线图

    # -----------------------------------------------------------------
    print('start testing!')

    model.eval()
    test_labels = []
    test_predictions = []

    with torch.no_grad():
        for data_label in test_loader:
            data, labels = data_label
            data, labels = data.to(device), labels.to(device)
            output, feat = model(data)
            test_predictions.extend(output.view(-1).tolist())
            test_labels.extend(labels.view(-1).tolist())

    # Calculate MAE, MSE, RMSE, and R^2
    test_predictions = [float(i) for i in test_predictions]
    test_labels = [float(i) for i in test_labels]
    mae = mean_absolute_error(test_labels, test_predictions)
    mse = mean_squared_error(test_labels, test_predictions)
    rmse = mse ** 0.5
    r2 = r2_score(test_labels, test_predictions)
    
    print(f'Test MAE: {mae:.4f}')
    print(f'Test R^2: {r2:.4f}')
    print(f'Test MSE: {mse:.4f}')
    print(f'Test RMSE: {rmse:.4f}')

    plt.figure(figsize=(8, 8))
    plt.scatter(test_labels, test_predictions, color='deepskyblue', edgecolor='black', s=50, alpha=0.8,
                label='Predictions vs. Actual')
    plt.plot([min(test_labels), max(test_labels)], [min(test_labels), max(test_labels)], 'k--', lw=1, label='Ideal fit')
    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predictions', fontsize=12)
    plt.title(f'All Cathode Prediction vs Actual Comparison',
              fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5, color='grey')
    plt.savefig(f'cathode_{target}.png')
    plt.show()



