import torch
import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from gatmodel1 import GATModel
from dataset_crystal_sl import CrystalDataLoader
from matplotlib.font_manager import FontProperties


if __name__ == '__main__':
    # 数据初始化
    target = 'average_voltage'
    # 'average_voltage', 'energy_grav', 'capacity_grav'
    all_structures = []
    all_labels = []

    # Cathode Data
    working_ions = ['Li', 'Na', 'Mg', 'Ca', 'K', 'Zn', 'Al']
    for working_ion in working_ions:
        filename = f'../data_all/{working_ion}_cathode_{target}.pkl'
        with open(filename, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
        structures = data['structures']
        labels = data[target]
        print(f'num of {working_ion} structures loaded: {len(structures)}')
        all_structures.extend(structures)
        all_labels.extend(labels)

    # 模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GATModel(num_layers=10).to(device)
    model_save_path = 'chk_5e6_avgvol.pth'
    model.load_state_dict(torch.load(model_save_path, map_location=device))

    print(f'loading data finished!')

    # 特征提取函数
    def extract_features(loader, model):
        features = []
        model.eval()
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                output, feat = model(data)
                features.append(feat.cpu().numpy())
        return np.concatenate(features)

    # 特征提取和降维
    all_features = []
    dataloader = CrystalDataLoader(all_structures, labels=None,
                                   batch_size=1, num_workers=3,
                                   augmentation=False).get_data_loaders()
    features = extract_features(dataloader, model)
    all_features.append(features)

    all_features = np.concatenate(all_features, axis=0)
    tsne = TSNE(n_components=2, random_state=42)# 123
    tsne_results = tsne.fit_transform(all_features)

    font = FontProperties()
    font.set_size(12)  # 设置字体大小为14
    font.set_family('Arial')  # 设置字体为Arial

    plt.figure(figsize=(7, 5))
    norm = plt.Normalize(min(all_labels), max(all_labels))
    cmap = cm.viridis
    sc = plt.scatter(tsne_results[:, 0], tsne_results[:, 1],
                     c=all_labels, edgecolor='black',
                     cmap=cmap, norm=norm, alpha=0.8) #1不完全透明，0完全透明

    plt.colorbar(sc, label='Average Voltage (V)')
    plt.title(f'Feature Representations - Average Voltage',
              fontsize=14,fontname='Arial')
    plt.xlabel('t-SNE 1', fontsize=12, fontname='Arial')
    plt.ylabel('t-SNE 2', fontsize=12, fontname='Arial')
    plt.savefig(f'cathode_visualization_{target}.png')
    plt.show()
    '''
    target = 'capacity_grav'
    num of Li structures loaded: 2346
    num of Na structures loaded: 303
    num of Mg structures loaded: 402
    num of Ca structures loaded: 425
    num of K structures loaded: 100
    num of Zn structures loaded: 351
    num of Al structures loaded: 84
    '''