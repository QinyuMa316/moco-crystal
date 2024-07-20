
import random
import numpy as np

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data, Dataset, DataLoader

from pymatgen.optimization.neighbors import find_points_in_spheres
from pymatgen.core import Structure
from pymatgen.core.operations import SymmOp

from DefaultElement import DEFAULT_ELEMENTS

import warnings
warnings.filterwarnings("ignore")

def gaussian_expansion(distances, initial=0.0, final=5.0, num_centers=100, width=0.5):
    """
    高斯径向展开函数
    参数:
        bond_dists: 原子间的键（边）距离，形状为[N]，N为距离数量
        initial: 第一个高斯基函数中心的位置
        final: 最后一个高斯基函数中心的位置
        num_centers: 高斯基函数的数量
        width: 高斯基函数的宽度
    返回:
        展开后的距离向量，形状为[N, num_centers]
    """
    centers = torch.linspace(initial, final, num_centers).to(distances.device)# 创建高斯中心
    if width is None:
        width = 1.0 / torch.diff(centers).mean()  # 如果没有指定宽度，使用默认计算方法
    # 计算距离与所有中心之间的差异
    diff = distances[:, None] - centers[None, :]
    # 执行高斯展开
    expanded_distances = torch.exp(-width * (diff ** 2))
    return expanded_distances

def structure_to_graph(structure, cutoff):
    element_types = DEFAULT_ELEMENTS
    numerical_tol = 1.0e-8
    pbc = np.array([1, 1, 1], dtype=int)
    lattice_matrix = structure.lattice.matrix
    cart_coords = structure.cart_coords
    # Find points within spheres
    src_id, dst_id, images, bond_dist = find_points_in_spheres(
        cart_coords, cart_coords, r=cutoff, pbc=pbc, lattice=lattice_matrix, tol=numerical_tol)
    # Exclude self-loops and very close duplicates
    exclude_self = (src_id != dst_id) | (bond_dist > numerical_tol)
    src_id, dst_id, images, bond_dist = src_id[exclude_self], dst_id[exclude_self], images[exclude_self], bond_dist[
        exclude_self]
    # Create edge index
    edge_index = torch.tensor([src_id, dst_id], dtype=torch.long)
    # Create pbc offset
    pbc_offset = torch.tensor(images, dtype=torch.float)
    # Create node type
    node_type = np.array([element_types.index(site.specie.symbol) for site in structure])
    node_type = torch.tensor(node_type, dtype=torch.long)
    # Create fractional coordinates
    frac_coords = torch.tensor(structure.frac_coords, dtype=torch.float)
    # Create positions
    pos = torch.tensor(structure.cart_coords, dtype=torch.float)  # Cartesian coordinates for node positions

    # Create Data object
    data = Data(edge_index=edge_index, pos=pos, x=node_type, frac_coords=frac_coords,
                pbc_offset=pbc_offset)
    # data = Data(edge_index=edge_index, pos=pos, x=node_type, frac_coords=frac_coords, pbc_offset=pbc_offset)

    # compute_pair_vector_and_distance
    node_pos = data.pos
    edge_index = data.edge_index
    src, dst = edge_index[0], edge_index[1]
    vector = node_pos[dst] - node_pos[src] + data.pbc_offset
    distances = vector.norm(dim=1)

    # 新增：在考虑了PBC偏移量之后再次过滤距离
    within_cutoff_after_pbc = distances <= cutoff
    edge_index = edge_index[:, within_cutoff_after_pbc]
    pbc_offset = pbc_offset[within_cutoff_after_pbc]
    distances = distances[within_cutoff_after_pbc]

    # 更新Data对象
    data.edge_index = edge_index
    data.pbc_offset = pbc_offset
    data.edge_attr = distances

    # Apply Gaussian expansion
    edge_attr = gaussian_expansion(distances)
    data.edge_attr = edge_attr
    return data

def generate_augmented_structure(structure: Structure) -> Structure:
    """
    对给定的晶体结构应用旋转、镜像、扰动和扩胞的数据增强技术。
    :param structure: 输入的晶体结构。
    :return: 增强后的晶体结构。
    """
    # 复制结构以避免修改原始数据
    s = structure.copy()

    # 随机选择增强方法
    methods = ['rotation', 'mirroring', 'perturbation', 'supercell']
    chosen_methods = random.sample(methods, k=random.randint(1, len(methods)))  # 随机选择一种或多种方法

    if 'rotation' in chosen_methods:
        # 随机选择旋转轴和角度
        axis = random.choice([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
        angle = random.uniform(0, 360)  # 旋转角度
        s.apply_operation(SymmOp.from_axis_angle_and_translation(axis, angle))

    if 'mirroring' in chosen_methods:
        # 随机选择一个镜像轴
        axis = random.choice([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
        symmop = SymmOp.from_xyz_string(f"-x, y, z" if axis == (1, 0, 0) else "x, -y, z" if axis == (0, 1, 0) else "x, y, -z")
        s.apply_operation(symmop)

    if 'perturbation' in chosen_methods:
        distance = random.uniform(0.01, 0.05)  # 随机扰动距离
        s.perturb(distance)

    if 'supercell' in chosen_methods:
        # 随机选择扩胞倍数
        scale = random.randint(2, 3)  # 选择2到3之间的整数倍扩胞
        s.make_supercell([scale, scale, scale])  # 应用扩胞操作
    return s

class CrystalDataset(Dataset):
    def __init__(self, structures, labels, augmentation=True, transform=None):
        self.structures = structures
        self.labels = labels
        self.augmentation = augmentation
        self.transform = transform
        self._indices = range(len(self.structures))
    def len(self):
        return len(self.structures)

    def get(self, idx):
        structure = self.structures[idx]
        if self.augmentation:
            augmented_structure = generate_augmented_structure(structure)
            data = structure_to_graph(structure=augmented_structure, cutoff=5.0)
        else:
            data = structure_to_graph(structure=structure, cutoff=5.0)
        if self.labels:
            labels = torch.tensor([self.labels[idx]], dtype=torch.float)
            return data, labels
        return data

class CrystalDataLoader(object):
    def __init__(self, structures, labels, batch_size, num_workers,
                 valid_size = 0.0, test_size = 0.0, augmentation = True):
        super(object, self).__init__()
        self.structures = structures
        self.labels = labels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.augmentation = augmentation

    def get_train_validation_data_loaders(self, dataset):
        # obtain training indices that will be used for validation
        num_dataset = len(dataset)
        indices = list(range(num_dataset))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_dataset))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        valid_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        return train_loader, valid_loader

    def get_train_validation_test_data_loaders(self, dataset):
        num_train = len(dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        test_split = int(np.floor(self.test_size * num_train))
        valid_split = test_split + int(np.floor(self.valid_size * (num_train - test_split)))

        train_idx, valid_idx, test_idx = indices[valid_split:], indices[test_split:valid_split], indices[:test_split]
        # print(f'len of train_idx: {len(train_idx)}')
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        # print(f'len of train_loader: {len(train_loader)}')
        valid_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        test_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=test_sampler,
                                 num_workers=self.num_workers, drop_last=True)

        return train_loader, valid_loader, test_loader

    def get_data_loaders(self):
        dataset = CrystalDataset(structures=self.structures, labels = self.labels, augmentation=self.augmentation)

        if self.valid_size != 0:
            if self.test_size != 0:
                train_loader, valid_loader, test_loader = self.get_train_validation_test_data_loaders(dataset)
                return train_loader, valid_loader, test_loader
            else:
                train_loader, valid_loader = self.get_train_validation_data_loaders(dataset)
                return train_loader, valid_loader
        else:
            dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)
            return dataloader


