import time
from collections import Counter
from fractions import Fraction
from pymatgen.core import Structure
# pip install chgnet==0.3.2
import warnings
import json
from tqdm import tqdm
warnings.filterwarnings("ignore")

type_ = 'P2'
num_tm_ = '3'


# ------------------------------------------------------------

from dataset_crystal_sl import CrystalDataLoader
from gatmodel2 import GATModel
import torch

def get_property_predictions(strucuture_list):
    dataloader = CrystalDataLoader(structures=strucuture_list, labels=None, batch_size=1, num_workers=5,
                                   augmentation=False).get_data_loaders()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    properties = ['average_voltage', 'capacity_grav'] #, 'energy_grav']
    all_predictions_dict = {}
    for property in properties:
        model = GATModel(num_layers=10).to(device)
        model_save_path_old = f'check_point/check_point_{property}.pth'
        model.load_state_dict(torch.load(model_save_path_old, map_location=device))
        model.eval()
        predictions = []
        with torch.no_grad():
            # for data in dataloader:
            for data in tqdm(dataloader, desc=f"Predicting {property}"):  # 使用tqdm显示进度条
                data = data.to(device)
                output, feat = model(data)
                prediction = output.cpu().numpy().item()
                predictions.append(prediction)
        all_predictions_dict[property] = predictions
    return all_predictions_dict


# ------------------------------------------------------------

def read_entry_dict_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        entry_dict = json.load(file)
    new_entry_dict = {}
    for key, entry in entry_dict.items():
        new_entry_dict[key] = {'structure' : Structure.from_dict(entry)}
    return new_entry_dict

def save_entry_dict_to_json(entry_dict, filename):
    for key, entry in entry_dict.items():
        entry['structure'] = entry['structure'].as_dict()
        # 对 entry_dict 进行修改时，直接对其结构进行修改可能会对其他引用到该字典的地方产生影响
    with open(filename, 'w') as f:
        json.dump(entry_dict, f, indent=4)
        print(f'{len(entry_dict)} entries saved to {filename}')


if __name__ == '__main__':
    # new_unique_structures_dict = {}
    # files = ['P2_NaNiO2_supercell_doping_1.cif','P2_NaNiO2_supercell_doping_2.cif']
    # for file in files:
    #     struct = Structure.from_file(file)
    #     dict_key = '_'.join([str(sp) for sp in struct.species])
    #     new_unique_structures_dict[dict_key] = struct

    filename = f'{type_}_NaTMO2_{num_tm_}tm_doping.json'
    entry_dict = read_entry_dict_from_json(filename)
    print(f'Number of entries: {len(entry_dict)}')

    # [1] get formula and element composition
    start_time = time.time()
    structure_list = []
    new_entry_dict = {}
    for key, entry in entry_dict.items():
        # formula, na_place_composition, tm_place_composition = transform_formula(key)
        struct = entry['structure']
        structure_list.append(struct)
        new_entry_dict[key] = {'structure': struct}

    end_time = time.time()
    total_duration = (end_time - start_time)/60
    print(f'finish getting formula, total duration: {total_duration:.2f} min')

    # [3] get prediction
    start_time = time.time()
    all_predictions_dict = get_property_predictions(structure_list)
    for i, key in enumerate(new_entry_dict.keys()):
        for property in all_predictions_dict:
            new_entry_dict[key][property] = all_predictions_dict[property][i]
    end_time = time.time()
    total_duration = (end_time - start_time) / 60
    print(f'finish getting battery properties predictions, '
          f'total duration: {total_duration:.2f} min')

    # [4] save files
    file_to_save = f'{type_}_NaTMO2_{num_tm_}tm_props.json'
    save_entry_dict_to_json(entry_dict=new_entry_dict,
                            filename=file_to_save)
    # [5] 获取 average_voltage & gravimetric_capacity top25% entries
    with open(file_to_save, 'r', encoding='utf-8') as file:
        entry_dict_props = json.load(file)

    # 140896 (tm_place_doping_3)
    # 提取同时满足前25% 'average_voltage' 和 'capacity_grav' 的条目
    def extract_top_entry(entry_dict, properties):
        top_25_percent_keys_sets = []
        for property_name in properties:
            sorted_entries = sorted(entry_dict.items(), key=lambda x: x[1][property_name], reverse=True)
            top_25_percent_count = int(len(sorted_entries) * 0.25)
            top_25_percent_keys = set(entry[0] for entry in sorted_entries[:top_25_percent_count])
            top_25_percent_keys_sets.append(top_25_percent_keys)
        combined_top_25_percent_keys = set.intersection(*top_25_percent_keys_sets)
        top_entry_dict = {key: entry_dict[key] for key in combined_top_25_percent_keys}
        return top_entry_dict

    top_entry_dict = extract_top_entry(entry_dict = entry_dict_props,
                                       properties=['average_voltage', 'capacity_grav'])

    file_to_save_filtered = f'{type_}_NaTMO2_{num_tm_}tm_props_top25.json'
    with open(file_to_save_filtered, 'w') as f:
        json.dump(top_entry_dict, f, indent=4)
    print(f'num of {len(top_entry_dict)} top 25% entries saved to {file_to_save_filtered}')
    '''
    entry_dict:
    'structure'
    'average_voltage', 
    'capacity_grav'
    # 'formula'
    # 'na_place_composition'
    # 'tm_place_composition'
    '''


'''
# 元素周期表顺序
periodic_table_order = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                        'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
                        'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
                        'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
                        'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
                        'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
                        'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

# 获取元素在周期表中的位置
def get_element_index(element):
    try:
        return periodic_table_order.index(element)
    except ValueError:
        return len(periodic_table_order)


def get_element_composition(elements):
    element_set = set(elements)
    return '-'.join(sorted(element_set, key=get_element_index))

def transform_formula(formula, normalization_factor=6):
    # 拆分字符串为列表
    elements = formula.split('_')
    # 提取最后12个元素
    o_elements = elements[-12:]
    # 提取中间部分的6个元素
    middle_elements = elements[-18:-12]
    # 提取剩余的元素
    remaining_elements = elements[:-18]
    # 获取元素组成
    na_place_composition = get_element_composition(remaining_elements)
    tm_place_composition = get_element_composition(middle_elements)
    # 将最后12个元素转化为字符串
    o_str = 'O' + str(len(o_elements) // normalization_factor)
    # 合并中间部分和剩余部分的元素，并进行归一化
    middle_str = '[' + merge_elements(middle_elements, normalization_factor) + ']'
    remaining_str = '[' + merge_elements(remaining_elements, normalization_factor) + ']'
    # 生成最终的公式
    if type_ == 'P2':
        transformed_formula = 'P2-' + remaining_str + middle_str + o_str
    elif type_ == 'O3':
        transformed_formula = 'O3-' + remaining_str + middle_str + o_str
    return transformed_formula, na_place_composition, tm_place_composition

def merge_elements(elements, normalization_factor):
    counter = Counter(elements)
    # TODO: SORT
    sorted_elements = sorted(counter.items(), key=lambda x: get_element_index(x[0]))
    merged_str = ''.join(f'{element}{Fraction(count, normalization_factor)}' for element, count in sorted_elements)
    return merged_str

for key, entry in entry_dict.items():
    formula, na_place_composition, tm_place_composition = transform_formula(key)
'''


