from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from pymatgen.core.composition import Element
import json
from tqdm import tqdm
import warnings
import time
warnings.filterwarnings("ignore")


type_ = 'P2'
num_tm_ = '3'

def na_place_doping(struct_dict):
    # 定义元素替换
    element_to_replace = 'Na'
    na_list = ['Mg', 'K', 'Ca']
    # TODO na place doping改成包含原始没有进行na doping的结构
    new_struct_dict = struct_dict.copy()
    # 遍历字典中的每个结构
    # for key, structure in struct_dict.items():
    for key, structure in tqdm(struct_dict.items()):
        new_structure = structure.copy()
        # 找到所有Na原子的索引
        na_indices = [i for i, site in enumerate(new_structure) if site.species_string == element_to_replace]
        for replacement_element in na_list:
            # 生成所有可能的Na替换为Mg的结构
            all_structures = []
            for i in na_indices:
                temp_structure = new_structure.copy()
                temp_structure[i] = Element(replacement_element)
                all_structures.append(temp_structure)
            # 使用StructureMatcher去除重复结构
            matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, comparator=ElementComparator())
            grouped_structures = matcher.group_structures(all_structures)
            # 提取唯一结构并添加到总字典中
            unique_structures = [group[0] for group in grouped_structures]
            for struct in unique_structures:
                # new dict_key special for p2
                #　Na_(4.5, 0.9, 2.7)_Na_(7.5, 0.9, 2.7)_Na_(-0.0, 1.7, 8.0)_Na_(3.0, 1.7, 8.0)_Na_(6.0, 1.7, 8.0)
                # _Ni_Ni_Ni_Ni_Ni_Ni_O_O_O_O_O_O_O_O_O_O_O_O
                # dict_key = '_'.join([site.species_string for i, site in enumerate(struct)])
                dict_key = '_'.join([
                    f"Na_({', '.join([f'{coord:.1f}' for coord in site.coords])})" if site.species_string == 'Na' else site.species_string
                    for site in struct
                ])
                new_struct_dict[dict_key] = struct
    return new_struct_dict

def read_entry_dict_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        entry_dict = json.load(file)
    # 用一个新的字典存储转换后的结构
    struct_dict = {}
    for key, entry in entry_dict.items():
        struct_dict[key] = Structure.from_dict(entry['structure'])
        # updated_entry_dict[key]['structure'] = Structure.from_dict(entry['structure'])
    return struct_dict


if __name__ == '__main__':

    # entry_dict[dict_key] =
    # {'structure','formula','na_place_composition',
    # 'tm_place_composition', 'average_voltage', 'capacity_grav'}

    file_to_process = f'{type_}_NaTMO2_{num_tm_}tm_props_top20.json'
    struct_dict = read_entry_dict_from_json(file_to_process)
    print(f'num of {len(struct_dict)} entries have been loaded')
    start_time = time.time()
    new_struct_dict = na_place_doping(struct_dict)

    # save
    for key, struct in new_struct_dict.items():
        # print(type(struct))
        new_struct_dict[key] = struct.as_dict()

    file_to_save = f'{type_}_NaTMO2_{num_tm_}tm_props_top20_na_doping.json'
    with open(file_to_save, 'w') as f:
        json.dump(new_struct_dict, f, indent=4)
        print(f'{len(new_struct_dict)} entries saved to {file_to_save}')
    # new_struct_dict [dict_key] = struct_dict
    end_time = time.time()
    duration = (end_time - start_time)/60
    print(f'total duration: {duration:.2f} minutes')

