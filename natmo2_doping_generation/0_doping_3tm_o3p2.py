import warnings
import pickle
# pip install chgnet==0.3.2
from chgnet.model import StructOptimizer
from itertools import combinations, permutations
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from pymatgen.core.composition import Element
from multiprocessing import Process, Pool, Manager
from tqdm import tqdm
import queue
import time
import json
import os

from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.computed_entries import ComputedStructureEntry
import warnings
warnings.filterwarnings("ignore")

type_ = 'O3'
num_tm_ = '3'


def save_entry_dict_to_json(entry_dict, filename):
    entry_dict_to_save = {}
    for formula, entry in entry_dict.items():
        entry_dict_to_save[formula] = entry.as_dict()
    with open(filename, 'w') as f:
        json.dump(entry_dict_to_save, f, indent = 4)
        print(f'{len(entry_dict)} entries saved to {filename}')


def save_neutral_structures(entry_dict):
    # 创建一个新的字典来保存电中性的结构
    new_entry_dict = {}
    # 遍历所有结构
    for key, entry in entry_dict.items():
        structure = entry
        # 计算电荷总和
        total_charge = 0
        for site in structure:
            for element, occu in site.species.items():
                if hasattr(element, 'oxi_state'):
                    total_charge += element.oxi_state * occu
        # 检查电荷总和是否为0
        if total_charge == 0:
            new_entry_dict[key] = structure
    return new_entry_dict


def get_unique_structures(structure, unique_permutations):
    all_structures = []
    # print(f'unique_permutations : \n{unique_permutations}')
    for perm in unique_permutations:
        new_structure = structure.copy()
        tm_indices = [i for i, site in enumerate(new_structure) if site.species_string == 'Ni']
        for i, tm_index in enumerate(tm_indices):
            new_structure[tm_index] = perm[i]
        all_structures.append(new_structure)
    # print(f'num of all structures: {len(all_structures)}')
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, comparator=ElementComparator())
    grouped_structures = matcher.group_structures(all_structures)
    unique_structures = [group[0] for group in grouped_structures]
    # print(f'num of unique structures: {len(unique_structures)}')
    return unique_structures


def tm_place_doping(tm_combo, base_unique_structures, base_comb):
    new_unique_structures = []
    for struct in base_unique_structures:
        temp_structure = struct.copy()
        for site in temp_structure:
            if site.species_string == base_comb[0]:
                site.species = {Element(tm_combo[0]): 1.0}
            elif site.species_string == base_comb[1]:
                site.species = {Element(tm_combo[1]): 1.0}
            elif site.species_string == base_comb[2]:
                site.species = {Element(tm_combo[2]): 1.0}
        new_unique_structures.append(temp_structure)
    return new_unique_structures

def na_place_vacancy(structure_list):
    na_place_vacancy_structures = []
    for struct in structure_list:
        new_structure = struct.copy()
        # 找到所有Na原子的索引
        na_indices = [i for i, site in enumerate(new_structure) if site.species_string == 'Na']
        all_structures = []
        # 删除所有可能的一个Na原子
        for i in na_indices:
            temp_structure = new_structure.copy()
            temp_structure.remove_sites([i])
            all_structures.append(temp_structure)
        # 使用StructureMatcher去除重复结构
        matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, comparator=ElementComparator())
        grouped_structures = matcher.group_structures(all_structures)
        # 提取唯一结构并添加到总字典中
        unique_structures = [group[0] for group in grouped_structures]
        na_place_vacancy_structures += unique_structures
    return na_place_vacancy_structures


if __name__ == "__main__":

    if type_ == 'P2':
        structure = Structure.from_file('P2_NaNiO2.cif')
        structure.make_supercell([3, 1, 1])
    if type_ == 'O3':
        structure = Structure.from_file('O3_NaNiO2.cif')
        structure.make_supercell([3, 2, 1])
    # Warning! O3 type: 3,2,1 ; P2 type: 3,1,1

    # update: get base unique structure list
    base_tm_list = ['Ar', 'Kr', 'Xe']
    base_tm_combinations = list(combinations(base_tm_list, 3))
    base_unique_structures = []
    for base_tm_combo in base_tm_combinations:
        for first in range(1, 5):  # [1, 4] first=6-1-1,1,1
            for second in range(1, 6 - first):  # [1, 5-first] # first,second=6-1-first,1
                third = 6 - first - second
                tm_list = ([base_tm_combo[0]] * first + [base_tm_combo[1]] * second + [base_tm_combo[2]] * third)
                base_unique_permutations = set(permutations(tm_list))
                unique_structures = get_unique_structures(structure, base_unique_permutations)
                # na place vacancy
                if type_ == 'P2':
                    unique_structures = na_place_vacancy(unique_structures)
                base_unique_structures += unique_structures
    print(f'num of base unique structures: {len(base_unique_structures)}')

    #
    tm_list1 = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd']  # transition metal-redox active
    tm_list2 = ['Li', 'Mg', 'Al', 'Ca', 'In', 'Sn', 'Sb', 'Te', 'Bi']  # redox inactive
    tm_list = tm_list1 + tm_list2
    # tm_combinations = list(combinations(tm_list, 3))
    # 生成包含至少一个 tm_list1 元素的 3 元素组合
    tm_combinations = [combo for combo in combinations(tm_list, 3) if any(tm in tm_list1 for tm in combo)]
    print(f'num of 3 TMs combination : {len(tm_combinations)}')

    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, comparator=ElementComparator())

    start_time = time.time()

    entry_dict = {}
    file_count = 0
    for tm_combo in tqdm(tm_combinations):
        print(f'tm_combination : {tm_combo}')
        unique_structures = tm_place_doping(tm_combo, base_unique_structures, base_comb=base_tm_list)
        # print(f'num of unique structures : {len(unique_structures)}')
        for struct in unique_structures:
            # dict_key = '_'.join([str(sp) for sp in struct.species])
            dict_key = '_'.join([
                f"Na_({', '.join([f'{coord:.1f}' for coord in site.coords])})" if site.species_string == 'Na' else site.species_string
                for site in struct
            ])
            entry_dict[dict_key] = struct
    # [1] tm place doping
    end_time = time.time()
    total_time = (end_time - start_time) / 60
    print(f'tm place doping finished, total duration {total_time:.2f} min. start saving ...')
    save_entry_dict_to_json(entry_dict=entry_dict,
                            filename=f'{type_}_NaTMO2_{num_tm_}tm_doping.json')


