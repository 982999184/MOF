import torch
import numpy as np
import linecache
import math
import warnings
import os
import csv
from torch.autograd import Variable
from torch import nn
from tqdm import tqdm
from colorama import Fore


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def Res18(x_name):
    model = torch.load('model_sort_sel.pht')
    model = model.module.eval()
    model = nn.DataParallel(model)
    model.to(device)
    warnings.filterwarnings("ignore")
    vector_size = 223
    result_in = []
    for filename in tqdm(iterable=x_name, bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.BLUE, Fore.RESET)):
        f_filename = path + "\\" + filename
        i = 1
        while linecache.getline(f_filename, i).split()[0] != '_cell_length_a':
            i += 1
            if i > 10000:
                print(filename, 'count')
                break
        matrix_a = np.zeros([vector_size + 1, vector_size + 1], dtype=int)
        matrix_b = np.zeros([vector_size + 1, vector_size + 1], dtype=int)
        matrix_c = np.zeros([vector_size + 1, vector_size + 1], dtype=int)
        data_a = float(linecache.getline(f_filename, i).split()[1])
        data_b = float(linecache.getline(f_filename, i + 1).split()[1])
        data_c = float(linecache.getline(f_filename, i + 2).split()[1])
        data_alpha = float(linecache.getline(f_filename, i + 3).split()[1])
        data_beta = float(linecache.getline(f_filename, i + 4).split()[1])
        data_gamma = float(linecache.getline(f_filename, i + 5).split()[1])
        alpha_neg = 1 if data_alpha > 90 else -1
        beta_neg = 1 if data_beta > 90 else -1
        gamma_neg = 1 if data_gamma > 90 else -1
        max_a_xy = max(data_a + data_b * abs(math.cos(math.radians(data_gamma))),
                       data_b * math.sin(math.radians(data_gamma)))
        max_b_yz = max(data_b + data_c * abs(math.cos(math.radians(data_alpha))),
                       data_c * math.sin(math.radians(data_alpha)))
        max_c_xz = max(data_c + data_a * abs(math.cos(math.radians(data_beta))),
                       data_a * math.sin(math.radians(data_beta)))

        atom_map = {'P': 1, 'Ba': 2, 'Mg': 3, 'Mn': 4, 'Zr': 5, 'Cd': 6, 'Ni': 7, 'In': 7,
                    'Cr': 9, 'Co': 10, 'Fe': 11, 'I': 12, 'Br': 13, 'V': 14, 'S': 15, 'Cl': 16,
                    'F': 17, 'Zn': 18, 'Cu': 19, 'N': 20, 'H': 21, 'C': 22, 'O': 23}

        i += 16
        while linecache.getline(f_filename, i) != '\n':
            temp_line = linecache.getline(f_filename, i)
            atom = temp_line.split()[1]
            temp_x = float(temp_line.split()[3]) if float(temp_line.split()[3]) > 0 else float(temp_line.split()[3]) + 1
            temp_y = float(temp_line.split()[4]) if float(temp_line.split()[4]) > 0 else float(temp_line.split()[4]) + 1
            temp_z = float(temp_line.split()[5]) if float(temp_line.split()[5]) > 0 else float(temp_line.split()[5]) + 1
            i += 1
            xy_x = int(
                (temp_x * data_a + data_b * temp_y * math.cos(math.radians(data_gamma))) * vector_size / max_a_xy)
            xy_x_fix = int(
                (temp_x * data_a + (temp_y - 1) * data_b * math.cos(math.radians(data_gamma))) * vector_size / max_a_xy)
            xy_y = int(temp_y * data_b * math.sin(math.radians(data_gamma)) * vector_size / max_a_xy)

            yz_x = int(
                (temp_y * data_b + data_c * temp_z * math.cos(math.radians(data_alpha))) * vector_size / max_b_yz)
            yz_x_fix = int(
                (temp_y * data_b + (temp_z - 1) * data_c * math.cos(math.radians(data_alpha))) * vector_size / max_b_yz)
            yz_y = int(temp_z * data_c * math.sin(math.radians(data_alpha)) * vector_size / max_b_yz)

            xz_x = int((temp_z * data_c + data_a * temp_x * math.cos(math.radians(data_beta))) * vector_size / max_c_xz)
            xz_x_fix = int(
                (temp_z * data_c + (temp_x - 1) * data_a * math.cos(math.radians(data_beta))) * vector_size / max_c_xz)
            xz_y = int(temp_x * data_a * math.sin(math.radians(data_beta)) * vector_size / max_c_xz)
            if atom in atom_map.keys():
                if gamma_neg != 1:
                    matrix_a[xy_x][xy_y] += atom_map[atom] * 5
                else:
                    matrix_a[xy_x_fix][xy_y] += atom_map[atom] * 5
                if alpha_neg != 1:
                    matrix_b[yz_x][yz_y] += atom_map[atom] * 5
                else:
                    matrix_b[yz_x_fix][yz_y] += atom_map[atom] * 5
                if beta_neg != 1:
                    matrix_c[xz_x][xz_y] += atom_map[atom] * 5
                else:
                    matrix_c[xz_x_fix][xz_y] += atom_map[atom] * 5
            else:
                if gamma_neg != 1:
                    matrix_a[xy_x][xy_y] += 57.5
                else:
                    matrix_a[xy_x_fix][xy_y] += 57.5
                if alpha_neg != 1:
                    matrix_b[yz_x][yz_y] += 57.5
                else:
                    matrix_b[yz_x_fix][yz_y] += 57.5
                if beta_neg != 1:
                    matrix_c[xz_x][xz_y] += 57.5
                else:
                    matrix_c[xz_x_fix][xz_y] += 57.5

        a_new = matrix_a[np.newaxis, :, :]
        b_new = matrix_b[np.newaxis, :, :]
        c_new = matrix_c[np.newaxis, :, :]
        comb = np.concatenate([a_new, b_new, c_new])
        temp = torch.from_numpy(comb)
        temp = Variable(torch.unsqueeze(temp, dim=0).float(), requires_grad=False)
        result_s = model(temp).squeeze().detach().cpu()
        result_in.append(result_s)
    return result_in


path = '.\\data\\test_cif'
data_list = np.array(os.listdir(path))
x_local = []
for i in data_list:
    x_local.append(i)
x_local = np.array(x_local)

result = Res18(x_local)
with open('predict.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    i = 0
    temp_row = []
    while x_local[i]:
        temp = [x_local[i], result[i]]
        i = i + 1
        writer.writerow(temp)