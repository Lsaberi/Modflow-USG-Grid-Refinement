import numpy as np
import pandas as pd
import os

def read_USG_rch_file(fn,start_yr = 1929):
    # returns well stress period data in dict form, (1 based nodes because who uses zero based for Cell ID's?)
    with open(fn, 'r') as file:
        stress_period_data = {}
        sp = 0
        lines = file.readlines()
        cnt = 0
        while lines[cnt].startswith('#'):
            cnt += 1

        if len(lines[cnt].split()) == 2:
            ipakcb = lines[cnt].split()[1]

        maxwels = lines[cnt].split()[0]

        cnt += 1
        max_rch_nodes = int(lines[cnt])
        # print(max_rch_nodes)
        cnt += 2

        data = []
        year = start_yr  # 1929#, 2011
        rch_spd = {}
        rch_mult_spd = {}
        node_spd = {}
        conc_spd = {}
        sp = 1
        while cnt < len(lines) - 1 and len(lines[cnt].split()) > 1:
            num_rch_nodes = float(lines[cnt].split()[1])
            cnt += 1
            if lines[cnt].split()[0] == '-1':
                rch_spd[year] = rch_spd[year - 1]
                node_spd[year] = node_spd[year - 1]
                rch_mult_spd[year] = rch_mult_spd[year - 1]

            else:
                rch_mult = float(lines[cnt].split()[1])
                rch_data = []
                val = np.ceil(num_rch_nodes / 10)
                for row in range(int(val)):  # todo need work but works for now
                    cnt += 1
                    line = [float(item) for item in lines[cnt].split()]
                    rch_data.append(line)
                rch_spd[year] = rch_data
                rch_mult_spd[year] = rch_mult
                cnt += 1

                node_data = []
                for row in range(int(val)):
                    cnt += 1
                    line = [float(item) for item in lines[cnt].split()]
                    node_data.append(line)
                node_spd[year] = node_data
                cnt += 1

                conc_data = []
                for row in range(int(val)):
                    cnt += 1
                    line = [float(item) for item in lines[cnt].split()]
                    conc_data.append(line)
                conc_spd[year] = conc_data
                cnt += 1

            year += 1
            sp += 1

        return rch_spd, rch_mult_spd, node_spd, conc_spd

def write_USG_rch_file(fn,rch_spd,rch_mult_spd,rch_node_spd, conc_node_spd):
    max_nodes = 0
    for key in rch_spd.keys():
        if not isinstance(rch_spd[key],int):
            flatten = [item for sublist in rch_spd[key] for item in sublist]
            if len(flatten) > max_nodes:
                max_nodes = len(flatten)
    f = open(fn,'w')
    header = ['# MODFLOW-USGs Recharge Package\n',' 2 50 CONC\n',f' {max_nodes}\n', '1\n', f'1 {max_nodes} INCONC\n']
    for line in header:
        f.write(line)

    for year in rch_spd.keys():
        if rch_spd[year] == -1:
            f.write('-1 -1\n')
        else:
            rch_mult = rch_mult_spd[year]
            nnodes = len(rch_spd[year])
            f.write(f' 1 {max_nodes}\n')
            f.write(f'INTERNAL  {rch_mult}  (FREE)   -1   Recharge by Node for Period 1 YR{year}\n')
            rch_data = rch_spd[year]
            node_data = rch_node_spd[year]
            conc_data = conc_node_spd[year]

            for line in rch_data:
                # f.write('{}\n'.format(' '.join([str(x) for x in line]))) '{:.6e} '
                f.write('{}\n'.format(' '.join(['{:.6e}'.format(x) for x in line]))) #'{:.6e} '


            f.write(f'INTERNAL  1  (FREE)   -1   Recharge Nodes YR{year}\n')
            for line in node_data:
                f.write('{}\n'.format(' '.join([str(x) for x in line])))

            f.write(f'INTERNAL  1  (FREE)   -1   Recharge Conc by Node YR{year}\n')
            for line in conc_data:
                f.write('{}\n'.format(' '.join([str(x) for x in line])))

    f.close()