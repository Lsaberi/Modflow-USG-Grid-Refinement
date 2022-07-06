import os
import pandas as pd
import numpy as np
import geopandas as gpd
pd.set_option('display.max_columns', 10)
import flopy
from collections import OrderedDict
import shutil
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
from scipy import stats


def translate_array(old_array, new2old, old2new):
    new_array = np.zeros(len(new2old))
    z = 0

    for new_node in new2old.keys():
        # new_node = i+1
        old_node = new2old[new_node]
        new_array[new_node-1] = old_array[old_node - 1]
        if old_node == 0:
            z+=1

    # data = []
    # for oldnode

    return new_array


def scratch_main():
    # this function has useful stuff but don't need to use in the code, instead use the main function at the bottom
    oldmf = flopy.modflow.Modflow.load('gma12.nam',model_ws = os.path.join('model_ws','base.wel.fix.1929-2010.pst18'), version='mfusg',check=False)#, load_only=['disu','lpf','bas6'])
    old_nodelay = [2221, 19089, 16185, 17218, 21941, 23315, 24786, 29084, 30954, 34123]
    new_nodelay = [2221, 22389, 22011, 23626, 29309, 30683, 32154, 36452, 38322, 41491]
    new_nodelay = [2221, 22389, 22011, 23626, 29309, 30683, 32154, 36452, 38322, 41491]
    nodes = sum(new_nodelay)
    print('Number of New Nodes ', nodes)

    # exit()

    grid_df = pd.read_csv(os.path.join('input_data','New_Grid_v4_Node_Lookup.csv'))

    fake_mf = flopy.modflow.Modflow('empty',version='mfusg')
    fake_dis = flopy.modflow.ModflowDisU.load(os.path.join('input_data','RefineVistaRidge3.dis'),fake_mf,check=False)
    fake_top, fake_botm = fake_dis.top.array, fake_dis.bot.array

    print(fake_dis.iac.array.shape)

    # bas first
    old_bas = oldmf.bas6
    ibound = old_bas.ibound.array

    mf = flopy.modflow.Modflow('rf.gma12',model_ws=os.path.join('model_ws','base.rf.RefGMA12'),version='mfusg')
    bas = flopy.modflow.ModflowBas(mf,ibound=1,strt=2000)

    old_disu = oldmf.disu

    old_top, old_botm = old_disu.top.array, old_disu.bot.array

    old_thickness = old_top - old_botm

    print(len(old_thickness))
    old_thickness = old_thickness[old_thickness >0]
    print(len(old_thickness))

    # exit()

    print(old_disu)

    print(old_disu.iac.array.shape)

    print(old_disu.iac.array.shape)


    disu = flopy.modflow.ModflowDisU(mf,nodes=sum(new_nodelay),nlay=10,njag=1868600,ivsd=1,nper=1,nstp=1,perlen=365.25,steady=[True],nodelay=new_nodelay,itmuni=4,lenuni=1,
        iac=fake_dis.iac,ja=fake_dis.ja,ivc=fake_dis.ivc,cl12=fake_dis.cl12,fahl=fake_dis.fahl)

    # disu.write_file()
    # bas.write_file()

    usg_shp = gpd.read_file(os.path.join('GIS','input_shapefiles','MODFLOW_USG_Grid_2d.shp'))

    nlay = 10
    for lay in range(nlay):
        lay_shp = gpd.read_file(os.path.join('..','Refine_Hydrostrat_wVistaRidge_v3','Grid',f'New_Grid_v3_Lyr{lay+1}_short.shp'))
        lay_shp = lay_shp[lay_shp['newnode'] >0]
        lay_shp.set_index('newnode',inplace=True)
        lay_shp['Top'] = fake_top[lay_shp.index-1]
        lay_shp['Botm'] = fake_botm[lay_shp.index-1]

        lay_shp.to_file(os.path.join('GIS','output_shapefiles',f'disu_lay{lay+1}.shp'))

        # exit()


def get_new2old(grid_df):
    df = grid_df[grid_df['NewNode'] > 0]

    # df.drop_duplicates(subset=['NewNodeShort'],inplace=True)
    df.sort_values(inplace=True,by='NewNode')
    df.set_index('NewNode',inplace=True)
    df_dict = df.to_dict()
    new2old = {node:df_dict['OriginalNode'][node] for node in df.index}
    # new2old = {}

    old2new = {node:[] for node in df['OriginalNode']}
    for newnode in new2old.keys():
        oldnode = new2old[newnode]
        old2new[oldnode].append(newnode)
    # i=0
    # for node in old2new.keys():
    #     if len(old2new[node]) >1:
    #         i+=1
    #         print(node, old2new[node])
    #
    # print(len(old2new),i)
    #
    # exit()
    return new2old, old2new


def list2lists(ls, by=25):
    data = []
    subdata = []
    itr = 0
    for i in ls:
        subdata.append(i)
        itr+=1
        if itr == by:
            itr=0
            data.append(subdata)
            subdata = []
    return data

def convert_lpf(new2old,old2new,new_nodelay,ogmodel_ws, model_ws):
    nodelay = [10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268,
               10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268]

    nodelay_cs = np.cumsum(nodelay)

    Ext_dir = os.path.join(ogmodel_ws, 'EXT')
    rf_ExtDri = os.path.join(model_ws, 'EXT')
    if not os.path.exists(rf_ExtDri):
        os.makedirs(rf_ExtDri)
    for f in os.listdir(Ext_dir):
        if f.split('.')[1] == 'txt':
            laynum = int(f.split('_')[0][-2])
            data = []
            # df = pd.read_csv(os.path.join(Ext_dir, f), header=None, names=['col1'], sep='\n')
            # df = [df[i][0] for i in range(len(df))]
            df = np.loadtxt(os.path.join(Ext_dir, f))
            L = df.tolist()
            df2 = pd.DataFrame(L, columns=['col1'], index=[i for i in range(len(L))])
            if laynum == 0:
                df2['nodenumber'] = range(1,(nodelay_cs[laynum]+1))
            else:
                df2['nodenumber'] = range(nodelay_cs[laynum-1]+1, nodelay_cs[laynum]+1)
            for i, dfrow in df2.iterrows():
                oldnode_num = df2['nodenumber'][i]
                for n in old2new[oldnode_num]:
                    # for n in old2new[oldnode]:
                    row = [df2['col1'][i], int(n)]
                    data.append(row)
            df_updated = pd.DataFrame(data, columns=['col1', 'nodenumber'])
            df_final = df_updated['col1']

            # df_updated.to_csv(os.path.join(rf_ExtDri, f'{f}'),index=False, header=False)
            f = open(os.path.join(rf_ExtDri, f'{f}'), 'w')
            for i in range(len(df_final)):
                string_val = "{:.7e}".format(df_final.iloc[i])
                f.write(f'{string_val.rjust(15)}\n')

            f.close()

    shutil.copy(os.path.join(ogmodel_ws, 'oahu.lpf'), os.path.join(model_ws, 'rf.oahu.lpf'))
    print('Done Converting LPF')

def convert_bas(new2old,old2new,new_nodelay,og_model_ws, model_ws,converge=False,pred=False):

    ibound = np.ones(np.sum(new_nodelay))
    # strt = 2000.
    nlay = 25
    f = open(os.path.join(model_ws,'rf.oahu.bas'),'w')
    header = ['# MODFLOW-USGs Basic Package\n','#\n','#\n',' UNSTRUCTURED FREE\n']
    if converge:
        header = ['# MODFLOW-USGs Basic Package\n', '#\n', '#\n', ' UNSTRUCTURED FREE CONVERGE\n']

    f.writelines(header)

    props = {'IBOUND':ibound}
    cs_new_nodelay = np.cumsum(new_nodelay)
    for lay in range(nlay):
        for prop in props.keys():
            array = props[prop]
            f.write(f'INTERNAL  1  (FREE)  -1  {prop} Layer {lay+1}\n')
            if lay == 0:
                node_start = 0
            else:
                node_start = cs_new_nodelay[lay-1]
            itr = 0
            string = ''
            for node in range(node_start,cs_new_nodelay[lay]):
                string = str(array[node])+' '
                string = ' '+ f'{int(array[node])}' # 14.7e
                f.write(string)
                itr += 1
                if itr == 10:
                    f.write('\n')
                    itr = 0
                    # string = ''
            if itr >0:
                f.write('\n')
    f.write('-1.0e+30\n')
    # for lay in range(nlay):
    #     f.write(f'CONSTANT {strt}                    Starting Heads Layer {lay+1}\n')
    if pred:
        for lay in range(nlay):
            # f.write('       -62     1.000(10e12.4)                   -1')
            f.write(''.join(['-62'.rjust(10), '1.000(10e12.4)'.rjust(19), '-1'.rjust(21),'\n']))
    f.close()

    print('Done Converting BAS')

def convert_drn(new2old,old2new,og_model_ws, model_ws,start_yr=1929):
    import drainfile_reader

    drn_spd, drn_df = drainfile_reader.read_USG_drn_file(os.path.join(og_model_ws,'oahu.drn'),start_yr=1929)
    # drn_df = drn_df.loc[drn_df['Year']==1929]
    # print(drn_df.head())
    data = []
    for i, dfrow in drn_df.iterrows():
        oldnode = dfrow['Node']
        stage, cond = dfrow['Stage'], dfrow['Cond']
        year = dfrow['Year']
        for node in old2new[oldnode]:
            row = [year, int(node), float(stage),float(cond)/len(old2new[oldnode])]
            data.append(row)
    drn_df = pd.DataFrame(data,columns=['Year','Node','Stage','Cond'])

    formats = OrderedDict([('Node', '{:>10d}'.format),
                           ('Stage', '{:>10d}'.format),
                           ('Cond', '{:>10d}'.format)])
    years = np.arange(1929,2046)

    pak_spd = {}
    for year in years:
        temp_df = drn_df.loc[drn_df['Year'] == year]
        pak_spd[year] = temp_df[list(formats.keys())].to_records(index=False)
    # for sp in range(1, nper):
    #     pak_spd[1929 + sp] = -1

    drainfile = os.path.join(model_ws, 'rf.oahu.drn')
    max_wells = 0
    for yr in pak_spd.keys():
        if not isinstance(pak_spd[yr],int):
            if len(pak_spd[yr]) > max_wells:
                max_wells = len(pak_spd[yr])

    with open(drainfile, 'w') as f:
        f.write('# MODFLOW-USGs drain Package\n')
        # f.write(f'{max_wells} NOPRINT\n')
        f.write('{}\n'.format(''.join([f'{max_wells}'.rjust(2), '50'.rjust(3), 'NOPRINT'.rjust(10)])))
        for y, rows in sorted(pak_spd.items()):
            if not isinstance(pak_spd[y],int):
                f.write('{}\n'.format(''.join([str(len(rows)).rjust(2), '0'.rjust(2)])))
                for r in rows:
                    # f.write('{}\n'.format(' '.join([str(x) for x in r])))
                    f.write('{}\n'.format(' '.join([str(r[0]).rjust(4), str(r[1]).rjust(11), str(r[2]).rjust(14)])))
            else:
                f.write(f'-1 0\n')

    print('Done Converting DRN Package')

def write_nam(model_ws):

    lines = ['LIST 7 rf.oahu.lst\n',
             'BAS6  19 rf.oahu.bas\n',
             'SMS  23 rf.oahu.sms\n',
             'DISU  29 rf.oahu.dis\n',
             'OC  22 rf.oahu.oc\n',
             'RCH  18 rf.oahu.rch\n',
             'WEL  21 rf.oahu.cln.wel\n',
             'CLN  24 rf.oahu.cln\n',
             'DRN  13 rf.oahu.drn\n',
             'CHD  17 rf.oahu.chd\n',
             'LPF  11 rf.oahu.lpf\n',
             'DATA 166 oahu_FlowReduction.dat\n',
             'BCT  26 rf.oahu.bct\n',
             'DDF  27 rf.oahu.ddf\n',
             'DATA(BINARY) 51 rf.oahu.cbb\n',
             'DATA(BINARY) 32 rf.oahu.con\n',
             'DATA(BINARY) 50 rf.oahu.cbb\n',
             'DATA(BINARY) 30 rf.oahu.hds\n',
             'DATA(BINARY) 31 rf.oahu.ddn\n',
             'DATA(BINARY) 62 initHeadFile2.hds\n',
             'DATA(BINARY) 63 pssConFile.con\n',
             ]

    f = open(os.path.join(model_ws,'rf.oahu.nam'),'w')
    f.writelines(lines)
    f.close()

def convert_oc():

    shutil.copy(os.path.join(ogmodel_ws, 'oahu.oc'), os.path.join(model_ws, 'rf.oahu.oc'))

def convert_rch(new2old,old2new,ogmodel_ws=os.path.join('model_ws', 'oahu_ft_00', 'model2'), model_ws='', start_yr=1929):
    import USG_rch
    # old_rch_spd, old_rch_mult_spd, old_node_spd= USG_rch.read_USG_rch_file(os.path.join(r'C:\Users\lsaberi\Dropbox\INTERA\GitHub\refined_beforeChanges\refined_czwx_2022\model_ws\base.wel.fix.1929-2010.pst18','gma12.rch'), start_yr=start_yr)
    old_rch_spd, old_rch_mult_spd, old_node_spd, old_conc_spd  = USG_rch.read_USG_rch_file(os.path.join(ogmodel_ws,'oahu.rch'), start_yr=start_yr)
    old_node2old_rch = {}
    old_node2old_conc = {}

    years = np.arange(1929,2046)

    rch_mult_spd = {}
    new_rch_spd, new_node_spd, new_conc_spd = {}, {}, {}
    for year in years:
        for i, nodes in enumerate(old_node_spd[year]):
            for j, node in enumerate(nodes):
                old_node2old_rch[node] = old_rch_spd[year][i][j]
                old_node2old_conc[node] = old_conc_spd[year][i][j]

        new_rch_spd[year] = []
        new_node_spd[year] = []
        new_conc_spd[year] = []
        for oldnode in old_node2old_rch.keys():
            newnodes = old2new[oldnode]
            for newnode in newnodes:
                new_rch_spd[year].append(old_node2old_rch[oldnode])
                new_conc_spd[year].append(old_node2old_conc[oldnode])
                new_node_spd[year].append(newnode)
        new_rch_spd[year] = list2lists(new_rch_spd[year],by=10)
        new_conc_spd[year] = list2lists(new_conc_spd[year], by=10)
        new_node_spd[year] = list2lists(new_node_spd[year],by=10)

        # for sp in range(1, nper):
        #     new_rch_spd[1929 + sp] = -1

        rch_mult_spd[year] = 1

    USG_rch.write_USG_rch_file(os.path.join(model_ws,'rf.oahu_ft.rch'),new_rch_spd,rch_mult_spd,new_node_spd, new_conc_spd)

    print('Done Converting RCH File')

def convert_chd():

def convert_wel():


def convert_hist(run=False):
    grid_df = pd.read_csv(os.path.join('input_data', 'rf.node.lookup.csv'))
    new2old, old2new = get_new2old(grid_df)

    ogmodel_ws = os.path.join('model_ws', 'oahu_ft_00', 'model2')

    new_nodelay = [11765, 11765, 11765, 11765, 11765, 11765, 11765, 11765, 11765, 11765, 11765, 11765, 11765,
                   11765, 11765, 11765, 11765, 11765, 11765, 11765]
    nodelay_fn = open(os.path.join('model_ws', 'RH_gridgen_ws', 'qtg.nodesperlay.dat'))

    new_nodelay = [int(node) for node in nodelay_fn.readlines()[0].split()]

    if not os.path.exists(model_ws): os.makedirs(model_ws)

    # wellnode = 214642 # location of PW-13 in simsboro
    # wel_name='Wel.24hr.test.wel'

    # make_welfile(model_ws, nper, wellnode=wellnode,SPandQ=SPandQ, wel_name=wel_name)

    nper = 118
    perlen, steady = [365.25] * nper, [False] * nper
    perlen[0] = 1
    steady[0] = True

    shutil.copy(os.path.join('model_ws', 'RH_gridgen_ws', 'mfusg.disu'), os.path.join(model_ws, 'rf.oahu.dis'))
    shutil.copy(os.path.join('model_ws', 'RH_gridgen_ws', 'rf.oahu_ft.gsf'), os.path.join(model_ws, 'rf.oahu.gsf'))
    shutil.copy(os.path.join('model_ws', 'RH_gridgen_ws', 'rf.oahu_ft.gnc'), os.path.join(model_ws, 'rf.oahu.gnc'))
    shutil.copy(os.path.join(ogmodel_ws, 'oahu_FlowReduction.dat'), os.path.join(model_ws, 'oahu_FlowReduction.dat'))
    shutil.copy(os.path.join(ogmodel_ws, 'initHeadFile2.hds'), os.path.join(model_ws, 'initHeadFile2.hds'))
    shutil.copy(os.path.join(ogmodel_ws, 'pssConFile.con'), os.path.join(model_ws, 'pssConFile.con'))
    shutil.copy(os.path.join(ogmodel_ws, 'oahu.sms'), os.path.join(model_ws, 'rf.oahu.sms'))
    # get_other_files(model_ws)

    write_nam(model_ws)

    # convert_wel(new2old,old2new,new_nodelay,ogmodel_ws,model_ws,1929)

    # convert_riv(new2old,old2new,ogmodel_ws,model_ws,1929)
    # convert_ghb(new2old,old2new,ogmodel_ws,model_ws,1929)

    convert_rch(new2old, old2new, ogmodel_ws=os.path.join('model_ws', 'oahu_ft_00', 'model2'), model_ws=model_ws,
                start_yr=1929)
    convert_lpf(new2old, old2new, new_nodelay, ogmodel_ws, model_ws)
    convert_bas(new2old, old2new, new_nodelay, ogmodel_ws, model_ws, converge=False, pred=True)
    convert_drn(new2old, old2new, ogmodel_ws, model_ws, start_yr=1929)
    write_nam(model_ws)
    convert_oc()

    # convert_wel(new2old,old2new,new_nodelay,ogmodel_ws,model_ws,1929)
    print('Done Conversion')

    if run:
        mf = flopy.mfusg.MfUsg.load('rf.oahu_ft.nam',
                                    model_ws=model_ws,
                                    version='mfusg', check=False, forgive=True,
                                    exe_name=os.path.join('bin', 'windows', 'mfusg.exe'),
                                    load_only=['disu'])  # , load_only=['disu','lpf','bas6'])
        success, buff = mf.run_model()
        print(success, buff)

        # plot_hydrograph(model_ws, nper, new_nodelay, perlen, SPandQ, wellnode)



from shapely.geometry import LineString



def hdobj2data(hdsobj):
    hds = []
    kstpkpers = hdsobj.get_kstpkper()
    for kstpkper in kstpkpers:
        data = hdsobj.get_data(kstpkper=kstpkper)
        fdata = []
        for lay in range(len(data)):
            fdata += data[lay].tolist()
        hds.append(fdata)

    return np.array(hds)

def hds2shp():
    import spatialpy

    rf_model_ws = os.path.join('model_ws','rf.base.wel.fix.1929-2010.pst18')
    rf_hdsobj = flopy.utils.HeadUFile(os.path.join(rf_model_ws,'rf.gma12.hds'))
    rf_hds = hdobj2data(rf_hdsobj)
    rf_usg_shp = gpd.read_file(os.path.join('GIS','input_shapefiles','RF_MODFLOW_USG_Grid_2d.shp'))
    rf_usg_shp['geometry'] = rf_usg_shp['geometry'].centroid

    shp_dir = os.path.join('GIS', 'output_shapefiles', f'rf.head.contours')
    if not os.path.exists(shp_dir):
        os.makedirs(shp_dir)

    res = 5280
    nlay = 10
    for sp in [1,82]:
        for lay in range(nlay):
            temp_shp = rf_usg_shp.loc[rf_usg_shp[f'nodeL{lay+1}'] > 0]
            temp_shp[f'hds.sp{sp}'] = temp_shp.apply(lambda i: rf_hds[sp-1,i[f'nodeL{lay+1}']-1],axis=1)

            ml = spatialpy.interp2d(temp_shp,f'hds.sp{sp}',res)

            array = ml.interpolate_2D()

            co_gdf = ml.get_contours(array,base=-200,interval=10)
            co_gdf.to_file(os.path.join(shp_dir,f'rf.hds.lay{lay+1}.sp{sp}.shp'))

    model_ws = os.path.join('model_ws','base.wel.fix.1929-2010.pst18')
    hdsobj = flopy.utils.HeadUFile(os.path.join(model_ws,'gma12.fix.pst18.hds'))
    hds = hdobj2data(hdsobj)
    usg_shp = gpd.read_file(os.path.join('GIS','input_shapefiles','MODFLOW_USG_Grid_2d.shp'))
    usg_shp['geometry'] = usg_shp['geometry'].centroid

    shp_dir = os.path.join('GIS', 'output_shapefiles', f'head.contours')
    if not os.path.exists(shp_dir):
        os.makedirs(shp_dir)

    res = 5280
    nlay = 10
    for sp in [1, 82]:
        for lay in range(nlay):
            temp_shp = usg_shp.loc[usg_shp[f'nodeL{lay+1}'] > 0]
            temp_shp[f'hds.sp{sp}'] = temp_shp.apply(lambda i: hds[sp-1,i[f'nodeL{lay+1}']-1],axis=1)

            ml = spatialpy.interp2d(temp_shp,f'hds.sp{sp}',res)

            array = ml.interpolate_2D()

            co_gdf = ml.get_contours(array,base=-200,interval=10)
            co_gdf.to_file(os.path.join(shp_dir,f'hds.lay{lay+1}.sp{sp}.shp'))

def ghb2shp():
    import ghbfile_reader
    shpdict = {'':'MODFLOW_USG_Grid_2d.shp','rf.':'RF_MODFLOW_USG_Grid_2d.shp'}
    for key in shpdict.keys():
        node_info = pd.read_csv(os.path.join('input_data', f'{key}node.info'), delim_whitespace=True)
        node_info_dict = node_info.set_index('nodenumber').to_dict()

        shp = shpdict[key]
        model_ws = os.path.join('model_ws',f'{key}base.wel.fix.1929-2010.pst18')

        ghb_spd, ghb_df = ghbfile_reader.read_USG_ghb_file(os.path.join(model_ws, f'{key}gma12.ghb'), start_yr=1929)
        ghb_df = ghb_df.loc[ghb_df['Year'] == 1929]
        ghb_df['Stage'] = ghb_df['Stage'].astype(float)
        ghb_df['Cond'] = ghb_df['Cond'].astype(float)

        ghb_df['layer'] = [node_info_dict['layer'][node] for node in ghb_df['Node'].tolist()]
        del ghb_df['Date']
        usg_shp = gpd.read_file(os.path.join('GIS','input_shapefiles',shp))

        shp_dir = os.path.join('GIS', 'output_shapefiles', f'{key}ghb.shp')
        if not os.path.exists(shp_dir):
            os.makedirs(shp_dir)

        nlay = 10
        for lay in range(nlay):
            temp_ghb_df = ghb_df.loc[ghb_df['layer'] == lay+1]
            if len(temp_ghb_df) > 0:
                temp_shp = usg_shp.loc[usg_shp[f'nodeL{lay+1}'] > 0]
                temp_shp_dict = temp_shp.set_index(f'nodeL{lay+1}').to_dict()
                temp_ghb_df['geometry'] = temp_ghb_df.apply(lambda i: temp_shp_dict['geometry'][i['Node']],axis=1)
                temp_ghb_df = gpd.GeoDataFrame(temp_ghb_df, crs=usg_shp.crs)

                temp_ghb_df.to_file(os.path.join(shp_dir,f'{key}ghb.lay{lay+1}.sp1.shp'))

def riv2shp():
    import riverfile_reader
    shpdict = {'':'MODFLOW_USG_Grid_2d.shp','rf.':'RF_MODFLOW_USG_Grid_2d.shp'}
    for key in shpdict.keys():
        node_info = pd.read_csv(os.path.join('input_data', f'{key}node.info'), delim_whitespace=True)
        node_info_dict = node_info.set_index('nodenumber').to_dict()

        shp = shpdict[key]
        model_ws = os.path.join('model_ws',f'{key}base.wel.fix.1929-2010.pst18')

        riv_spd, riv_df = riverfile_reader.read_USG_riv_file(os.path.join(model_ws, f'{key}gma12.riv'), start_yr=1929)
        riv_df = riv_df.loc[riv_df['Year'] == 1929]
        riv_df['Stage'] = riv_df['Stage'].astype(float)
        riv_df['Cond'] = riv_df['Cond'].astype(float)
        riv_df['Rbot'] = riv_df['Rbot'].astype(float)

        riv_df['layer'] = [node_info_dict['layer'][node] for node in riv_df['Node'].tolist()]
        del riv_df['Date']
        usg_shp = gpd.read_file(os.path.join('GIS','input_shapefiles',shp))

        shp_dir = os.path.join('GIS', 'output_shapefiles', f'{key}riv.shp')
        if not os.path.exists(shp_dir):
            os.makedirs(shp_dir)

        nlay = 10
        for lay in range(nlay):
            temp_riv_df = riv_df.loc[riv_df['layer'] == lay+1]
            if len(temp_riv_df) > 0:
                temp_shp = usg_shp.loc[usg_shp[f'nodeL{lay+1}'] > 0]
                temp_shp_dict = temp_shp.set_index(f'nodeL{lay+1}').to_dict()
                temp_riv_df['geometry'] = temp_riv_df.apply(lambda i: temp_shp_dict['geometry'][i['Node']],axis=1)
                temp_riv_df = gpd.GeoDataFrame(temp_riv_df, crs=usg_shp.crs)

                temp_riv_df.to_file(os.path.join(shp_dir,f'{key}riv.lay{lay+1}.sp1.shp'))
                
def drn2shp():
    import drainfile_reader
    shpdict = {'':'MODFLOW_USG_Grid_2d.shp','rf.':'RF_MODFLOW_USG_Grid_2d.shp'}
    for key in shpdict.keys():
        node_info = pd.read_csv(os.path.join('input_data', f'{key}node.info'), delim_whitespace=True)
        node_info_dict = node_info.set_index('nodenumber').to_dict()

        shp = shpdict[key]
        model_ws = os.path.join('model_ws',f'{key}base.wel.fix.1929-2010.pst18')

        drn_spd, drn_df = drainfile_reader.read_USG_drn_file(os.path.join(model_ws, f'{key}gma12.drn'), start_yr=1929)
        drn_df = drn_df.loc[drn_df['Year'] == 1929]
        drn_df['Stage'] = drn_df['Stage'].astype(float)
        drn_df['Cond'] = drn_df['Cond'].astype(float)


        drn_df['layer'] = [node_info_dict['layer'][node] for node in drn_df['Node'].tolist()]
        del drn_df['Date']
        usg_shp = gpd.read_file(os.path.join('GIS','input_shapefiles',shp))

        shp_dir = os.path.join('GIS', 'output_shapefiles', f'{key}drn.shp')
        if not os.path.exists(shp_dir):
            os.makedirs(shp_dir)

        nlay = 10
        for lay in range(nlay):
            temp_drn_df = drn_df.loc[drn_df['layer'] == lay+1]
            if len(temp_drn_df) > 0:
                temp_shp = usg_shp.loc[usg_shp[f'nodeL{lay+1}'] > 0]
                temp_shp_dict = temp_shp.set_index(f'nodeL{lay+1}').to_dict()
                temp_drn_df['geometry'] = temp_drn_df.apply(lambda i: temp_shp_dict['geometry'][i['Node']],axis=1)
                temp_drn_df = gpd.GeoDataFrame(temp_drn_df, crs=usg_shp.crs)

                temp_drn_df.to_file(os.path.join(shp_dir,f'{key}drn.lay{lay+1}.sp1.shp'))


if __name__ == '__main__':
    ogmodel_ws = os.path.join('model_ws', 'oahu_ft_00', 'model2')
    model_ws = os.path.join('model_ws', 'rf.oahu_ft')
    if not os.path.exists(model_ws):
        os.makedirs(model_ws)
    grid_df = pd.read_csv(os.path.join('input_data', 'rf.node.lookup.csv'))
    new2old, old2new = get_new2old(grid_df=grid_df)
    nodelay_fn = open(os.path.join('model_ws', 'RH_gridgen_ws', 'qtg.nodesperlay.dat'))
    new_nodelay = [int(node) for node in nodelay_fn.readlines()[0].split()]
    # convert_wel(new2old, old2new, new_nodelay, ogmodel_ws, model_ws, 1929)
    # convert_hist(run=False)
    # convert_rch(new2old,old2new,ogmodel_ws=os.path.join('model_ws', 'oahu_ft_00', 'model2'), model_ws=model_ws, start_yr=1929)
    # convert_lpf(new2old,old2new,new_nodelay,ogmodel_ws, model_ws)
    # convert_bas(new2old, old2new, new_nodelay, ogmodel_ws, model_ws, converge=False, pred=True)
    # convert_drn(new2old, old2new, ogmodel_ws, model_ws, start_yr=1929)
    # write_nam(model_ws)
    # convert_oc()


