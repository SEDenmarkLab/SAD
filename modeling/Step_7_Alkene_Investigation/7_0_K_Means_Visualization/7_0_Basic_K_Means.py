import numpy as np
import pandas as pd
from sklearn import feature_selection as fs
from sklearn import preprocessing as pp
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import decomposition as dr
from sklearn import metrics
from scipy.spatial.distance import cdist
import pickle
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.PropertyMol import PropertyMol
import os
from sklearn.pipeline import Pipeline
from glob import glob
from bidict import bidict
from kneed import KneeLocator
import warnings

def find_canonical_smiles(original_smiles: str):
    mol = Chem.MolFromSmiles(original_smiles)
    can_smiles = Chem.MolToSmiles(mol, canonical=True)
    return can_smiles

def visualize_similarity_mols(name, mol_list, similarity_type):

    obj_name = name
    if len(mol_list) != 0:
        _img = Draw.MolsToGridImage(mol_list, molsPerRow=5, subImgSize=(400,400), useSVG=True,returnPNG=False, highlightAtomLists=None, highlightBondLists=None,  legends=[f'{i.GetProp("_Name")}\nSimilarity = {i.GetProp(f"_{similarity_type}")}' for i in mol_list], maxMols=20000)
        with open(f'{obj_name}.svg', 'w') as f:
            f.write(_img.data)

def give_unique_react(df: pd.DataFrame):
    df_iso = df[['Reactant ID', 'Reactant SMILES']]
    return df_iso.drop_duplicates()

def create_mol_list(react_df: pd.DataFrame):
    id = react_df['Reactant ID'].values
    smiles_str = react_df['Reactant SMILES'].values

    mol_list = list()

    for i, smiles in zip(id, smiles_str):
        mol = Chem.MolFromSmiles(smiles)
        mol.SetProp("_Name", i)
        mol_list.append(mol)

    return mol_list

def eval_distortion(full_df:pd.DataFrame, fs_pipe:Pipeline, max_clusters=None, name='test', plot_distortion=False):

    inertias = list()
    distortions = list()

    if max_clusters:
        K = range(1, max_clusters+1)
    else:
        K = range(1, full_df.shape[0]+1)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, random_state=42).fit(fs_pipe)
        #This returns the distortion (average euclidean squared distance from the centroid of the respective clusters)
        distortions.append(sum(np.min(cdist(fs_pipe, kmeanModel.cluster_centers_, 'euclidean'),axis=1)**2) / fs_pipe.shape[0])

        #This returns the inertia (sum of squared distances of samples to their closest cluster center)
        inertias.append(kmeanModel.inertia_)
    
    if plot_distortion:
        fig = plt.figure()
        #Based on Distortion
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Distortion')
        plt.title(f'Distortion Plot vs. Clusters {name}')
        fig.savefig(f'{name}_distortion.png')

        # fig = plt.figure()
        # #Based on Inertia
        # plt.plot(K, inertias, 'bx-')
        # plt.xlabel('Number of Clusters')
        # plt.ylabel('Inertia')
        # plt.title(f'Inertia Plot vs. Clusters {name}')
        # fig.savefig(f'{name}_inertia.png')
    return (list(K),distortions)

def kneed_eval_distortion(clusters: list, distortions:list, name='test'):
    'Utilize Kneed Algorithm to choose an elbow point on the plot. Returns the knee value chosen'

    x_val = clusters
    y_val = distortions

    assert len(x_val) == len(y_val), f'clusters != distortions:\n{clusters}\n{distortions}'

    # knees = []
    # sensitivity = x_val
    # for s in sensitivity:
    #     kl = KneeLocator(x_val, y_val, curve="convex", direction="decreasing", S=s)
    #     knees.append(kl.knee)
    # print(f'These are all the knees at various sensitivities')
    # print(knees)

    kl = KneeLocator(x_val, y_val, curve='convex', direction='decreasing', online=True)

    # kl.plot_knee_normalized()
    fig = plt.figure()
    plt.ylim([0,1])
    plt.title(f'Elbow with Kneed Distortion {name}', fontsize=16)
    plt.xlabel('Number of Clusters', fontsize=12)
    plt.ylabel('Normalized Data', fontsize=12)
    plt.plot(x_val, y_val/np.max(y_val), 'bx-')
    plt.plot(x_val, kl.y_difference, 'r-')

    plt.axvline(kl.knee)
    plt.text(kl.knee+0.1,0.5, f'{kl.knee}')
    plt.legend(['Normalized Distortion', 'Y difference', 'Maximum Difference'])
    fig.savefig(f'{name}_Kneed.png')
    plt.close()
    return kl.knee

def visualize_k_clusters(full_df:pd.DataFrame, fs_pipe: Pipeline, k: int, mol_dict: dict, draw='2D', pic_folder_label='cluster_pics', name='test'):
    #This does the same thing for each set of clusters
    print(f'Evaluating {k} clusters for {name} for {draw} space')
    #This top fit is being used to compute the centroids. This is under the assumption that these are the same when they are fit and transformed!!!
    kmeanModel = KMeans(n_clusters=k, random_state=42).fit(fs_pipe)
    #This transforms the reduced array to a cluster distance space
    trans = KMeans(n_clusters=k, random_state=42).fit_transform(fs_pipe)

    pred = KMeans(n_clusters=k, random_state=42).fit_predict(fs_pipe)

    assert (ulabel := np.unique(pred).shape[0]) == k, f"Unique Predictions ({ulabel}) != k ({k})"
   
    fig = plt.figure()
    if draw == '2D':
        ax = fig.add_subplot(111)
        _pca_test = dr.PCA(n_components=2, random_state=42).fit(fs_pipe)
        pca_pipe = dr.PCA(n_components=2, random_state=42).fit_transform(fs_pipe)
    if draw == '3D':
        ax = fig.add_subplot(111, projection='3d')
        _pca_test = dr.PCA(n_components=3, random_state=42).fit(fs_pipe)
        pca_pipe = dr.PCA(n_components=3, random_state=42).fit_transform(fs_pipe)
    
    ax.grid(False)
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title(f'K-Means Visualization (k={k}) {name}')

    color_name = [
        'tab:blue',
        'tab:orange',
        'tab:green',
        'tab:red',
        'tab:purple',
        'tab:brown',
        'tab:pink',
        'tab:gray',
        'midnightblue',
        'tab:cyan',
        'lightcoral',
        'chocolate',
        'darkgoldenrod',
        'khaki',
        'greenyellow',
        'darkolivegreen',
        'deepskyblue',
        'lightpink',
        'tab:olive',
    ]
    
    color_dict = {i:name for i,name in enumerate(color_name)}

    print(f'{np.sum(_pca_test.explained_variance_ratio_)} is the explained variance ratio')
    # color_label = [f'{idx}' for idx in range(k)]
    color_label = list()
    for i in range(0,k):
        full_cent = kmeanModel.cluster_centers_[i].reshape(1,fs_pipe.shape[1])
        pca_cent_coords = _pca_test.transform(full_cent)
        # print(pca_cent_coords)
        coords = pca_pipe[np.where(pred == i)[0]]

        if draw == '2D':
            scatter = ax.scatter(coords[:,0],coords[:,1], color=color_dict[i], alpha=0.6)
            scatter.set_label(f'{i}')
            color_label.append(scatter)
            ax.scatter(pca_cent_coords[:,0],pca_cent_coords[:,1], color=color_dict[i], alpha=1)

        if draw == '3D':
            scatter = ax.scatter(coords[:,0],coords[:,1],coords[:,2], color=color_dict[i], alpha=0.6)
            scatter.set_label(f'{i}')
            color_label.append(scatter)
            ax.scatter(pca_cent_coords[:,0],pca_cent_coords[:,1],pca_cent_coords[:,2], color=color_dict[i], alpha=1)

        # plt.scatter(coords[:,0],coords[:,1])
        # plt.scatter(pca_cent_coords[:,0],pca_cent_coords[:,1], label='k')

        d = trans[:, i][np.where(pred == i)[0]]
        sort_d = np.argsort(d)
        sort_dist = d[sort_d]

        #This creates a list with labels based on the now sorted array from above. Originally I had converted it to a numpy array, but there seems to be a general push back from the community to iterating over a numpy array, so I'm continuing with a list for now
        cent_series = full_df.index[np.where(pred == i)[0]]
        cent_mol = np.array([mol_dict[name] for name in cent_series])
        sort_cent_mol = cent_mol[sort_d]

        assert d.shape == cent_mol.shape, f'd is not the same size as cent_mol! {d}\n{cent_mol}'
        #This zips the ordered list of centroid mol objects with the ordered list of distances
        full_legend = list()
        mol_and_dist = tuple(zip(sort_cent_mol,sort_dist))
        for tup in mol_and_dist:
            mol = tup[0]
            dist = tup[1]
            #This creates a legend with a vertical label. First row is the name, and the second is the distance from the centroid out to 4 values
            final_name = mol.GetProp("_Name") + '\n' + f'cent_dist = {dist:.4f}'
            full_legend.append(final_name)

        _img = Draw.MolsToGridImage(sort_cent_mol.tolist(), molsPerRow=5, subImgSize=(400,400), useSVG=True, legends=full_legend, maxMols=100)
        if not os.path.isdir(pic_folder_label):
            os.makedirs(pic_folder_label)
        open(f'./{pic_folder_label}/{k}_clusters_{i}_centroid.svg', 'w').write(_img.data)

    lgd = ax.legend(handles=color_label, bbox_to_anchor=(1.1,1))

    if draw == '2D':
        fig.savefig(f'KMeans Visualization k{k} {name}', bbox_extra_artists = [lgd], bbox_inches='tight')
    else:
        plt.show()

all_dfs = glob('*.csv')

db_df = pd.read_csv('p8_reduced_database_column_update_1001.csv')
alkene_iso = give_unique_react(db_df)
can_alkene_smiles = alkene_iso['Reactant SMILES'].values
alkene_mol_list = create_mol_list(alkene_iso)
alkene_mol_dict = bidict({mol.GetProp("_Name") : mol for mol in alkene_mol_list})

warnings.filterwarnings('ignore')

for csv in all_dfs:
    if csv == 'p8_reduced_database_column_update_1001.csv':
        continue
    full_df = pd.read_csv(csv, index_col=0)

    if full_df.shape[0] > 50:
        max_clusters = 50
    else:
        max_clusters = full_df.shape[0]

    df_name = ' '.join(csv.split('_')[2:4])
    print(df_name)
    pipe = Pipeline(steps=[
        ('scaler', pp.MinMaxScaler()),
        ('vt', fs.VarianceThreshold(threshold=0))
    ])

    print(f'There were {pipe["vt"].fit(full_df).n_features_in_}')
    print(f'There were {np.count_nonzero(pipe["vt"].fit(full_df).get_support())} features remaining')


    clusters, distortions = eval_distortion(
        full_df=full_df,
        fs_pipe=pipe.fit_transform(full_df),
        max_clusters=max_clusters, 
        name=df_name
    )

    knee_val = kneed_eval_distortion(
        clusters = clusters,
        distortions=distortions,
        name=df_name
    )

    visualize_k_clusters(
        full_df=full_df,
        fs_pipe=pipe.fit_transform(full_df),
        k=knee_val,
        mol_dict=alkene_mol_dict,
        draw='2D',
        pic_folder_label=f'{df_name}_k_means_pic_{knee_val}',
        name=df_name
    )
