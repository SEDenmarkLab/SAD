import numpy as np
import pandas as pd
from sklearn import feature_selection as fs
from sklearn import manifold as man
from sklearn import preprocessing as pp
from matplotlib import pyplot as plt
from sklearn import decomposition as dr
from sklearn.pipeline import Pipeline
from umap import UMAP

def visualize_pca_score_plots(full_df:pd.DataFrame, fs_pipe: Pipeline, color_dict: dict, marker_dict:dict, dimensions=2, name='test'):

    fig = plt.figure()
    if dimensions == 2:
        ax = fig.add_subplot(111)
        _pca_test = dr.PCA(n_components=2, random_state=42).fit(fs_pipe)
        pca_pipe = _pca_test.transform(fs_pipe)
    if dimensions == 3:
        ax = fig.add_subplot(111, projection='3d')
        _pca_test = dr.PCA(n_components=3, random_state=42).fit(fs_pipe)
        pca_pipe = _pca_test.transform(fs_pipe)
    
    ax.grid(False)
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title(f'PCA Visualization {dimensions}D {name}')

    #This is the series used for coloring the correct points
    label_ser = full_df['Olefin Type'].values

    print(f'{np.sum(_pca_test.explained_variance_ratio_)} is the explained variance ratio of the {dimensions}D space')

    mono_ser = np.where(label_ser == 'mono')[0]
    gem_di_ser = np.where(label_ser == 'gem_di')[0]
    cis_di_ser = np.where(label_ser == 'cis_di')[0]
    trans_di_ser = np.where(label_ser == 'trans_di')[0]
    tri_ser = np.where(label_ser == 'tri')[0]
    tetra_ser = np.where(label_ser == 'tetra')[0]

    all_alkene_types = ['mono', 'gem_di', 'cis_di', 'trans_di', 'tri', 'tetra']
    all_alkene_dict = dict(zip(all_alkene_types, [mono_ser, gem_di_ser, cis_di_ser, trans_di_ser, tri_ser, tetra_ser]))
    legend_handles = list()
    for alk_type, idx in all_alkene_dict.items():

        if dimensions == 2:
            scatter = ax.scatter(pca_pipe[idx,0],pca_pipe[idx,1], c=color_dict[alk_type], marker=marker_dict[alk_type], alpha=0.5)
            scatter.set_label(alk_type)
            legend_handles.append(scatter)
            # scatter = ax.scatter(pca_pipe[:,0],pca_pipe[:,1], c=colors, m=markers, alpha=0.5)
            #This is a fancy looking way for creating the legends, this could easily be done with DF groupby
            # lp = lambda i: ax.plot([], color=color_dict[i], label=i, ls="", m=marker_dict[i])[0]

        if dimensions == 3:
            scatter = ax.scatter(pca_pipe[idx,0],pca_pipe[idx,1],pca_pipe[idx,2], c=color_dict[alk_type], marker=marker_dict[alk_type], alpha=0.5)
            scatter.set_label(alk_type)
            legend_handles.append(scatter)
            # scatter = ax.scatter(pca_pipe[:,0],pca_pipe[:,1],pca_pipe[:,2], c=colors, m=markers, alpha=0.5)
        #This is a fancy looking way for creating the legends, this could easily be done with DF groupby
        # lp = lambda i: ax.plot([],[], color=color_dict[i], label=i, ls="", m=marker_dict[i])[0]
    
    # handles = [lp(i) for i in color_dict.keys()]
    #CAn adjust bbox_to_anchor to move legend
    # plt.add_legend
    # lgd = ax.legend(handles=handles, bbox_to_anchor=(1.1,1))
    lgd = ax.legend(handles=legend_handles, bbox_to_anchor=(1.1,1))

    if dimensions == 2:
        fig.savefig(f'PCA Visualization {dimensions}D {name}',bbox_extra_artists = [lgd], bbox_inches='tight')
    plt.show()

def visualize_tsne_score_plots(full_df:pd.DataFrame, fs_pipe: Pipeline, color_dict: dict, marker_dict:dict, dimensions=2, metric='euclidean', name='test'):

    fig = plt.figure()
    if dimensions == 2:
        ax = fig.add_subplot(111)
        TSNE_pipe = man.TSNE(n_components=2, metric=metric, random_state=42).fit_transform(fs_pipe)
    if dimensions == 3:
        ax = fig.add_subplot(111, projection='3d')
        TSNE_pipe = man.TSNE(n_components=3, metric=metric, random_state=42).fit_transform(fs_pipe)
    
    ax.grid(False)
    plt.xlabel('TSNE1')
    plt.ylabel('TSNE2')
    plt.title(f'TSNE Visualization {dimensions}D {metric} {name}')

    #This is the series used for coloring the correct points
    label_ser = full_df['Olefin Type'].values

    mono_ser = np.where(label_ser == 'mono')[0]
    gem_di_ser = np.where(label_ser == 'gem_di')[0]
    cis_di_ser = np.where(label_ser == 'cis_di')[0]
    trans_di_ser = np.where(label_ser == 'trans_di')[0]
    tri_ser = np.where(label_ser == 'tri')[0]
    tetra_ser = np.where(label_ser == 'tetra')[0]

    all_alkene_types = ['mono', 'gem_di', 'cis_di', 'trans_di', 'tri', 'tetra']
    all_alkene_dict = dict(zip(all_alkene_types, [mono_ser, gem_di_ser, cis_di_ser, trans_di_ser, tri_ser, tetra_ser]))
    legend_handles = list()
    for alk_type, idx in all_alkene_dict.items():

        if dimensions == 2:
            scatter = ax.scatter(TSNE_pipe[idx,0],TSNE_pipe[idx,1], c=color_dict[alk_type], marker=marker_dict[alk_type], alpha=0.5)
            scatter.set_label(alk_type)
            legend_handles.append(scatter)

        if dimensions == 3:
            scatter = ax.scatter(TSNE_pipe[idx,0],TSNE_pipe[idx,1],TSNE_pipe[idx,2], c=color_dict[alk_type], marker=marker_dict[alk_type], alpha=0.5)
            scatter.set_label(alk_type)
            legend_handles.append(scatter)
    
    lgd = ax.legend(handles=legend_handles, bbox_to_anchor=(1.1,1))

    if dimensions == 2:
        fig.savefig(f'TSNE Visualization {dimensions}D {metric} {name}',bbox_extra_artists = [lgd], bbox_inches='tight')
    plt.show()

def visualize_umap_score_plots(full_df:pd.DataFrame, fs_pipe: Pipeline, color_dict: dict, marker_dict:dict, dimensions=2, metric='euclidean', min_dist=0.1, name='test'):

    fig = plt.figure()
    if dimensions == 2:
        ax = fig.add_subplot(111)
        _UMAP_test = UMAP(n_components=2, metric=metric, random_state=42).fit(fs_pipe)
        UMAP_pipe = _UMAP_test.transform(fs_pipe)
    if dimensions == 3:
        ax = fig.add_subplot(111, projection='3d')
        _UMAP_test = UMAP(n_components=3, metric=metric, random_state=42).fit(fs_pipe)
        UMAP_pipe = _UMAP_test.transform(fs_pipe)
    
    ax.grid(False)
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    # plt.title(f'UMAP Visualization {dimensions}D {metric} {name}')
    plt.title(f'UMAP Visualization {metric} {name} (Min Dist = {min_dist})')

    #This is the series used for coloring the correct points
    label_ser = full_df['Olefin Type'].values

    mono_ser = np.where(label_ser == 'mono')[0]
    gem_di_ser = np.where(label_ser == 'gem_di')[0]
    cis_di_ser = np.where(label_ser == 'cis_di')[0]
    trans_di_ser = np.where(label_ser == 'trans_di')[0]
    tri_ser = np.where(label_ser == 'tri')[0]
    tetra_ser = np.where(label_ser == 'tetra')[0]

    all_alkene_types = ['mono', 'gem_di', 'cis_di', 'trans_di', 'tri', 'tetra']
    all_alkene_dict = dict(zip(all_alkene_types, [mono_ser, gem_di_ser, cis_di_ser, trans_di_ser, tri_ser, tetra_ser]))
    legend_handles = list()
    for alk_type, idx in all_alkene_dict.items():

        if dimensions == 2:
            scatter = ax.scatter(UMAP_pipe[idx,0],UMAP_pipe[idx,1], c=color_dict[alk_type], marker=marker_dict[alk_type], alpha=0.5)
            scatter.set_label(alk_type)
            legend_handles.append(scatter)
            # scatter = ax.scatter(UMAP_pipe[:,0],UMAP_pipe[:,1], c=colors, m=markers, alpha=0.5)
            #This is a fancy looking way for creating the legends, this could easily be done with DF groupby
            # lp = lambda i: ax.plot([], color=color_dict[i], label=i, ls="", m=marker_dict[i])[0]

        if dimensions == 3:
            scatter = ax.scatter(UMAP_pipe[idx,0],UMAP_pipe[idx,1],UMAP_pipe[idx,2], c=color_dict[alk_type], marker=marker_dict[alk_type], alpha=0.5)
            scatter.set_label(alk_type)
            legend_handles.append(scatter)
            # scatter = ax.scatter(UMAP_pipe[:,0],UMAP_pipe[:,1],UMAP_pipe[:,2], c=colors, m=markers, alpha=0.5)
        #This is a fancy looking way for creating the legends, this could easily be done with DF groupby
        # lp = lambda i: ax.plot([],[], color=color_dict[i], label=i, ls="", m=marker_dict[i])[0]
    
    # handles = [lp(i) for i in color_dict.keys()]
    #CAn adjust bbox_to_anchor to move legend
    # plt.add_legend
    # lgd = ax.legend(handles=handles, bbox_to_anchor=(1.1,1))
    lgd = ax.legend(handles=legend_handles, bbox_to_anchor=(1.1,1))

    if dimensions == 2:
        fig.savefig(f'UMAP Visualization {dimensions}D {metric} {min_dist} {name}.png',bbox_extra_artists = [lgd], bbox_inches='tight')
    plt.show()

# full_desc_df = pd.read_csv('Step_6_Full_Aligned_React_Desc_DF_w_Olefin_Type.csv', index_col=0)
full_desc_df = pd.read_csv('Step_6_No_Frags_React_Desc_DF_w_Olefin_Type.csv', index_col=0)
file_name = 'No Frags'
metric='euclidean'
dimension=2

pipe = Pipeline(steps=[
    ('scaler', pp.MinMaxScaler()),
    ('vt', fs.VarianceThreshold(threshold=0))
])

label_remove_df = full_desc_df[full_desc_df.columns[:-1]]

print(f'There were {pipe["vt"].fit(label_remove_df).n_features_in_} features in.')
print(f'There were {np.count_nonzero(pipe["vt"].fit(label_remove_df).get_support())} features remaining')

#These provide the color labels
color_dict = {'mono': 'tab:grey',
'gem_di': 'tab:orange',
'cis_di': 'tab:green',
'trans_di': 'tab:red',
'tri': 'tab:purple',
'tetra': 'tab:blue'}

marker_dict = {'mono': 'o',
'gem_di': '^',
'cis_di': 'v',
'trans_di': '*',
'tri': 'x',
'tetra': 'd'}

# visualize_pca_score_plots(
#     full_df=full_desc_df,
#     fs_pipe = pipe.fit_transform(label_remove_df),
#     color_dict=color_dict,
#     marker_dict=marker_dict,
#     dimensions=dimension,
#     name=file_name
# )
visualize_tsne_score_plots(
    full_df=full_desc_df,
    fs_pipe = pipe.fit_transform(label_remove_df),
    color_dict=color_dict,
    marker_dict=marker_dict,
    dimensions=dimension,
    metric=metric,
    name=file_name
)
# visualize_umap_score_plots(
#     full_df=full_desc_df,
#     fs_pipe = pipe.fit_transform(label_remove_df),
#     color_dict=color_dict,
#     marker_dict=marker_dict,
#     dimensions=dimension,
#     metric=metric,
#     name=file_name
# )