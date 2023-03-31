import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import _alkene_type_filter as fil
from glob import glob
from datetime import date

today = date.today()

d1 = today.strftime("%m/%d/%Y")
d2 = today.strftime("%m_%d_%Y")

csv_file_name = '1000_Entry_Database_p8.csv'
df = pd.read_csv(csv_file_name)

# print(df[fil.find_mono(df)]["ee (%)"].min())

fig = plt.figure()
g = sns.violinplot(data=df,
    bw = 0.5,  # type: ignore
    # x = 'ddG er (kcal/mol)',
    x = 'ee (%)',
    y = 'Olefin Type',
    scale = 'width',
    order = ['mono','gem_di','trans_di','cis_di','tri', 'tetra'],
    orient = 'h',
    inner='box',
    saturation = 1,
    cut=0
)
g.set(
    title = f'{d1} Database {df.shape[0]} entries',
    # title = f'{d1} Database',
    yticklabels = [f'mono ({df[fil.find_mono(df)].shape[0]})',f'gem_di ({df[fil.find_gem_di(df)].shape[0]})',f'trans_di ({df[fil.find_trans_di(df)].shape[0]})',f'cis_di ({df[fil.find_cis_di(df)].shape[0]})',f'tri ({df[fil.find_tri(df)].shape[0]})', f'tetra ({df[fil.find_tetra(df)].shape[0]})'],
    xticks=[0,10,20,30,40,50,60,70,80,90,100]
)
# rcParams['figure.figsize'] = 40, 12
plt.tight_layout()
fig.savefig(f'{d2}_database.png')
# fig.savefig(f'{d2}_database_ddg.png')
# plt.show()