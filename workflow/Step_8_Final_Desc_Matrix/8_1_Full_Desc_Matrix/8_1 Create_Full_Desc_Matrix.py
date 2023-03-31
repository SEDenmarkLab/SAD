import numpy as np
import pandas as pd

react_desc_df = pd.read_csv('Step_6_Full_Aligned_React_Desc_DF_w_Olefin_Type.csv', index_col=0)
db_df = pd.read_csv('p8_reduced_database_column_update_1001.csv')

#This line replaces anything with "Not Reported" in the column with 0.0
fixed_temp_val = np.where(db_df['Temperature ©'].values != 'Not Reported', db_df['Temperature ©'].values, 0.0)

new_temp_ser = pd.Series(data=fixed_temp_val, index=db_df['Temperature ©'].index)
float_temp_ser = new_temp_ser.astype(float)

db_df['Temperature ©'] = float_temp_ser
assert (temp_nan := np.count_nonzero(db_df['Temperature ©'].isna())) == 0, f'There are {temp_nan} nan values in the temperature series.'


#This recalculates the unknown temp values ddG er (kcal/mol)
temps = db_df["Temperature ©"].values
er = db_df['er'].values
ddG_vals = (temps+273.15)*(8.314)*np.log(er)*(0.000239)


####Creates Dataframe that averages all reported values of a reactant####
ddG_df = pd.DataFrame(data=ddG_vals, columns=['ddG er (kcal/mol)'], index=db_df.index)
react_id_ddG_df = pd.concat([db_df['Reactant ID'],ddG_df], axis=1)
averaged_ddG_by_react = react_id_ddG_df.groupby('Reactant ID').mean().reset_index()
average_ddG_react_ser = pd.Series(data=averaged_ddG_by_react['ddG er (kcal/mol)'].values, index=averaged_ddG_by_react['Reactant ID'].values, name='ddG er (kcal/mol)')

small_db_df = db_df[['Reactant ID','Product ID','Solvent 1 ID','Solvent 2 ID', 'Sol1_Sol2_Ratio', 'Oxidant ID', 'Catalyst ID','Temperature ©', 'er']]

#These are assertions to make sure there are not problems in direct comparisons
assert np.count_nonzero(small_db_df['er'].isna().values) == 0, f'ddG er has nan values: {small_db_df[small_db_df["er"].isna()]}'
# assert fixed_small_df['Solvent 1 ID'].nunique() == 1, f'There are more than one solvent1 values: {fixed_small_df["Solvent 1 ID"].unique()}'
# assert fixed_small_df['Solvent 2 ID'].nunique() == 1, f'There are more than one solvent2 values: {fixed_small_df["Solvent 2 ID"].unique()}'
# assert fixed_small_df['Oxidant ID'].nunique() == 1, f'There are more than one oxidant ID values: {fixed_small_df["Oxidant ID"].unique()}'

#Right now, there are missing values in the reactant descriptor due to some weird descriptor issues, so for now, it will be an inner join
final_desc_df = pd.concat([react_desc_df,average_ddG_react_ser],join='inner',axis=1)

print(final_desc_df)

final_desc_df.to_csv(f'Step_8_Altered_Temp_Scaled_ddG_Full_Desc_Matrix_Avg.csv')



