from pprint import pprint
from glob import glob
from rdkit import Chem
import yaml
import pandas as pd
import math
import numpy as np

def add_entry_to_database(csv_file_name:str, naming_yaml_file_name: str, overwrite_prev_files=False):
    '''
    This is the first prototype for the creation of a semiautomatic workflow to try and parse literature data as quick as possible. This asks a variety of questions to give a format such that
    it will create a series to fill in the columns in the format of "fixed_database_columns.csv." A Yaml File is used to store names and react/cat/ox/solvent IDs, it is possible to create nicknames
    that will then be correlated to certain smiles strings. All SMILES strings are stored/updated in the library with their canonical representation, and any products listed within this library
    should be the true smiles representation. I STILL NEED TO MAKE CHANGES TO DIASTEREOSELECTIVITY SUCH THAT 0 AND N/A ARE SEPARATED!!!!!! Otherwise it's pretty robust and should work.
    '''

    original_df = pd.read_csv(csv_file_name)

    def find_canonical_smiles(original_smiles: str):
        mol = Chem.MolFromSmiles(original_smiles)
        can_smiles = Chem.MolToSmiles(mol, canonical=True)
        return can_smiles

    with open(naming_yaml_file_name) as f:
        yml = yaml.safe_load(f)

    all_values = []

    # reactant
    react_name = input("1a. What is reactant name? (Hit Enter if you don't want to add a common name to the library) ")
    #This pulls the smiles and react id from the common name
    if react_name in yml['react_can_smiles_map']:
        can_react_smiles = yml['react_can_smiles_map'][react_name]
        react_id = yml['react_id_smiles_map'][can_react_smiles]
        print('This reactant has been seen before, here are the series it has been seen in:')
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(original_df[np.vectorize(lambda x: react_id == x)(original_df['Reactant ID'])])
        pot_dup_react = input('Would you like to restart? (Type "y" for yes, Hit [Enter] for No) ')
        if pot_dup_react == "y":
            final_df = None
            yml = None
            return final_df, yml
        
    #If there is no common name you want to add, this creates the canonical smiles and then it adds the new smiles and reactant id to the dictionary
    elif react_name == '':
        react_smiles = input('1b. What is the reactant smiles? ')
        can_react_smiles = find_canonical_smiles(react_smiles)
        #This checks to see if the canonical smiles is already assigned to a reactant id in the dictionary
        if can_react_smiles in yml['react_id_smiles_map']:
            react_id = yml['react_id_smiles_map'][can_react_smiles]
            print('This reactant has been seen before, here are the series it has been seen in:')
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
                print(original_df[np.vectorize(lambda x: react_id == x)(original_df['Reactant ID'])])
            pot_dup_react = input('Would you like to restart? (Type "y" for yes, Hit [Enter] for No) ')
            if pot_dup_react == "y":
                final_df = None
                yml = None
                return final_df, yml
            
        #If it is not in the library, it assigns it a new reactant id
        else:
            yml['react_id_smiles_map'][can_react_smiles] = f'react_{len(yml["react_id_smiles_map"])}'
            react_id = yml['react_id_smiles_map'][can_react_smiles]
            if react_id in original_df['Reactant ID']:
                print('This reactant has been seen before, here are the series it has been seen in:')
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
                    print(original_df[np.vectorize(lambda x: react_id == x)(original_df['Reactant ID'])])
                pot_dup_react = input('Would you like to restart? (Type "y" for yes, Hit [Enter] for No) ')
                if pot_dup_react == "y":
                    final_df = None
                    yml = None
                    return final_df, yml
                
    #This adds the common name with a canonical smiles string and a new reactant with the canonical smiles string and reactant id to the dictionary
    else:
        react_smiles = input('1c. What is the reactant smiles? ')
        can_react_smiles = find_canonical_smiles(react_smiles)
        yml['react_can_smiles_map'][react_name] = can_react_smiles
        #this checks to see if the canonical smiles is already assigned to a reactant id in the dictionary
        if can_react_smiles in yml['react_id_smiles_map']:
            react_id = yml['react_id_smiles_map'][can_react_smiles]
            print('This reactant has been seen before, here are the series it has been seen in:')
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
                print(original_df[np.vectorize(lambda x: react_id == x)(original_df['Reactant ID'])])
            pot_dup_react = input('Would you like to restart? (Type "y" for yes, Hit [Enter] for No) ')
            if pot_dup_react == "y":
                final_df = None
                yml = None
                return final_df, yml
            
        else:
            yml['react_id_smiles_map'][can_react_smiles] = f'react_{len(yml["react_id_smiles_map"])}'
            react_id = yml['react_id_smiles_map'][can_react_smiles]

    all_values.extend([react_id,can_react_smiles])

    # Product
    prod_name = input("2a. What is the IUPAC product name? (Hit Enter if you don't want to add a common name to the library) ")
    if prod_name in yml['prod_can_smiles_map']:
        can_prod_smiles = yml['prod_can_smiles_map'][prod_name]
        prod_id = yml['prod_id_smiles_map'][can_prod_smiles]
        print('This product has been seen before, here are the series it has been seen in:')
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(original_df[np.vectorize(lambda x: prod_id == x)(original_df['Product ID'])])
        pot_dup_prod = input('Would you like to restart? (Type "y" for yes, Hit [Enter] for No) ')
        if pot_dup_prod == "y":
            final_df = None
            yml = None
            return final_df, yml
        
    elif prod_name == '':
        prod_smiles = input('2b. What is the product smiles? ')
        can_prod_smiles = find_canonical_smiles(prod_smiles)
        if can_prod_smiles in yml['prod_id_smiles_map']:
            prod_id = yml['prod_id_smiles_map'][can_prod_smiles]
            print('This product has been seen before, here are the series it has been seen in:')
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
                print(original_df[np.vectorize(lambda x: prod_id == x)(original_df['Product ID'])])
            pot_dup_prod = input('Would you like to restart? (Type "y" for yes, Hit [Enter] for No) ')
            if pot_dup_prod == "y":
                final_df = None
                yml = None
                return final_df, yml
        else:
            yml['prod_id_smiles_map'][can_prod_smiles] = f'prod_{len(yml["prod_id_smiles_map"])}'
            prod_id = yml['prod_id_smiles_map'][can_prod_smiles]

    else:
        prod_smiles = input('2c. What is the product smiles? ')
        can_prod_smiles = find_canonical_smiles(prod_smiles)
        yml['prod_can_smiles_map'][prod_name] = can_prod_smiles
        if can_prod_smiles in yml['prod_id_smiles_map']:
            prod_id = yml['prod_id_smiles_map'][can_prod_smiles]
            print('This product has been seen before, here are the series it has been seen in:')
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
                print(original_df[np.vectorize(lambda x: prod_id == x)(original_df['Product ID'])])
            pot_dup_prod = input('Would you like to restart? (Type "y" for yes, Hit [Enter] for No) ')
            if pot_dup_prod == "y":
                final_df = None
                yml = None
                return final_df, yml
        else:
            yml['prod_id_smiles_map'][can_prod_smiles] = f'prod_{len(yml["prod_id_smiles_map"])}'
            prod_id = yml['prod_id_smiles_map'][can_prod_smiles]

    all_values.extend([prod_id, can_prod_smiles])

    #Catalyst Questions
    cat_question_1 = input('3a. What is the commercial catalyst mixture? (Type "a" (DHQ-PHAL mix), "b" (DHQD-PHAL mix), or "c" (non-standard/non-commercial mixture)) ')
    if cat_question_1 == 'a':
        cat_id = 'cat_0'
        cat_des = 'alpha'
        cat_standard = 'Yes'
        cat_notes = ''
    elif cat_question_1 == 'b':
        cat_id = 'cat_1'
        cat_des = 'beta'
        cat_standard = 'Yes'
        cat_notes = ''
    elif cat_question_1 == 'c':
        ####Changed in part 4#####
        # cat_question_2 = input('3b. Is this using a non-standardized mixture of DHQ or DHQD PHAL with the same contents as a normal AD-mix? (Enter yes or no)? ')
        cat_question_2 = input('3b.  Is this still using of DHQ (alpha) or DHQD PHAL (beta) (Enter yes or no)? ')
        if cat_question_2 == 'yes':
            cat_question_3 = input('3c. What ligand is used (Type "a" (DHQ-PHAL mix) or "b" (DHQD-PHAL mix)? ')
            if cat_question_3 == 'a':
                cat_id = 'cat_0'
                cat_des = 'alpha'
                cat_standard = 'No'
            elif cat_question_3 == 'b':
                cat_id = 'cat_1'
                cat_des = 'beta'
                cat_standard = 'No'
            else:
                raise ValueError('Please answer question 3c. with a or b when deciding on a standardized mixture')
            cat_question_4 = input('3d. Please enter additional notes (consider entering something normalized such as "new metal: OsO4", "no base", "new oxidant: NMO", "multiple additives: NaHCO3") ')
            cat_notes = cat_question_4
        elif cat_question_2 == 'No':
            raise ValueError('Question 3b. is not currently implemented for non-alpha or non-beta mixtures')
        else:
            raise ValueError('Please answer Question 3b. non-standardized mixture of DHQ or DHQD (a or b) with a "Yes" or "No"')
    else:
        raise ValueError('Question 3a. not calibrated for catalysts that are not alpha, beta, or non-standard mixture (a, b, c)')

    all_values.extend([cat_id,cat_des,cat_standard,cat_notes])

    #Solvent 1
    sol1_name = input("4a. What is solvent 1 name? (Hit Enter if you don't want to add a common name to the library) ")
    if sol1_name in yml['sol1_can_smiles_map']:
        can_sol1_smiles = yml['sol1_can_smiles_map'][sol1_name]
        sol1_id = yml['sol1_id_smiles_map'][can_sol1_smiles]
    elif sol1_name == '':
        sol1_smiles = input('4b. What is the solvent 1 smiles? ')
        can_sol1_smiles = find_canonical_smiles(sol1_smiles)
        if can_sol1_smiles in yml['sol1_id_smiles_map']:
            sol1_id = yml['sol1_id_smiles_map'][can_sol1_smiles]
        else:
            yml['sol1_id_smiles_map'][can_sol1_smiles] = f'sol1_{len(yml["sol1_id_smiles_map"])}'
            sol1_id = yml['sol1_id_smiles_map'][can_sol1_smiles]
    else:
        sol1_smiles = input('4c. What is the solvent 1 smiles? ')
        can_sol1_smiles = find_canonical_smiles(sol1_smiles)
        yml['sol1_can_smiles_map'][sol1_name] = can_sol1_smiles
        if can_sol1_smiles in yml['sol1_id_smiles_map']:
            sol1_id = yml['sol1_id_smiles_map'][can_sol1_smiles]
        else:
            yml['sol1_id_smiles_map'][can_sol1_smiles] = f'sol1_{len(yml["sol1_id_smiles_map"])}'
            sol1_id = yml['sol1_id_smiles_map'][can_sol1_smiles]

    all_values.extend([sol1_id,can_sol1_smiles])

    #solvent 2
    sol2_name = input("5a. What is solvent 2 name? (Type None if there is not a second solvent) (Hit Enter if you don't want to add a common name to the library) ")
    if sol2_name == 'None':
        can_sol2_smiles = 'None'
        sol2_id = 'None'
    elif sol2_name in yml['sol2_can_smiles_map']:
        can_sol2_smiles = yml['sol2_can_smiles_map'][sol2_name]
        sol2_id = yml['sol2_id_smiles_map'][can_sol2_smiles]
    elif sol2_name == '':
        sol2_smiles = input('5b. What is the solvent 2 smiles? ')
        can_sol2_smiles = find_canonical_smiles(sol2_smiles)
        if can_sol2_smiles in yml['sol2_id_smiles_map']:
            sol2_id = yml['sol2_id_smiles_map'][can_sol2_smiles]
        else:
            yml['sol2_id_smiles_map'][can_sol2_smiles] = f'sol2_{len(yml["sol2_id_smiles_map"])}'
            sol2_id = yml['sol2_id_smiles_map'][can_sol2_smiles]
    else:
        sol2_smiles = input('5c. What is the solvent 2 smiles? ')
        can_sol2_smiles = find_canonical_smiles(sol2_smiles)
        yml['sol2_can_smiles_map'][sol2_name] = can_sol2_smiles
        if can_sol2_smiles in yml['sol2_id_smiles_map']:
            sol2_id = yml['sol2_id_smiles_map'][can_sol2_smiles]
        else:
            yml['sol2_id_smiles_map'][can_sol2_smiles] = f'sol2_{len(yml["sol2_id_smiles_map"])}'
            sol2_id = yml['sol2_id_smiles_map'][can_sol2_smiles]

    all_values.extend([sol2_id,can_sol2_smiles])

    #Solvent Ratio
    if sol2_name == 'None':
        sol_ratio = 'None'
    else:
        sol_ratio = eval(input('What is the ratio of solvent 1 to solvent 2? Input as ratio (i.e. 7/1 or 3/2) '))

    all_values.append(sol_ratio)

    #Oxidant
    if cat_standard == 'Yes':
        can_ox_smiles = yml['ox_can_smiles_map']['standard']
        ox_id = yml['ox_id_smiles_map'][can_ox_smiles]

    else:
        ox_name = input("6a. What is the oxidant name? (Hit Enter if you don't want to add a common name to the library) ")
        if ox_name in yml['ox_can_smiles_map']:
            can_ox_smiles = yml['ox_can_smiles_map'][ox_name]
            ox_id = yml['ox_id_smiles_map'][can_ox_smiles]

        elif ox_name == '':
            ox_smiles = input('6b. What is the oxidant smiles? ')
            can_ox_smiles = find_canonical_smiles(ox_smiles)
            if ox_smiles in yml['ox_id_smiles_map']:
                ox_id = yml['ox_id_smiles_map'][can_ox_smiles]
            else:
                yml['ox_id_smiles_map'][can_ox_smiles] = f'ox_{len(yml["ox_id_smiles_map"])}'

        else:
            ox_smiles = input('6c. What is the oxidant smiles? ')
            can_ox_smiles = find_canonical_smiles(ox_smiles)
            yml['ox_can_smiles_map'][ox_name] = can_ox_smiles
            if can_ox_smiles in yml['ox_id_smiles_map']:
                ox_id = yml['ox_id_smiles_map'][can_ox_smiles]
            else:
                yml['ox_id_smiles_map'][can_ox_smiles] = f'ox_{len(yml["ox_id_smiles_map"])}'
                ox_id = yml['ox_id_smiles_map'][can_ox_smiles]

    all_values.extend([ox_id,can_ox_smiles])

    #Oxidant solution
    if cat_des in ['alpha','beta']:
        can_ox_sol_smiles = 'None'
        ox_sol_id = 'None'
    else:
        ox_sol_name = input("7a. What is the oxidant solution's name? (Hit Enter if you don't want to add a common name to the library) ")
        if ox_sol_name in yml['ox_sol_can_smiles_map']:
            can_ox_sol_smiles = yml['ox_sol_can_smiles_map'][ox_sol_name]
            ox_sol_id = yml['ox_sol_id_smiles_map'][can_ox_sol_smiles]

        elif ox_sol_name == '':
            ox_smiles = input("What is the oxidant solution's smiles? ")
            can_ox_sol_smiles = find_canonical_smiles(ox_smiles)
            if ox_smiles in yml['ox_sol_id_smiles_map']:
                ox_sol_id = yml['ox_sol_id_smiles_map'][can_ox_sol_smiles]
            else:
                yml['ox_sol_id_smiles_map'][can_ox_sol_smiles] = f'ox_{len(yml["ox_sol_id_smiles_map"])}'

        else:
            ox_smiles = input("7b. What is the oxidant solution's smiles? ")
            can_ox_sol_smiles = find_canonical_smiles(ox_smiles)
            yml['ox_sol_can_smiles_map'][ox_sol_name] = can_ox_sol_smiles
            if can_ox_sol_smiles in yml['ox_sol_id_smiles_map']:
                ox_sol_id = yml['ox_sol_id_smiles_map'][can_ox_sol_smiles]
            else:
                yml['ox_sol_id_smiles_map'][can_ox_sol_smiles] = f'ox_{len(yml["ox_sol_id_smiles_map"])}'
                ox_sol_id = yml['ox_sol_id_smiles_map'][can_ox_sol_smiles]

    all_values.extend([ox_sol_id,can_ox_sol_smiles])

    #Oxidant to Oxidant solution
    if cat_des in ['alpha','beta']:
        ox_to_ox_sol = 'None'
    else:
        ox_to_ox_sol = input('8a. What is the ratio of oxidant to the oxidant solution (Do not give "1:1", give as float or int) (Hit [Enter] if none reported) ')
        if ox_to_ox_sol == '':
            pass
        else:
            ox_to_ox_sol = float(ox_to_ox_sol)

    all_values.append(ox_to_ox_sol)

    #Additive
    adtv_name = input("9a. What is the additive solution's name? (Hit [Enter] if there is no additive) ")
    if adtv_name  == '':
        can_adtv_smiles = 'None'
        adtv_id = 'None'

    elif adtv_name in yml['adtv_can_smiles_map']:
        can_adtv_smiles = yml['adtv_can_smiles_map'][adtv_name]
        adtv_id = yml['adtv_id_smiles_map'][can_adtv_smiles]

    elif adtv_name == '':
        adtv_smiles = input("9b. What is the additive's smiles? ")
        can_adtv_smiles = find_canonical_smiles(adtv_smiles)
        if adtv_smiles in yml['adtv_id_smiles_map']:
            adtv_id = yml['adtv_id_smiles_map'][can_adtv_smiles]
        else:
            yml['adtv_id_smiles_map'][can_adtv_smiles] = f'adtv_{len(yml["adtv_id_smiles_map"])}'

    else:
        adtv_smiles = input("9c. What is the additive solution's smiles? ")
        can_adtv_smiles = find_canonical_smiles(adtv_smiles)
        yml['adtv_can_smiles_map'][adtv_name] = can_adtv_smiles
        if can_adtv_smiles in yml['adtv_id_smiles_map']:
            adtv_id =yml['adtv_id_smiles_map'][can_adtv_smiles]
        else:
            yml['adtv_id_smiles_map'][can_adtv_smiles] = f'adtv_{len(yml["adtv_id_smiles_map"])}'
            adtv_id = yml['adtv_id_smiles_map'][can_adtv_smiles]

    all_values.extend([adtv_id, can_adtv_smiles])

    #Temperature (Celsius)
    temp_question = input("10. What is the temperature of the reaction? (Give in Celsius, or if not reported, Hit [Enter]) ")

    if temp_question == '':
        temp="Not Reported"
    else:
        try:
            temp = float(temp_question)
        except:
            raise ValueError('Question 10: Unable to convert temperature into float and value given not blank')


    #Time (hours)
    time_question = input("11. How long was the reaction run (in hours, or if not reported, hit [Enter])? ")
    if time_question == '':
        time = "Not Reported"
    else:
        try:
            time = float(time_question)
        except:
                raise ValueError('Question 11: Unable to convert time into float and value given not blank')

    # #substrate amount (mmol)
    # sub_amount = float(input('12. What is the amount of substrate used (in mmol)? '))

    all_values.extend([temp,time])


    # #Conversion
    # conv_question = input('13. What is the conversion of the reaction? (Give in %, if not given, type None) ')
    # try:
    #     conv = float(conv_question)
    # except:
    #     if conv_question != 'None':
    #         raise ValueError('Question 13 Error: Unable to convert conversion into float and value given not None')
    #     else:
    #         conv = "Not Reported"

    #Yield
    perc_prod_question = input('14. What is the yield of this reaction? (Give answer as %, if not given, hit [Enter]) ')
    if perc_prod_question == '':
        perc_prod = "Not Reported"
    else:
        try:
            perc_prod = float(perc_prod_question)
        except:
            raise ValueError('Question 14 Error: Unable to convert percent product into float and value given not blank')

    all_values.extend([perc_prod])

    # #Molarity
    # molarity_question = input('15a. Is the molarity given? (Type yes or no) ')
    # if molarity_question == 'yes':
    #     molarity = input('15b. What is the molarity of the reaction? ')
    # elif molarity_question == 'no':
    #     amount_of_sol = eval(input('15c. How much solvent was added? (Add up the total amount of solvent in the vessel in mL (inputs can be given as a math function (i.e. 50+50)) '))
    #     molarity = sub_amount/amount_of_sol
    # else:
    #     raise ValueError('Question 15a Error: Please give a valid answer for the molarity question (i.e. 1 or 0) ')

    # all_values.extend([conv, perc_prod])


    #Diastereoselectivity
    de_question_1 = input('16a. Does this reaction form a compound with more than 1 stereocenter? (If Yes, type "y", if No, hit [Enter]) ')
    if de_question_1 == 'y':
        multi_stereo = 'Yes'
        de_question_2 = input('17a. Does this reaction have any chiral centers/stereochemistry in the reactant? (If Yes, type "y", if No, hit [Enter]) ')
        if de_question_2 == '':
            de = 'N/A'
            dr = 'N/A'
            ddG_dr = 'N/A'
            ee_question = float(input('17b. What is the ee of this reaction? (Give as a percent) '))
            if 0 <= ee_question <= 100:
                ee = ee_question
                er = (50+(ee/2))/(50-(ee/2))
            else:
                raise ValueError('Question 17b Error: Only calibrated for ee percentages')
            
        elif de_question_2 == 'y':
            de = float(input('17c. What is the de of this reaction? (Give as a percent) '))
            dr = (50+(de/2))/(50-(de/2))
            if isinstance(temp,str):
                ddG_dr = float("nan")
            else:
                ddG_dr = ((-1)*(temp+273.15)*(8.314))*math.log(dr)*(0.000239)
        else:
            raise ValueError('Question 17b Error: Only calibrated for "y" and hit [Enter] arguments ')
        
    elif de_question_1 == '':
        multi_stereo = 'No'
        de = 'N/A'
        dr = 'N/A'
        ddG_dr = 'N/A'
        ee_question = float(input('17c. What is the ee of this reaction? (Give as a percent) '))
        if 0 <= ee_question <= 100:
            ee = ee_question
            er = (50+(ee/2))/(50-(ee/2))
        else:
            # raise ValueError('Question 16a Error: Only calibrated for ee or er')
            raise ValueError('Question 167 Error: Only calibrated for ee percentages')
    else:
        raise ValueError('Question 17a Error: Only accepts "y" or [Enter] arguments ')
 
    if isinstance(temp,str):
        ddG_er = float("nan")
    else:
        ddG_er = ((-1)*(temp+273.15)*(8.314))*math.log(er)*(0.000239)
    all_values.extend([ee,er,ddG_er, multi_stereo, de, dr, ddG_dr])

    #Olefin class
    olefin_type = input('18 What type of alkene is this? (Options are: Mono, Cis, Trans, Gem, Tri, Tetra) ')
    if olefin_type not in ['Mono' , 'Cis' , 'Trans' , 'Gem' , 'Tri' , 'Tetra']:
        raise ValueError('Question 18 Error: Must be equal to Mono, Cis, Trans, Gem, Tri, Tetra')

    #DOI
    doi = input('20. What is the DOI of this paper? ')
    all_values.extend([olefin_type,doi])

    #Additional notes (Added in p4)
    ad_notes = input('21. Do you have additional notes about this paper? Hit [Enter] if No (consider normalizing remarks, such as "product cyclizes") ')
    all_values.extend([ad_notes])
    all_values_np = np.array(all_values)

    if len(all_values) != len(original_df.columns):
        raise ValueError('The code did not populate all_values to align with all possible columns')
    else:
        #This creates a series where the index is the original dataframe's column
        new_ser = pd.Series(all_values_np,index=original_df.columns)
        temp_update_df = new_ser.to_frame()
        #Have to transpose series such that the columns match
        update_df = temp_update_df.transpose()
        #This adds the new information to the end of the excel file
        final_df =pd.concat([original_df,update_df], ignore_index=True)
        no_doi_or_note_test_df = final_df[final_df.columns[:-2]]
        duplicates = no_doi_or_note_test_df[no_doi_or_note_test_df.duplicated(keep=False)]
        print(duplicates.values)
        dup_test_df = no_doi_or_note_test_df.drop_duplicates()

        #These are the duplicates
        ###This is attempting to find if everything outside of the doi and additional notes matches
        if dup_test_df.index.shape != final_df.index.shape:
            print('This data has already been recorded in the database (other than the doi/additional notes). Here are the duplicate values:')
            print(duplicates)
            dup_question = input('Do these look like duplicates? (Hit [Enter] to return to the beginning, otherwise type "No" ')
            if dup_question == '':
                final_df = None
                yml = None
                return final_df, yml

        if overwrite_prev_files:
            final_df.to_csv(csv_file_name,index=False)
            with open(naming_yaml_file_name, 'wt') as f:
                yaml.dump(yml,f)
            print(f'{csv_file_name} and {naming_yaml_file_name} overwritten! Converting final_df and yml to None and returning both')
            final_df = None
            yml = None
            return final_df, yml
        else:
            print('Returning updated dataframe and yml file')
            return final_df,yml

'''
This workflow is dependent on the following naming scheme:

SAD_Database_{number}.csv
SAD_Database_{number}.yaml

This will search for the highest number of these and use that one as the starting point.
It will then iterate upwards until it reaches a maximum number. This script is not optimized
by any means, but allows consistent enumeration of new entries to the database. You can also 
implement shortcuts, which have already been done for certain entries and can be viewed in SAD_Database.yaml:

m = methanesulfonamide (additive)
k = potassium ferricyanide (oxidant)
t = tBuOH (Solvent 1)
w = water (Solvent 2)

You can add names to most categories continue using those as references for easier tabulation.
Many IUPAC names are stored for a variety of reactants and products.

'''

all_csvs = glob('*.csv')
all_yamls = glob('*.yaml')
csv_yaml_max = max(list(int(csv.split('.csv')[0].split('_')[5]) for csv in all_csvs))
# raise ValueError()
csv_file_name = f'SAD_Database_{csv_yaml_max}.csv'
yml_file_name = f'SAD_Database_{csv_yaml_max}.yaml'
new_file_idx = csv_yaml_max + 1

print(f'Using {csv_file_name} as the starting file')

#This allows a limit to be set for how many you are willing to do
while new_file_idx < 1100:
    print(f'Creating new reaction and new yaml/csv files with index {new_file_idx}')
    final_df, yml = add_entry_to_database(csv_file_name=csv_file_name, naming_yaml_file_name=yml_file_name, overwrite_prev_files=False)
    if not isinstance(final_df, pd.DataFrame):
        print(f'final_df not registered as an instance of a Dataframe, restarting new yaml/csv files\n')
        continue
    else:
        print('Current data added to row')
        print(final_df)
        print(final_df.iloc[new_file_idx-2])
        final_df.to_csv(f'SAD_Database_{new_file_idx}.csv',index=False)
        with open(f'SAD_Database_{new_file_idx}.yaml', 'wt') as f:
            yaml.dump(yml,f)
        csv_file_name = f'SAD_Database_{new_file_idx}.csv'
        yml_file_name = f'SAD_Database_{new_file_idx}.yaml'
        new_file_idx += 1