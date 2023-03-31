import pandas as pd
import numpy as np

def find_mono(df:pd.DataFrame) -> np.ndarray:
    '''
    This will find a conditional filter for mono-substituted alkenes
    '''
    scaffold_type = df['Olefin Type']
    
    return np.vectorize(lambda x: "mono" in x)(scaffold_type)

def find_gem_di(df:pd.DataFrame) -> np.ndarray:
    '''
    This will find a conditional filter for gem-di-substituted alkenes
    '''
    scaffold_type = df['Olefin Type']
    
    return np.vectorize(lambda x: "gem_di" in x)(scaffold_type)

def find_cis_di(df:pd.DataFrame) -> np.ndarray:
    '''
    This will find a conditional filter for cis-di-substituted alkenes
    '''
    scaffold_type = df['Olefin Type']
    
    return np.vectorize(lambda x: "cis_di" in x)(scaffold_type)

def find_trans_di(df:pd.DataFrame) -> np.ndarray:
    '''
    This will find a conditional filter for trans-di-substituted alkenes
    '''
    scaffold_type = df['Olefin Type']
    
    return np.vectorize(lambda x: "trans_di" in x)(scaffold_type)

def find_tri(df:pd.DataFrame) -> np.ndarray:
    '''
    This will find a conditional filter for tri-substituted alkenes
    '''
    scaffold_type = df['Olefin Type']
    
    return np.vectorize(lambda x: "tri" in x)(scaffold_type)

def find_tetra(df:pd.DataFrame) -> np.ndarray:
    '''
    This will find a conditional filter for tetra-substituted alkenes
    '''
    scaffold_type = df['Olefin Type']
    
    return np.vectorize(lambda x: "tetra" in x)(scaffold_type)