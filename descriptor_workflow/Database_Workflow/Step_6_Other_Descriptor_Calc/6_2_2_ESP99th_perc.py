import molli as ml
from pprint import pprint
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from pathlib import Path
from tqdm import tqdm

'''
This script:
1. Takes the cache formed from the ESP calculation
2. Loads the outputs of the calculation
3. Reads the ESP values of the entire grid and assigns NWESPMin, NWESPMax, and 99ESPMax

'''

mlib = ml.MoleculeLibrary("6_2_1_DB_All_ESPFrags_Calc.mlib")
mlib_correct = ml.MoleculeLibrary(
    "6_2_2_DB_ESPFrags_Corrected.mlib", readonly=False, overwrite=True
)

with mlib.reading(), mlib_correct.writing():
    for file in tqdm(glob("./cache/output/*.out")):

        #Loads molli output file
        out = ml.pipeline.JobOutput.load(file)
        k = Path(file).stem
        if k in mlib:
            m = mlib[k]
        else:
            continue

        #Reads the grid from the ESP calculation
        res = out.files[f"esp.grid"]
        grid = res.decode()
        gc = np.loadtxt(grid.splitlines(), skiprows=1, usecols=(3)) * 2625.5  # kJ
        # Assigns new descriptor for 99th percentile of ESPMax
        m.attrib["NWESPmin"] = np.min(gc)
        m.attrib["NWESPMax"] = np.max(gc)
        m.attrib["99ESPMax"] = np.percentile(gc, 99)

        mlib_correct[k] = m

"""
The following values were redone separately from this script

failed = [
    "react_458_Q1",  # Iodine problems solved with 6-311G* basis set calculation
    "react_574_Q1",  # Redone manually with noautoz and 6-311G* basis set calculation
    "react_440_Q1",  # Redone manually with noautoz and 6-311G* basis set calculation
    "react_714_Q1",  # Redone manually with noautoz and 6-311G* basis set calculation
    "react_54_Q1",  # Redone manually with noautoz and 6-311G* basis set calculation
    "react_199_Q1",  # Redone manually with noautoz and 6-311G* basis set calculation
    "react_80_Q1",  # Iodine problems solved with 6-311G* basis set calculation
    "react_247_Q1",  # Redone manually with noautoz and 6-311G* basis set calculation
]

"""
