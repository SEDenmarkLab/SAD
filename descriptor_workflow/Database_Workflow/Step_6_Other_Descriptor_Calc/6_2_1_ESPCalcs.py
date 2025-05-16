import molli as ml
from molli.pipeline.nwchem import NWChemDriver


source = ml.MoleculeLibrary("6_1_DB_All_ESPFrags.mlib")

target = ml.MoleculeLibrary(
    "6_2_1_DB_All_ESPFrags_Calc.mlib",
    overwrite=True,
    readonly=False,
    comment="We did it!",
)

nwchem = NWChemDriver(
    nprocs=4,
)

ml.pipeline.jobmap(
    nwchem.calc_espmin_max_m,
    source,
    target,
    cache_dir="./cache/",
    n_workers=32,
    scratch_dir="/scratch/blakeo2/Dihydroxylation_Project/Redo_Original_Workflow/scratch",
    progress=True,
    verbose=True,
    kwargs={"basis": "6-31g*", "functional": "b3lyp", "optimize_geometry": False},
)
