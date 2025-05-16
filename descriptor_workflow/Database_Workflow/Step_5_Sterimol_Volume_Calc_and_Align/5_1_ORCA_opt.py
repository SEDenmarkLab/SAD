import molli as ml
from molli.pipeline.orca import ORCADriver

source = ml.MoleculeLibrary("4_DB_mols_w_H.mlib")

target = ml.MoleculeLibrary(
    "5_1_DB_mols_OPT.mlib",
    overwrite=True,
    readonly=False,
    comment="We did it!",
)

orca = ORCADriver(
    nprocs=32,
    memory=28800,
    # envars={"OMP_NUM": "/opt/share/orca/scripts"}
)

#Runs a basic optimization calculation on the alkene database
ml.pipeline.jobmap_sge(
    orca.basic_calc_m,
    source,
    target,
    cache_dir="./DB_orca_cache/",
    scratch_dir="/scratch/blakeo2/Dihydroxylation_Project/DB_orca",
    progress=True,
    verbose=True,
    kwargs={
        "keywords": "b3lyp def2-svp opt rijcosx def2/j nopop miniprint","orca_suffix": "-mca btl ^sm"
    },
    qsub_header="#$ -q all.q -pe orte 32\n",
)
