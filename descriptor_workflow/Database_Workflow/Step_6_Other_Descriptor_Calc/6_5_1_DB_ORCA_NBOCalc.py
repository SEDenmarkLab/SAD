import molli as ml
from molli.pipeline.orca import ORCADriver

# Do NBO calculations
source = ml.MoleculeLibrary("5_3_DB_OPT_AlignMaxVol.mlib")

target = ml.MoleculeLibrary(
    "5_4_0_DB_NBOCalc.mlib",
    overwrite=True,
    readonly=False,
    comment="We did it!",
)

orca = ORCADriver(
    nprocs=8,
    memory=48000,
    # envars={"OMP_NUM": "/opt/share/orca/scripts"}
)

ml.pipeline.jobmap(
    orca.basic_calc_m,
    source,
    target,
    cache_dir="./6_4_NBOCache/",
    n_workers=8,
    scratch_dir="/scratch/blakeo2/Dihydroxylation_Project/DB_Orca",
    progress=True,
    verbose=True,
    kwargs={
        "keywords": "b3lyp def2-svp nbo rijcosx def2/j nopop miniprint",
    },
)