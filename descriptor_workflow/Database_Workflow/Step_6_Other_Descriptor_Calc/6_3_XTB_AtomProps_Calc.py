import molli as ml
from molli.pipeline.xtb import XTBDriver

source = ml.MoleculeLibrary("5_3_DB_OPT_AlignMaxVol.mlib")

target = ml.MoleculeLibrary(
    "6_3_DB_OPT_XTBProps.mlib",
    overwrite=True,
    readonly=False,
    comment="We did it!",
)

xtb = XTBDriver(nprocs=16, memory=16*1024/2)

ml.pipeline.jobmap_sge(
    xtb.atom_properties_m,
    source,
    target,
    cache_dir="./xtb_cache/",
    scratch_dir="/scratch/blakeo2/Dihydroxylation_Project/AtomPropsCalc",
    progress=True,
    verbose=True,
    kwargs={
        "method": "gfn2",
        "accuracy": 1.0,
    },
    qsub_header="#$ -pe orte 16\n",
)