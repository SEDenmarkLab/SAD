import molli as ml
from molli.pipeline.crest import CrestDriver

source = ml.MoleculeLibrary("5_3_DB_OPT_AlignMaxVol.mlib")

target = ml.ConformerLibrary(
    "6_4_1_DB_CRESTSearch.clib",
    overwrite=True,
    readonly=False,
    comment="We did it!",
)

crest = CrestDriver(nprocs=16, memory=16*1024/2)

ml.pipeline.jobmap_sge(
    crest.conformer_search,
    source,
    target,
    cache_dir="./cache/",
    scratch_dir="/scratch/blakeo2/Dihydroxylation_Project/DB_crest",
    progress=True,
    verbose=True,
    kwargs={
        "method": "gfnff",
        "temp": 298.15,
        "ewin": 15.0,
        "chk_topo": True,
    },
    qsub_header="#$ -pe orte 16\n",
)
