import molli as ml
from molli.pipeline.crest import CrestDriver

source = ml.ConformerLibrary("6_4_1_DB_CRESTSearch.clib")

target = ml.ConformerLibrary(
    "6_4_2_DB_CRESTScreen.clib",
    overwrite=True,
    readonly=False,
    comment="We did it!",
)

crest = CrestDriver(nprocs=16, memory=16*1024/2)

ml.pipeline.jobmap_sge(
    crest.conformer_screen,
    source,
    target,
    cache_dir="./screen_cache/",
    scratch_dir="/scratch/blakeo2/Dihydroxylation_Project/DB_crest",
    progress=True,
    verbose=True,
    kwargs={
        "method": "gfn2",
        "temp": 298.15,
        "ewin": 15.0,
        "chk_topo": True,
    },
    qsub_header="#$ -pe orte 16\n",
)
