import molli as ml

col = ml.Collection.from_zip('almost_all_alkenes_opt_gfnff_conf.zip')

xtb =ml.XTBDriver('sxtb', scratch_dir='/scratch/blakeo2/Dihydroxylation_Project', nprocs=4)
concur = ml.Concurrent(col, backup_dir='conf_props_pkl', concurrent=24, update=5, timeout=None)

_opt = concur(xtb.conformer_atom_props)(
    method='gfn2', 
    accuracy = 1.0, 
    backup_dir='conf_props_pkl')

