import molli as ml

def generate_crest_conformers(
        col: ml.Collection,
        nprocs: int, 
        concurrent: int,
        backup_dir = './backup_dir',
        scratch_dir = '.',
        method='gfnff', 
        temp = 298.15,
        ewin = 6.0,
        charge = 0,
        mdlen=20, 
        mddump=250, 
        vbdump = 1.0,
        chk_topo=False, 
        constr_val_angles=[]):
    """
    Generates conformers using crest from a zip file and returns a new zip file
    You need to set the number of processors to use and how many jobs you want to do at once.
    mdlen refers to length of metadynamic simulations (in picoseconds).
    mddump refers to frequency in which coordinates are written to the trajectory file (in femtoseconds).
    chk_topo can be used to sidestep potential changes in topology if geometry looks super odd (default False).
    The method can be either gfn1, gfn2, gfnff, or gfn2//gfnff.
    Constr_val_angles can be used  
    """

    #You should change the scratch directory whenever you're using this
    crest = ml.CRESTDriver("screst", scratch_dir=scratch_dir, nprocs=nprocs)

    concur = ml.Concurrent(col, backup_dir=backup_dir, concurrent=concurrent, update=5, timeout=6000)
    _opt = concur(crest.conformer_search)(
        method=method, 
        temp=temp,
        ewin=ewin,
        charge=charge,
        mdlen=mdlen, 
        mddump=mddump,
        vbdump=vbdump,
        chk_topo=chk_topo, 
        constr_val_angles=constr_val_angles)

    conf_col = ml.Collection(f'{col.name}_{method}_conf', _opt)
    
    print(f'Succesfully generated CREST conformers for all structures in {col.name}')
    
    return conf_col

col = ml.Collection.from_zip('almost_all_alkenes_opt.zip')

conf_col = generate_crest_conformers(
    col = col,
    nprocs = 4, 
    concurrent = 24,
    backup_dir = './crest_backup',
    scratch_dir = '/scratch/blakeo2/Dihydroxylation_Project/crest_calc',
    method='gfnff', 
    temp = 298.15,
    ewin = 6.0,
    charge = 0,
    mdlen=20, 
    mddump=250, 
    vbdump = 1.0,
    chk_topo=False, 
    constr_val_angles=[],
)



conf_col.to_zip(f'{conf_col.name}.zip')