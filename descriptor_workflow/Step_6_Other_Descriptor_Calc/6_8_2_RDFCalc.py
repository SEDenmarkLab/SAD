import molli as ml
import numpy as np
from tqdm import tqdm

def _prop_rdf(
    conf: ml.Molecule,
    ref_atom: int,
    inc_size: float,
    const: float,
    num_spheres: int,
    radial_scaling: int =None,
    prop: str = "disp",
    ):

    #Establishes atom of interst
    center = conf.get_atom_coord(ref_atom)

    #Max radius is the constant times the number of incremental size increases
    #This is minus 1 because anything above the 9th sphere is considered a part of the 10th sphere
    max_radius = const + inc_size*(num_spheres - 1)

    #This creates 10 concentric spheres in the same style as Ian's original code
    sphere_refs = np.arange(const+inc_size, max_radius, inc_size)
    
    #This calculates the distances of the atoms towards the center
    dists = np.linalg.norm(conf.coords - center, axis=1)

    # sphere_assignments = defaultdict(list)
    sphere_assignments = {i: list() for i in range(num_spheres)}

    for i,d in enumerate(dists):

        #Considers radius of sphere n < dist <= radius of sphere n + 1
        sphere_idx = np.searchsorted(sphere_refs, d, side='left')

        #Finds dispersion value
        prop_val = conf.attrib[prop][i]

        if radial_scaling:
            prop_val = prop_val / (d**radial_scaling)

        #If it is above the constant but not outside the number of spheres, it gets counted
        sphere_assignments[sphere_idx].append(prop_val)
    
    #Sums Property Values for each sphere and returns the dictionary of values
    return {f'{sphere_idx}':np.sum(prop_vals) for sphere_idx, prop_vals in sphere_assignments.items()}

def calc_alk_rdf(
    ref_mlib: ml.MoleculeLibrary, 
    align_clib: ml.ConformerLibrary, 
    new_mlib: ml.ConformerLibrary,
    inc_size: float = 0.90,
    const: float = 1.8,
    num_spheres: int = 10,
    radial_scaling: int =0,
    prop: str = "disp",):
    print(f'''
Reference Library: {ref_mlib}
Old Conformer Library: {align_clib}
New Molecule Library Written To: {new_mlib}
''')
    with ref_mlib.reading(), align_clib.reading(), new_mlib.writing():
        for name in tqdm(ref_mlib):

            m = ref_mlib[name]

            #Isolates Alkene Carbons
            # c0, c1 = [m.get_atom(i) for i in m.attrib['C Order']]
            c0, c1 = m.attrib['C Order']
            ens = align_clib[name]
            #Find the atom property of interest and assign the array to the ensemble which will make it available for every conformer
            ens.attrib[prop] = np.array([float(a.attrib[prop]) for a in m.atoms])

            c0_desc = {f"{i}": list() for i in range(num_spheres)}
            c1_desc = {f"{i}": list() for i in range(num_spheres)}

            for conformer in ens:
                #Calculates C0 and C1 RDF
                c0_conf_res = _prop_rdf(
                    conformer,
                    ref_atom=c0,
                    inc_size=inc_size,
                    const=const,
                    num_spheres=num_spheres,
                    radial_scaling=radial_scaling,
                    prop=prop
                )
                c1_conf_res = _prop_rdf(
                    conformer,
                    ref_atom=c1,
                    inc_size=inc_size,
                    const=const,
                    num_spheres=num_spheres,
                    radial_scaling=radial_scaling,
                    prop=prop
                )

                #Appends the summed values to their respective lists
                for idx, val in c0_conf_res.items():
                    # print(idx)
                    # print(val)

                    c0_desc[idx].append(val)
                    # raise ValueError()
                for idx, val in c1_conf_res.items():
                    c1_desc[idx].append(val)

            # Calculates the final RDF then converts the numpy sum back to floats
            c0_rdf = {f'{sphere_idx}':np.mean(prop_vals).item() for sphere_idx, prop_vals in c0_desc.items()}
            c1_rdf = {f'{sphere_idx}':np.mean(prop_vals).item() for sphere_idx, prop_vals in c1_desc.items()}

            m.get_atom(c0).attrib['C0 RDF Series'] = c0_rdf
            m.get_atom(c1).attrib['C1 RDF Series'] = c1_rdf

            m.attrib["RDF Series"] = {'C0 RDF': c0_rdf, "C1 RDF": c1_rdf}

            new_mlib[name] = m

max_mlib = ml.MoleculeLibrary("5_6_Realign_MaxVol.mlib")
max_clib = ml.ConformerLibrary("5_7_0_CRESTScreen_MaxAlign.clib")
new_max_mlib = ml.MoleculeLibrary("5_7_1_RDF_Realign_MaxVol.mlib", readonly=False, overwrite=True)

calc_alk_rdf(ref_mlib=max_mlib, align_clib=max_clib, new_mlib=new_max_mlib)

bfs_mlib = ml.MoleculeLibrary("5_6_Realign_BFSVol.mlib")
bfs_clib = ml.ConformerLibrary("5_7_0_CRESTScreen_BFSAlign.clib")
new_bfs_mlib = ml.MoleculeLibrary("5_7_1_RDF_Realign_BFSVol.mlib", readonly=False, overwrite=True)

calc_alk_rdf(ref_mlib=bfs_mlib, align_clib=bfs_clib, new_mlib=new_bfs_mlib)