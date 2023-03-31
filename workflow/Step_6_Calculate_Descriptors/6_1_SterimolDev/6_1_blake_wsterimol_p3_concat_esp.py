import numpy as np
import pickle
import pandas as pd
import molli as ml
import yaml
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial import ConvexHull

with open('cpk_full_vdw_dict.yaml') as f:
    yml = yaml.safe_load(f)

def rotation_matrix(v1, v2, tol=1.0e-6):
    """
    Rotation Matrix (vector-to-vector definition)
    Code from Alex Shved, Denmark Laboratory
    ---

    Computes a 3x3 rotation matrix that transforms v1/|v1| -> v2/|v2|
    tol detects a situation where dot(v1, v2) ~ -1.0
    and returns a diag(-1,-1,1) matrix instead. NOTE [Is this correct?!]
    This may be improved by figuring a more correct asymptotic behavior, but works for the present purposes.

    returns matrix [R], that satisfies the following equation:
    v1 @ [R] / |v1| == v2 / |v2|

    Inspired by
    https://en.wikipedia.org/wiki/Rotation_matrix
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

    """
    if not v1.size == v2.size == 3:
        raise ValueError("Vectors must have a size of 3")

    v1n = np.array(v1) / np.linalg.norm(v1)
    v2n = np.array(v2) / np.linalg.norm(v2)

    I = np.eye(3)

    Ux = np.outer(v1n, v2n) - np.outer(v2n, v1n)
    c = np.dot(v1n, v2n)

    if c <= -1 + tol:
        return np.diag([-1, -1, 1])

    else:
        return I + Ux + Ux @ Ux / (1 + c)

def angle_between_vectors(v1, v2):
    '''
    Used to calculate the angle between two vectors (used to check that the b_vectors and l_vectors are orthogonal)
    '''
    # calculate the dot product
    dot_product = np.dot(v1, v2)
    #   print(f'dot product is {dot_product}')
    # calculate the magnitudes of the vectors
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    #   print(f'mag1 is {magnitude_v1}')
    #   print(f'mag2 is {magnitude_v2}')
    angle = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))
    #   print(f'angle is {angle}')
    # calculate the angle between the vectors
    return angle

def calc_convex_hull(full_b_arr,l_vec):
    '''
    Calculates a convex hull for the b_vector array and returns the indices of the vertices generated:

    This creates a rotation matrix by rotating an l_vector (which is orthogonal to any b_vectors) to be along the z axis.
    This then rotates all the vectors in the b_vector array to the xy plane using the same rotation matrix.
    It then calculates a convex hull using only the x,y coordinates and returns the array of indices corresponding to the vertices of the convex hull.
    This should translate directly to original b_vector array snice the vertex indices are based on the original input array (full_b_arr).
    '''
    r = rotation_matrix(l_vec, np.array([0,0,1]))
    new_vecs = full_b_arr @ r
    vec_arr_2d = new_vecs[:,0:2]
    hull = ConvexHull(vec_arr_2d)
    vertex_arr = hull.vertices

    return vertex_arr

def plot_sterimol_result(all_coord, all_bond_loc, extend_v_a1n_arr, b_vectors, b1, b5, l_vectors, l_max, name='sterimol', all=True, planar=True, bond_coord=True, extend=True, plot_b=True,plot_l=True):
    '''
    Plots all the vectors created:

    All Coordinates = original coordinates (colored green)
    Extended Coordinates = original coordinates + vdw radii extension (colored orange)
    B vectors = all points representing b vectors should be normalized to a plane (colored black)
    L vectors = all points representing l vectors should be normalized to a plane (colored blue)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(False)

    #This plots all of the original coordinates in green
    if all:
        ax.scatter(all_coord[:,0],all_coord[:,1], all_coord[:,2], color='green')
    #This plots all of the original coordinates extended by their respective van der Waals radii in orange
    if extend:
        ax.scatter(extend_v_a1n_arr[:,0],extend_v_a1n_arr[:,1], extend_v_a1n_arr[:,2], color='orange')
    #This plots all the points representing l vectors as single blue dots along the axis
    if plot_l:
        ax.scatter(l_vectors[:,0],l_vectors[:,1], l_vectors[:,2],color='blue')
        #This plots the l vector with the longest magnitude along the primary axis as a red line
        l_mags = np.linalg.norm(l_vectors, axis=1)
        l_idx = np.where(l_mags == np.max(l_max))[0]
        l_vec = Line3DCollection([[[0,0,0],l_vectors[l_idx][0]]], colors='red', linestyle='solid')
        ax.add_collection3d(l_vec)
    #This plots all the vectors as single black dots along the plane that has been created
    if plot_b:
        ax.scatter(b_vectors[:,0],b_vectors[:,1], b_vectors[:,2],color='black')
        #This plots the b1 and b5 vectors with the shortest/longest magnitudes orthogonal to the primary axis as red lines
        b_mags = np.linalg.norm(b_vectors, axis=1)
        if planar:
            b_mag = b1
            b1_vec = Line3DCollection([[[0,0,0],[0,0,b_mag]]], colors='red', linestyle='solid')
            ax.add_collection3d(b1_vec)
        else:
            b1_idx = np.where(b_mags == b1)[0]
            b1_vec = Line3DCollection([[[0,0,0],b_vectors[b1_idx][0]]], colors='red', linestyle='solid')
            ax.add_collection3d(b1_vec)
        b5_idx = np.where(b_mags == b5)[0]
        b5_vec = Line3DCollection([[[0,0,0],b_vectors[b5_idx][0]]], colors='red', linestyle='solid')
        ax.add_collection3d(b5_vec)


    #This line can be used to create a solid black lines representing the structure
    if bond_coord:
        for bond in all_bond_loc:
            bond_a1 = bond[0]
            bond_a2 = bond[1]
            line_seg = Line3DCollection([[all_coord[bond_a1],all_coord[bond_a2]]], colors='black', linestyle='solid')
            ax.add_collection3d(line_seg)

    # fig.savefig(f'{name}.png')
    plt.show()

def dftd_cn(a1:ml.dtypes.Atom, mol: ml.Molecule, elem_num: dict, g_cov_rad: dict, conf_num:int =-1):
    '''
    This is the coordination number defined by Grimme in "https://doi.org/10.1063/1.3382344".
    All covalent radii for metals will be scaled down by 10%, and is built to error if the element used is not 1-94.
    Designed for use with CPK Rules.
    
    Needs a bit further stress testing before being fully confirmed as good to go with reference to conformer geometry, but current implementation seems to be working
    '''

    k1 = 16
    k2 = 4/3 # SAS sez: better compute and have a float but what do I know

    atom_num = elem_num
    atom_arr = np.array(mol.atoms)

    atom_elem_arr = np.array([atom_num[atom.symbol] for atom in atom_arr])
    assert 0 < np.max(atom_elem_arr) < 94, f'The following elements do not fall between 1 to 94: {atom_arr[~((atom_elem_arr <= 94) & (0 < atom_elem_arr))]}'

    ra1_cov = g_cov_rad[a1.symbol]

    #This creates an "atom2" array where the atom is not equal to atom1
    a2_arr = atom_arr[np.where(atom_arr != a1)]

    #This creates an array matching grimme covalent radii to each atom in atom2
    ra2_cov_arr = np.array([g_cov_rad[atom.symbol] for atom in a2_arr])
    #This adds the scalar ra1_cov to every atom in the arr of ra2_cov_arr

    #This is an array of all of the magnitudes of the vector a1 -> a2 multiplied by the constant k2
    #Coordinates of atom1 (shape = (1,3) ) and array of atom2 (shape = (n_atoms-1,3) )
    a1_coords = mol.get_subgeom([a1],conformer=conf_num).coord
    a2_coord_arr = mol.get_subgeom(a2_arr,conformer=conf_num).coord
    #This line finds the magnitude of each row with shape of (n_atoms-1,1) (new radius vector)
    rab = np.linalg.norm(a1_coords-a2_coord_arr, axis=1)
    # print(rab)
    #This line reshapes the radius array to match the shape of the rco matrix (shape = (n_atoms-1,) )
    # rab = np.reshape(rab,(len(mol.atoms)-1,)) # SAS sez: dis stoopid. FY Blake

    inn = -k1*(k2*(ra1_cov + ra2_cov_arr)/rab - 1)  # SAS sez: it looks cool if it's just like in the paper.

    e = 1 + np.exp(inn)
    cn = np.sum(1/e)

    return cn

def assign_cpk(atom_list: list, cn:float, ster_cpk_match: dict):
    '''
    Adapted from Rob Paton Group's sterimol/CPK matcher
    See: https://github.com/bobbypaton/wSterimol 
    '''
    cpk_radii = list()

    atoms_sym_list = [atom.symbol for atom in atom_list]
    for sym in atoms_sym_list:
        if sym == 'O':
            if cn < 1.5:
                cpk_radii.append(ster_cpk_match['O2'])
            else:
                cpk_radii.append(ster_cpk_match['O'])
        elif sym == 'S':
            if cn < 2.5:
                cpk_radii.append(ster_cpk_match['S'])
            elif 2.5 < cn < 5.5:
                cpk_radii.append(ster_cpk_match['S4'])
            else:
                cpk_radii.append(ster_cpk_match['S1'])
        elif sym == 'N':
            if cn > 2.5:
                cpk_radii.append(ster_cpk_match['N'])
            else:
                cpk_radii.append(ster_cpk_match['C6/N6'])
        elif sym == 'C':
            if cn < 2.5:
                cpk_radii.append(ster_cpk_match['C3'])
            elif 2.5 < cn < 3.5:
                cpk_radii.append(ster_cpk_match['C6/N6'])
            else:
                cpk_radii.append(ster_cpk_match['C'])
        elif sym == 'Cl':
            cpk_radii.append(ster_cpk_match['C1'])
        elif sym == 'Br':
            cpk_radii.append(ster_cpk_match['B1'])
        elif sym in ['H','P','F','I']:
            cpk_radii.append(ster_cpk_match[sym])
        else:
            raise KeyError(f'Atom is not supported by the CPK Model : {sym}')
    
    assert '' not in cpk_radii, f'CPK Radii List Built Incorrectly: {cpk_radii}'

    cpk_radii_dict = dict(zip(atom_list,cpk_radii))

    return cpk_radii_dict

def calc_sterimol(mol: ml.Molecule, a1: ml.dtypes.Atom, a2: ml.dtypes.Atom, yml_props:dict, radii='bondi', calc_vol=False, conf_num=-1, print_outcome=True, plot_result=False):
    '''
    This calculates steirmol based on either bond vdw or cpk radii.

    A dictionary is necessary to supply for this function that contains the vdw_radius, real cov_rad, element number, grimme_corrected cov_radius, and sterimol-> cpk match

    This returns 3 values:

    --> b1 (shortest distance perpendicular to the primary axis of attachment)

    --> b5 (longest distance perpendicular to the primary axis of attachment)

    --> l (total distance following the primary axis)

    FUNDAMENTAL CONCEPTS:
    + For B1 and B5:

        Consider that the vector is drawn to the center of each atom: 

        --> This means along this vector, there is another vector that is also equal to this vdw_radius that is projected into the plane.

        --> This implies that the longest possible vector that can be created is the one that fills in the other half of the radius vector.

        --> If each vector is extended by one van der Waals radii (i.e. vdw_radii*unit vector), this should equal the maximum vector in that direction.

        --> These vectors can then be projected onto the plane of us looking down the primary axis

        If a vector projection is nearly orthogonal to the plane, it will appear very short when projected to the plane

        --> This does not allow for easy calculation of B1/B5, as this will not represent the overlapping spheres very well and the true distance to the edge of these spheres.

        --> This can be solved using a ConvexHull approach, where a minimal surface of the vectors is calculated to create the "edges of the radii"

        --> The smallest distance to a vertex in the convex hull should represent B1, while the largest distance should represent B5
        
        Corollary: When looking down the primary axis at the 2D image, it is impossible for the shortest/longest vectors to not be represented at the edge of the sphere.
        
    + Dealing with L

        Calculation of L follows similar logic as 1, but it must fall along the original axis defined by atom1 and atom2:

        --> All that needs to occur is the projection of every vector onto the normal vector, which has already been done with the original "proj" calculation for B1/B5.

        --> All of the vectors now fall on the axis.
        
        --> Since L is defined as the total distance along this primary axis starting at atom2, to get the final vectors, v1v2 can be subtracted giving all possible L vectors.

        --> The magnitude of the longest vector should be equal to L.

        Assumption: Vectors do not need to be reflected to all be oriented in the same direction since only the magnitude to atom2 matters.
    '''

    bondi_rad = yml_props['vdw_radius']
    atom_num = yml_props['elem_num']
    g_cov_rad = yml_props['grimme_cov_rad']
    ster_cpk_match = yml_props['ster_cpk_match']

    if radii == 'cpk':
        cn = dftd_cn(a1=a1, mol=mol, elem_num=atom_num, g_cov_rad=g_cov_rad, conf_num=-1)
        sterimol_dict = assign_cpk(atom_list=mol.atoms, cn=cn, ster_cpk_match=ster_cpk_match)
    elif radii == 'bondi':
        sterimol_dict = {atom: bondi_rad[atom.symbol] for atom in mol.atoms}
    else:
        raise KeyError('Only calibrated for use with "cpk" or "bondi" vdw radii.')

    all_coords_geom = mol.get_subgeom(mol.atoms, conformer=conf_num)
    all_coords_geom.set_origin(mol.get_atom_idx(a1))
    all_coords = all_coords_geom.coord

    a1_coord = all_coords[mol.get_atom_idx(a1)]
    a2_coord = all_coords[mol.get_atom_idx(a2)]

    #The original vector and unit vector connecting a1 (attachment point) and a2 (first real atom)
    v_a12 = np.subtract(a2_coord,a1_coord)
    
    mag_v_a12 = np.linalg.norm(v_a12)
    uv_a12 = np.divide(v_a12, mag_v_a12)

    #This is used to create a matrix of coordinates that are not atom1 or atom2
    no_a1_a2_atom_list = [atom for atom in mol.atoms if (atom != a1) and (atom != a2)]

    if no_a1_a2_atom_list == list():
        b1 = 0
        b5 = 0
        l = 0
        if not calc_vol:
            return b1, b5, l
        else:
            vol = 0
            return b1, b5, l, vol
        
    no_a1_a2_atom_mask = [True if (atom != a1) and (atom != a2) else False for atom in mol.atoms ]
    num_atoms_no_a1_a2 = len(no_a1_a2_atom_list)
    no_a1_a2_coord_arr = all_coords[no_a1_a2_atom_mask]

    #This creates an array of vectors atom 1 -> atom n-2, as well as unit vectors
    v_a1n_arr = np.subtract(no_a1_a2_coord_arr,a1_coord)
    v_a1n_arr_mags =  np.reshape(np.linalg.norm(v_a1n_arr, axis=1),newshape=(num_atoms_no_a1_a2,1))
    uv_a1n_arr = np.divide(v_a1n_arr,v_a1n_arr_mags)

    # This matches the atom_list without a1 and a2 to the vdw_radii, multiplies the unit vectors by these radii
    vdw_radii_match = np.reshape(np.array([sterimol_dict[atom] for atom in no_a1_a2_atom_list]),newshape=(num_atoms_no_a1_a2,1))
    uv_a1n_radii_extend = np.multiply(uv_a1n_arr,vdw_radii_match)
    full_v_a1n_arr = np.add(no_a1_a2_coord_arr,uv_a1n_radii_extend)

    #The array of coordinates must be flipped so the shape is of (3, n) to do the dot product correctly in case of mismatched arrays, it is then immediately flipped back
    flip = np.transpose(full_v_a1n_arr)
    v_a12_dot_v_a1an = np.reshape(np.transpose(np.dot(v_a12,flip)), newshape=(num_atoms_no_a1_a2,1))
    comp_a12_an = np.divide(v_a12_dot_v_a1an,mag_v_a12)

    #This calculates the projection of all vectors atom 1 -> atom n-2 onto the unit vector atom 1 -> atom 2
    proj_a1n = np.multiply(uv_a12,comp_a12_an)

    #L is measured as distance along the primary axis
    full_l_vectors = proj_a1n
    full_l_vectors_mags = np.linalg.norm(full_l_vectors, axis=1)
    l = np.max(full_l_vectors_mags)
    l_vec = full_l_vectors[np.where(full_l_vectors_mags == l)[0]][0]

    #This places removes the projection of the full array of vectors -> v_a12 to place all the vectors orthogonal for calculation of the b vectors
    full_b_vectors = np.subtract(full_v_a1n_arr,proj_a1n)
    full_b_vectors_mag = np.linalg.norm(full_b_vectors, axis=1)

    #This calculates the convex hull and its vertices after being projected onto the xy plane to allow for calculation of b1
    try:
        red_vertex_arr = calc_convex_hull(full_b_arr=full_b_vectors, l_vec=full_l_vectors[0])
    except Exception as e:
        red_b_vectors = full_b_vectors
        red_b_vectors_mag = full_b_vectors_mag
        # print(e)
        # print(red_b_vectors_mag)
    else:
        red_b_vectors = full_b_vectors[red_vertex_arr]
        red_b_vectors_mag = np.linalg.norm(red_b_vectors,axis=1)

    b1 = np.min(red_b_vectors_mag)
    b5 = np.max(red_b_vectors_mag)

    #This is used to check that b1_vec and b5_vec are still matching after the convex hull calculation
    b1_vec = full_b_vectors[np.where(full_b_vectors_mag == b1)[0],:][0]
    b5_vec = full_b_vectors[np.where(full_b_vectors_mag == b5)[0],:][0]

    # print(b1)
    # print(angle_between_vectors(b1_vec,b5_vec))
    # print(abs(np.dot(b1_vec,b5_vec)))
    # print(np.multiply(np.linalg.norm(b1_vec),np.linalg.norm(b5_vec))*0.996)
    old_b1 = False

    #Checking that b1_vec and b5_vec are not antiparallel
    # if b1 < 0.01 or (abs(np.dot(b1_vec,b5_vec)) >= np.multiply(np.linalg.norm(b1_vec),np.linalg.norm(b5_vec))*0.996):
    #     print(f'planar molecule or planar b1_b5')
    #     old_b1 = b1
    #     dummy_b1 = np.cross(v_a12,b5_vec)
    #     uv_dummy_b1 = np.divide(dummy_b1,np.linalg.norm(dummy_b1))
    #     b1 = sterimol_dict[a2]
    #     b1_vec = np.add(a1_coord,np.multiply(uv_dummy_b1,b1))    
    #     print(b1_vec)

    assert (angle_check := angle_between_vectors(b1_vec,l_vec)) < 1.59, f'angle between b1 and l = {angle_check}' 
    assert (angle_check := angle_between_vectors(b5_vec,l_vec)) < 1.59, f'angle between b5 and l = {angle_check}' 
    assert round(b1,6) == round(np.linalg.norm(b1_vec),6), f'b1 is {b1}, b1_vec_norm = {np.linalg.norm(b1_vec)}'
    assert round(b5,6) == round(np.linalg.norm(b5_vec),6), f'b1 is {b5}, b1_vec_norm = {np.linalg.norm(b5_vec)}'


    if print_outcome:
        print(f'For molecule {mol.name}, conformer {conf_num}, with atom1 ({a1}) and atom2 ({a2}), and {radii} vdw_type, B1 = {b1}, B5 = {b5}, L = {l}')

    if plot_result:
        all_bond_locs = list()
        for i,bond in enumerate(mol.bonds):
            all_bond_locs.append((mol.get_atom_idx(bond.a1),mol.get_atom_idx(bond.a2)))

        if old_b1:
            plot_sterimol_result(all_coord=all_coords, all_bond_loc=all_bond_locs,extend_v_a1n_arr=full_v_a1n_arr, b_vectors=full_b_vectors, b1=b1,b5=b5, l_vectors=full_l_vectors, l_max=l, name=f'{mol.name}_{conf_num}_sterimol', 
            planar=True,all=True,bond_coord=True,extend=True,plot_b=True,plot_l=True)
        else:
            plot_sterimol_result(all_coord=all_coords, all_bond_loc=all_bond_locs,extend_v_a1n_arr=full_v_a1n_arr, b_vectors=full_b_vectors, b1=b1,b5=b5, l_vectors=full_l_vectors, l_max=l, name=f'{mol.name}_{conf_num}_sterimol', 
        planar=False,all=True,bond_coord=True,extend=True,plot_b=True,plot_l=True)
        
        #This can be used to just plot all vectors related to b_vectors and l_vectors
        # plot_sterimol_result(all_coord=all_coords, all_bond_loc=all_bond_locs,extend_v_a1n_arr=full_v_a1n_arr, b_vectors=full_b_vectors, b1=b1,b5=b5, l_vectors=full_l_vectors, l_max=l, name=f'{mol.name}_{conf_num}_sterimol', 
        # all=False,bond_coord=False,extend=False,plot_b=True,plot_l=True)

    # print(mol.name)
    # print(b1,b5,l)
    # print(b1_vec)
    # print(b5_vec)
    # print(l_vec)
    # raise ValueError()
    if calc_vol:
        #This creates a 3x3 matrix vertically
        conf_d_array = np.vstack((b1_vec, b5_vec, l_vec))
        #The determinant of a 3x3 vertically stacked matrix is the triple product ((V1 x V2) * V3)
        vol = np.abs(np.linalg.det(conf_d_array))
        return b1,b5,l, vol
    else:
        return b1, b5, l

def boltzmann_weight(energy_arr: np.array, temp: float):
    '''
    This accepts an energy array in units of hartree, and it returns an array of weights corresponding to the boltzmann distribution based on Joules/mol.
    '''
    #Constants
    r = 8.314 # J K−1 mol−1
    hartree_to_Jmol = 2625499.639 # hartree to J/mol
    r_t = r* temp

    #Array converted to Joules 
    j_arr = np.multiply(energy_arr, hartree_to_Jmol)
    scale_energy = np.min(j_arr)
    scaled_arr = np.subtract(j_arr, scale_energy)

    ##Boltzmann Distribution Energies with results in J/mol instead of J/molecule
    distrib_val = np.exp(np.divide(np.negative(scaled_arr), r_t))

    b_weights = np.divide(distrib_val,np.sum(distrib_val))

    return b_weights

def calc_wsterimol(mol:ml.Molecule, a1: ml.dtypes.Atom, a2: ml.dtypes.Atom, yml_props:dict, energy_arr: np.array, sterimol_outcome=True, radii='bondi', temp: float = 298.16):
    '''
    This calculates wsterimol based on a boltzmann weighted average. 
    Sterimol can be calculated using "cpk" radii or "bondi" radii.
    This will return a dictionary corresponding to the name of each fragment in the order of b1, b5, l
    '''

    wsterimol_dict = dict()

    #This will become a matrix defined by B1, B5, and L
    all_vals = np.zeros(shape=(len(mol.conformers), 3))

    for i,c in enumerate(mol.conformers):
        b1, b5, l = calc_sterimol(mol=mol, a1=a1, a2=a2, yml_props=yml_props,radii=radii,conf_num=i, outcome=sterimol_outcome)
        all_vals[i] = b1,b5,l
        raise ValueError()
    
    #This finds the weighted array based on an energy input and temperature
    weight_arr = boltzmann_weight(energy_arr=energy_arr, temp=temp)

    wsterimol_result = np.average(all_vals, weights=weight_arr, axis=0)

    #This writes the result to a dictionary
    wsterimol_dict[mol.name] = wsterimol_result

    return wsterimol_dict

col = ml.Collection.from_zip('SAD_Step_5_All_Molli_Frags_H_3_01_2023.zip')

sterimol_dict = dict()

# csv_name='hydrogen_test'
# esp_pkl_name='esp_test'

csv_name = 'Step_6_1000_Entry_Sterimol_ESP_Frag_desc_df'
esp_pkl_name = 'Step_6_1000_Entry_ESP_Calc.pkl'

for i,mol in enumerate(col.molecules):
    # print(mol.name)
    # if 'react_767' in mol.name:
    #     print(mol.name)
        # with open(f'{mol.name}.xyz', 'w') as f:
        #     f.write(mol.to_xyz())
    assert mol.atoms[0].label == 'H', f'first atom is {mol.atoms[0].label}'
    b1, b5, l, vol = calc_sterimol(mol=mol, a1=mol.atoms[0], a2=mol.atoms[1], yml_props=yml, calc_vol=True, radii='bondi',conf_num=-1, print_outcome=False,plot_result=False)
    sterimol_params = np.array([b1,b5,l, vol])
    # fix_name = '_'.join(mol.name.split('_')[:-1])
    fix_name = mol.name
    # print(sterimol_params)
    # print(fix_name)
    # raise ValueError()
    sterimol_dict[fix_name] = sterimol_params
    # print(sterimol_dict)

# print(sterimol_dict)
# raise ValueError()
# assert len(col.molecules) == len(sterimol_dict), f'{len(col.molecules)} molecules, {len(sterimol_dict)}'

# raise ValueError()
# print(sterimol_dict['react_96'])
sterimol_df = pd.DataFrame(data=sterimol_dict.values(),columns=['b1','b5','l','vol'], index=sterimol_dict.keys())

# print(sterimol_df)

with open(f'{esp_pkl_name}', 'rb') as f:
    esp_dict = pickle.load(f)

new_esp_dict = dict()
new_esp_dict.update((key,value) for key, value in esp_dict.items())
# print(new_esp_dict)

esp_df = pd.DataFrame(data=new_esp_dict.values(),columns=['espmin','espmax'], index=new_esp_dict.keys())

react_desc_df = pd.concat([sterimol_df,esp_df], axis=1)
print(react_desc_df)

# raise ValueError()
react_desc_df.to_csv(f'{csv_name}.csv')


