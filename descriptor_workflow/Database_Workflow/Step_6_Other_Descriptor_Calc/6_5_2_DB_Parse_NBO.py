import molli as ml
from glob import glob
from pathlib import Path

def nbo_parse_values(output: str):
    """
    This parses the NBO population analysis. This returns 4 pieces of data in the following format:

    - orb_homo_lumo = (homo, lumo)

    - nat_charge_dict = {atom number : natural charge}

    - pert_final_list = [((atom1,atom2), second order energy (kcal))] (Note, this only stores the acceptor orbital)

    - nbo_orb_final_list = [(Bond Type (BD or BD*), (atom1_idx, atom2_idx), atom bond order, NBO orbital energy value (ev) )]

    THIS WILL NOT FUNCTION AS EXPECTED WITH >100 ATOMS DUE TO LIMITATIONS IN THE
    CURRENT PARSING
    
    """
    # print(output)
    nbo_file_split = output.split("\n")

    # Find the indices corresponding to unique pieces of the nbo output
    for idx, line in enumerate(nbo_file_split):
        # print(line)
        if line == "ORBITAL ENERGIES":
            first_orb = idx + 4
            # print("hello?")
            continue
        if line == "Now starting NBO....":
            last_orb = idx - 1
        if line == " Summary of Natural Population Analysis:":
            first_charge = idx + 6
        if line == "                                 Natural Population":
            last_charge = idx - 4
        if (
            line
            == " SECOND ORDER PERTURBATION THEORY ANALYSIS OF FOCK MATRIX IN NBO BASIS"
        ):
            first_pert = idx + 8
        if line == " NATURAL BOND ORBITALS (Summary):":
            last_pert = idx - 3
            first_orb_energy = idx + 7
        if line == " $CHOOSE":
            last_orb_energy = idx - 9

    # Orbital Parsing
    orb_list = nbo_file_split[first_orb : last_orb + 1]
    orb_dict = dict()
    for line in orb_list:
        orb_nums, orb_occs, orb_ehs, orb_evs = " ".join(line.split()).split(" ")
        orb_num, orb_occ, orb_ehs, orb_ev = (
            int(orb_nums),
            float(orb_occs),
            float(orb_ehs),
            float(orb_evs),
        )
        orb_dict[orb_num] = orb_ev
        if float(orb_occ) == 0:
            homo = orb_dict[orb_num - 1]
            lumo = orb_dict[orb_num]
            orb_homo_lumo = (homo, lumo)
            break

    # Natural Charge Parsing
    nat_charge_list = nbo_file_split[first_charge : last_charge + 1]
    nat_charge_dict = dict()
    for line in nat_charge_list:
        atom, atom_nums, nat_charges, core, valence, rydberg, tot = " ".join(
            line.split()
        ).split(" ")
        atom_num, nat_charge = int(atom_nums) - 1, float(nat_charges)
        nat_charge_dict[atom_num] = nat_charge

    # Second Order Energy Contribution Parsing
    pert_list = nbo_file_split[first_pert : last_pert + 1]
    if pert_list[0] == " within unit  1":
        pert_list = nbo_file_split[first_pert + 1 : last_pert + 2]
    pert_final_list = list()
    for line in pert_list:
        parse_pert_list = " ".join(line.split()).split(" ")
        if len(parse_pert_list) < 10:
            continue
        no_parse_orbitals = False
        for pert in parse_pert_list:
            if "RY" in pert:
                no_parse_orbitals = True
            if "CR" in pert:
                no_parse_orbitals = True
            if "LV" in pert:
                no_parse_orbitals = True

        if no_parse_orbitals:
            continue

        # Necessary because the acceptor sometimes has an additional resonance structure that contains a -#
        if "-" in parse_pert_list[-6]:
            acceptor_idx_1 = int(parse_pert_list[-6].split("-")[0]) - 1
        elif "+" in parse_pert_list[-6]:
            acceptor_idx_1 = int(parse_pert_list[-6].split("+")[0]) - 1
        # This only happens if the really small columns end up collapsed into one
        elif len(parse_pert_list[-5]) >= 4:
            if "-" in parse_pert_list[-5]:
                acceptor_idx_1 = int(parse_pert_list[-5].split("-")[0]) - 1
            if "+" in parse_pert_list[-5]:
                acceptor_idx_1 = int(parse_pert_list[-5].split("+")[0]) - 1
        else:
            acceptor_idx_1 = int(parse_pert_list[-6]) - 1

        if "-" in parse_pert_list[-4]:
            acceptor_idx_2 = int(parse_pert_list[-4].split("-")[0]) - 1
        elif "+" in parse_pert_list[-4]:
            acceptor_idx_2 = int(parse_pert_list[-4].split("-")[0]) - 1
        else:
            acceptor_idx_2 = int(parse_pert_list[-4]) - 1

        acceptor_bond = (acceptor_idx_1, acceptor_idx_2)

        hyp_conj_kcal = parse_pert_list[-3]

        pert_final_list.append((acceptor_bond, hyp_conj_kcal))

    nbo_orb_energy_list = nbo_file_split[first_orb_energy : last_orb_energy + 1]
    nbo_orb_final_list = list()
    for line in nbo_orb_energy_list:
        parse_nbo_list = " ".join(line.split()).split(" ")
        if len(parse_nbo_list) < 8:
            continue
        if parse_nbo_list[1] == "BD*(":
            bond_type = "BD*"
        else:
            bond_type = parse_nbo_list[1]

        if bond_type == "BD":
            try:
                float(parse_nbo_list[-1])
            except:
                if "-" in parse_nbo_list[-6]:
                    b_atom1_idx = int(parse_nbo_list[-6].split("-")[0]) - 1
                elif "+" in parse_nbo_list[-6]:
                    b_atom1_idx = int(parse_nbo_list[-6].split("-")[0]) - 1
                elif len(parse_nbo_list[-5]) >= 4:
                    if "-" in parse_nbo_list[-5]:
                        b_atom1_idx = int(parse_nbo_list[-5].split("-")[0]) - 1
                    if "+" in parse_nbo_list[-5]:
                        b_atom1_idx = int(parse_nbo_list[-5].split("+")[0]) - 1
                else:
                    b_atom1_idx = int(parse_nbo_list[-6]) - 1

                if "-" in parse_nbo_list[-4]:
                    b_atom2_idx = int(parse_nbo_list[-4].split("-")[0]) - 1
                elif "+" in parse_nbo_list[-4]:
                    b_atom2_idx = int(parse_nbo_list[-4].split("+")[0]) - 1
                elif len(parse_nbo_list[-4]) >= 4:
                    if "-" in parse_nbo_list[-4]:
                        b_atom1_idx = int(parse_nbo_list[-4].split("-")[0]) - 1
                    elif "+" in parse_nbo_list[-4]:
                        b_atom1_idx = int(parse_nbo_list[-4].split("+")[0]) - 1
                    else:
                        b_atom1_idx = int(parse_nbo_list[-4]) - 1
                else:
                    b_atom2_idx = int(parse_nbo_list[-4]) - 1
            else:
                if "-" in parse_nbo_list[-5]:
                    b_atom1_idx = int(parse_nbo_list[-5].split("-")[0]) - 1
                elif "+" in parse_nbo_list[-5]:
                    b_atom1_idx = int(parse_nbo_list[-5].split("-")[0]) - 1
                elif len(parse_nbo_list[-4]) >= 4:
                    if "-" in parse_nbo_list[-4]:
                        b_atom1_idx = int(parse_nbo_list[-4].split("-")[0]) - 1
                    if "+" in parse_nbo_list[-4]:
                        b_atom1_idx = int(parse_nbo_list[-4].split("+")[0]) - 1
                else:
                    b_atom1_idx = int(parse_nbo_list[-5]) - 1

                if "-" in parse_nbo_list[-3]:
                    b_atom2_idx = int(parse_nbo_list[-3].split("-")[0]) - 1
                elif "+" in parse_nbo_list[-3]:
                    b_atom2_idx = int(parse_nbo_list[-3].split("+")[0]) - 1
                elif len(parse_nbo_list[-3]) >= 4:
                    if "-" in parse_nbo_list[-3]:
                        b_atom1_idx = int(parse_nbo_list[-3].split("-")[0]) - 1
                    elif "+" in parse_nbo_list[-3]:
                        b_atom1_idx = int(parse_nbo_list[-3].split("+")[0]) - 1
                    else:
                        b_atom1_idx = int(parse_nbo_list[-3]) - 1
                else:
                    b_atom2_idx = int(parse_nbo_list[-3]) - 1

            b_bond_order = parse_nbo_list[3][0]

            nbo_b_energy = float(parse_nbo_list[-2])

            b_bond_info = ("BD", (b_atom1_idx, b_atom2_idx), b_bond_order, nbo_b_energy)

            nbo_orb_final_list.append(b_bond_info)

        elif bond_type == "BD*":
            if len(parse_nbo_list[2]) >= 4:
                if "-" in parse_nbo_list[3]:
                    ab_atom1_idx = int(parse_nbo_list[3].split("-")[0]) - 1
                elif "+" in parse_nbo_list[3]:
                    ab_atom1_idx = int(parse_nbo_list[3].split("-")[0]) - 1
                elif len(parse_pert_list[3]) >= 3:
                    if "-" in parse_nbo_list[3]:
                        ab_atom1_idx = int(parse_nbo_list[3].split("-")[0]) - 1
                    elif "+" in parse_nbo_list[3]:
                        ab_atom1_idx = int(parse_nbo_list[3].split("+")[0]) - 1
                else:
                    ab_atom1_idx = int(parse_nbo_list[3]) - 1

                if "-" in parse_nbo_list[-3]:
                    ab_atom2_idx = int(parse_nbo_list[-3].split("-")[0]) - 1
                elif "+" in parse_nbo_list[-3]:
                    ab_atom2_idx = int(parse_nbo_list[-3].split("-")[0]) - 1
                else:
                    ab_atom2_idx = int(parse_nbo_list[-3]) - 1
            else:
                if "-" in parse_nbo_list[4]:
                    ab_atom1_idx = int(parse_nbo_list[4].split("-")[0]) - 1
                elif "+" in parse_nbo_list[4]:
                    ab_atom1_idx = int(parse_nbo_list[4].split("-")[0]) - 1
                elif len(parse_pert_list[4]) >= 4:
                    if "-" in parse_nbo_list[4]:
                        ab_atom1_idx = int(parse_nbo_list[4].split("-")[0]) - 1
                    elif "+" in parse_nbo_list[4]:
                        ab_atom1_idx = int(parse_nbo_list[4].split("+")[0]) - 1
                else:
                    ab_atom1_idx = int(parse_nbo_list[4]) - 1

                if "-" in parse_nbo_list[-3]:
                    ab_atom2_idx = int(parse_nbo_list[-3].split("-")[0]) - 1
                elif "+" in parse_nbo_list[-3]:
                    ab_atom2_idx = int(parse_nbo_list[-3].split("-")[0]) - 1
                else:
                    ab_atom2_idx = int(parse_nbo_list[-3]) - 1

            ab_bond_order = parse_nbo_list[2][0]

            nbo_ab_energy = float(parse_nbo_list[-1])

            ab_bond_info = (
                "BD*",
                (ab_atom1_idx, ab_atom2_idx),
                ab_bond_order,
                nbo_ab_energy,
            )

            nbo_orb_final_list.append(ab_bond_info)

        elif bond_type == "CR":
            continue

        elif bond_type == "LP":
            continue

        elif bond_type == "RY":
            break

    return orb_homo_lumo, nat_charge_dict, pert_final_list, nbo_orb_final_list

mlib = ml.MoleculeLibrary('6_5_1_DB_NBOCalc.mlib')
nbo_mlib = ml.MoleculeLibrary('6_5_2_DB_NBOAdded.mlib', readonly=False, overwrite=True)

nbo_dict = dict()

with mlib.reading(), nbo_mlib.writing():
    for file in glob("./6_4_NBOCache/output/*.out"):
        res = ml.pipeline.JobOutput.load(file)
        name = Path(file).stem
        m = mlib[name]

        orb_homo_lumo, nat_charge_dict, pert_final_list, nbo_orb_final_list = (
        nbo_parse_values(res.stdouts["orca"])
        )

        m.attrib["HOMO_LUMO"] = orb_homo_lumo
        m.attrib["Perturbation Energies"] = pert_final_list
        m.attrib["NBO Orbital Energies"] = nbo_orb_final_list

        for a in m.atoms:
            a.attrib["Natural Charge"] = nat_charge_dict[m.get_atom_index(a)]

        nbo_mlib[name] = m
