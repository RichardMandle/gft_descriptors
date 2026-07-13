#!/usr/bin/env python3

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import AllChem

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import pandas as pd

# custom colourmaps
import cmasher as cmr
import re
from pathlib import Path
import numpy as np
import glob

'''
helpers.py

functions for use with orca_charge GFT notebook.

Function (purpose)
text_colour_for_value (decides if colour somewhere needs to be flipped black --> white based on luminance)
'''

### constants - probably get these from scikit...
HARTREE_TO_KJMOL = 2625.499638
R_GAS = 8.31446261815324 
DIPOLE_AU_TO_DEBYE = 2.541746473

def text_colour_for_value(value, sc, threshold=0.45):
    '''
    Used to decide if we need to flip the colour of text labels from black to white...
    How - use percieved luminescence values from here https://www.w3.org/WAI/GL/wiki/Relative_luminance

    if our rgb values for our data on the cmap fall below this, then flip the colour.
    '''
    r, g, b, _ = sc.cmap(sc.norm(value))

    # perceived luminance; values are 0–1
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

    return "white" if luminance < threshold else "black"

def detect_orca_charge_model(out_file):
    with open(out_file, "r", errors="ignore") as f:
        text = f.read().lower()

    hits = []
    keys = {
        "chelpg": "chelpg",
        "hirshfeld": "hirshfeld",
        "mulliken atomic charges": "mulliken",
        "loewdin atomic charges": "loewdin",
        "mbis analysis": "mbis"
    }

    for k, v in keys.items():
        idx = text.rfind(k)
        if idx != -1:
            hits.append((idx, v))

    if not hits:
        return "charge"

    hits.sort()
    return hits[-1][1]

def parse_orca_charges(out_file, charge_model="auto"):
    if charge_model == "auto":
        charge_model = detect_orca_charge_model(out_file)

    with open(out_file, "r", errors="ignore") as f:
        lines = f.readlines()

    heading_indices = []
    for i, line in enumerate(lines):
        low = line.lower()
        if charge_model == "chelpg" and "chelpg" in low and "charge" in low:
            heading_indices.append(i)
        elif charge_model == "hirshfeld" and "hirshfeld" in low:
            heading_indices.append(i)
        elif charge_model == "mulliken" and "mulliken atomic charges" in low:
            heading_indices.append(i)
        elif charge_model == "loewdin" and "loewdin atomic charges" in low:
            heading_indices.append(i)
        elif charge_model == "mbis" and "mbis analysis" in low:
            heading_indices.append(i)
        elif charge_model == "resp" and "resp charges" in low:
            if "generation" not in low:
                if "calculated..." not in low:
                    heading_indices.append(i)
            
    if not heading_indices:
        raise RuntimeError(f"Could not find {charge_model} charges in {out_file}")

    line_re = re.compile(r"^\s*(\d+)\s+([A-Za-z]{1,3})\s*:?\s+(-?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)")

    for start in reversed(heading_indices):
        atoms = []
        charges = []
        started = False
        misses = 0

        for line in lines[start + 1:start + 500]:
        
            # MBIS charge table ends cleanly at TOTAL
            if started and line.strip().upper().startswith("TOTAL"):
                break
        
            # Also useful safety: stop if we accidentally reach another MBIS subsection
            if started and line.strip().upper().startswith("MBIS "):
                break
        
            m = line_re.match(line)
            if m:
                atoms.append(m.group(2))
                charges.append(float(m.group(3)))
                started = True
                misses = 0
                continue
        
            if started:
                misses += 1
                if misses >= 12:
                    break
    
            if len(charges) >= 2:
                return atoms, np.array(charges, dtype=float), charge_model
    
        raise RuntimeError(f"Failed to parse {charge_model} charges from {out_file}")
    
def mol_from_orca_xyz(xyz_file, charge=0, determine_bond_orders=False):
    '''
    Build an rdkit mol object from an xyz file {in this case, one written by ORCA}

    args:
    xyz_files    - obvious
    charge       - charge on molecule
    determine..  - try to figure out bond orders

    returns:
    mol          - an rdkit mol object, read from the xyz file
    '''
    with open(xyz_file, "r") as f:
        xyz_block = f.read()

    mol = Chem.MolFromXYZBlock(xyz_block)

    if mol is None:
        raise RuntimeError(f"Could not read XYZ file: {xyz_file}")

    rdDetermineBonds.DetermineConnectivity(mol)
    if determine_bond_orders:
        # Very cool rdkit feature - https://greglandrum.github.io/rdkit-blog/posts/2022-12-18-introducing-rdDetermineBonds.html
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
    return mol

def plot_molecular_signal(
    mol, signal,
    title=None,
    signal_label="Atomic charge",
    cmap="coolwarm",
    show_atom_indices=True,
    show_atom_symbols=True,
    show_charge_values=False,
    circle_scale=200, figsize=(8, 4), dpi=200, pad_frac=0.12, v_ax_scale = 1, usr_v_max = None, usr_v_min = None,
):
    signal = np.asarray(signal, dtype=float)
    
    if mol is None:
        raise ValueError("mol is None")
    if len(signal) != mol.GetNumAtoms():
        raise ValueError(f"Signal length ({len(signal)}) does not match number of atoms ({mol.GetNumAtoms()})")

    mol2d = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol2d)
    conf = mol2d.GetConformer()

    xy = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(mol2d.GetNumAtoms())])

    vmax = np.nanmax(np.abs(signal)) * v_ax_scale
    if vmax == 0 or not np.isfinite(vmax):
        vmax = 1.0
    vmin = -vmax

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for bond in mol2d.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color="black", linewidth=1.5, zorder=1)
    
    sizes = circle_scale * (0.35 + np.abs(signal) / vmax)
    if circle_scale == 200:
        sizes = circle_scale * (sizes / sizes)# normalise it
    
    if usr_v_max:
        vmax = usr_v_max
    if usr_v_min:
        vmin = usr_v_min
    
    sc = ax.scatter(
        xy[:, 0], xy[:, 1], c=signal, s=sizes, cmap=cmap,
        vmin=vmin, vmax=vmax, edgecolors="black", linewidths=0.8, zorder=2
    )

    for i, atom in enumerate(mol2d.GetAtoms()):
        label = ""
        if show_atom_symbols:
            label += atom.GetSymbol()
        if show_atom_indices:
            label += str(i)

        if label:
            text_colour = text_colour_for_value(signal[i], sc)
            ax.text(xy[i, 0], xy[i, 1], label, ha="center", va="center", fontsize=7, color=text_colour, zorder=3)
            
        if show_charge_values:
            text_colour = text_colour_for_value(signal[i], sc)
            ax.text(xy[i, 0], xy[i, 1] - 0.25, f"{signal[i]:+.2f}", ha="center", va="top", fontsize=6, color=text_colour, zorder=3)

    xmin, xmax = xy[:, 0].min(), xy[:, 0].max()   
    ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
    dx = xmax - xmin
    dy = ymax - ymin
    ax.set_xlim(xmin - pad_frac * max(dx, 1.0), xmax + pad_frac * max(dx, 1.0))
    ax.set_ylim(ymin - pad_frac * max(dy, 1.0), ymax + pad_frac * max(dy, 1.0))

    ax.set_aspect("equal")
    ax.axis("off")
    if title is not None:
        ax.set_title(title)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02, shrink=0.4, location='bottom')
    cbar.set_label(signal_label)

    return fig

def get_available_charge_modes(out_file):
    with open(out_file, "r", errors="ignore") as f:
        lines = f.readlines()
    hits = []    
    for line in lines:
        if " charges" in line.lower() or " analysis" in line.lower():
            candidate = line.split(' ')[0]
            if candidate == candidate.upper(): # Orca seems to give charges in upper case
                if candidate not in hits:
                    hits.append(candidate)

    if hits == []:
        print("No charge modes found")
        return
    
    print("*"*50)                
    print("Identified Charge Modes (caution!)")                
    for h in hits:
        print(h)
    print("*"*50)

def compute_graph_charge_spectrum_from_orca(mol, charges, atom_mode="all", center=False, normalize=False):
    charges = np.asarray(charges, dtype=float)

    if mol.GetNumAtoms() != len(charges):
        raise ValueError(f"Mol has {mol.GetNumAtoms()} atoms but charges has {len(charges)} values")

    A = Chem.GetAdjacencyMatrix(mol).astype(float)

    if atom_mode == "heavy":
        keep = np.array([a.GetSymbol().upper() != "H" for a in mol.GetAtoms()])
        A = A[np.ix_(keep, keep)]
        charges = charges[keep]
    elif atom_mode != "all":
        raise ValueError("atom_mode must be 'all' or 'heavy'")

    if center:
        charges = charges - charges.mean()

    d = A.sum(axis=1)
    with np.errstate(divide="ignore"):
        inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-15))

    D_inv_sqrt = np.diag(inv_sqrt)
    L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt

    eigenvalues, eigenvectors = np.linalg.eigh(L)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    coeffs = eigenvectors.T @ charges
    power_spectrum = coeffs**2

    if normalize:
        total = power_spectrum.sum()
        if total > 0:
            power_spectrum = power_spectrum / total

    return eigenvalues, power_spectrum

def spectrum_features(
    eigenvalues,
    power_spectrum,
    high_freq=1.5,
    v_high_freq=1.9,
    zero_tol=1e-8,
    eps=1e-15,
):
    """
    Extract scalar descriptors from a GFT power spectrum.

    Keeps old keys:
        total_power
        high_freq_fraction_lambda
        very_high_freq_fraction_lambda
        spectral_centroid

    Adds more robust/useful keys:
        total_power_nonzero
        high_freq_power
        very_high_freq_power
        high_freq_fraction_nonzero_lambda
        very_high_freq_fraction_nonzero_lambda
        dominant_lambda
        dominant_power
        dominant_power_fraction
        spectral_entropy
        spectral_entropy_norm
        spectral_spread

    Does GFT banding:
    """

    eig = np.asarray(eigenvalues, dtype=float)
    power = np.asarray(power_spectrum, dtype=float)

    if eig.shape != power.shape:
        raise ValueError(
            f"eigenvalues and power_spectrum must have same shape; "
            f"got {eig.shape} and {power.shape}"
        )

    finite = np.isfinite(eig) & np.isfinite(power)
    eig = eig[finite]
    power = power[finite]

    # avoid weird negative numerical noise in power spectra
    power = np.clip(power, 0.0, None)

    total_power_all = power.sum()

    nonzero_mask = eig > zero_tol
    eig_nz = eig[nonzero_mask]
    power_nz = power[nonzero_mask]

    total_power_nonzero = power_nz.sum()

    hf_mask = eig > high_freq
    vhf_mask = eig > v_high_freq

    hf_power = power[hf_mask].sum()
    vhf_power = power[vhf_mask].sum()

    if total_power_all > eps:
        hf_frac = hf_power / total_power_all
        vhf_frac = vhf_power / total_power_all
        spectral_centroid = np.sum(eig * power) / total_power_all
    else:
        hf_frac = np.nan
        vhf_frac = np.nan
        spectral_centroid = np.nan

    if total_power_nonzero > eps:
        hf_frac_nz = hf_power / total_power_nonzero
        vhf_frac_nz = vhf_power / total_power_nonzero

        p_nz = power_nz / total_power_nonzero

        spectral_centroid_nz = np.sum(eig_nz * p_nz)
        spectral_spread = np.sqrt(np.sum(((eig_nz - spectral_centroid_nz) ** 2) * p_nz))

        spectral_entropy = -np.sum(p_nz * np.log(p_nz + eps))

        if len(p_nz) > 1:
            spectral_entropy_norm = spectral_entropy / np.log(len(p_nz))
        else:
            spectral_entropy_norm = 0.0

        dom_local_idx = np.argmax(power_nz)
        dominant_lambda = eig_nz[dom_local_idx]
        dominant_power = power_nz[dom_local_idx]
        dominant_power_fraction = dominant_power / total_power_nonzero

    else:
        hf_frac_nz = np.nan
        vhf_frac_nz = np.nan
        spectral_centroid_nz = np.nan
        spectral_spread = np.nan
        spectral_entropy = np.nan
        spectral_entropy_norm = np.nan
        dominant_lambda = np.nan
        dominant_power = np.nan
        dominant_power_fraction = np.nan

    return {
        # backwards-compatible keys
        "total_power": total_power_all,
        "high_freq_fraction_lambda": hf_frac,
        "very_high_freq_fraction_lambda": vhf_frac,
        "spectral_centroid": spectral_centroid,

        "high_freq_power": hf_power,
        "very_high_freq_power": vhf_power,

        "total_power_nonzero": total_power_nonzero,
        "high_freq_fraction_nonzero_lambda": hf_frac_nz,
        "very_high_freq_fraction_nonzero_lambda": vhf_frac_nz,
        "spectral_centroid_nonzero": spectral_centroid_nz,

        "dominant_lambda": dominant_lambda,
        "dominant_power": dominant_power,
        "dominant_power_fraction": dominant_power_fraction,

        "spectral_entropy": spectral_entropy,
        "spectral_entropy_norm": spectral_entropy_norm,
        "spectral_spread": spectral_spread,
    }


def plot_gft(eigenvalues, power_spectrum, title="Power Spectrum - GFT", figsize=(4, 3), dpi=200):

    # find centroid, as above:
    total = power_spectrum.sum()
    centroid = (eigenvalues * power_spectrum).sum() / total if total > 0 else np.nan
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    markerline, stemlines, baseline = ax.stem(eigenvalues, power_spectrum, linefmt="blue", markerfmt="o", bottom=0)

    markerline.set_markerfacecolor("white")
    markerline.set_markeredgecolor("red")
    markerline.set_markersize(8)

    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

    ax.axvline(1, linestyle="--", color="grey", label=r"$\lambda$ = 1", alpha = 0.5)
    ax.axvline(centroid, linestyle="--", color="black", alpha = 0.5,
               label=rf"$\langle \lambda \rangle_{{|C_k|^2}}$")
    
    ax.grid(alpha=0.5, linestyle="--")
    ax.set_xlabel(r"Eigenvalue ($\lambda$)")
    ax.set_ylabel(r"$|C_k|^2$")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    return fig

def full_workflow(xyz_file, out_file, charge = 0, determine_bo = True, cmap = 'viridis_r', charge_model = 'auto', atom_mode = 'all', print_charges = False,
                 normalise_eigs=False, centre_eigs=False, show_atom_indices=False, show_atom_symbols=True, show_charge_values = False, circle_scale = 200, 
                 v_ax_scale = 1, dpi = 200, **plot_kwargs):
    
    mol = mol_from_orca_xyz(xyz_file, charge=charge, determine_bond_orders=determine_bo)
    
    atoms, charges, charge_model = parse_orca_charges(out_file, charge_model=charge_model)

    if print_charges:
        print('charges:')
        print(charges)
        
    fig = plot_molecular_signal(mol, charges, cmap = cmap,
                                signal_label=f"${charge_model.upper()}$ charge",
                                show_atom_indices=show_atom_indices, show_atom_symbols=show_atom_symbols, show_charge_values = show_charge_values, 
                               circle_scale = circle_scale, v_ax_scale = v_ax_scale, dpi = dpi, **plot_kwargs)
    plt.show()
    
    eigenvalues, power_spectrum = compute_graph_charge_spectrum_from_orca(
        mol, charges, atom_mode=atom_mode, center=centre_eigs, normalize=normalise_eigs)
    
    #ok = spectrum_features(eigenvalues, power_spectrum)
    #for o in ok.keys():
    #    print(f" {o} = {ok[o]}")
    
    fig = plot_gft(eigenvalues, power_spectrum, title=f"{charge_model.upper()} charge GFT")
    plt.show()

    return eigenvalues, power_spectrum, mol, charges

def plot_gft_mode_contribution_from_files(
    xyz_file,
    out_file,
    charge_model="auto",
    mode_idx=None,
    center_signal=True,
    skip_zero_mode=True,
    **plot_kwargs,
):
    '''
    - reads xyz into RDKit mol
    - parses ORCA charges
    - computes graph spectrum
    - plots chosen or dominant GFT mode contribution
    '''
    
    mol = mol_from_orca_xyz(xyz_file)

    atoms, charges, detected_model = parse_orca_charges(
        out_file,
        charge_model=charge_model,
    )

    if len(charges) != mol.GetNumAtoms():
        raise ValueError(
            f"Parsed {len(charges)} charges but molecule has {mol.GetNumAtoms()} atoms"
        )

    fig, info = plot_gft_mode_contribution(
        mol,
        charges,
        mode_idx=mode_idx,
        center_signal=center_signal,
        skip_zero_mode=skip_zero_mode,
        title_prefix=f"{detected_model.upper()} mode contribution",
        **plot_kwargs,
    )

    info["mol"] = mol
    info["charges"] = charges
    info["charge_model"] = detected_model
    info["atoms"] = atoms

    return fig, info

def plot_gft_mode_contribution(
    mol,
    signal,
    eigvals=None,
    eigvecs=None,
    mode_idx=None,
    center_signal=False,
    skip_zero_mode=True,
    title_prefix="Mode contribution",
    **plot_kwargs,
):
    signal = np.asarray(signal, dtype=float)

    if eigvals is None or eigvecs is None:
        eigvals, eigvecs = get_graph_spectrum(mol)

    signal_used = signal - signal.mean() if center_signal else signal
    coeffs = compute_gft(signal, eigvecs, center=center_signal)

    if mode_idx is None:
        mode_idx = choose_mode_from_gft(coeffs, skip_zero_mode=skip_zero_mode)

    lam = eigvals[mode_idx]

    vec = fix_eigenvector_sign(eigvecs[:, mode_idx])

    coeff = np.dot(vec, signal_used)
    contrib = coeff * vec

    fig = plot_molecular_signal(
        mol,
        contrib,
        title=f"{title_prefix} {mode_idx} (lambda={lam:.3f}, coeff={coeff:+.3f})",
        signal_label="Mode contribution",
        **plot_kwargs,
    )

    info = {
        "mode_idx": mode_idx,
        "lambda": lam,
        "coefficient": coeff,
        "contribution": contrib,
        "coeffs": coeffs,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
    }

    return fig, info

def mol_to_adjacency(mol):
    n = mol.GetNumAtoms()
    A = np.zeros((n, n), dtype=float)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        A[i, j] = 1.0
        A[j, i] = 1.0

    return A

def mol_to_normalised_laplacian(mol):
    A = mol_to_adjacency(mol)

    d = A.sum(axis=1)

    with np.errstate(divide="ignore"):
        inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-15))

    D_inv_sqrt = np.diag(inv_sqrt)

    L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt

    return L

def get_graph_spectrum(mol):
    '''
    matches compute_graph_charge_spectrum_from_orca():
    uses the symmetric normalised graph Laplacian.
    '''
    L = mol_to_normalised_laplacian(mol)

    eigvals, eigvecs = np.linalg.eigh(L)

    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    return eigvals, eigvecs

def compute_gft(signal, eigvecs, center=True):
    signal = np.asarray(signal, dtype=float)
    if center:
        signal = signal - signal.mean()
    coeffs = eigvecs.T @ signal
    return coeffs

def choose_mode_from_gft(coeffs, skip_zero_mode=True):
    coeffs = np.asarray(coeffs, dtype=float)
    start = 1 if skip_zero_mode else 0
    if start >= len(coeffs):
        raise ValueError("No modes available after skipping zero mode.")
    return start + np.argmax(np.abs(coeffs[start:]))

def fix_eigenvector_sign(vec):
    vec = np.asarray(vec, dtype=float).copy()
    imax = np.argmax(np.abs(vec))
    if vec[imax] < 0:
        vec = -vec
    return vec

def top_k_indices(x, k=4):
    return sorted(range(len(x)), key=x.__getitem__, reverse=True)[:k]

### MULTICONFORMER STUFF
def _float_list(s):
    return [float(x) for x in re.findall(
        r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?", s)]

def _normalise_symbols(symbols):
    return [s[0].upper() + s[1:].lower() for s in symbols]

def read_xyz_symbols(xyz_file):
    with open(xyz_file, "r", errors="ignore") as f:
        lines = f.readlines()

    n_atoms = int(lines[0].strip())
    return [lines[i].split()[0] for i in range(2, 2 + n_atoms)]

def find_conformer_file_pairs(
    conformer_dir,
    xyz_glob="*.xyz",
    out_glob="*.out",
    recursive=False,
):
    '''
    minimal stem-matching conformer finder.

    Expects e.g.
        conf_000.xyz  conf_000.out
        conf_001.xyz  conf_001.out
        ...

    returns a list of dicts with name, xyz_file, out_file.
    '''
    
    conformer_dir = Path(conformer_dir)

    globber = conformer_dir.rglob if recursive else conformer_dir.glob

    xyz_files = sorted(globber(xyz_glob))
    out_files = sorted(globber(out_glob))

    xyz_by_stem = {p.stem: p for p in xyz_files}

    pairs, skipped = [], []

    for out_file in out_files:
        xyz_file = xyz_by_stem.get(out_file.stem)

        if xyz_file is None:
            skipped.append(out_file)
            continue

        pairs.append({
            "name": out_file.stem,
            "xyz_file": xyz_file,
            "out_file": out_file,
        })

    if skipped:
        print(f"Warning: skipped {len(skipped)} .out files with no matching .xyz")

    if not pairs:
        raise RuntimeError(f"No matching xyz/out conformer pairs found in {conformer_dir}")

    return pairs

def parse_orca_scf_energy(out_file):
    '''
    read the final SCF energy from an ORCA output file.

    regex target:
        FINAL SINGLE POINT ENERGY     -xxx.xxxxx

    returns energy in Hartree.
    '''
    with open(out_file, "r", errors="ignore") as f:
        lines = f.readlines()

    final_sp_re = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)")

    energies = []

    for line in lines:
        m = final_sp_re.search(line)
        if m:
            energies.append(float(m.group(1)))

    if energies:
        return energies[-1]

    raise RuntimeError(f"Could not find FINAL SINGLE POINT ENERGY in {out_file}")


def parse_orca_dipole(out_file):
    '''
    read the last ORCA dipole moment block if present.

    returns:
        dipole_vec_au
        dipole_mag_au
        dipole_mag_debye

    averages of vector components are only meaningful if conformers are in a common frame.
    so make sure to use PAF if you plan on using these (principal alignment frame?)
    '''
    with open(out_file, "r", errors="ignore") as f:
        lines = f.readlines()

    starts = [
        i for i, line in enumerate(lines)
        if line.strip().upper() == "DIPOLE MOMENT"
    ]

    result = {
        "dipole_vec_au": np.array([np.nan, np.nan, np.nan]),
        "dipole_mag_au": np.nan,
        "dipole_mag_debye": np.nan,
    }

    if not starts:
        return result

    for start in reversed(starts):
        block = lines[start:start + 120]

        for line in block:
            low = line.lower()
            nums = _float_list(line)

            if "total dipole moment" in low and len(nums) >= 3:
                result["dipole_vec_au"] = np.array(nums[-3:], dtype=float)

            elif "magnitude" in low and "a.u" in low and nums:
                result["dipole_mag_au"] = nums[-1]

            elif "magnitude" in low and "debye" in low and nums:
                result["dipole_mag_debye"] = nums[-1]

        if np.any(np.isfinite(result["dipole_vec_au"])) or np.isfinite(result["dipole_mag_debye"]):
            if not np.isfinite(result["dipole_mag_debye"]) and np.all(np.isfinite(result["dipole_vec_au"])):
                result["dipole_mag_debye"] = (
                    np.linalg.norm(result["dipole_vec_au"]) * DIPOLE_AU_TO_DEBYE)

            return result
    return result


def parse_orca_quadrupole(out_file):
    '''
    read the last ORCA quadrupole moment block if present.

    returns components in the order they are written:
        XX, YY, ZZ, XY, XZ, YZ

    as with dipoles, beware orientation! 

    Note - while this works, its not really used anywhere.
    '''
    with open(out_file, "r", errors="ignore") as f:
        lines = f.readlines()

    starts = [
        i for i, line in enumerate(lines)
        if "QUADRUPOLE MOMENT" in line.upper()
    ]

    result = {"quadrupole_components_au": np.array([np.nan] * 6, dtype=float)}

    if not starts:
        return result

    for start in reversed(starts):
        block = lines[start:start + 200]

        for line in block:
            low = line.strip().lower()
            nums = _float_list(line)

            if low.startswith("total") and len(nums) >= 6:
                result["quadrupole_components_au"] = np.array(nums[-6:], dtype=float)
                return result

            if "total quadrupole moment" in low and len(nums) >= 6:
                result["quadrupole_components_au"] = np.array(nums[-6:], dtype=float)
                return result

    return result


def boltzmann_populations_from_energies(energies_hartree, T=298.15):
    '''
    turn absolute conformer energies in Ha into Boltzmann populations at the given
    value of T.
    '''
    energies_hartree = np.asarray(energies_hartree, dtype=float)

    rel_kjmol = (energies_hartree - np.nanmin(energies_hartree)) * HARTREE_TO_KJMOL

    beta_weights = np.exp(-(rel_kjmol * 1000.0) / (R_GAS * T))
    populations = beta_weights / np.nansum(beta_weights)

    return populations, rel_kjmol


def weighted_nanmean(values, weights):
    '''
    weighted, nan-tolerant mean.
    '''
    arr = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if arr.ndim == 1:
        mask = np.isfinite(arr)
        if not np.any(mask):
            return np.nan
        return np.sum(arr[mask] * weights[mask]) / np.sum(weights[mask])

    w = weights.reshape((weights.size,) + (1,) * (arr.ndim - 1))
    mask = np.isfinite(arr)

    numerator = np.nansum(arr * w, axis=0)
    denominator = np.sum(mask * w, axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        out = numerator / denominator

    out = np.where(denominator > 0, out, np.nan)
    return out


def print_conformer_summary(records):
    '''
    print some chat to the terminal / notebook about conformers - 
    energies, populations etc
    '''
    
    print("")
    print("Conformer summary")
    print("-" * 90)
    print(f"{'name':25s} {'E / Eh':>18s} {'dE / kJ mol-1':>16s} {'pop':>10s} {'mu / D':>10s}")
    print("-" * 90)

    for r in records:
        print(
            f"{r['name']:25s} "
            f"{r['energy_hartree']:18.10f} "
            f"{r['relative_energy_kjmol']:16.3f} "
            f"{r['population']:10.4f} "
            f"{r['dipole_mag_debye']:10.3f}"
        )

    print("-" * 90)
    print("")


def multi_conformer_charge_workflow(
    conformer_dir,
    xyz_glob="*.xyz",
    out_glob="*.out",
    recursive=False,

    charge=0,
    determine_bo=False,
    charge_model="auto",
    T=298.15,

    atom_mode="all",
    normalise_eigs=False,
    centre_eigs=False,
    high_freq = 1.5,   
    v_high_freq = 1.9,

    strict_atom_order=True,

    plot=True,
    report=True,
    cmap="magma",
    circle_scale=200,
    v_ax_scale=1,
    show_atom_indices=False,
    show_atom_symbols=True,
    show_charge_values=False,

    plt_v_max = None,
    plt_v_min = None,
):
    '''
    minimal multi-conformer wrapper.

    reuses a lot of functions:
        mol_from_orca_xyz
        parse_orca_charges
        plot_molecular_signal
        compute_graph_charge_spectrum_from_orca
        spectrum_features
        plot_gft
        high_freq
        v_high_freq

    returns a dict containing per-conformer records, populations,
    averaged charges, averaged moments, and GFT data.

    This is a pretty big function so we'll keep tabs with some in-line comments.
    '''

    pairs = find_conformer_file_pairs(
        conformer_dir,
        xyz_glob=xyz_glob,
        out_glob=out_glob,
        recursive=recursive,
    )

    records = []
    charges_by_conformer = []
    dipole_vecs_au = []
    dipole_mags_debye = []
    quadrupoles_au = []

    ref_symbols, ref_mol, used_charge_model= None, None, None

    for pair in pairs:
        name = pair["name"]
        xyz_file = pair["xyz_file"]
        out_file = pair["out_file"]

        energy_hartree = parse_orca_scf_energy(out_file)

        atoms_from_charges, charges, this_charge_model = parse_orca_charges(
            out_file,
            charge_model=charge_model,
        )

        if used_charge_model is None:
            used_charge_model = this_charge_model
        elif this_charge_model != used_charge_model:
            raise RuntimeError(
                f"Charge model changed between files: "
                f"{used_charge_model} vs {this_charge_model} in {out_file}"
            )

        xyz_symbols = read_xyz_symbols(xyz_file)

        # Given conformers will originate from a GOAT calculation or similar, we should be 
        # OK wiht atom ordering, but lets preemt that possiblity with a check
        if strict_atom_order:
            if len(xyz_symbols) != len(atoms_from_charges):
                raise RuntimeError(
                    f"Atom count mismatch in {name}: "
                    f"XYZ has {len(xyz_symbols)}, charges have {len(atoms_from_charges)}")

            if _normalise_symbols(xyz_symbols) != _normalise_symbols(atoms_from_charges):
                raise RuntimeError(
                    f"Atom-order/symbol mismatch in {name}. "
                    f"This would make conformer charge averaging unsafe.")

        if ref_symbols is None:
            ref_symbols = xyz_symbols
            ref_mol = mol_from_orca_xyz(
                xyz_file,
                charge=charge,
                determine_bond_orders=determine_bo)
        else:
            if strict_atom_order:
                if _normalise_symbols(xyz_symbols) != _normalise_symbols(ref_symbols):
                    raise RuntimeError(
                        f"{name} does not have the same atom ordering as the first conformer.")

        dipole = parse_orca_dipole(out_file)
        quadrupole = parse_orca_quadrupole(out_file)

        charges_by_conformer.append(charges)
        dipole_vecs_au.append(dipole["dipole_vec_au"])
        dipole_mags_debye.append(dipole["dipole_mag_debye"])
        quadrupoles_au.append(quadrupole["quadrupole_components_au"])

        # Append hte results to a dict for later retreival.
        records.append({
            "name": name,
            "number_of_confs": len(charges_by_conformer),
            "xyz_file": str(xyz_file),
            "out_file": str(out_file),
            "energy_hartree": energy_hartree,
            "charge_model": this_charge_model,
            "dipole_vec_au": dipole["dipole_vec_au"],
            "dipole_mag_debye": dipole["dipole_mag_debye"],
            "quadrupole_components_au": quadrupole["quadrupole_components_au"],
            "high_frequency": high_freq,
            "very_high_frequency": v_high_freq,
        })

    energies = np.array([r["energy_hartree"] for r in records], dtype=float)
    populations, rel_kjmol = boltzmann_populations_from_energies(energies, T=T)

    for r, p, de in zip(records, populations, rel_kjmol):
        r["population"] = p
        r["relative_energy_kjmol"] = de

    if report:
        print_conformer_summary(records)

    charges_by_conformer = np.vstack(charges_by_conformer)
    average_charges = populations @ charges_by_conformer

    dipole_vecs_au = np.vstack(dipole_vecs_au)
    quadrupoles_au = np.vstack(quadrupoles_au)

    average_dipole_vec_au = weighted_nanmean(dipole_vecs_au, populations)
    average_dipole_mag_debye = weighted_nanmean(np.array(dipole_mags_debye), populations)
    average_quadrupole_components_au = weighted_nanmean(quadrupoles_au, populations)

    eigenvalues, power_from_average_charges = compute_graph_charge_spectrum_from_orca(
        ref_mol,
        average_charges,
        atom_mode=atom_mode,
        center=centre_eigs,
        normalize=normalise_eigs,
    )

    features_from_average_charges = spectrum_features(
        eigenvalues,
        power_from_average_charges,
        high_freq = high_freq, v_high_freq = v_high_freq
    )

    conformer_power_spectra = []

    for charges in charges_by_conformer:
        eig_i, ps_i = compute_graph_charge_spectrum_from_orca(
            ref_mol,
            charges,
            atom_mode=atom_mode,
            center=centre_eigs,
            normalize=normalise_eigs,
        )
        # This shouldn't happen, but lets preemt that it _might_ happen
        if not np.allclose(eig_i, eigenvalues):
            raise RuntimeError("Unexpected eigenvalue mismatch between conformers")

        conformer_power_spectra.append(ps_i)

    conformer_power_spectra = np.vstack(conformer_power_spectra)
    population_average_power_spectrum = populations @ conformer_power_spectra

    features_from_population_average_power = spectrum_features(
        eigenvalues,
        population_average_power_spectrum,
        high_freq=high_freq,
        v_high_freq=v_high_freq,
    )

    # For debugging really, we might want to print out some of the properties to inspect them on the fly...
    if report:
        print("Population-averaged scalar properties")
        print("-" * 50)
        print(f"Temperature / K: {T:.2f}")
        print(f"Average dipole magnitude / D: {average_dipole_mag_debye:.4f}")
        print("")
        print("Features from GFT of population-averaged charges")
        for k, v in features_from_average_charges.items():
            print(f"  {k} = {v}")
        print("")
        print("Features from population-averaged GFT power")
        for k, v in features_from_population_average_power.items():
            print(f"  {k} = {v}")
        print("-" * 50)

    # Again, we might just want to plot and have a look. Which is the right way to average? Average the
    # charges, or average the power spectrum? I don't know, so lets do both and see.
    if plot:
        fig = plot_molecular_signal(
            ref_mol,
            average_charges,
            cmap=cmap,
            signal_label=f"Boltzmann-averaged {used_charge_model.upper()} charge",
            show_atom_indices=show_atom_indices,
            show_atom_symbols=show_atom_symbols,
            show_charge_values=show_charge_values,
            circle_scale=circle_scale,
            v_ax_scale=v_ax_scale,
            usr_v_max = plt_v_max,
            usr_v_min = plt_v_min,
        )
        plt.show()

        fig = plot_gft(
            eigenvalues,
            power_from_average_charges,
            title=f"{used_charge_model.upper()} GFT of averaged charges",
        )
        plt.show()

        fig = plot_gft(
            eigenvalues,
            population_average_power_spectrum,
            title=f"Population-averaged {used_charge_model.upper()} GFT power",
        )
        plt.show()

    return {
        "records": records,
        "temperature_K": T,
        "used_charge_model": used_charge_model,

        "mol": ref_mol,

        "energies_hartree": energies,
        "relative_energies_kjmol": rel_kjmol,
        "populations": populations,

        "charges_by_conformer": charges_by_conformer,
        "average_charges": average_charges,

        "dipole_vecs_au": dipole_vecs_au,
        "average_dipole_vec_au_frame_dependent": average_dipole_vec_au,
        "average_dipole_mag_debye": average_dipole_mag_debye,

        "quadrupoles_au": quadrupoles_au,
        "average_quadrupole_components_au_frame_dependent": average_quadrupole_components_au,

        "eigenvalues": eigenvalues,
        "power_from_average_charges": power_from_average_charges,
        "population_average_power_spectrum": population_average_power_spectrum,
        "features_from_average_charges": features_from_average_charges,
        "features_from_population_average_power": features_from_population_average_power,

        "high_frequency": high_freq,
        "very_high_frequency": v_high_freq,
    }

## This is just a basic single conformer version of hte multi_conformer workflow, above
# I think its probably preferable to have just one function and to figure out
# automatically (or on the fly) if we have conformers of the same molecule or different
# molecules. could do it by just looking at the structure of the atoms in the .xyz (for example)
def single_conformer_charge_workflow(
    xyz_file,
    out_file=None,

    charge=0,
    determine_bo=False,
    charge_model="auto",
    T=298.15,

    atom_mode="all",
    normalise_eigs=False,
    centre_eigs=False,
    high_freq=1.5,
    v_high_freq=1.9,

    plot=True,
    report=True,
    cmap="magma",
    circle_scale=200,
    v_ax_scale=1,
    show_atom_indices=False,
    show_atom_symbols=True,
    show_charge_values=False,

    plt_v_max = None,
    plt_v_min = None,
):
    '''
    a signle-conformer version of multi_conformer_charge_workflow.

    it returns the same style of dictionary, so parse_key_results(),
    plotting, dataframe generation, etc. can be reused.

    see logic and comments above, its basically the same.

    Might be ways to combine the logic a bit more.
    '''

    xyz_file = Path(xyz_file)

    if out_file is None:
        out_file = xyz_file.with_suffix(".out")
    else:
        out_file = Path(out_file)

    name = str(out_file).split('/')[-1]

    energy_hartree = parse_orca_scf_energy(out_file)

    atoms_from_charges, charges, used_charge_model = parse_orca_charges(
        out_file,
        charge_model=charge_model)

    xyz_symbols = read_xyz_symbols(xyz_file)

    if len(xyz_symbols) != len(atoms_from_charges):
        raise RuntimeError(
            f"Atom count mismatch in {name}: "
            f"XYZ has {len(xyz_symbols)}, charges have {len(atoms_from_charges)}")

    if _normalise_symbols(xyz_symbols) != _normalise_symbols(atoms_from_charges):
        raise RuntimeError(
            f"Atom-order/symbol mismatch in {name}. "
            f"This would make charge mapping unsafe.")

    mol = mol_from_orca_xyz(
        xyz_file,
        charge=charge,
        determine_bond_orders=determine_bo,)

    dipole = parse_orca_dipole(out_file)
    quadrupole = parse_orca_quadrupole(out_file)

    # TO DO - I never added the logic for the power averaged GFT...
    
    eigenvalues, power_from_average_charges = compute_graph_charge_spectrum_from_orca(
        mol,
        charges,
        atom_mode=atom_mode,
        center=centre_eigs,
        normalize=normalise_eigs)

    features_from_average_charges = spectrum_features(
        eigenvalues,
        power_from_average_charges,
        high_freq=high_freq,
        v_high_freq=v_high_freq)

    population_average_power_spectrum = power_from_average_charges.copy()

    features_from_population_average_power = spectrum_features(
        eigenvalues,
        population_average_power_spectrum,
        high_freq=high_freq,
        v_high_freq=v_high_freq)

    populations = np.array([1.0])
    energies = np.array([energy_hartree])
    rel_kjmol = np.array([0.0])

    records = [{
        "name": name,
        "xyz_file": str(xyz_file),
        "out_file": str(out_file),
        "energy_hartree": energy_hartree,
        "relative_energy_kjmol": 0.0,
        "population": 1.0,
        "charge_model": used_charge_model,
        "dipole_vec_au": dipole["dipole_vec_au"],
        "dipole_mag_debye": dipole["dipole_mag_debye"],
        "quadrupole_components_au": quadrupole["quadrupole_components_au"],
        "high_frequency": high_freq,
        "very_high_frequency": v_high_freq,
    }]

    if report:
        print_conformer_summary(records)

        print("Single-conformer scalar properties")
        print("-" * 50)
        print(f"Temperature / K: {T:.2f}")
        print(f"Dipole magnitude / D: {dipole['dipole_mag_debye']:.4f}")
        print("")
        print("Features from GFT of charges")
        for k, v in features_from_average_charges.items():
            print(f"  {k} = {v}")
        print("-" * 50)

    if plot:
        fig = plot_molecular_signal(
            mol,
            charges,
            cmap=cmap,
            signal_label=f"{used_charge_model.upper()} charge",
            show_atom_indices=show_atom_indices,
            show_atom_symbols=show_atom_symbols,
            show_charge_values=show_charge_values,
            circle_scale=circle_scale,
            v_ax_scale=v_ax_scale,
            usr_v_max = plt_v_max,
            usr_v_min = plt_v_min,
        )
        plt.show()

        fig = plot_gft(
            eigenvalues,
            power_from_average_charges,
            title=f"{used_charge_model.upper()} GFT of charges",
        )
        plt.show()

    return {
        "records": records,
        "temperature_K": T,
        "used_charge_model": used_charge_model,

        "mol": mol,

        "energies_hartree": energies,
        "relative_energies_kjmol": rel_kjmol,
        "populations": populations,

        "charges_by_conformer": np.array([charges]),
        "average_charges": charges,

        "dipole_vecs_au": np.array([dipole["dipole_vec_au"]]),
        "average_dipole_vec_au_frame_dependent": dipole["dipole_vec_au"],
        "average_dipole_mag_debye": dipole["dipole_mag_debye"],

        "quadrupoles_au": np.array([quadrupole["quadrupole_components_au"]]),
        "average_quadrupole_components_au_frame_dependent": quadrupole["quadrupole_components_au"],

        "eigenvalues": eigenvalues,
        "power_from_average_charges": power_from_average_charges,
        "population_average_power_spectrum": population_average_power_spectrum,
        "features_from_average_charges": features_from_average_charges,
        "features_from_population_average_power": features_from_population_average_power,
        
        "high_frequency": high_freq,
        "very_high_frequency": v_high_freq,
    }


def plot_df_scatter(df, x_col, y_col, nem_v_smectic = False, alpha =(0.8, 0.9), xlabel=None, ylabel=None, xlim=None, edge_colour = 'k', sizes = (90, 120)):
    '''
    simple plotter for making a scatter of polar and apolar nematics (optional: smectics)
    and plotting a couple of GFT descriptors against each other. 
    '''
    mask_polar = df["interesting"] == True
    mask_other = df["interesting"] == False

    if nem_v_smectic:
        mask_pnematic = mask_polar & (df["nematic"] == True)
        mask_psmectic = mask_polar & (df["nematic"] == False)
    
    plt.figure(figsize=(4, 2.5), dpi=200)

    plt.scatter(
        df.loc[mask_other, x_col],
        df.loc[mask_other, y_col],
        s=sizes[0],
        label="apolar phases",
        alpha=alpha[0], edgecolor=edge_colour,
    )
    if not nem_v_smectic:
        plt.scatter(
            df.loc[mask_polar, x_col],
            df.loc[mask_polar, y_col],
            s=sizes[1],
            label="polar nematic",
            alpha=alpha[1], edgecolor=edge_colour,
        )

    if nem_v_smectic:     
        plt.scatter(
            df.loc[mask_pnematic, x_col],
            df.loc[mask_pnematic, y_col],
            s=sizes[1],
            label="polar nematic",
            alpha=alpha[1],
            edgecolor=edge_colour,
        )
    
        plt.scatter(
            df.loc[mask_psmectic, x_col],
            df.loc[mask_psmectic, y_col],
            s=sizes[1],
            label="polar smectic",
            alpha=alpha[1],
            edgecolor=edge_colour,
        )

    if xlim:
        plt.xlim(xlim)
    
    plt.xlabel(xlabel)
    if not xlabel:
        plt.xlabel(x_col.replace('_'," "))
        
    plt.ylabel(ylabel)
    if not ylabel:
        plt.ylabel(y_col.replace('_'," "))
        
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

def parse_key_results(results, results_key="features_from_average_charges", multi=True):
    """
    This just gets the key results out of a results dict.
    we can change results_key to something else incase we have other charges
    multi = True/False just turns off/on multi conformer mode
    """
    result = results

    if multi:
        filename = "_".join(
            result["records"][0]["out_file"].split("/")[-2].split("_")[0:2]
        )
    else:
        filename = "_".join(
            result["records"][0]["out_file"].split("/")[-1].split("_")[0:3]
        )

    ave_dipole = result.get("average_dipole_mag_debye", np.nan)

    features = result.get(results_key, {})

    high_frequency = result.get("high_frequency", np.nan)
    very_high_frequency = result.get("very_high_frequency", np.nan)

    total_power = features.get("total_power", np.nan)
    total_power_nonzero = features.get("total_power_nonzero", total_power)

    hf_frac = features.get("high_freq_fraction_lambda", np.nan)
    vhf_frac = features.get("very_high_freq_fraction_lambda", np.nan)

    hf_frac_nz = features.get("high_freq_fraction_nonzero_lambda", hf_frac)
    vhf_frac_nz = features.get("very_high_freq_fraction_nonzero_lambda", vhf_frac)

    hf_power = features.get("high_freq_power", total_power * hf_frac)
    vhf_power = features.get("very_high_freq_power", total_power * vhf_frac)

    spec_cent = features.get("spectral_centroid", np.nan)
    spec_cent_nz = features.get("spectral_centroid_nonzero", spec_cent)

    dominant_lambda = features.get("dominant_lambda", np.nan)
    dominant_power = features.get("dominant_power", np.nan)
    dominant_power_fraction = features.get("dominant_power_fraction", np.nan)

    spectral_entropy = features.get("spectral_entropy", np.nan)
    spectral_entropy_norm = features.get("spectral_entropy_norm", np.nan)
    spectral_spread = features.get("spectral_spread", np.nan)

    return {
        "filename": filename,

        "high_frequency": high_frequency,
        "very_high_frequency": very_high_frequency,

        "average_dipole_mag_debye": ave_dipole,

        "total_power": total_power,
        "total_power_nonzero": total_power_nonzero,

        "high_freq_fraction_lambda": hf_frac,
        "very_high_freq_fraction_lambda": vhf_frac,

        "high_freq_fraction_nonzero_lambda": hf_frac_nz,
        "very_high_freq_fraction_nonzero_lambda": vhf_frac_nz,

        "high_freq_power": hf_power,
        "very_high_freq_power": vhf_power,

        "spectral_centroid": spec_cent,
        "spectral_centroid_nonzero": spec_cent_nz,
        "spectral_spread": spectral_spread,

        "dominant_lambda": dominant_lambda,
        "dominant_power": dominant_power,
        "dominant_power_fraction": dominant_power_fraction,

        "spectral_entropy": spectral_entropy,
        "spectral_entropy_norm": spectral_entropy_norm,

        # Frankly I don't really like the combined descriptors
        "dipole_x_high_freq_fraction": ave_dipole * hf_frac,
        "dipole_x_very_high_freq_fraction": ave_dipole * vhf_frac,
        "dipole_x_high_freq_power": ave_dipole * hf_power,
        "dipole_x_very_high_freq_power": ave_dipole * vhf_power,
        "dipole_x_high_freq_fraction_nonzero": ave_dipole * hf_frac_nz,
        "dipole_x_very_high_freq_fraction_nonzero": ave_dipole * vhf_frac_nz,
        "dipole_x_spectral_centroid_nonzero": ave_dipole * spec_cent_nz,
    }

def get_cols():
    '''
    just return a load of expected columns for mapping results into a df

    seems needless, but it saves space.
    '''
    columns = ['filename', 'high_frequency', 'very_high_frequency', 'average_dipole_mag_debye', 'total_power', 'total_power_nonzero', 'high_freq_fraction_lambda', 'very_high_freq_fraction_lambda', 'high_freq_fraction_nonzero_lambda', 'very_high_freq_fraction_nonzero_lambda', 'high_freq_power', 'very_high_freq_power', 'spectral_centroid', 'spectral_centroid_nonzero', 'spectral_spread', 'dominant_lambda', 'dominant_power', 'dominant_power_fraction', 'spectral_entropy', 'spectral_entropy_norm', 'dipole_x_high_freq_fraction', 'dipole_x_very_high_freq_fraction', 'dipole_x_high_freq_power', 'dipole_x_very_high_freq_power', 'dipole_x_high_freq_fraction_nonzero', 'dipole_x_very_high_freq_fraction_nonzero', 'dipole_x_spectral_centroid_nonzero']

    return columns