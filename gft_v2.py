#!/usr/bin/env python3
'''
utilities for graph-Fourier analysis of atom-centred ORCA properties.
'''

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import re
import warnings

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds

# constants; get from scipy
from scipy import constants
HARTREE_TO_KJMOL = constants.physical_constants['Hartree energy'][0] * constants.Avogadro / 1000 # Ha --> kJ mol; factor of 1000, J --> kJ mol!
R_GAS = constants.gas_constant
DIPOLE_AU_TO_DEBYE = cst.physical_constants['atomic unit of electric dipole mom.'][0] * cst.c * 10e20

_FLOAT = r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[EeDd][+-]?\d+)?"

PROPERTY_ALIASES = {
    "polar": "atompol",
    "polarisability": "atompol",
    "polarizability": "atompol",
    "atomic_polarisability": "atompol",
    "atomic_polarizability": "atompol",
    "atomic polarizability": "atompol",
    "atomic polarisability": "atompol",
    "atompol": "atompol",
    "resp": "resp",
    "chelpg": "chelpg",
    "hirshfeld": "hirshfeld",
    "mulliken": "mulliken",
    "loewdin": "loewdin",
    "lowdin": "loewdin",
    "mbis": "mbis",
}

SCALAR_PROPERTY_NAMES = {
    "resp",
    "chelpg",
    "hirshfeld",
    "mulliken",
    "loewdin",
    "mbis",
}

# data containers
@dataclass
class AtomPropertyData:
    '''Atom-centred property data parsed from one ORCA output file.

    <<PARAMS>>
    name
        Canonical property name, e.g. ``resp``, ``mbis`` or ``atompol``.
    atoms
        Atomic symbols in ORCA atom order.
    indices
        ORCA atom indices.
    components
        Mapping from component name to one value per atom. Scalar charge
        schemes use the component name ``value``. Atomic polarizability uses
        ``xx``, ``yy``, ``zz``, ``xy``, ``xz``, ``yz`` and ``iso``.
    units
        Printed ORCA units.
    totals
        Optional molecular totals printed beneath the atomic table.
    metadata
        Additional parser information.
    '''

    name: str
    atoms: tuple[str, ...]
    indices: np.ndarray
    components: dict[str, np.ndarray]
    units: str | None = None
    totals: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = canonical_property_name(self.name)
        self.indices = np.asarray(self.indices, dtype=int)
        self.atoms = tuple(_normalise_symbols(self.atoms))
        self.components = {
            str(key).lower(): np.asarray(value, dtype=float)
            for key, value in self.components.items()
        }

        n_atoms = len(self.atoms)
        if self.indices.shape != (n_atoms,):
            raise ValueError(
                f"indices has shape {self.indices.shape}; expected ({n_atoms},)"
            )

        for key, values in self.components.items():
            if values.shape != (n_atoms,):
                raise ValueError(
                    f"Component {key!r} has shape {values.shape}; "
                    f"expected ({n_atoms},)"
                )

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def available_components(self) -> tuple[str, ...]:
        return tuple(self.components)

    @property
    def default_component(self) -> str:
        if self.name == "atompol":
            return "iso"
        if "value" in self.components:
            return "value"
        if len(self.components) == 1:
            return next(iter(self.components))
        raise ValueError(
            f"No unambiguous default component for {self.name}: "
            f"{self.available_components}"
        )

    def resolve_component(self, component: str | None = None) -> str:
        '''
        resolve component aliases and return the canonical scalar component name
        '''
        
        key = self.default_component if component is None else component.lower()
        
        aliases = {
            "charge": "value",
            "q": "value",
            "isotropic": "iso",
            "alpha_iso": "iso",
        }
        
        key = aliases.get(key, key)
        if key == "trace" and self.name == "atompol":
            return key
        if key in self.components:
            return key
        raise KeyError(
            f"Component {component!r} is unavailable for {self.name}. "
            f"Available components: {', '.join(self.available_components)}"
        )

    def scalar(self, component: str | None = None, *, copy: bool = True) -> np.ndarray:
        '''
        get one scalar atomic signal suitable for a conventional GFT
        '''
        key = self.resolve_component(component)
        if key == "trace":
            values = self.components["xx"] + self.components["yy"] + self.components["zz"]
        else:
            values = self.components[key]
        return values.copy() if copy else values

    @property
    def tensor(self) -> np.ndarray:
        '''
        obtain atomic symmetric 3x3 tensors for atomic polarizability
        '''
        required = ("xx", "yy", "zz", "xy", "xz", "yz")
        missing = [key for key in required if key not in self.components]
        if missing:
            raise AttributeError(
                "A full tensor is unavailable; missing components: "
                + ", ".join(missing)
            )

        tensor = np.zeros((self.n_atoms, 3, 3), dtype=float)
        tensor[:, 0, 0] = self.components["xx"]
        tensor[:, 1, 1] = self.components["yy"]
        tensor[:, 2, 2] = self.components["zz"]
        tensor[:, 0, 1] = tensor[:, 1, 0] = self.components["xy"]
        tensor[:, 0, 2] = tensor[:, 2, 0] = self.components["xz"]
        tensor[:, 1, 2] = tensor[:, 2, 1] = self.components["yz"]
        return tensor

    def as_dataframe(self) -> pd.DataFrame:
        data: dict[str, Any] = {
            "index": self.indices,
            "atom": self.atoms,
        }
        data.update(self.components)
        return pd.DataFrame(data)


@dataclass(frozen=True)
class ConformerFilePair:
    name: str
    xyz_file: Path
    out_file: Path


@dataclass
class GraphBasis:
    '''
    normalised graph-Laplacian basis for one molecular graph
    '''

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    adjacency: np.ndarray
    laplacian: np.ndarray
    keep_mask: np.ndarray
    atom_mode: str


@dataclass
class GFTSpectrum:
    eigenvalues: np.ndarray
    coefficients: np.ndarray
    power_spectrum: np.ndarray
    signal: np.ndarray
    features: dict[str, float]

# helper functions
def canonical_property_name(property_name: str) -> str:
    key = str(property_name).strip().lower()
    if key == "auto":
        return key
    try:
        return PROPERTY_ALIASES[key]
    except KeyError as exc:
        allowed = sorted(set(PROPERTY_ALIASES.values()))
        raise ValueError(
            f"Unknown atom-centred property {property_name!r}. "
            f"Expected one of: {', '.join(allowed)}, or 'auto'."
        ) from exc


def _orca_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _float_list(text: str) -> list[float]:
    return [_orca_float(x) for x in re.findall(_FLOAT, text)]


def _normalise_symbols(symbols: Iterable[str]) -> list[str]:
    return [str(s)[0].upper() + str(s)[1:].lower() for s in symbols]


def _read_lines(path: str | Path) -> list[str]:
    with open(path, "r", errors="ignore") as handle:
        return handle.readlines()


def text_colour_for_value(value: float, sc: Any, threshold: float = 0.45) -> str:
    '''
    automatically pick black or white annotation text from mapped colour luminance.

    basically; judge if black or white text would be easier to read. Thats it.
    '''
    r, g, b, _ = sc.cmap(sc.norm(value))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < threshold else "black"

#NEW!!! parsing atom properties from ORCA output
def find_available_atom_properties(out_file: str | Path) -> list[str]:
    '''
    return whatever atom-centred properties we find, ordered by final occurrence.

    The order reflects where each property's final complete/recognisable block
    appears in the file. This preserves the old ``auto`` behaviour of gfv_v1; the last
    available property is selected.

    For example, for electrical properties we might consult this part of the manual:
    https://orca-manual.mpi-muelheim.mpg.de/contents/spectroscopyproperties/electric.html
    '''
    
    lines = _read_lines(out_file)
    hits: list[tuple[int, str]] = []

    for i, line in enumerate(lines):
        low = line.lower()

        if "resp charges" in low and "generation" not in low and "calculated..." not in low:
            hits.append((i, "resp"))
        if "chelpg" in low and "charge" in low:
            hits.append((i, "chelpg"))
        if "hirshfeld" in low:
            hits.append((i, "hirshfeld"))
        if "mulliken atomic charges" in low:
            hits.append((i, "mulliken"))
        if "loewdin atomic charges" in low or "löwdin atomic charges" in low:
            hits.append((i, "loewdin"))
        if "mbis analysis" in low:
            hits.append((i, "mbis"))

        # The sum line marks a complete atomic-polarizability table and is more
        # reliable across ORCA versions than any one heading phrase.
        if re.match(r"^\s*sum\s+polar\.\s*\(a\.u\.\)\s*:", low):
            hits.append((i, "atompol"))

    latest: dict[str, int] = {}
    for index, name in hits:
        latest[name] = max(index, latest.get(name, -1))

    return [name for name, _ in sorted(latest.items(), key=lambda item: item[1])]


def get_avail_atom_props(out_file: str | Path) -> str:
    '''
    a backwards-compatible singular property detector
    '''
    available = find_available_atom_properties(out_file)
    return available[-1] if available else "None"


def get_available_charge_modes(out_file: str | Path) -> list[str]:
    '''
    a backwards-compatible replacement for the old print-only helper
    '''
    available = [
        name for name in find_available_atom_properties(out_file)
        if name in SCALAR_PROPERTY_NAMES
    ]
    if not available:
        print("No charge modes found")
        return []

    print("*" * 50)
    print("Identified charge modes")
    
    for name in available:
        print(name.upper())
    print("*" * 50)
    
    return available


def _scalar_property_heading_indices(lines: Sequence[str], property_name: str) -> list[int]:
    indices: list[int] = []

    for i, line in enumerate(lines):
        low = line.lower()
        if property_name == "chelpg" and "chelpg" in low and "charge" in low:
            indices.append(i)
        elif property_name == "hirshfeld" and "hirshfeld" in low:
            indices.append(i)
        elif property_name == "mulliken" and "mulliken atomic charges" in low:
            indices.append(i)
        elif property_name == "loewdin" and (
            "loewdin atomic charges" in low or "löwdin atomic charges" in low
        ):
            indices.append(i)
        elif property_name == "mbis" and "mbis analysis" in low:
            indices.append(i)
        # RESP is a bit awkward in that ORCA prints something like "generation of RESP"
        # and "calculated RESP charges in ...."; so we sidestep that with loops.
        elif property_name == "resp" and "resp charges" in low:
            if "generation" not in low and "calculated..." not in low:
                indices.append(i)

    return indices


def _parse_scalar_atom_property(
    lines: Sequence[str],
    property_name: str,
    source: str | Path,
) -> AtomPropertyData:
    heading_indices = _scalar_property_heading_indices(lines, property_name)
    if not heading_indices:
        raise RuntimeError(f"Could not find {property_name} atom-centred data in {source}")

    row_re = re.compile(
        rf"^\s*(\d+)\s+([A-Za-z]{{1,3}})\s*:?\s*({_FLOAT})"
    )

    # we want the final valid table - ORCA outputs may contain repeated blocks.
    for start in reversed(heading_indices):
        indices: list[int] = []
        atoms: list[str] = []
        values: list[float] = []
        started = False
        misses = 0

        for line in lines[start + 1 : start + 1000]:
            stripped_upper = line.strip().upper()

            if started and stripped_upper.startswith("TOTAL"):
                break
            if started and stripped_upper.startswith("MBIS "):
                break

            match = row_re.match(line)
            if match:
                indices.append(int(match.group(1)))
                atoms.append(match.group(2))
                values.append(_orca_float(match.group(3)))
                started = True
                misses = 0
                continue

            if started:
                misses += 1
                if misses >= 12:
                    break

        if len(values) >= 2:
            return AtomPropertyData(
                name=property_name,
                atoms=tuple(atoms),
                indices=np.asarray(indices, dtype=int),
                components={"value": np.asarray(values, dtype=float)},
                units="e",
                metadata={"source": str(source), "heading_line": start + 1},
            )

    raise RuntimeError(f"Failed to parse {property_name} atom-centred data from {source}")


def _parse_atomic_polarizability(
    lines: Sequence[str],
    source: str | Path,
) -> AtomPropertyData:
    '''
    NEW!!

    Read the "atomic polarizability" table from an orca output.

    This is generated with something like:
    %elprop
    Polar True
    PolarAtom True
    end

    You can also get atom centred dipoles (Dipole True, DipoleAtom True) and quad.s
    (Quadrupole True, Quadrupole Atom True) but these are not something I've looked at.
    '''

    # the regex looks like this because the tensor and isotropic value are all given on a single line
    # we need to grab 6 floats for the tensor and 1 float for the isotropic value per atom (row)
    row_re = re.compile(
        rf'''
        ^\s*
        (\d+)\s*-\s*([A-Za-z]{{1,3}})\s*:\s*
        ({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s+
        ({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s+
        iso\s*=\s*({_FLOAT})\s*$
        ''',
        re.IGNORECASE | re.VERBOSE,
    )

    sum_re = re.compile(
        rf'''
        ^\s*sum\s+polar\.\s*\(a\.u\.\)\s*:\s*
        ({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s+
        ({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s+
        iso\s*=\s*({_FLOAT})\s*$
        ''',
        re.IGNORECASE | re.VERBOSE,
    )

    sum_indices = [i for i, line in enumerate(lines) if sum_re.match(line)]
    if not sum_indices:
        raise RuntimeError(
            f"Could not find a complete atomic-polarizability table in {source}; "
            "expected a 'Sum polar. (a.u.)' line."
        )

    component_names = ("xx", "yy", "zz", "xy", "xz", "yz", "iso")

    for end in reversed(sum_indices):
        total_match = sum_re.match(lines[end])
        assert total_match is not None

        # skip the separator/header lines immediately above the sum, then read the
        # contiguous atom rows backwards.
        j = end - 1
        skipped = 0
        while j >= 0 and row_re.match(lines[j]) is None and skipped < 40:
            j -= 1
            skipped += 1

        rows: list[tuple[str, ...]] = []
        while j >= 0:
            match = row_re.match(lines[j])
            if match is None:
                break
            rows.append(match.groups())
            j -= 1

        rows.reverse()
        if not rows:
            continue

        indices = np.asarray([int(row[0]) for row in rows], dtype=int)
        atoms = tuple(row[1] for row in rows)
        matrix = np.asarray(
            [[_orca_float(value) for value in row[2:]] for row in rows],
            dtype=float,
        )
        components = {
            name: matrix[:, column]
            for column, name in enumerate(component_names)
        }
        totals_array = np.asarray(
            [_orca_float(value) for value in total_match.groups()],
            dtype=float,
        )
        totals = {
            name: float(totals_array[column])
            for column, name in enumerate(component_names)
        }

        calculated_iso = (components["xx"] + components["yy"] + components["zz"]) / 3.0
        if not np.allclose(components["iso"], calculated_iso, rtol=1e-7, atol=1e-6):
            raise RuntimeError(
                f"Atomic polarizability table in {source} failed the check "
                "iso = (xx + yy + zz) / 3."
            )

        return AtomPropertyData(
            name="atompol",
            atoms=atoms,
            indices=indices,
            components=components,
            units="a.u.",
            totals=totals,
            metadata={"source": str(source), "sum_line": end + 1},
        )

    raise RuntimeError(
        f"Found an atomic-polarizability sum in {source}, but could not parse its atom rows."
    )


def parse_orca_atom_property(
    out_file: str | Path,
    property_name: str = "auto",
) -> AtomPropertyData:
    
    '''
    parse one complete atom-centred property table from an ORCA output.

    ``property_name='auto'`` preserves the historical behaviour and chooses the
    last recognised property in the file. For outputs containing both charges
    and atomic polarizabilities, explicit selection is strongly recommended.

    The idea here is that in <<FUTURE>> we could add any other atom centred
    properties we wanted fairly eaisly, with a regex as above. The old code was
    very much set up to work with partial charges only.
    '''
    
    out_file = Path(out_file)
    lines = _read_lines(out_file)

    requested = canonical_property_name(property_name)
    if requested == "auto":
        available = find_available_atom_properties(out_file)
        if not available:
            raise RuntimeError(f"No supported atom-centred properties found in {out_file}")
        requested = available[-1]

    # <<FUTURE>> IF we wanted to add more atom centred properties, we'd start here.
    if requested == "atompol":
        return _parse_atomic_polarizability(lines, out_file)
    if requested in SCALAR_PROPERTY_NAMES:
        return _parse_scalar_atom_property(lines, requested, out_file)

    raise AssertionError(f"Unhandled canonical property name: {requested}")


def parse_orca_atom_props(
    out_file: str | Path,
    property_name: str = "auto",
    component: str | None = None,
    *,
    return_data: bool = False,
):
    '''
    A backward-compatible parser wrapper.

    defaults to return  three-tuple used in gft_v1 ``(atoms, scalar_values, property_name)``. 
    
    for the atomic polarizability the default scalar is ``iso``. Set ``return_data=True`` to obtain the complete
    :class:`AtomPropertyData` object instead, i.e. the tensor.
    
    '''
    data = parse_orca_atom_property(out_file, property_name=property_name)
    if return_data:
        return data
    return list(data.atoms), data.scalar(component), data.name


# Molecular stuff and GFT
def read_xyz_symbols(xyz_file: str | Path) -> list[str]:
    lines = _read_lines(xyz_file)
    if not lines:
        raise RuntimeError(f"Empty XYZ file: {xyz_file}")
    n_atoms = int(lines[0].strip())
    atom_lines = lines[2 : 2 + n_atoms]
    if len(atom_lines) != n_atoms:
        raise RuntimeError(f"Incomplete XYZ file: {xyz_file}")
    return [line.split()[0] for line in atom_lines]


def mol_from_orca_xyz(
    xyz_file: str | Path,
    charge: int = 0,
    determine_bond_orders: bool = False,
) -> Chem.Mol:
    with open(xyz_file, "r") as handle:
        xyz_block = handle.read()

    mol = Chem.MolFromXYZBlock(xyz_block)
    if mol is None:
        raise RuntimeError(f"Could not read XYZ file: {xyz_file}")

    rdDetermineBonds.DetermineConnectivity(mol)
    if determine_bond_orders:
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
    return mol


def mol_to_adjacency(mol: Chem.Mol) -> np.ndarray:
    return Chem.GetAdjacencyMatrix(mol).astype(float)


def mol_to_normalised_laplacian(mol: Chem.Mol) -> np.ndarray:
    adjacency = mol_to_adjacency(mol)
    degree = adjacency.sum(axis=1)
    with np.errstate(divide="ignore"):
        inv_sqrt_degree = 1.0 / np.sqrt(np.maximum(degree, 1e-15))
    d_inv_sqrt = np.diag(inv_sqrt_degree)
    return np.eye(adjacency.shape[0]) - d_inv_sqrt @ adjacency @ d_inv_sqrt


def build_graph_basis(mol: Chem.Mol, atom_mode: str = "all") -> GraphBasis:
    atom_mode = atom_mode.lower()
    if atom_mode not in {"all", "heavy"}:
        raise ValueError("atom_mode must be 'all' or 'heavy'")

    full_adjacency = mol_to_adjacency(mol)
    if atom_mode == "all":
        keep_mask = np.ones(mol.GetNumAtoms(), dtype=bool)
    else:
        keep_mask = np.asarray(
            [atom.GetAtomicNum() != 1 for atom in mol.GetAtoms()],
            dtype=bool,
        )

    adjacency = full_adjacency[np.ix_(keep_mask, keep_mask)]
    degree = adjacency.sum(axis=1)
    with np.errstate(divide="ignore"):
        inv_sqrt_degree = 1.0 / np.sqrt(np.maximum(degree, 1e-15))
    d_inv_sqrt = np.diag(inv_sqrt_degree)
    laplacian = np.eye(adjacency.shape[0]) - d_inv_sqrt @ adjacency @ d_inv_sqrt

    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    order = np.argsort(eigenvalues)

    return GraphBasis(
        eigenvalues=eigenvalues[order],
        eigenvectors=eigenvectors[:, order],
        adjacency=adjacency,
        laplacian=laplacian,
        keep_mask=keep_mask,
        atom_mode=atom_mode,
    )


def get_graph_spectrum(mol: Chem.Mol, atom_mode: str = "all") -> tuple[np.ndarray, np.ndarray]:
    basis = build_graph_basis(mol, atom_mode=atom_mode)
    return basis.eigenvalues, basis.eigenvectors


def compute_gft(
    signal: Sequence[float],
    eigvecs: np.ndarray,
    center: bool = True,
) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"signal must be one-dimensional; got shape {values.shape}")
    if eigvecs.shape[0] != values.size:
        raise ValueError(
            f"Signal has {values.size} values but eigenvectors expect {eigvecs.shape[0]}"
        )
    if center:
        values = values - values.mean()
    return eigvecs.T @ values


def compute_gft_spectrum(
    mol: Chem.Mol,
    signal: Sequence[float],
    *,
    atom_mode: str = "all",
    center: bool = False,
    normalize: bool = False,
    basis: GraphBasis | None = None,
    high_freq: float = 1.5,
    v_high_freq: float = 1.9,
) -> GFTSpectrum:
    
    values = np.asarray(signal, dtype=float)
    if values.shape != (mol.GetNumAtoms(),):
        raise ValueError(
            f"Molecule has {mol.GetNumAtoms()} atoms but signal has shape {values.shape}"
        )

    if basis is None:
        basis = build_graph_basis(mol, atom_mode=atom_mode)
    elif basis.atom_mode != atom_mode:
        raise ValueError(
            f"Provided basis uses atom_mode={basis.atom_mode!r}, not {atom_mode!r}"
        )

    used_signal = values[basis.keep_mask]
    transformed_signal = used_signal - used_signal.mean() if center else used_signal.copy()
    coefficients = basis.eigenvectors.T @ transformed_signal
    power = coefficients**2

    if normalize:
        total = power.sum()
        if total > 0:
            power = power / total

    features = spectrum_features(
        basis.eigenvalues,
        power,
        high_freq=high_freq,
        v_high_freq=v_high_freq,
    )

    return GFTSpectrum(
        eigenvalues=basis.eigenvalues.copy(),
        coefficients=coefficients,
        power_spectrum=power,
        signal=transformed_signal,
        features=features,
    )


def gft_from_properties(
    mol: Chem.Mol,
    properties: Sequence[float],
    atom_mode: str = "all",
    center: bool = False,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    '''Backward-compatible two-array GFT wrapper.'''
    result = compute_gft_spectrum(
        mol,
        properties,
        atom_mode=atom_mode,
        center=center,
        normalize=normalize,
    )
    return result.eigenvalues, result.power_spectrum


def spectrum_features(
    eigenvalues: Sequence[float],
    power_spectrum: Sequence[float],
    high_freq: float = 1.5,
    v_high_freq: float = 1.9,
    zero_tol: float = 1e-8,
    eps: float = 1e-15,
) -> dict[str, float]:
    eig = np.asarray(eigenvalues, dtype=float)
    power = np.asarray(power_spectrum, dtype=float)

    if eig.shape != power.shape:
        raise ValueError(
            f"eigenvalues and power_spectrum must have the same shape; "
            f"got {eig.shape} and {power.shape}"
        )

    finite = np.isfinite(eig) & np.isfinite(power)
    eig = eig[finite]
    power = np.clip(power[finite], 0.0, None)

    total_power_all = float(power.sum())
    nonzero_mask = eig > zero_tol
    eig_nz = eig[nonzero_mask]
    power_nz = power[nonzero_mask]
    total_power_nonzero = float(power_nz.sum())

    hf_power = float(power[eig > high_freq].sum())
    vhf_power = float(power[eig > v_high_freq].sum())

    if total_power_all > eps:
        hf_frac = hf_power / total_power_all
        vhf_frac = vhf_power / total_power_all
        spectral_centroid = float(np.sum(eig * power) / total_power_all)
    else:
        hf_frac = vhf_frac = spectral_centroid = np.nan

    if total_power_nonzero > eps:
        p_nz = power_nz / total_power_nonzero
        hf_frac_nz = hf_power / total_power_nonzero
        vhf_frac_nz = vhf_power / total_power_nonzero
        spectral_centroid_nz = float(np.sum(eig_nz * p_nz))
        spectral_spread = float(
            np.sqrt(np.sum(((eig_nz - spectral_centroid_nz) ** 2) * p_nz))
        )
        spectral_entropy = float(-np.sum(p_nz * np.log(p_nz + eps)))
        spectral_entropy_norm = (
            float(spectral_entropy / np.log(len(p_nz))) if len(p_nz) > 1 else 0.0
        )
        dominant_index = int(np.argmax(power_nz))
        dominant_lambda = float(eig_nz[dominant_index])
        dominant_power = float(power_nz[dominant_index])
        dominant_power_fraction = dominant_power / total_power_nonzero
    else:
        hf_frac_nz = vhf_frac_nz = spectral_centroid_nz = np.nan
        spectral_spread = spectral_entropy = spectral_entropy_norm = np.nan
        dominant_lambda = dominant_power = dominant_power_fraction = np.nan

    return {
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


def choose_mode_from_gft(coeffs: Sequence[float], skip_zero_mode: bool = True) -> int:
    coeffs_array = np.asarray(coeffs, dtype=float)
    start = 1 if skip_zero_mode else 0
    if start >= len(coeffs_array):
        raise ValueError("No modes available after skipping the zero mode.")
    return int(start + np.argmax(np.abs(coeffs_array[start:])))


def fix_eigenvector_sign(vec: Sequence[float]) -> np.ndarray:
    values = np.asarray(vec, dtype=float).copy()
    imax = int(np.argmax(np.abs(values)))
    if values[imax] < 0:
        values = -values
    return values


def top_k_indices(x: Sequence[float], k: int = 4) -> list[int]:
    values = np.asarray(x)
    return np.argsort(values)[::-1][:k].tolist()


# scalar modifiers and conformer weights 
def parse_orca_scf_energy(out_file: str | Path) -> float:
    lines = _read_lines(out_file)
    pattern = re.compile(rf"FINAL SINGLE POINT ENERGY\s+({_FLOAT})")
    energies = [
        _orca_float(match.group(1))
        for line in lines
        if (match := pattern.search(line)) is not None
    ]
    if not energies:
        raise RuntimeError(f"Could not find FINAL SINGLE POINT ENERGY in {out_file}")
    return energies[-1]


def parse_orca_dipole(out_file: str | Path) -> dict[str, Any]:
    lines = _read_lines(out_file)
    starts = [i for i, line in enumerate(lines) if line.strip().upper() == "DIPOLE MOMENT"]

    result: dict[str, Any] = {
        "dipole_vec_au": np.array([np.nan, np.nan, np.nan], dtype=float),
        "dipole_mag_au": np.nan,
        "dipole_mag_debye": np.nan,
    }
    if not starts:
        return result

    for start in reversed(starts):
        block = lines[start : start + 120]
        candidate = result.copy()

        for line in block:
            low = line.lower()
            numbers = _float_list(line)
            if "total dipole moment" in low and len(numbers) >= 3:
                candidate["dipole_vec_au"] = np.asarray(numbers[-3:], dtype=float)
            elif "magnitude" in low and "a.u" in low and numbers:
                candidate["dipole_mag_au"] = numbers[-1]
            elif "magnitude" in low and "debye" in low and numbers:
                candidate["dipole_mag_debye"] = numbers[-1]

        if np.any(np.isfinite(candidate["dipole_vec_au"])) or np.isfinite(
            candidate["dipole_mag_debye"]
        ):
            if not np.isfinite(candidate["dipole_mag_debye"]) and np.all(
                np.isfinite(candidate["dipole_vec_au"])
            ):
                candidate["dipole_mag_debye"] = (
                    np.linalg.norm(candidate["dipole_vec_au"]) * DIPOLE_AU_TO_DEBYE
                )
            return candidate

    return result


def parse_orca_quadrupole(out_file: str | Path) -> dict[str, np.ndarray]:
    lines = _read_lines(out_file)
    starts = [i for i, line in enumerate(lines) if "QUADRUPOLE MOMENT" in line.upper()]
    result = {"quadrupole_components_au": np.full(6, np.nan, dtype=float)}

    for start in reversed(starts):
        for line in lines[start : start + 200]:
            low = line.strip().lower()
            numbers = _float_list(line)
            if (low.startswith("total") or "total quadrupole moment" in low) and len(numbers) >= 6:
                result["quadrupole_components_au"] = np.asarray(numbers[-6:], dtype=float)
                return result
    return result


def boltzmann_populations_from_energies(
    energies_hartree: Sequence[float],
    T: float = 298.15,
) -> tuple[np.ndarray, np.ndarray]:
    energies = np.asarray(energies_hartree, dtype=float)
    if energies.ndim != 1 or energies.size == 0:
        raise ValueError("energies_hartree must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(energies)):
        raise ValueError("All conformer energies must be finite")
    if T <= 0:
        raise ValueError("Temperature must be greater than zero kelvin")

    relative_kjmol = (energies - energies.min()) * HARTREE_TO_KJMOL
    exponent = -(relative_kjmol * 1000.0) / (R_GAS * T)
    exponent -= exponent.max()  # numerical stability!!
    raw_weights = np.exp(exponent)
    populations = raw_weights / raw_weights.sum()
    return populations, relative_kjmol


def weighted_nanmean(values: Sequence[Any], weights: Sequence[float]) -> np.ndarray | float:
    array = np.asarray(values, dtype=float)
    weights_array = np.asarray(weights, dtype=float)

    if array.shape[0] != weights_array.size:
        raise ValueError(
            f"First values dimension ({array.shape[0]}) does not match "
            f"number of weights ({weights_array.size})"
        )

    if array.ndim == 1:
        mask = np.isfinite(array) & np.isfinite(weights_array)
        if not np.any(mask):
            return np.nan
        return float(np.sum(array[mask] * weights_array[mask]) / np.sum(weights_array[mask]))

    expanded_weights = weights_array.reshape(
        (weights_array.size,) + (1,) * (array.ndim - 1)
    )
    mask = np.isfinite(array)
    numerator = np.nansum(array * expanded_weights, axis=0)
    denominator = np.sum(mask * expanded_weights, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        output = numerator / denominator
    return np.where(denominator > 0, output, np.nan)


# Conformers
# This part basically assumes that conformers are named similarly and will be in a given directory
# For me, I generally put my work something like compound_name/conformers/conf_xxx.yyy; where xxx
# is a number (e.g. 000) and yyy is the .out, .xyz. The code is written to reflect that convention.
def find_conformer_file_pairs(
    conformer_dir: str | Path,
    xyz_glob: str = "*.xyz",
    out_glob: str = "*.out",
    recursive: bool = False,
) -> list[dict[str, Any]]:
    
    '''
    get matching XYZ/OUT files by the filestem.

    A list of dictionaries is returned so as to be compatible with the gft_v1 code
    ... but is this optimal? probably not, we might do it differently.
    '''
    
    directory = Path(conformer_dir)
    globber = directory.rglob if recursive else directory.glob
    xyz_by_stem = {path.stem: path for path in globber(xyz_glob)}
    out_files = sorted(globber(out_glob))

    pairs: list[dict[str, Any]] = []
    skipped: list[Path] = []
    for out_file in out_files:
        xyz_file = xyz_by_stem.get(out_file.stem)
        if xyz_file is None:
            skipped.append(out_file)
            continue
        pairs.append(
            {
                "name": out_file.stem,
                "xyz_file": xyz_file,
                "out_file": out_file,
            }
        )

    if skipped:
        print(f"Warning: skipped {len(skipped)} .out files with no matching .xyz")
    if not pairs:
        raise RuntimeError(f"No matching XYZ/OUT conformer pairs found in {directory}")
    return pairs


def _coerce_pairs(
    pairs: Sequence[ConformerFilePair | Mapping[str, Any] | Sequence[Any]],
) -> list[ConformerFilePair]:
    coerced: list[ConformerFilePair] = []
    for item in pairs:
        if isinstance(item, ConformerFilePair):
            coerced.append(item)
        elif isinstance(item, Mapping):
            coerced.append(
                ConformerFilePair(
                    name=str(item.get("name", Path(item["out_file"]).stem)),
                    xyz_file=Path(item["xyz_file"]),
                    out_file=Path(item["out_file"]),
                )
            )
        else:
            sequence = list(item)
            if len(sequence) == 2:
                xyz_file, out_file = sequence
                name = Path(out_file).stem
            elif len(sequence) == 3:
                name, xyz_file, out_file = sequence
            else:
                raise ValueError(
                    "Each pair must be ConformerFilePair, a mapping, or a "
                    "2/3-item sequence"
                )
            coerced.append(
                ConformerFilePair(str(name), Path(xyz_file), Path(out_file))
            )
    if not coerced:
        raise ValueError("At least one conformer file pair is required")
    return coerced


def _stack_property_components(
    property_data: Sequence[AtomPropertyData],
) -> dict[str, np.ndarray]:
    component_sets = [set(data.components) for data in property_data]
    common = set.intersection(*component_sets)
    return {
        component: np.vstack([data.components[component] for data in property_data])
        for component in sorted(common)
    }


def _validate_atom_order(
    name: str,
    xyz_symbols: Sequence[str],
    property_data: AtomPropertyData,
) -> None:
    xyz_norm = _normalise_symbols(xyz_symbols)
    property_norm = _normalise_symbols(property_data.atoms)
    if len(xyz_norm) != len(property_norm):
        raise RuntimeError(
            f"Atom count mismatch in {name}: XYZ has {len(xyz_norm)}, "
            f"but {property_data.name} has {len(property_norm)} values"
        )
    if xyz_norm != property_norm:
        raise RuntimeError(
            f"Atom-order/symbol mismatch in {name}; property mapping is unsafe."
        )


def run_orca_gft_workflow(
    pairs: Sequence[ConformerFilePair | Mapping[str, Any] | Sequence[Any]],
    *,
    charge: int = 0,
    determine_bo: bool = False,
    property_name: str = "auto",
    property_component: str | None = None,
    T: float = 298.15,
    atom_mode: str = "all",
    normalise_eigs: bool = False,
    centre_eigs: bool = False,
    high_freq: float = 1.5,
    v_high_freq: float = 1.9,
    strict_atom_order: bool = True,
    plot: bool = True,
    report: bool = True,
    cmap: str = "magma",
    circle_scale: float = 200,
    v_ax_scale: float = 1,
    show_atom_indices: bool = False,
    show_atom_symbols: bool = True,
    show_property_values: bool = False,
    plt_v_max: float | None = None,
    plt_v_min: float | None = None,
) -> dict[str, Any]:
    
    '''execute the same ORCA-property GFT workflow for one or many conformers.

    <<ARGS>>
    pairs
        One or more matching XYZ/OUT file pairs.
    property_name
        ``resp``, ``mbis``, another supported charge scheme, ``atompol``, or
        ``auto``.
    property_component
        Scalar component used for the GFT. Charge schemes default to ``value``;
        atomic polarizability defaults to ``iso``. Tensor components such as
        ``xx`` can be selected, but are coordinate-frame dependent.

    <<NOTES>>
    Two conformer-averaged spectra are returned:

    * ``power_from_weighted_property_values``: Boltzmann-average the scalar
      atomic signal first, then calculate one GFT.
    * ``population_average_power_spectrum``: calculate each conformer's GFT
      first, then Boltzmann-average the power spectra.
    '''
    conformers = _coerce_pairs(pairs)

    records: list[dict[str, Any]] = []
    parsed_property_data: list[AtomPropertyData] = []
    scalar_signals: list[np.ndarray] = []
    dipole_vecs_au: list[np.ndarray] = []
    dipole_mags_debye: list[float] = []
    quadrupoles_au: list[np.ndarray] = []

    ref_symbols: list[str] | None = None
    ref_mol: Chem.Mol | None = None
    used_property_name: str | None = None
    used_component: str | None = None

    for conformer in conformers:
        energy_hartree = parse_orca_scf_energy(conformer.out_file)
        data = parse_orca_atom_property(conformer.out_file, property_name=property_name)
        component = data.resolve_component(property_component)
        scalar = data.scalar(component)

        if used_property_name is None:
            used_property_name = data.name
            used_component = component
        elif data.name != used_property_name:
            raise RuntimeError(
                f"Property changed between files: {used_property_name} vs "
                f"{data.name} in {conformer.out_file}"
            )

        if component != used_component:
            raise RuntimeError(
                f"Property component changed between files: {used_component} vs {component}"
            )

        xyz_symbols = read_xyz_symbols(conformer.xyz_file)
        if strict_atom_order:
            _validate_atom_order(conformer.name, xyz_symbols, data)

        if ref_symbols is None:
            ref_symbols = xyz_symbols
            ref_mol = mol_from_orca_xyz(
                conformer.xyz_file,
                charge=charge,
                determine_bond_orders=determine_bo,
            )
        elif strict_atom_order and _normalise_symbols(xyz_symbols) != _normalise_symbols(ref_symbols):
            raise RuntimeError(
                f"{conformer.name} does not have the same atom ordering as the first conformer."
            )

        dipole = parse_orca_dipole(conformer.out_file)
        quadrupole = parse_orca_quadrupole(conformer.out_file)

        parsed_property_data.append(data)
        scalar_signals.append(scalar)
        dipole_vecs_au.append(dipole["dipole_vec_au"])
        dipole_mags_debye.append(float(dipole["dipole_mag_debye"]))
        quadrupoles_au.append(quadrupole["quadrupole_components_au"])

        records.append(
            {
                "name": conformer.name,
                "xyz_file": str(conformer.xyz_file),
                "out_file": str(conformer.out_file),
                "energy_hartree": energy_hartree,
                "property_name": data.name,
                "property_component": component,
                "available_property_components": data.available_components,
                "dipole_vec_au": dipole["dipole_vec_au"],
                "dipole_mag_debye": dipole["dipole_mag_debye"],
                "quadrupole_components_au": quadrupole["quadrupole_components_au"],
                "high_frequency": high_freq,
                "very_high_frequency": v_high_freq,
            }
        )

    assert ref_mol is not None
    assert used_property_name is not None
    assert used_component is not None

    if used_property_name == "atompol" and used_component not in {"iso", "trace"} and len(conformers) > 1:
        warnings.warn(
            f"Atomic polarizability component {used_component!r} is frame-dependent. "
            "Conformer averaging is only physically meaningful if all geometries use a "
            "common aligned frame. The isotropic component is frame-invariant.",
            RuntimeWarning,
            stacklevel=2,
        )

    energies = np.asarray([record["energy_hartree"] for record in records], dtype=float)
    populations, relative_energies = boltzmann_populations_from_energies(energies, T=T)

    for record, population, delta_e in zip(records, populations, relative_energies):
        record["population"] = float(population)
        record["relative_energy_kjmol"] = float(delta_e)
        record["number_of_confs"] = len(records)

    scalar_signals_array = np.vstack(scalar_signals)
    weighted_property_values = populations @ scalar_signals_array

    component_arrays = _stack_property_components(parsed_property_data)
    weighted_property_components = {
        component: populations @ values
        for component, values in component_arrays.items()
    }

    dipole_vecs_array = np.vstack(dipole_vecs_au)
    quadrupoles_array = np.vstack(quadrupoles_au)
    average_dipole_vec_au = weighted_nanmean(dipole_vecs_array, populations)
    average_dipole_mag_debye = weighted_nanmean(dipole_mags_debye, populations)
    average_quadrupole_components_au = weighted_nanmean(quadrupoles_array, populations)

    basis = build_graph_basis(ref_mol, atom_mode=atom_mode)
    weighted_signal_gft = compute_gft_spectrum(
        ref_mol,
        weighted_property_values,
        atom_mode=atom_mode,
        center=centre_eigs,
        normalize=normalise_eigs,
        basis=basis,
        high_freq=high_freq,
        v_high_freq=v_high_freq,
    )

    conformer_spectra = [
        compute_gft_spectrum(
            ref_mol,
            signal,
            atom_mode=atom_mode,
            center=centre_eigs,
            normalize=normalise_eigs,
            basis=basis,
            high_freq=high_freq,
            v_high_freq=v_high_freq,
        )
        for signal in scalar_signals_array
    ]
    conformer_power_spectra = np.vstack(
        [spectrum.power_spectrum for spectrum in conformer_spectra]
    )
    population_average_power_spectrum = populations @ conformer_power_spectra
    features_from_population_average_power = spectrum_features(
        basis.eigenvalues,
        population_average_power_spectrum,
        high_freq=high_freq,
        v_high_freq=v_high_freq,
    )

    if report:
        print_conformer_summary(records)
        print("Population-averaged atom-centred property")
        print("-" * 60)
        print(f"Property: {used_property_name}")
        print(f"GFT component: {used_component}")
        print(f"Temperature / K: {T:.2f}")
        print(f"Average dipole magnitude / D: {average_dipole_mag_debye:.4f}")
        print("\nFeatures from GFT of the population-averaged signal")
        for key, value in weighted_signal_gft.features.items():
            print(f"  {key} = {value}")
        print("\nFeatures from the population-averaged GFT power")
        for key, value in features_from_population_average_power.items():
            print(f"  {key} = {value}")
        print("-" * 60)

    if plot:
        label = _property_signal_label(used_property_name, used_component, averaged=True)
        fig = plot_molecular_signal(
            ref_mol,
            weighted_property_values,
            cmap=cmap,
            signal_label=label,
            show_atom_indices=show_atom_indices,
            show_atom_symbols=show_atom_symbols,
            show_property_values=show_property_values,
            circle_scale=circle_scale,
            v_ax_scale=v_ax_scale,
            usr_v_max=plt_v_max,
            usr_v_min=plt_v_min,
        )
        plt.show()

        fig = plot_gft(
            basis.eigenvalues,
            weighted_signal_gft.power_spectrum,
            title=f"{used_property_name.upper()} ({used_component}) GFT of averaged signal",
        )
        plt.show()

        if len(conformers) > 1:
            fig = plot_gft(
                basis.eigenvalues,
                population_average_power_spectrum,
                title=f"Population-averaged {used_property_name.upper()} ({used_component}) GFT power",
            )
            plt.show()

    return {
        "records": records,
        "temperature_K": T,
        "used_property_name": used_property_name,
        "used_property_component": used_component,
        "available_property_components": parsed_property_data[0].available_components,
        "property_units": parsed_property_data[0].units,
        "mol": ref_mol,
        "graph_basis": basis,
        "energies_hartree": energies,
        "relative_energies_kjmol": relative_energies,
        "populations": populations,
        "parsed_property_data": parsed_property_data,
        "property_components_by_conformer": component_arrays,
        "weighted_property_components": weighted_property_components,
        "properties_by_conformer": scalar_signals_array,
        "weighted_property_values": weighted_property_values,
        "dipole_vecs_au": dipole_vecs_array,
        "average_dipole_vec_au_frame_dependent": average_dipole_vec_au,
        "average_dipole_mag_debye": average_dipole_mag_debye,
        "quadrupoles_au": quadrupoles_array,
        "average_quadrupole_components_au_frame_dependent": average_quadrupole_components_au,
        "eigenvalues": basis.eigenvalues,
        "gft_coefficients_from_weighted_property_values": weighted_signal_gft.coefficients,
        "power_from_weighted_property_values": weighted_signal_gft.power_spectrum,
        "conformer_gft_coefficients": np.vstack(
            [spectrum.coefficients for spectrum in conformer_spectra]
        ),
        "conformer_power_spectra": conformer_power_spectra,
        "population_average_power_spectrum": population_average_power_spectrum,
        "features_from_weighted_property_values": weighted_signal_gft.features,
        "features_from_population_average_power": features_from_population_average_power,
        "high_frequency": high_freq,
        "very_high_frequency": v_high_freq,
    }


def multi_conformer_workflow(
    conformer_dir: str | Path,
    xyz_glob: str = "*.xyz",
    out_glob: str = "*.out",
    recursive: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    pairs = find_conformer_file_pairs(
        conformer_dir,
        xyz_glob=xyz_glob,
        out_glob=out_glob,
        recursive=recursive,
    )
    return run_orca_gft_workflow(pairs, **kwargs)


def single_conformer_workflow(
    xyz_file: str | Path,
    out_file: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    xyz_path = Path(xyz_file)
    out_path = xyz_path.with_suffix(".out") if out_file is None else Path(out_file)
    pair = ConformerFilePair(out_path.stem, xyz_path, out_path)
    return run_orca_gft_workflow([pair], **kwargs)


def single_conformer_charge_workflow(
    xyz_file: str | Path,
    out_file: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    '''Deprecated compatibility alias for :func:`single_conformer_workflow`.'''
    warnings.warn(
        "single_conformer_charge_workflow() is deprecated; use "
        "single_conformer_workflow().",
        DeprecationWarning,
        stacklevel=2,
    )
    return single_conformer_workflow(xyz_file, out_file=out_file, **kwargs)


def full_workflow(
    xyz_file: str | Path,
    out_file: str | Path,
    charge: int = 0,
    determine_bo: bool = True,
    cmap: str = "viridis_r",
    property_name: str = "auto",
    property_component: str | None = None,
    atom_mode: str = "all",
    print_properties: bool = False,
    normalise_eigs: bool = False,
    centre_eigs: bool = False,
    show_atom_indices: bool = False,
    show_atom_symbols: bool = True,
    show_property_values: bool = False,
    circle_scale: float = 200,
    v_ax_scale: float = 1,
    dpi: int = 200,
    plot: bool = True,
    report: bool = False,
    **plot_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, Chem.Mol, np.ndarray]:
    '''Backward-compatible single-conformer notebook workflow.

    The newer :func:`single_conformer_workflow` returns the complete results
    dictionary. This wrapper retains the historical four-value return and
    accepts the old plotting arguments.
    '''
    result = single_conformer_workflow(
        xyz_file,
        out_file=out_file,
        charge=charge,
        determine_bo=determine_bo,
        property_name=property_name,
        property_component=property_component,
        atom_mode=atom_mode,
        normalise_eigs=normalise_eigs,
        centre_eigs=centre_eigs,
        plot=False,
        report=report,
    )

    if print_properties:
        print("property values:")
        print(result["weighted_property_values"])

    if plot:
        component = result["used_property_component"]
        used_name = result["used_property_name"]
        fig = plot_molecular_signal(
            result["mol"],
            result["weighted_property_values"],
            cmap=cmap,
            signal_label=_property_signal_label(used_name, component),
            show_atom_indices=show_atom_indices,
            show_atom_symbols=show_atom_symbols,
            show_property_values=show_property_values,
            circle_scale=circle_scale,
            v_ax_scale=v_ax_scale,
            dpi=dpi,
            **plot_kwargs,
        )
        plt.show()

        fig = plot_gft(
            result["eigenvalues"],
            result["power_from_weighted_property_values"],
            title=f"{used_name.upper()} ({component}) GFT power spectrum",
            dpi=dpi,
        )
        plt.show()

    return (
        result["eigenvalues"],
        result["power_from_weighted_property_values"],
        result["mol"],
        result["weighted_property_values"],
    )


# plotting and related functions 
def _property_signal_label(
    property_name: str,
    component: str,
    *,
    averaged: bool = False,
) -> str:
    prefix = "Boltzmann-averaged " if averaged else ""
    if property_name == "atompol":
        return f"{prefix}atomic polarizability ({component}, a.u.)"
    return f"{prefix}{property_name.upper()} charge / e"


def plot_molecular_signal(
    mol: Chem.Mol,
    signal: Sequence[float],
    title: str | None = None,
    signal_label: str = "Atomic property",
    cmap: str = "coolwarm",
    show_atom_indices: bool = True,
    show_atom_symbols: bool = True,
    show_property_values: bool = False,
    circle_scale: float = 200,
    figsize: tuple[float, float] = (8, 4),
    dpi: int = 200,
    pad_frac: float = 0.12,
    v_ax_scale: float = 1,
    usr_v_max: float | None = None,
    usr_v_min: float | None = None,
    symmetric: bool | None = None,
):
    values = np.asarray(signal, dtype=float)
    if mol is None:
        raise ValueError("mol is None")
    if values.shape != (mol.GetNumAtoms(),):
        raise ValueError(
            f"Signal shape {values.shape} does not match {mol.GetNumAtoms()} atoms"
        )
    if not np.any(np.isfinite(values)):
        raise ValueError("Signal contains no finite values")

    mol2d = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol2d)
    conformer = mol2d.GetConformer()
    xy = np.asarray(
        [
            [conformer.GetAtomPosition(i).x, conformer.GetAtomPosition(i).y]
            for i in range(mol2d.GetNumAtoms())
        ]
    )

    finite_values = values[np.isfinite(values)]
    if symmetric is None:
        symmetric = bool(np.nanmin(finite_values) < 0 < np.nanmax(finite_values))

    if symmetric:
        vmax = float(np.nanmax(np.abs(values)) * v_ax_scale)
        vmax = 1.0 if vmax == 0 or not np.isfinite(vmax) else vmax
        vmin = -vmax
    else:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        if vmin == vmax:
            pad = max(abs(vmin) * 0.05, 1.0)
            vmin -= pad
            vmax += pad
        else:
            midpoint = 0.5 * (vmin + vmax)
            half_range = 0.5 * (vmax - vmin) * v_ax_scale
            vmin, vmax = midpoint - half_range, midpoint + half_range

    if usr_v_max is not None:
        vmax = usr_v_max
    if usr_v_min is not None:
        vmin = usr_v_min

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for bond in mol2d.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        ax.plot(
            [xy[i, 0], xy[j, 0]],
            [xy[i, 1], xy[j, 1]],
            color="black",
            linewidth=1.5,
            zorder=1,
        )

    max_abs = float(np.nanmax(np.abs(values)))
    if max_abs > 0 and circle_scale != 200:
        sizes = circle_scale * (0.35 + np.abs(values) / max_abs)
    else:
        sizes = np.full(values.shape, circle_scale, dtype=float)

    scatter = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=values,
        s=sizes,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="black",
        linewidths=0.8,
        zorder=2,
    )

    for i, atom in enumerate(mol2d.GetAtoms()):
        label = ""
        if show_atom_symbols:
            label += atom.GetSymbol()
        if show_atom_indices:
            label += str(i)

        if label:
            ax.text(
                xy[i, 0],
                xy[i, 1],
                label,
                ha="center",
                va="center",
                fontsize=7,
                color=text_colour_for_value(values[i], scatter),
                zorder=3,
            )

        if show_property_values:
            ax.text(
                xy[i, 0],
                xy[i, 1] - 0.25,
                f"{values[i]:+.2f}",
                ha="center",
                va="top",
                fontsize=6,
                color=text_colour_for_value(values[i], scatter),
                zorder=3,
            )

    xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
    ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
    dx, dy = xmax - xmin, ymax - ymin
    ax.set_xlim(xmin - pad_frac * max(dx, 1.0), xmax + pad_frac * max(dx, 1.0))
    ax.set_ylim(ymin - pad_frac * max(dy, 1.0), ymax + pad_frac * max(dy, 1.0))
    ax.set_aspect("equal")
    ax.axis("off")
    if title is not None:
        ax.set_title(title)

    colourbar = fig.colorbar(
        scatter,
        ax=ax,
        fraction=0.05,
        pad=0.02,
        shrink=0.4,
        location="bottom",
    )
    colourbar.set_label(signal_label)
    return fig


def plot_gft(
    eigenvalues: Sequence[float],
    power_spectrum: Sequence[float],
    title: str = "Power Spectrum - GFT",
    figsize: tuple[float, float] = (4, 3),
    dpi: int = 200,
):
    eigenvalues_array = np.asarray(eigenvalues, dtype=float)
    power_array = np.asarray(power_spectrum, dtype=float)
    if eigenvalues_array.shape != power_array.shape:
        raise ValueError("eigenvalues and power_spectrum must have the same shape")

    total = power_array.sum()
    centroid = (
        float(np.sum(eigenvalues_array * power_array) / total)
        if total > 0
        else np.nan
    )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    markerline, _, _ = ax.stem(eigenvalues_array, power_array, bottom=0)
    markerline.set_markerfacecolor("white")
    markerline.set_markersize(8)

    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.axvline(1, linestyle="--", alpha=0.5, label=r"$\lambda=1$")
    if np.isfinite(centroid):
        ax.axvline(
            centroid,
            linestyle="--",
            alpha=0.5,
            label=r"$\langle \lambda \rangle_{|C_k|^2}$",
        )
    ax.grid(alpha=0.5, linestyle="--")
    ax.set_xlabel(r"Eigenvalue ($\lambda$)")
    ax.set_ylabel(r"$|C_k|^2$")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_gft_mode_contribution(
    mol: Chem.Mol,
    signal: Sequence[float],
    eigvals: np.ndarray | None = None,
    eigvecs: np.ndarray | None = None,
    mode_idx: int | None = None,
    center_signal: bool = False,
    skip_zero_mode: bool = True,
    title_prefix: str = "Mode contribution",
    **plot_kwargs: Any,
):
    values = np.asarray(signal, dtype=float)
    if values.shape != (mol.GetNumAtoms(),):
        raise ValueError("Mode-contribution plotting currently requires atom_mode='all'")

    if eigvals is None or eigvecs is None:
        eigvals, eigvecs = get_graph_spectrum(mol, atom_mode="all")

    signal_used = values - values.mean() if center_signal else values
    coeffs = compute_gft(signal_used, eigvecs, center=False)
    if mode_idx is None:
        mode_idx = choose_mode_from_gft(coeffs, skip_zero_mode=skip_zero_mode)

    eigenvector = fix_eigenvector_sign(eigvecs[:, mode_idx])
    coefficient = float(np.dot(eigenvector, signal_used))
    contribution = coefficient * eigenvector
    lam = float(eigvals[mode_idx])

    fig = plot_molecular_signal(
        mol,
        contribution,
        title=f"{title_prefix} {mode_idx} (lambda={lam:.3f}, coeff={coefficient:+.3f})",
        signal_label="Mode contribution",
        **plot_kwargs,
    )
    return fig, {
        "mode_idx": mode_idx,
        "lambda": lam,
        "coefficient": coefficient,
        "contribution": contribution,
        "coeffs": coeffs,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
    }


def plot_gft_mode_contribution_from_files(
    xyz_file: str | Path,
    out_file: str | Path,
    property_name: str = "auto",
    property_component: str | None = None,
    mode_idx: int | None = None,
    center_signal: bool = True,
    skip_zero_mode: bool = True,
    **plot_kwargs: Any,
):
    mol = mol_from_orca_xyz(xyz_file)
    data = parse_orca_atom_property(out_file, property_name=property_name)
    signal = data.scalar(property_component)
    if len(signal) != mol.GetNumAtoms():
        raise ValueError(
            f"Parsed {len(signal)} atomic values but molecule has {mol.GetNumAtoms()} atoms"
        )

    fig, info = plot_gft_mode_contribution(
        mol,
        signal,
        mode_idx=mode_idx,
        center_signal=center_signal,
        skip_zero_mode=skip_zero_mode,
        title_prefix=f"{data.name.upper()} mode contribution",
        **plot_kwargs,
    )
    info.update(
        {
            "mol": mol,
            "properties": signal,
            "property_data": data,
            "property_name": data.name,
            "property_component": data.resolve_component(property_component),
            "atoms": list(data.atoms),
        }
    )
    return fig, info


# reporting and associated helper functions
def print_conformer_summary(records: Sequence[Mapping[str, Any]]) -> None:
    print("\nConformer summary")
    print("-" * 90)
    print(f"{'name':25s} {'E / Eh':>18s} {'dE / kJ mol-1':>16s} {'pop':>10s} {'mu / D':>10s}")
    print("-" * 90)
    for record in records:
        print(
            f"{str(record['name']):25s} "
            f"{float(record['energy_hartree']):18.10f} "
            f"{float(record['relative_energy_kjmol']):16.3f} "
            f"{float(record['population']):10.4f} "
            f"{float(record['dipole_mag_debye']):10.3f}"
        )
    print("-" * 90)


def parse_key_results(
    results: Mapping[str, Any],
    results_key: str = "features_from_weighted_property_values",
    multi: bool = True,
) -> dict[str, Any]:
    records = results["records"]
    first_out = Path(records[0]["out_file"])

    if multi:
        parent_parts = first_out.parent.name.split("_")
        filename = "_".join(parent_parts[:2]) or first_out.parent.name
    else:
        filename = "_".join(first_out.stem.split("_")[:3])

    average_dipole = results.get("average_dipole_mag_debye", np.nan)
    features = results.get(results_key, {})
    high_frequency = results.get("high_frequency", np.nan)
    very_high_frequency = results.get("very_high_frequency", np.nan)

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

    return {
        "filename": filename,
        "property_name": results.get("used_property_name"),
        "property_component": results.get("used_property_component"),
        "high_frequency": high_frequency,
        "very_high_frequency": very_high_frequency,
        "average_dipole_mag_debye": average_dipole,
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
        "spectral_spread": features.get("spectral_spread", np.nan),
        "dominant_lambda": features.get("dominant_lambda", np.nan),
        "dominant_power": features.get("dominant_power", np.nan),
        "dominant_power_fraction": features.get("dominant_power_fraction", np.nan),
        "spectral_entropy": features.get("spectral_entropy", np.nan),
        "spectral_entropy_norm": features.get("spectral_entropy_norm", np.nan),
        "dipole_x_high_freq_fraction": average_dipole * hf_frac,
        "dipole_x_very_high_freq_fraction": average_dipole * vhf_frac,
        "dipole_x_high_freq_power": average_dipole * hf_power,
        "dipole_x_very_high_freq_power": average_dipole * vhf_power,
        "dipole_x_high_freq_fraction_nonzero": average_dipole * hf_frac_nz,
        "dipole_x_very_high_freq_fraction_nonzero": average_dipole * vhf_frac_nz,
        "dipole_x_spectral_centroid_nonzero": average_dipole * spec_cent_nz,
    }


def get_cols() -> list[str]:
    return [
        "filename",
        "property_name",
        "property_component",
        "high_frequency",
        "very_high_frequency",
        "average_dipole_mag_debye",
        "total_power",
        "total_power_nonzero",
        "high_freq_fraction_lambda",
        "very_high_freq_fraction_lambda",
        "high_freq_fraction_nonzero_lambda",
        "very_high_freq_fraction_nonzero_lambda",
        "high_freq_power",
        "very_high_freq_power",
        "spectral_centroid",
        "spectral_centroid_nonzero",
        "spectral_spread",
        "dominant_lambda",
        "dominant_power",
        "dominant_power_fraction",
        "spectral_entropy",
        "spectral_entropy_norm",
        "dipole_x_high_freq_fraction",
        "dipole_x_very_high_freq_fraction",
        "dipole_x_high_freq_power",
        "dipole_x_very_high_freq_power",
        "dipole_x_high_freq_fraction_nonzero",
        "dipole_x_very_high_freq_fraction_nonzero",
        "dipole_x_spectral_centroid_nonzero",
    ]


def plot_df_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    nem_v_smectic: bool = False,
    alpha: tuple[float, float] = (0.8, 0.9),
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim: tuple[float, float] | None = None,
    edge_colour: str = "k",
    sizes: tuple[float, float] = (90, 120),
) -> None:

    '''
    This had a fairly limited use case; plot some GFT derived descriptor against some
    other GFT descriptor. Probably not htat useful.
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
        alpha=alpha[0],
        edgecolor=edge_colour,
    )

    if not nem_v_smectic:
        plt.scatter(
            df.loc[mask_polar, x_col],
            df.loc[mask_polar, y_col],
            s=sizes[1],
            label="polar nematic",
            alpha=alpha[1],
            edgecolor=edge_colour,
        )
    else:
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

    if xlim is not None:
        plt.xlim(xlim)
    plt.xlabel(xlabel if xlabel is not None else x_col.replace("_", " "))
    plt.ylabel(ylabel if ylabel is not None else y_col.replace("_", " "))
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


__all__ = [
    "AtomPropertyData",
    "ConformerFilePair",
    "GraphBasis",
    "GFTSpectrum",
    "find_available_atom_properties",
    "get_avail_atom_props",
    "get_available_charge_modes",
    "parse_orca_atom_property",
    "parse_orca_atom_props",
    "read_xyz_symbols",
    "mol_from_orca_xyz",
    "mol_to_adjacency",
    "mol_to_normalised_laplacian",
    "build_graph_basis",
    "get_graph_spectrum",
    "compute_gft",
    "compute_gft_spectrum",
    "gft_from_properties",
    "spectrum_features",
    "choose_mode_from_gft",
    "fix_eigenvector_sign",
    "top_k_indices",
    "parse_orca_scf_energy",
    "parse_orca_dipole",
    "parse_orca_quadrupole",
    "boltzmann_populations_from_energies",
    "weighted_nanmean",
    "find_conformer_file_pairs",
    "run_orca_gft_workflow",
    "multi_conformer_workflow",
    "single_conformer_workflow",
    "single_conformer_charge_workflow",
    "full_workflow",
    "plot_molecular_signal",
    "plot_gft",
    "plot_gft_mode_contribution",
    "plot_gft_mode_contribution_from_files",
    "print_conformer_summary",
    "parse_key_results",
    "get_cols",
    "plot_df_scatter",
]
