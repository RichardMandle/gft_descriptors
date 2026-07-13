# Graph Fourier Analysis of Molecular Charge Distributions

## Scientific background
The paper by Palacio-Betancur and Jackson (https://pubs.acs.org/doi/10.1021/jacs.5c18760), and the repo by the same authors (https://github.com/TheJacksonLab/PolarNematic) use graph Fourier transform (GFT) analysis of atom-centred molecular charge distributions to predict the presence/absence of polar nematic order. 

<br><br>
Here, we develop our own implementation of this GFT analysis using atomic partial charges computed with Orca (https://www.faccts.de/docs/orca/6.1/tutorials/prop/charges.html). Our workflow can work with single geometries:

``` import gft_helpers as gft

result = gft.single_conformer_charge_workflow(
    xyz_file="example.xyz",
    out_file="example.out",
    charge_model="mbis",
    atom_mode="all",
    plot=True,
)
```

or multiple conformers generated with ORCA/GOAT (and laid out something like ```compound/conf_000.xyz, conf_000.xyz```):

```result = gft.multi_conformer_charge_workflow(
    "compound/",
    xyz_glob="*.xyz",
    out_glob="*.out",
    charge_model="mbis",
    T=298.15,
    plot=True,
)
```

This returns a bunch of ***GFT descriptors** but we don't use these much. Just looking at the power spectrum is _sometimes_ enough to discern polar versus non polar, nematic vs smectic. 


## Installation
I just used Conda: 
```
conda create -n molecular-gft python=3.12
conda activate molecular-gft
conda install -c conda-forge \
    rdkit numpy pandas matplotlib jupyter cmasher
```

cmasher isn't essential, but it gives some non-matplotlib perceptually uniform colourmaps which can be useful.

## Electronic Structure Calculations
The code will, as written, only work with output from Orca (we use Orca 6.1). For each molecule or conformer it expects both the .xyz file (geometry) and .out file containing the charges. A typical input file might look like this:

```! wb97x-3c mbis hirshfeld chelpg resp
%pal nprocs 4 end
%maxcore 4000
%method
	   MBIS_LARGEPRINT TRUE
	end
#conf_000
* xyzfile 0 1 conf_000.xyz
```

Generally, we generate conformers with GOAT / GFN2-xTB and then do a single point (as above) with composite DFT (wB97x-3C). RESP and MBIS charge models seem best.

## simple walkthrough:
See examples/UIO_confs for UIO (UUZGU-3-F; see Gibb et al., https://advanced.onlinelibrary.wiley.com/doi/10.1002/adma.73977?af=R) output and xyz files. Single conformer:

```
import gft_helpers as gft
import pandas as pd

dft_path = '/Users/phyrma/orca_sandbox/nf_gft/xIO_mbis'
m_o_i = 'UIO_confs/conf_000'

x_pth = f'{dft_path}//{m_o_i}.xyz'
o_pth = f'{dft_path}//{m_o_i}.out'

gft.get_available_charge_modes(o_pth)

eigs, sig, mol, charge = gft.full_workflow(x_pth, 
                                           o_pth, 
                                           determine_bo = True, 
                                           charge_model = 'resp', 
                                           atom_mode = 'all', 
                                           cmap='inferno', 
                                           circle_scale = 200, 
                                           v_ax_scale = 0.5, 
                                           dpi = 400)
```
The available charge models are printed ("MULLIKEN", "LOEWDIN", "HIRSHFELD", "MBIS", "RESP"). As we didn't specify ```plot = False``` it'll go ahead and show the scale of the MBIS charges on the molecular graph. 
<img width="2560" height="911" alt="image" src="https://github.com/user-attachments/assets/67c53794-b95d-49a5-b87b-ade95f107817" />

And we see the GFT charge spectrum:
<img width="780" height="580" alt="image" src="https://github.com/user-attachments/assets/47aaf415-adae-4fac-b6f5-d3cc228b3ce1" />


If rdkit struggles with bonding errors, just set ```determine_bo = False ```. 

gft_helpers can look at multiple conformers with a single function call:
```d = 'UIO_confs/'
dft_path = '/Users/phyrma/orca_sandbox/nf_gft/xIO_mbis'

d = f'{dft_path}/{d}'
_ = gft.multi_conformer_charge_workflow(
    d, xyz_glob="*.xyz", 
    out_glob="*.out",
    charge_model="resp", 
    determine_bo=True,
    T=298, atom_mode="heavy", 
    cmap="inferno",
    circle_scale=200, 
    v_ax_scale=1,
    normalise_eigs=False,
    show_atom_indices=False, 
    show_atom_symbols=True, 
    show_charge_values=False,
    plot = True, 
    report = False, 
    high_freq = 1.5, 
    v_high_freq = 1.99,
    plt_v_max=0.4,
    plt_v_min=-0.4)
```

With mapped charges: 
<img width="1280" height="455" alt="image" src="https://github.com/user-attachments/assets/c91dae7e-6aaf-4fb3-956a-1a68a14d1909" />
yielding the power spectrum:
<img width="780" height="580" alt="image" src="https://github.com/user-attachments/assets/07062f28-52c7-4ff1-b076-a4fb0194c55d" />


With a bit of glob and a basic loop, we can look at many molecules. We can label the "interesting" (i.e. polar) ones:

```
high_freq = 1.5
very_high_freq = 1.95

directories = glob.glob(f'{dft_path}/*/')

rows = []

for d in directories:
    result  = gft.multi_conformer_charge_workflow(
    d, xyz_glob="*.xyz", 
    out_glob="*.out",
    charge_model="mbis", 
    determine_bo=True,
    T=298, atom_mode="all", 
    cmap="plasma",
    circle_scale=150, 
    v_ax_scale=1,
    normalise_eigs=False,
    show_atom_indices=False, 
    show_atom_symbols=True, 
    show_charge_values=False,
    plot = False, 
    report = False, 
    high_freq = high_freq, 
    v_high_freq = very_high_freq,
    plt_v_max=0.4,
    plt_v_min=-0.4)
    
    print(d)
    quick_eig_plot(eigs = result['eigenvalues'], sig = result['population_average_power_spectrum'])
    
    rows.append(gft.parse_key_results(result, results_key = 'features_from_population_average_power', multi = True))
    
df = pd.DataFrame(rows, columns=gft.get_cols())

interesting_list = ["DIO_confs","UIO_confs", "PIO_confs"]
nematics = ["DIO_confs","UIO_confs"]

df["interesting"] = df["filename"].isin(interesting_list)
df["nematic"] = df["filename"].isin(nematics)
```

And compare the GFT power spectrum of a polar-nematic former (UIO, top) with a borderline-polar material (OIO) apolar-nematic former (BIO, bottom)

UIO (polar):
<img width="806" height="220" alt="image" src="https://github.com/user-attachments/assets/05efb40b-e4ba-4c03-a232-166f45f3dc91" />
OIO (edge-case):
<img width="806" height="220" alt="image" src="https://github.com/user-attachments/assets/962910eb-f7ba-4f06-83d3-ff537de61482" />
BIO (non-polar, but dipole > 13 Debye):
<img width="806" height="220" alt="image" src="https://github.com/user-attachments/assets/cf7ce8e2-8d6f-4ec2-afc9-1f3702b5e1b2" />


_This is research code under active development._

## Scientific motivation
Conventional molecular descriptors such as dipole moment describe the overall
magnitude and direction of charge separation, but do not fully describe how
positive and negative charge is distributed across a molecular framework.

Here, a molecule is represented as a graph:

- atoms are graph vertices;
- covalent bonds define graph edges;
- atom-centred partial charges form a signal on the graph.

The charge signal is projected onto the eigenvectors of the symmetric,
normalised graph Laplacian. The resulting GFT power spectrum describes the
distribution of electrostatic structure over graph-frequency modes.

Low graph frequencies describe slowly varying charge distributions, whereas
high graph frequencies describe rapid charge alternation between neighbouring
atoms. The approach is being explored as a source of molecular descriptors for
distinguishing apolar, polar-nematic, and polar-smectic liquid crystals.
