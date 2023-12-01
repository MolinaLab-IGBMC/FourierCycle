## FourierCycle--cell cycle dependence of mRNA metabolism using Fourier series approximation
The cell cycle is a highly regulated process that ensures the accurate replication and transmission of genetic information from one generation of cells to the next. In eukaryotic cells the cell cycle consists of four main phases--G1, S, G2, M. The advent of single-cell sequencing technologies, and recent advances in computational methodologies, such as [RNA velocity](http://velocyto.org/) and [DeepCycle](https://github.com/andreariba/DeepCycle) have made it possible to investigate the cell cycle with extremely high temporal resolution. In this work we showcase a novel computational approach that allows us to obtain the cell cycle dependence of mRNA metabolism for every gene.

### Biophysical model
We model the dynamics of mRNA metabolism by expressing the steps involved—transcription, splicing, degradation—as a system of coupled ordinary differential equations,

$\frac{\mathrm{d}u(t)}{\mathrm{d}t} = \alpha(t) - \beta u(t)$

$\frac{\mathrm{d}s(t)}{\mathrm{d}t} = \beta u(t) - \gamma(t) s(t)$.

We used Fourier series approximation to solve for $u(t)$, $s(t)$, $\alpha(t)$, $\gamma(t)$, and $\beta$. The mathematical details are in `fourier_solution.pdf`.

### Usage
To use the code one would need to run [DeepCycle](https://github.com/andreariba/DeepCycle) and obtain an AnnData object which contains layers: `Ms` and `Mu` and obs: `cell_cycle_theta`. The notebook `jupyter/fit_gene.ipynb` demonstrates the usage. The code has been tested on single-cell RNA-seq and single-nucleus RNA-seq datasets obtained from the Chromium platform of 10X Genomics.

### Dependencies
The code has the following dependencies:
> scipy
numpy
pandas
scikit-learn
anndata
gseapy
matplotlib
seaborn

### Contact
For questions and concerns contact mauliknariya[at]gmail[dot]com
