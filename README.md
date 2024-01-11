## Fourier—Cyclecell cycle dependence of mRNA metabolism using Fourier series approximation
The cell cycle is a highly regulated process that ensures the accurate replication and transmission of genetic information from one generation of cells to the next. In eukaryotic cells the cell cycle consists of four main phases—G1, S, G2, M. The advent of single-cell sequencing technologies, and recent advances in computational methodologies, such as [RNA velocity](http://velocyto.org/) and [DeepCycle](https://github.com/andreariba/DeepCycle) have made it possible to investigate the cell cycle with extremely high temporal resolution. In this work we showcase a novel computational approach that allows us to obtain the cell cycle dependence of mRNA metabolism for every gene.

### Biophysical model
We model the dynamics of mRNA metabolism—transcription, splicing, degradation—as a system of coupled ordinary differential equations (ODE),

$\frac{\mathrm{d}u(\theta)}{\mathrm{d}\theta} = \alpha(\theta) - \beta u(\theta)$

$\frac{\mathrm{d}s(\theta)}{\mathrm{d}\theta} = \beta u(\theta) - \gamma(\theta) s(\theta)$

where $u$ and $s$ are the unspliced and the spliced levels of mRNA respectively, typically obtained from sequencing data, $\alpha$, $\beta$, and $\gamma$ are the synthesis, splicing and degradation rates respectively. We use Fourier series approximation to solve the ODEs. The mathematical details are in `fourier_solution.pdf`.

### Usage
To use the code one would first need to run [DeepCycle](https://github.com/andreariba/DeepCycle) and obtain an AnnData object which contains the layers: `Mu` and `Ms` and obs: `cell_cycle_theta`, these are the model observables $u$, $s$, and $\theta$ respectively. The notebook `jupyter/fit_gene.ipynb` demonstrates how to obtain the rates for a given gene. The code has been tested on single-cell RNA-seq and single-nucleus RNA-seq datasets obtained from the 10X Genomics Chromium platform.

### Dependencies
The code has the following dependencies:
* scipy
* numpy
* pandas
* scikit-learn
* anndata
* gseapy
* matplotlib
* seaborn

### Contact
For questions and concerns contact mauliknariya[at]gmail[dot]com
