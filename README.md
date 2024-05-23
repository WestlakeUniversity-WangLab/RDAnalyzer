# RDAnalyzer
Reactive Diffusion Analyzer (RDAnalyzer)

RDAnalyzer is a Python module for analyzing ReaxFF MD trajectory, especially the hydroxide diffusion in AEM.

Features:
- Splitting trajectory according to user's setting
- Transforming text trajectory to Atoms, Molecules, and Reactions
- Analysis based on these Atoms, Molecules, and Reactions
- Calculation of drift length of hydroxide for obtaining conductivity


## Installation
RDAnalyzer is developed on Ubuntu with python 3.11

RDAnalyzer needs `scipy`, `numpy`, `matplotlib`, `ase` and `networkx`.

You can install these modules manually, and copy RDAnalyzer in the directory that same with trajectory files.

**or**

**Install with pip**

```pip install RDAnalyzer```

## Getting started
See `Tutorial.ipynb` for detailed usage about RDAnalyzer 

## References
- {}
