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

You can install these modules manually, and copy RDAnalyzer in the directory that same with trajectory files. Or

**Install with pip**

```pip install RDAnalyzer```

## Cite
- {L. Ma, T. Wang, Angew. Chem. Int. Ed. 2024, 63, e202403614. https://doi.org/10.1002/anie.202403614}
