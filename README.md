# Epydemix, the ABC of Epidemics
[![GitHub stars](https://img.shields.io/github/stars/epistorm/epydemix.svg?style=social)](https://github.com/epistorm/epydemix/stargazers)
[![PyPI Downloads](https://static.pepy.tech/badge/epydemix)](https://pepy.tech/projects/epydemix)
[![Read the Docs](https://readthedocs.org/projects/epydemix/badge/?version=latest)](https://epydemix.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Codecov](https://codecov.io/gh/epistorm/epydemix/branch/main/graph/badge.svg)
[![PyPI version](https://img.shields.io/pypi/v/epydemix.svg)](https://pypi.org/project/epydemix/)
[![medRxiv](https://img.shields.io/badge/preprint-medRxiv-orange)](https://www.medrxiv.org/content/10.1101/2025.05.07.25327151v1)

![Alt text](https://raw.githubusercontent.com/epistorm/epydemix/main/tutorials/img/epydemix-logo.png)


**[Documentation](https://epydemix.readthedocs.io/en/latest/)** | **[Website](https://www.epydemix.org/)** | **[Tutorials](https://github.com/epistorm/epydemix/tree/main/tutorials)**

**Epydemix** is a Python package for epidemic modeling. It provides tools to create, calibrate, and analyze epidemic models, allowing users to simulate the spread of infectious diseases using different compartmental models, contact layers, and calibration techniques. It is designed to be used in conjunction with the [epydemix-data](https://github.com/epistorm/epydemix-data/) package to load population and contact matrix data.


## Installation

To install the latest version of `epydemix` from [PyPI](https://pypi.org/project/epydemix/), use the following command in a terminal or command prompt—ideally within a virtual environment to avoid dependency conflicts:

```bash
pip install epydemix
```

---

## Quick Start

Once installed, you can start using **epydemix** in your Python scripts or Jupyter notebooks. Below an example to get started.

### Example: Creating and running a simple SIR model

```python
from epydemix import EpiModel
from epydemix.visualization import plot_quantiles

# Define a basic SIR model
model = EpiModel(
    name="SIR Model",
    compartments=["S", "I", "R"],  # Susceptible, Infected, Recovered
)

# Add transitions: infection and recovery
model.add_transition(source="S", target="I", params=(0.3, "I"), kind="mediated")
model.add_transition(source="I", target="R", params=0.1, kind="spontaneous")

# Run simulations
results = model.run_simulations(
    start_date="2024-01-01",
    end_date="2024-04-10",
    Nsim=100,
)

# Extract and plot quantiles of compartment counts
df_quantiles = results.get_quantiles_compartments()
plot_quantiles(df_quantiles, columns=["I_total", "S_total", "R_total"])
```

### Tutorials
We provide a series of tutorials to help you get started with **epydemix**.

- [Tutorial 1](https://github.com/epistorm/epydemix/blob/main/tutorials/01_Model_Definition_and_Simulation.ipynb): An Introduction to Model Definition and Simulation
- [Tutorial 2](https://github.com/epistorm/epydemix/blob/main/tutorials/02_Modeling_with_Population_Data.ipynb): Using Population Data from Epydemix Data
- [Tutorial 3](https://github.com/epistorm/epydemix/blob/main/tutorials/03_Modeling_Interventions.ipynb): Modeling Non-pharmaceutical Interventions
- [Tutorial 4](https://github.com/epistorm/epydemix/blob/main/tutorials/04_Model_Calibration_part1.ipynb): Model Calibration with ABC (Part 1)
- [Tutorial 5](https://github.com/epistorm/epydemix/blob/main/tutorials/05_Model_Calibration_part2.ipynb): Model Calibration with ABC (Part 2)
- [Tutorial 6](https://github.com/epistorm/epydemix/blob/main/tutorials/06_Advanced_Modeling_Features.ipynb): Advanced Modeling Features
- [Tutorial 7](https://github.com/epistorm/epydemix/blob/main/tutorials/07_Covid-19_Example.ipynb): COVID-19 Case Study
- [Tutorial 8](https://github.com/epistorm/epydemix/blob/main/tutorials/08_Multiple_Strains.ipynb): Modeling Multiple Strains
- [Tutorial 9](https://github.com/epistorm/epydemix/blob/main/tutorials/09_Vaccinations.ipynb): Modeling Vaccinations
- [Tutorial 10](https://github.com/epistorm/epydemix/blob/main/tutorials/10_Multiprocessing.ipynb): Speeding up Simulations and Calibration with Multiprocessing

---
## Epydemix Data

**epydemix** also provides access to a wealth of real-world population and contact matrix data through the [**epydemix_data**](https://github.com/epistorm/epydemix-data/) module. This dataset allows you to load predefined population structures, including age distribution and contact matrices for over 400 locations globally. You can use this data to create realistic simulations of disease spread in different geographies.

### Example of Loading Population Data

```python
from epydemix.population import load_epydemix_population

# Load population data for the United States using the Mistry 2021 contact matrix
population = load_epydemix_population(
    population_name="United_States",
    contacts_source="mistry_2021",
    layers=["home", "work", "school", "community"],
)

# Assign the loaded population to the epidemic model
model.set_population(population)
```

Epydemix can load data either locally from a folder or directly from online sources, making it easy to simulate a wide range of epidemic models on real population data.

For more information about the available population and contact matrices and to download the data, please visit the [dedicated repository](https://github.com/epistorm/epydemix-data/).

---
## Citation 
The preprint describing the development of Epydemix is available [here](https://www.medrxiv.org/content/10.1101/2025.05.07.25327151v1).
To reference our work, please use the following citation:
```
@article{gozzi2025epydemix,
	author = {Gozzi, Nicol{\`o} and Chinazzi, Matteo and Davis, Jessica T. and Gioannini, Corrado and Rossi, Luca and Ajelli, Marco and Perra, Nicola and Vespignani, Alessandro},
	title = {Epydemix: An open-source Python package for epidemic modeling with integrated approximate Bayesian calibration},
	elocation-id = {2025.05.07.25327151},
	year = {2025},
	doi = {10.1101/2025.05.07.25327151},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2025/05/08/2025.05.07.25327151},
	eprint = {https://www.medrxiv.org/content/early/2025/05/08/2025.05.07.25327151.full.pdf},
	journal = {medRxiv}
}
```

---
## License

This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for more details.

---
## Contact

For questions or issues, please open an issue on GitHub or contact the maintainer at `epydemix@isi.it`.
