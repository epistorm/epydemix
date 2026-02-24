Welcome to Epydemix's Documentation!
====================================

The source code for Epydemix is available on GitHub: `Epydemix Repository <https://github.com/epistorm/epydemix>`_.

.. image:: https://img.shields.io/github/stars/epistorm/epydemix.svg?style=social
   :target: https://github.com/epistorm/epydemix/stargazers
   :alt: GitHub stars

.. image:: https://static.pepy.tech/badge/epydemix
   :target: https://pepy.tech/projects/epydemix
   :alt: PyPI Downloads

.. image:: https://readthedocs.org/projects/epydemix/badge/?version=latest
   :target: https://epydemix.readthedocs.io/en/latest/?badge=latest
   :alt: Read the Docs

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0
   :alt: License: GPL v3

.. image:: https://codecov.io/gh/epistorm/epydemix/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/epistorm/epydemix
   :alt: Codecov

.. image:: https://img.shields.io/pypi/v/epydemix.svg
   :target: https://pypi.org/project/epydemix/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/Published%20in-PLOS%20Computational%20Biology-blue?logo=plos
   :target: https://doi.org/10.1371/journal.pcbi.1013735
   :alt: PLOS Computational Biology


.. image:: https://raw.githubusercontent.com/epistorm/epydemix/main/tutorials/img/epydemix-logo.png
   :width: 500px
   :align: center


**Epydemix** is a Python package for epidemic modeling. It provides tools to create, calibrate, and analyze epidemic models, allowing users to simulate the spread of infectious diseases using different compartmental models, contact layers, and calibration techniques. 

Features:
---------
- Define and simulate compartmental models (e.g., SIR, SEIR).
- Integrate real-world population data with contact matrices.
- Calibrate models using Approximate Bayesian Computation (ABC).
- Visualize simulation results with built-in plotting tools.
- Extensible framework for modeling interventions and policy scenarios.

Installation
------------

To install Epydemix, use the following command:

.. code-block:: bash

   pip install epydemix

Get started
----------

We provide a series of tutorials to help you get started with Epydemix:

- `Tutorial 1: Model Definition and Simulation <https://github.com/epistorm/epydemix/blob/main/tutorials/01_Model_Definition_and_Simulation.ipynb>`_
- `Tutorial 2: Using Population Data <https://github.com/epistorm/epydemix/blob/main/tutorials/02_Modeling_with_Population_Data.ipynb>`_
- `Tutorial 3: Modeling Interventions <https://github.com/epistorm/epydemix/blob/main/tutorials/03_Modeling_Interventions.ipynb>`_
- `Tutorial 4: Model Calibration with ABC (Part 1) <https://github.com/epistorm/epydemix/blob/main/tutorials/04_Model_Calibration_part1.ipynb>`_
- `Tutorial 5: Model Calibration with ABC (Part 2) <https://github.com/epistorm/epydemix/blob/main/tutorials/05_Model_Calibration_part2.ipynb>`_
- `Tutorial 6: Advanced Modeling Features <https://github.com/epistorm/epydemix/blob/main/tutorials/06_Advanced_Modeling_Features.ipynb>`_
- `Tutorial 7: COVID-19 Case Study <https://github.com/epistorm/epydemix/blob/main/tutorials/07_Covid-19_Example.ipynb>`_
- `Tutorial 8: Modeling Multiple Strains <https://github.com/epistorm/epydemix/blob/main/tutorials/08_Multiple_Strains.ipynb>`_
- `Tutorial 9: Modeling Vaccinations <https://github.com/epistorm/epydemix/blob/main/tutorials/09_Vaccinations.ipynb>`_
- `Tutorial 10: Speeding up Simulations and Calibration with Multiprocessing <https://github.com/epistorm/epydemix/blob/main/tutorials/10_Multiprocessing.ipynb>`_
- `Tutorial 11: Using Epistorm-Mix Contact Matrices (sex and race/ethnicity) <https://github.com/epistorm/epydemix/blob/main/tutorials/11_Epistorm_Mix_Matrices.ipynb>`_

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   epydemix.calibration
   epydemix.model
   epydemix.population
   epydemix.utils
   epydemix.visualization



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
