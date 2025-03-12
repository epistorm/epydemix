Welcome to Epydemix's Documentation!
====================================

The source code for Epydemix is available on GitHub: `Epydemix Repository <https://github.com/ngozzi/epydemix>`_.

.. image:: https://img.shields.io/github/stars/ngozzi/epydemix?style=social
   :target: https://github.com/ngozzi/epydemix
   :alt: GitHub Repo

.. image:: https://readthedocs.org/projects/epydemix/badge/?version=latest
   :target: https://epydemix.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
    :target: https://www.gnu.org/licenses/gpl-3.0
    :alt: License: GPL v3

.. image:: https://codecov.io/gh/ngozzi/epydemix/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/ngozzi/epydemix
   :alt: Code coverage status


.. image:: https://cdn.prod.website-files.com/67bde9057c9d78157874e100/67c1d1122127f0a9ce202197_epydemix-logo-p-500.png
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

- `Tutorial 1: Model Definition and Simulation <https://github.com/ngozzi/epydemix/blob/main/tutorials/1_Model_Definition_and_Simulation.ipynb>`_
- `Tutorial 2: Using Population Data <https://github.com/ngozzi/epydemix/blob/main/tutorials/2_Modeling_with_Population_Data.ipynb>`_
- `Tutorial 3: Modeling Interventions <https://github.com/ngozzi/epydemix/blob/main/tutorials/3_Modeling_Interventions.ipynb>`_
- `Tutorial 4: Model Calibration with ABC (Part 1) <https://github.com/ngozzi/epydemix/blob/main/tutorials/4_Model_Calibration_part1.ipynb>`_
- `Tutorial 5: Model Calibration with ABC (Part 2) <https://github.com/ngozzi/epydemix/blob/main/tutorials/5_Model_Calibration_part2.ipynb>`_
- `Tutorial 6: Advanced Modeling Features <https://github.com/ngozzi/epydemix/blob/main/tutorials/6_Advanced_Modeling_Features.ipynb>`_
- `Tutorial 7: COVID-19 Case Study <https://github.com/ngozzi/epydemix/blob/main/tutorials/7_Covid-19_Example.ipynb>`_


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
