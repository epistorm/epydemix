from setuptools import find_packages, setup

# Read the README file to use as the long description (optional)
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="epydemix",
    version="1.0.2",
    author="The Epydemix Developers",
    author_email="epydemix@isi.it",
    description="A Python package for epidemic modeling, simulation, and calibration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/epistorm/epydemix",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "evalidate>=2.0.3",
        "matplotlib>=3.7.3",
        "numpy>=1.23.5",
        "pandas>=2.0.3",
        "scipy>=1.10.1",
        "seaborn>=0.13.2",
        "setuptools>=68.2.0",
    ],
    entry_points={
        "console_scripts": [],
    },
)
