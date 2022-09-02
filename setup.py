
"""setup.py: setuptools control."""

import re
from setuptools import setup

version = re.search(
    '^__version__\s*=\s*"(.*)"',
    open('laff/laff.py').read(),
    re.M
    ).group(1)

with open("README.rst", "rb") as f:
    long_descr = f.read().decode("utf-8")

requirements = ["astropy>=4.3","pandas>=1.4.3","matplotlib>=3.5.2", "scipy>=1.8.0","numpy>=1.23", \
    "argparse>=1.4"]

setup(
    name = "laff",
    packages = ["laff"],
    entry_points = {
        "console_scripts": ['laff = laff.laff:main']
        },
    version = version,
    description = "Automated fitting of continuum and flares in GRB lightcurves.",
    long_description = long_descr,
    author = "Adam Hennessy",
    author_email = "ah724@leicester.ac.uk",
    url = "https://github.com/ajhenne/laff",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)