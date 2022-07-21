# -*- coding: utf-8 -*-


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


setup(
    name = "laff",
    packages = ["laff"],
    entry_points = {
        "console_scripts": ['laff = laff.laff:main']
        },
    version = version,
    description = "Python command line application bare bones template.",
    long_description = long_descr,
    author = "Adam Hennessy",
    author_email = "ah724@leicester.ac.uk",
    url = "http://gehrcke.de/2014/02/distributing-a-python-command-line-application",
    )


# find requires line from old_laff version and include the dependcies here.
# test if i can run this after uploading and also test if dependencies install