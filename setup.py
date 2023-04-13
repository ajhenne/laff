import re
from setuptools import setup, find_packages

version = re.search(
    '^__version__\s*=\s*"(.*)"',
    open('laff/__init__.py').read(),
    re.M
    ).group(1)

with open("README.md", "rb") as f:
    long_descr = f.read().decode("utf-8")

# requirements = ["astropy>=5.1","pandas>=1.4.3","matplotlib>=3.5.2", "scipy>=1.8.0","numpy>=1.23", \
#     "argparse>=1.4", "lmfit>=1.0.3"]
requirements = ["pandas>=1.4.0", "astropy>=5.1", "numpy>=1.23", "matplotlib>3.5.2"]

setup(
    name='laff',
    version=version,
    author='Adam Hennessy',
    author_email='ah724@alice2.le.ac.uk',
    description='Automated fitting of continuum and flares in GRB lightcurves.',
    long_description_content_type="text/markdown",
    long_description=long_descr,
    url = "https://github.com/ajhenne/laff",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)