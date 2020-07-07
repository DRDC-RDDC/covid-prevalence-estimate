from setuptools import setup, find_namespace_packages
import re

from os import path

with open("README.md") as f:
    long_description = f.read()

verstr = "unknown"
try:
    verstrline = open("covid_prevalence/_version.py", "rt").read()
except EnvironmentError:
    pass
else:
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(VSRE, verstrline, re.M)
    if mo:
        verstr = mo.group(1)
    else:
        raise RuntimeError("unable to find version in covid_prevalence/_version.py")

setup(
    name="covid_prevalence",
    author="Steven Horn",
    author_email="steven@horn.work",
    packages=find_namespace_packages(),
    url="https://gitlab.com/stevenhorn/covid-prevalence-estimate/",
    description="Bayesian estimation of COVID point prevalence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6.0",
    version=verstr,
    install_requires=["pymc3>=3.9.1", "matplotlib", "numpy", "pandas", "theano","GitPython","geopandas","geojson","folium>=0.11.0",],
)