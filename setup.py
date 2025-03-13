import tomllib
from setuptools import find_packages, setup

pyproject = dict()

with open("pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)

project = "project"
name = pyproject[project]["name"]
version = pyproject[project]["version"]
description = pyproject[project]["description"]
readme = pyproject[project]["readme"]
dependencies = pyproject[project]["dependencies"]
python_requires = pyproject[project]["requires-python"]

setup(
    name=name,
    version=version,
    description=description,
    author="Abner Tu",
    # author_email="",
    url="https://github.com/gin31259461/torch-opt-automl",
    packages=find_packages(),
    install_requires=dependencies,
    python_requires=python_requires,
)
