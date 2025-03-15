import tomllib
from setuptools import find_packages, setup

pyproject = dict()

with open("pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)

uv_project = "project"
name = pyproject[uv_project]["name"]
version = pyproject[uv_project]["version"]
description = pyproject[uv_project]["description"]
readme = pyproject[uv_project]["readme"]
dependencies = pyproject[uv_project]["dependencies"]
python_requires = pyproject[uv_project]["requires-python"]

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
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Machine Learning",
    ],
)
