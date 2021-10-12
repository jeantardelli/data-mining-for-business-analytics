from setuptools import setup, find_packages

setup(
    name="dmutils",
    version="0.0.2",
    description="A userful module for the book exercises and analysis",
    packages=find_packages("src"),
    package_dir={"" : "src"},
)
