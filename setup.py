from setuptools import find_packages, setup

setup(
    name="tfop_code",
    packages=find_packages(where="tfop"),
    package_dir={"": "tfop"},
)
