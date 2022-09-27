import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="harmonizome-helper",
    version="0.2.2",
    author="Charles R. Dai",
    author_email="charles.r.dai@gmail.com",
    description="A small package of helper functions for Harmonizome appyters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chrlsdai/harmonizome",
    install_requires=list(
        map(str.strip, open('requirements.txt', 'r').readlines())),
    packages=setuptools.find_packages(),
)
