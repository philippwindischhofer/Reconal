from setuptools import setup, find_packages

setup(
    name='reconal',
    packages=find_packages(),
    install_requires=[
        'Cython==3.0.11',
        'h5py==3.12.1',
        'numpy==2.1.3',
        'scipy==1.14.1',
        'pykonal @ git+https://github.com/mcb28/pykonal.git'
    ]
)