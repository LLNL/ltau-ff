from setuptools import setup, find_packages

setup(
    name='ltau_ff',
    version='0.0.1',
    packages=find_packages(),
    install_requires=['faiss-cpu==1.8.0'],  # may need to install separately
    scripts=[
        'scripts/ltau-ff-nequip-descriptors',
        'scripts/ltau-ff-mace-descriptors',
        'scripts/ltau-ff-nequip-minimizer',
        ]
)
