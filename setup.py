from glob import glob
from os.path import basename, splitext
from setuptools import setup, find_packages
# from distutils.core import setup, find_packages

requirements = ['torch~=1.8.0', 'fair-esm~=0.3.1', 'pytest', 'tqdm']
setup(
    name="pgen",
    version="0.2.0",
    description="Generating new protein sequence by gibbs sampling masked protein language models",
    author="Sean Johnson, Sarah Monaco, Kenneth Massie, Zaid Sayed",
    url="https://github.com/seanrjohnson/protein_gibbs_sampler",
    license="MIT",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    scripts=["src/pgen/pgen_esm.py", "src/pgen/pgen_msa.py", "src/pgen/pgen_esm_from_fasta.py"],
    # py_modules=[splitext(basename(path))[0] for path in glob('src/pgen/*.py')],
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
    # python_requires='>=3.6'
)