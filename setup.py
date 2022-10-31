from setuptools import setup

version = '1.0.0'

classifiers = [
    'Programming Language :: Python :: 3',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Topic :: Scientific/Engineering :: Information Analysis',
]

dependencies = [
    'numpy>=1.14.5',
    'pandas>=1.1.1',
    'scipy>=1.1.1',
    'tqdm>=3',
    'scikit-learn>=1.1.2',
]

with open('README.md', 'r') as f:
    readme = f.read()

setup(
    name = 'gstatsim',
    version = version,
    description = 'Geostatistics tools for interpolation and simulation.',
    long_description = readme,
    long_description_content_type = 'text/markdown',
    author = '(Emma) Mickey MacKie',
    author_email = 'emackie@ufl.edu',
    url = 'https://github.com/GatorGlaciology/GStatSim',
    license = 'MIT',
    classifiers = classifiers,
    py_modules = ['gstatsim'],
    python_requires = '>=3',
    install_requires = dependencies,
)
