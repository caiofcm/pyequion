import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()
LICENSE = (HERE / "LICENSE.md")#.read_text()

setup(
    name="pyequion",
    version='0.0.5',
    description="Chemical equilibrium for electrolytes in pure python",
    # packages=["pyequion"],
    url="https://github.com/caiofcm/pyequion",
    packages=find_packages(exclude=("tests",)),
    long_description=README,
    long_description_content_type="text/markdown",
    author="Caio Curitiba Marcellos",
    author_email="caiocuritiba@gmail.com",
    license='BSD 3-Clause',
    #license_file=LICENSE,
    email='caiocuritiba@gmail.com',
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'numpy',
        # 'numba>=0.45',
        # 'numba',
        'ordered-set',
        'commentjson',
        'scipy',
        'sympy',
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
    ],
    keywords=[
        "scientific",
        "chemical equilibrium",
    ]
    # package_data = {
    #     # If any package contains *.txt or *.rst files, include them:
    #     '': ['*.txt', '*.xml', '*.special', '*.huh'],
    # },
    # data_files = [
    #     ('', ['test.txt'])
    # ]
    # entry_points='''
    #     [console_scripts]
    #     createsignal=signal_generator.cli:main
    # ''',
)
