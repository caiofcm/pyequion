from setuptools import setup, find_packages

setup(
    name="pyequion",
    version='0.0.1',
    description="Chemical equilibrium for electrolytes in pure python",
    packages=["pyequion"],
    author="Caio Marcellos",
    license='BSD',
    email='caiocuritiba@gmail.com',
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'numpy',
        'numba>=0.45',
        'ordered-set',
        'commentjson',
        'scipy',
        'sympy',
    ],
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
