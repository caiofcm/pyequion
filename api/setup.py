from setuptools import setup, find_packages

setup(
    name="pyequion",
    version="0.0.1",
    description="Chemical equilibrium for electrolytes in pure python",
    packages=["pyequion"],
    author="Caio Marcellos, Elvis Soares",
    license="BSD",
    email="caiocuritiba@gmail.com",
    install_requires=[
        "pyequion",
        "flask",
        "Flask-JSONRPC",
        "dataclasses-json",
        "flask-cors",
        "simplejson",
    ]
    # entry_points='''
    #     [console_scripts]
    #     createsignal=signal_generator.cli:main
    # ''',
)
