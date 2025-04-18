==================
GW_GRB_Correlation
==================


.. image:: https://img.shields.io/pypi/v/gw_grb_correlation.svg
        :target: https://pypi.python.org/pypi/gw_grb_correlation

.. image:: https://img.shields.io/travis/CrazyAncestor, gulley-s/gw_grb_correlation.svg
        :target: https://travis-ci.com/CrazyAncestor, gulley-s/gw_grb_correlation

.. image:: https://readthedocs.org/projects/gw-grb-correlation/badge/?version=latest
        :target: https://gw-grb-correlation.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




this is a project studying cross-correlation between GW and GRB


* Free software: GNU General Public License v3
* Documentation: https://gw-grb-correlation.readthedocs.io.

Setup
--------
To download the packages needed, please run the command
$pip install -e .

If unable to download, please follow the following way to create a virtual environment to download
$sudo apt update
$sudo apt install python3-venv

Create a Virtual Environment in your project directory:
$python3 -m venv venv

Activate the Virtual Environment:

For Linux/macOS:
$source venv/bin/activate

For Windows:
$.\venv\Scripts\activate

Install dependencies from requirements.txt inside the virtual environment:
$pip install -e .
This ensures that the packages are installed locally in your virtual environment and not globally.

Unit Tests
--------
To run the Unit Tests
Use the unittest module to discover and run all tests in the tests folder:
$python -m unittest discover
Alternatively, you can run the specific test file:
$python -m unittest test_fermi_data.py


Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
