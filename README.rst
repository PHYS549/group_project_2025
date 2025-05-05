==================
GW_GRB_Correlation
==================
This is a project studying cross-correlation between GW and GRB.

* Free software: GNU General Public License v3

Installation
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

Demo Notebooks
--------
Explore GRB data:

1. Open demo/grb_data_exploration.ipynb

2. Run the cells to first download and preprocess the Fermi data

3. Optional: If you want to access the preprocessed data instead of downloading raw data from Fermi database, which is very long, you can download the preprocessed data from the google drive link in the notebook and upload it to the same directory as the notebook.

4. Run the cells following to visualize the data and explore the GRB data.

5. To run the time correlation analysis between GRB and GW, be sure that you already run the GW data preprocessing notebook first to get the totalgwdata.csv file and places it in the ./gw_data/ directory.

Train GRB location model:

1. Open demo/fermi_position_predictor.ipynb

2. Be sure to run the demo/grb_data_exploration.ipynb notebook first to get the preprocessed Fermi data.

3. Run the cells in the notebook to train the GRB localization model.

4. You may also run the cells to visualize the model performance and the GRB localization results.

Get GW data:

1. These files lead to getting a reduced GW data file, however this can file can be downloaded at (URL HERE). If further correlation studies are to be done, these files can be editied to create a new reduced GW data file

2. The data files are seperated into the three observing runs. These are files that have been created using Google Colab. Using Google Colab is thus recommended when running these files, as well as following the directions that are commented. Run files demo/gewtgwc1data.ipynb, demo/getgwc2data.ipynb, and demo/getgwc3data.ipynb

3. Run file demo/getgwtotaldata.ipynb.

Features
--------

1. Convenience for downloading and preprocessing Fermi GRB data.

2. Help visaulization of GRB data.

3. Provide a GRB localization model training method based on the Fermi data.

4. Provide an easy way to study the correlation between GRB and GW data.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
