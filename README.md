# Probing NO2 emission plumes of individual ships

Urban Computing 2023-2024, Group 2. Stijn Vleugels (vleugels@strw.leidenuniv.nl) & Bart Ouwehand (ouwehand@strw.leidenuniv.nl)

## Description

During this project we wanted to add upon a previous method by kurchaba et al. 2021 on determining the NO2 emission from individual ships. We used a Exponentially Modified Gaussian curve to model the plume and extract the emission rate and plume lifetime.


## Executing program

To download the TROPOMI data and regrid it in the region, first run the TROPOMI\_API.ipynb file and then the TROPOMI\_Regrid\_NO2.ipynb file.

After that, you need the AIS data which is unfortunately not publicly available. If you have this, the Data\_analysis.ipynb can be run to reproduce the results of our report.
