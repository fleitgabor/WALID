# WALID
**W**indowed **A**nisotropic **L**ocal **I**nverse **D**istance weighted (WALID) interpolation tool for riverbed mapping.

This is a Python-based tool to generate river digital elevation models (DEM) from cross-sectional data.
The /test_data folder contains inputs for testing.

- The **control.txt** file contains the user defined parameters of the interpolation algorithm in a _punched card_ fashion.
- A simple delimited text file (***.xyz**) contains the measured bathymetric data with a coordinate order [_x_, _y_, _z_].
- The **thaleg.xy** file contains the breakpoints of the river thalweg in a consecutive order [_x_, _y_].
- In case the interpolated DEM is required in a specific set of query poitns [_x_, _y_], the **query.xy** file should also be added to the directory where the script is executed.

The code provides outputs in standard text file and/or in ParaView file format.

**Reference:** Fleit. G (under review) Windowed Anisotropic Local Inverse Distance weighted (WALID) interpolation method for riverbed mapping. Environmental Modelling & Software.

Example:
<img width="1227" alt="Screenshot 2023-10-16 at 14 33 52" src="https://github.com/fleitgabor/WALID/assets/49308041/a5a6b49e-de30-4a4c-b915-d073907334b0">
