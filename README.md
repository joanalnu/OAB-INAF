OAB-INAF
---


# Aim of this project

# Structure of the repository

The content of this repository is divided into 3 main pilars: code, data and figures.

## DATA

- `table.csv`: GRB data from Ghirlanda et al. 2008 (including $E_\text{peak}$, $E_\text{iso}$ and $z$).
- `nsns_population_joan.hdf5`: data from BNS simulations from Colombo et al. 2025.
- `*.npy`: $\chi^2$-surface results stored as `NumPy` files by the single files for the joint contraints script.

## CODE

- `constraints_OmOL.py`: Computes the $(\Omega_m, \Omega_\Lambda)$ contraints with GRB data.
- `constraints_OmOL_animation.py`: Creates animation of the effect of extra scatter and DoF on the constraints..
- `constraints_OmOLOk.py`: Constraints for $(\Omega_m, \Omega_\Lambda, \Omega_k)$ hypersurface.
- `constraints_w0wa.py`: Constraints for $(w0, wa)$ surface.
- `constraints_OmOLH0.py`: Constraints for $(\Omega_m, \Omega_\Lambda, H_0)$ hypersurface.
- `Epeak_Eiso.ipynb`: Notebook to explore the $E_\text{peak}-E_\text{iso}$ relation at the beginnign of the project.

- `gw_hubble_diagram.ipynb`: Constructs Hubble diagram ($z$, $d_L$) from BNS merger simulations.
- `SGRB_Epeak_Eiso.ipynb`: Equivalent to `Epeak_Eiso.ipynb` for SGRB data from BNS merger simulations.
- `SGRB_constraints_OMOL.py`: Equivalent to `constraints_OMOL.py` for SGRB data from BNS merger simulations.
- `Epeak_Eiso_SGRB.py`: Exploration of an issue with parameter fitting.
