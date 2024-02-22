
# <img src="https://ffortin-sci-edu.github.io/pictures/CAGIRE_fake_logo.png" width="20%">  Image Simulator v1.1 

This is the official repository for the COLIBRI/CAGIRE image simulator.

CAGIRE is a near-infrared imager (J, H) that will be mounted on the COLIBRI telescope (San Pedro Martir, Mexico).
As a part of the ground segment of SVOM, CAGIRE is designed to perform systematic followups of gamma-ray burst afterglows.


A complete description of CAGIRE's specifications and performance is available in [Nouvel de la Fleche et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023ExA...tmp...39N/abstract).


## Generating a ramp

Modify the parameters in `simu_b.py` according to your needs, and run it.

Parameters include the target coordinates, photometric band, exposure time, whether or not to simulate a GRB, and output directory/file names.

The detector of CAGIRE is read every ~1.33s, hence for a 60s exposure the simulator will generate a ramp of 45 frames.


### Ramp with a GRB

A few GRB lightcurves are available in `/GRBs/`. When `addGRB=True` is specified in `simu_b.py`, the coordinates of the GRB will override the input coordinates automatically for the simulation. To produce a realistic image, CAGIRE's field of view is slightly offset from the GRB at a random angular distance.


### Multithreading

Some pieces of the code are parallelized, using a default thread count of 4 (`n_threads=4` in ImSimpyA.py). Use `n_threads=1` to revert back to a single-core execution (less ram usage, longer simulation time), or `n_threads=-1` to use all available threads (more ram usage, faster simulation time). For reference, on an Intel i7 13800H, the single-core generation of a 60s exposure (45 frames) takes 35 seconds. Execution of the code with 4 cores or more takes less than 16 seconds.



## Extra parameters
`/config/CAGIRE_ImSim_config.hjson` contains extra parameters to control which effect to simulate (cosmic rays, dark current, saturation, persistence...).

By default, all effects are taken into account to produce realistic images from CAGIRE.

## Requirements

- astropy
- scipy
- numpy
- joblib
- utils

On Ubuntu : `pip install astropy scipy numpy joblib utils`


## Version history

### v1.1 (current)

- Implemented a GRB lightcurve generator `./GRB_generator.py`:
    - produces a simulated lightcurve on demand if it does not already exist
    - can specify time of observation after trigger
    - can specify any other reshift than the real one
    - if other redshift, corrects for luminosity distance, time dilation and spectral redshift
    - generated lightcurves are stored in `./GRBs/nameofGRB` in subfolders according to redshift and observation bands

- Added a GRB database to feed the generator `./GRBs/GRBs.csv`
    - for now contains a small collection of high-redshift GRBs (z > 6)

- All additions above are implemented so that the user only has to modify `simu_b.py` and set the relevant parameters

Works on Ubuntu 22.04 / Python 3.10.

### v1.0

- Parallelized functions: addObjects, buildFrame, readDetector
- Outputs are now saved as extended fits
- Added keywords in Primary and Images necessary for the pre-proc
- Now using modern WCS in Images headers
- Field of View random offset corrected when generating GRB
- Updated gamma map for non-linearity (threshold at $\gamma \lt -10^{-5}$)

Works on Ubuntu 22.04 / Python 3.10.

### v0.9 (base)

Initial working version of the CAGIRE simulator by Alix Nouvel de la Fleche.

Available at https://github.com/alixdelafleche/simu.

Works on Ubuntu 20.04 / Python 3.8.


## Acknowledgements

Thanks to Alix Nouvel de la Fleche for the base version of the CAGIRE simulator. Thanks to David Corre for the ETC on which is based this project (https://github.com/dcorre/pyETC). Thanks to CAGIRE for (soon) capturing the GRBs infrared emission.

This repository is maintained by [Francis Fortin](mailto:francis.fortin@irap.omp.eu) (IRAP, CNES).

<img src="https://ffortin-sci-edu.github.io/pictures/IRAP_logo_midres.png" width="10%"> <img src="https://ffortin-sci-edu.github.io/pictures/CNES_logo_midres.png" width="7%">

