
# <img src="https://ffortin-sci-edu.github.io/pictures/CAGIRE_fake_logo.png" width="20%">  Image Simulator v1.0 

This is the official repository for the COLIBRI/CAGIRE image simulator.




## Generating a ramp

Modify the parameters in "simu_b.py" according to your needs, and run it.

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

### v1.0 (current)

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

Thanks to Alix Nouvel de la Fleche for the base version of the CAGIRE simulator. Thanks to David Corre for the ETC on which is based this project (https://github.com/dcorre/pyETC). Thanks to CAGIRE for capturing the GRBs infrared emission.

This repository is maintained by [Francis Fortin](mailto:francis.fortin@irap.omp.eu) (IRAP, CNES).

<img src="https://ffortin-sci-edu.github.io/pictures/IRAP_logo_midres.png" width="10%">
<img src="https://ffortin-sci-edu.github.io/pictures/CNES_logo_midres.png" width="7%">

