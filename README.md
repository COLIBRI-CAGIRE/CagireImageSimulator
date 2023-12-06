# CAGIRE Image Simulator v1.0

This is the official repository for the COLIBRI/CAGIRE image simulator.

## Generating a ramp

Modify the parameters in "simu_b.py" according to your needs and run it. Parameters include the target coordinates, photometric band, exposure time, whether or not to simulate a GRB, and output directory/file names.

### Multithreading

Some pieces of the code are parallelized, using a default thread count of 4 (n_threads=4 in ImSimpyA.py). Use n_threads=1 to revert back to a single-core execution (less ram usage, longer simulation time), or n_threads=-1 to use all available threads (more ram usage, faster simulation time). For reference, on an Intel i7 13800H, the single-core generation of a 60s exposure (45 frames) takes 35 seconds. Execution of the code with 4 cores or more takes less than 16 seconds.



## Extra parameters
`/config/CAGIRE_ImSim_config.hjson` contains extra parameters to control which effect to simulate (cosmic rays, dark current, saturation, persistance...).
By default, all effects are taken into accout to produce realistic images from CAGIRE.



## Acknowledgements

Thanks to Alix Nouvel de la Fleche for the base version of the CAGIRE simulator. Thanks to David Corre for the ETC on which is based this project (https://github.com/dcorre/pyETC). Thanks to CAGIRE for capturing the GRBs infrared emission.
