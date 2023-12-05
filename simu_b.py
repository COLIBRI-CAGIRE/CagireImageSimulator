# -*- coding: utf-8 -*-
import time
import numpy as np
import importlib as imp
import warnings
import os
warnings.filterwarnings('ignore')
import math
import gc

path = imp.util.find_spec('ImSimpyA').origin.rstrip("ImSimpyA.py")

start = time.perf_counter()

########################################################################################################################
# SET SIMULATION PARAMETERS
########################################################################################################################
seeing=1
moonage=7
band='H'                                     # Photometry band ('J' or 'H' )
Texp=1.331210                                # Frame duration (s)                / Fixed parameter
t_rampe =  5 #66.5                          # Ramp duration  (s)
RA_image = 214.477                           # Right Ascension                   / Not used if addGRB=True
DEC_image = -45.411                          # Declination                       / Not used if addGRB=True
nomrampe=  'my_simulation_1'                 # Base name for the output file
nomrampe+='_'+band                           # Add photometric band to base name
nomrampe+='_'+str(int(t_rampe//Texp))+"f"    # Add number of frames to base name

AddGRB = False                               # Simulate a GRB ?

tdeb_obs = 300                               # If GRB : time of observation after burst alert
cheminGRB = path+'GRBs/GRB90423_J.txt'       # Path to GRB lightcurve (coordinates and magnitudes with time)

nomPersistance =  'carte_persistance.fits'   # If persitance, name of the file with saturated pixels from previous acquisition
Treset = 0                                   # Duration between current and previous acquisition (since last reset)


output_dir = 'my_simulation'                 # Output directory
configFile = 'CAGIRE_ImSim_config.hjson'     # Configuration file

########################################################################################################################
########################################################################################################################

# Compute the number of frames in one ramp
Nfin=np.round(t_rampe/Texp)
Nframes=np.arange(0,Nfin,1)


# If GRB is simulated:
if AddGRB==True :
    # Begining of observations
    tdeb_obs=tdeb_obs- Texp

    # Fetching GRB data
    GRB= np.loadtxt(cheminGRB)
    tGRB = GRB[0]
    idebGRB = np.intersect1d(np.argwhere(tGRB>= tdeb_obs),np.argwhere(tGRB< tdeb_obs+Texp ))
    Nobs = [int(idebGRB+ x) for x in Nframes]
    magGRB=GRB[1]
    magGRB=magGRB[Nobs]
    raGRB = GRB[2,0]
    decGRB = GRB[3,0]
    
    dist = 26./2./60.
    while dist >= 26./2./60.:
        # Centering field of view randomly around the GRB
        DRA  = np.random.normal(loc=0.0, scale = 0.1*np.cos(decGRB), size = None)
        DDEC = np.random.normal(loc=0.0, scale = 0.1, size = None)
        dist = np.arccos(np.sin(decGRB)*np.sin(decGRB+DDEC)+np.cos(decGRB)*np.cos(decGRB+DDEC)*np.cos(DRA))
    RA_image= raGRB+DRA
    DEC_image= decGRB+DDEC
    print("Generated GRB located at ",raGRB,decGRB)
    print("GRB located at a radius of ",round(dist*60.,1)," arcmin away from center.\n")



#######################################################################################################################
# Simulating the image
#######################################################################################################################

from ImSimpyA import ImageSimulator_UTR

colibri_IS = ImageSimulator_UTR(configFile=configFile, name_telescope='colibri')

# Read the configfile
colibri_IS.readConfigs()

colibri_IS.config['seeing_zenith'] = seeing
colibri_IS.config['moon_age'] = moonage

colibri_IS.config['SourcesList']['generate']['radius'] = 21.7
colibri_IS.config['SourcesList']['generate']['catalog'] = 'II/246'

# Load existing PSF instead of computing new ones for speeding up the calculations
colibri_IS.config['PSF']['total']['method'] = 'load'

colibri_IS.config['nom'] = nomrampe
colibri_IS.config['nomPersistance'] = nomPersistance
colibri_IS.config['Treset'] = Treset

if AddGRB == True:
    colibri_IS.config["addGRB"] = 'yes'
    colibri_IS.config['SourcesToAdd']['gen']['listeMAG'] = magGRB
    colibri_IS.config['SourcesToAdd']['gen']['RA'] = raGRB
    colibri_IS.config['SourcesToAdd']['gen']['DEC'] = decGRB

else :
    colibri_IS.config["addGRB"] = 'no'

print('Image center localization: ',round(RA_image,5),round(DEC_image,5))
ra = RA_image
dec = DEC_image

colibri_IS.config['SourcesList']['generate']['RA'] = ra
colibri_IS.config['SourcesList']['generate']['DEC'] = dec
colibri_IS.config['RA'] = ra
colibri_IS.config['DEC'] = dec

colibri_IS.config['filter_band'] = band
#colibri_IS.config['PSF']['total']['file'] = 'total_PSF/%s/PSF_total_%s.fits' % (output_dir, band)
colibri_IS.config['PSF']['total']['file'] = 'total_PSF/PSF_total_%s.fits' % (band)
colibri_IS.config['exptime'] = Texp
colibri_IS.config['Nfin'] = Nfin


colibri_IS.config['output'] = output_dir + "/"+ nomrampe + '.fits'

# Run the Image Simulator
colibri_IS.simulate('data')

del colibri_IS
gc.collect()

t1=time.perf_counter()
print('\n\nTotal elapsed time: '+str(round(t1-start,1))+" s\n")

