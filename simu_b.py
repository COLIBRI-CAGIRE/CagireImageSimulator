# -*- coding: utf-8 -*-
import time
import numpy as np
import importlib as imp
import warnings
import os
warnings.filterwarnings('ignore')
import math
import gc
from GRB_generator import *
from astropy import coordinates as coord
from astropy import units as u
from astropy.wcs.wcs import WCS

path = imp.util.find_spec('ImSimpyA').origin.rstrip("ImSimpyA.py")

start = time.perf_counter()

########################################################################################################################
# SET SIMULATION PARAMETERS
########################################################################################################################
seeing=1                                     # Local seeing                         / [arcsec]
moonage=7                                    # Days since new moon                  / [d]
band='J'                                     # Photometry band ('J' or 'H' )

Texp=1.331210                                # Frame duration                        / [s], Fixed parameter
t_rampe = 60                                 # Ramp duration                         / [s]
nFrames = int(t_rampe/Texp)


RA_image = 56.871249                           # Right Ascension                       / [deg], Not used if addGRB=True
DEC_image = 24.1075                         # Declination                           / [deg], Not used if addGRB=True


AddGRB   = True                              # Simulate a GRB ?                      / [bool]
GRBname  = "GRB050904A"                      # Name of the GRB to simulate
z        = 8                                 # Redshift of the GRB. "default" automatically reverts to the real redshift.
tdeb_obs = 0.1279                            # Time of observation after trigger     / [d]

output_dir = 'my_simulation'                 # Output directory
nomrampe   = 'GRB050904A_initial'                      # Base name for the output file.

configFile = 'CAGIRE_ImSim_config.hjson'     # Configuration file

nomPersistance =  'carte_persistance.fits'   # File with saturated pixels from previous acquisition

Treset = 0                                   # Time between current and previous acquisition / [s]

########################################################################################################################

if z == "default":
    with open('./GRBs/GRBs.csv','r') as infile:
        for line in infile.readlines():
            data = line.rstrip('\n').split(',')
            if data[0] == GRBname:
                cols = data
    z=float(cols[12])
                                             # Path to GRB lightcurve (coordinates and magnitudes with time)  
cheminGRB = path + 'GRBs'+'\
/'+GRBname+ '\
/z'+str(z)+ '\
/'+band+    '\
/'+GRBname+'_'+band+'_'+str(tdeb_obs)+'T_'+str(nFrames)+'f_'+'z'+str(z)+'.txt' 



nomrampe  += '_'+band                        # Add photometric band to base name
if AddGRB:                                   # If GRB simulated, add time since trigger
    nomrampe += '_'+str(tdeb_obs)+'T'
nomrampe  += '_'+str(int(t_rampe//Texp))+"f" # Add number of frames to base name
if AddGRB:                                   # If GRB simulated, add redshift
    nomrampe += '_z'+str(z)

if AddGRB:
    print('Observing the GRB '+str(tdeb_obs*24.)+' hrs after trigger.')


########################################################################################################################
########################################################################################################################

# Compute the number of frames in one ramp
Nfin=np.round(t_rampe/Texp)
Nframes=np.arange(0,Nfin,1)


# If GRB is simulated:
if AddGRB==True :
    # Fetching GRB data
    if  os.path.exists(cheminGRB):
        GRB= np.loadtxt(cheminGRB)
        print('Found GRB lightcurve in database.')
    else:
        print('GRB lightcurve not found in database.\nGenerating a lightcurve using input parameters.\n')
        G = GRB(GRBname)
        G.generateGRBLightCurve4CAGIRE(band, z, tdeb_obs, t_rampe)
        GRB= np.loadtxt(cheminGRB)
    tGRB = GRB[0]
    #idebGRB = np.intersect1d(np.argwhere(tGRB>= tdeb_obs),np.argwhere(tGRB< tdeb_obs+Texp ))
    #Nobs = [int(idebGRB+ x) for x in Nframes]
    magGRB=GRB[1]
    #magGRB=magGRB[Nobs]
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
colibri_IS.config['PSF']['total']['file'] = 'total_PSF/PSF_total_%s.fits' % (band)
colibri_IS.config['exptime'] = Texp
colibri_IS.config['Nfin'] = Nfin


colibri_IS.config['output'] = output_dir + "/"+ nomrampe + '.fits'

# Run the Image Simulator
colibri_IS.simulate('data')

c = coord.SkyCoord(colibri_IS.config['SourcesToAdd']['gen']['RA'], colibri_IS.config['SourcesToAdd']['gen']['DEC'], unit=(u.deg, u.deg),frame='icrs')
w = WCS(colibri_IS.hdu_header)
world = np.array([[c.ra.deg, c.dec.deg]])
pix = w.all_world2pix(world, 1)
print("\n GRB located at pixels [",int(pix[0][0]),int(pix[0][1]),"]\n")

del colibri_IS
gc.collect()

t1=time.perf_counter()
print('\n\nTotal elapsed time: '+str(round(t1-start,1))+" s\n")

