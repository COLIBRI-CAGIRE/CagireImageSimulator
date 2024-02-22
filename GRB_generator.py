# Libraries
import numpy as np

import astropy.io.fits as fits
import astropy.constants as co

import matplotlib.pyplot as plt
import os

from operator import itemgetter

from scipy import constants as const
from scipy import signal
from scipy.integrate import quad as integrator
from scipy.interpolate import splrep,splev

from lmfit import Model

from scipy import odr

from scipy.integrate import quad as integrator

import sys

plt.ion()
plt.rc('font', **{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)

TDIT  = 1.331210                # [s]  / Read frequency of CAGIRE
nuLyb = co.c.value/91.18e-9     # [Hz] / Lyman break
nuLya = co.c.value/121.57e-9    # [Hz] / Lyman alpha

nu       = {'H': co.c.value/1.63e-6, 'J': co.c.value/1.22e-6} # Central frenquencies of observation bands
photband = {'H':1e6*co.c.value/np.array([1.630-0.5*.307,1.630+0.5*.307]), 'J':1e6*co.c.value/np.array([1.220-0.5*.213,1.220+0.5*.213])} # Lower and upper limits of observation bands
d2s      = 24.*3600.

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

class GRB:
    def __init__(self, name):
        cols = []
        with open('./GRBs/GRBs.csv','r') as infile:
            for line in infile.readlines():
                data = line.rstrip('\n').split(',')
                if data[0] == name:
                    cols = data
        
        if cols == []:
            print('Could not find GRB data. Aborting')
        else:
            print('Found '+cols[0]+' in the database.\n')
            self.name     = cols[0]
            self.ra       = float(cols[10])
            self.dec      = float(cols[11])
            self.t0       = float(cols[1])
            self.tobs     = {'J':float(cols[2]), 'H':float(cols[3])}
            self.tbreak   = float(cols[8])
            self.tobsmag  = {'J':float(cols[4]), 'H':float(cols[5])}
            self.alpha    = {'1':float(cols[6]), '2':float(cols[7])}
            self.index    = float(cols[9])
            self.redshift = float(cols[12])


    def generateGRBLightCurve(self, band, z2, ti=0.001, tf=1., ndata=100, aLya=-30.,photsample=100):
        
        
        mag0   = self.tobsmag[band]
        tobs   = self.tobs[band]/24.
        tbreak = self.tbreak
        a1, a2 = self.index, aLya
        b1, b2 = self.alpha['1'], self.alpha['2']
        nuobs = np.linspace(photband[band][0], photband[band][1], photsample)
        
        z1 = self.redshift
        
        Z1 = 1.+z1
        Z2 = 1.+z2
        Z  = Z1/Z2
        
        nub1 = nuLya/Z1
        nub2 = nuLya/Z2
        
        tb1  = tbreak
        tb2  = tbreak/Z
        
        ts = np.logspace(np.log10(ti), np.log10(tf), ndata)
        
        magz1 = mag0 -2.5*np.log10(smoothBPL(ts,tb1,-b1,-b2)/smoothBPL(tobs,tb1,-b1,-b2)) #GRB lightcurve in the observed band at z=z1 in observer frame
        
        mag0rf = mag0 -2.5*np.log10(4.*np.pi*self.lumDist(z1)**2*Z1) #Absolute magnitude of the reference measurement in the restframe of the GRB
        trf    = tobs/Z1
        nurf   = nuobs*Z1

        magrf  = mag0rf -2.5*np.log10(smoothBPL(ts,tb1/Z1,-b1,-b2)/smoothBPL(trf,tb1/Z1,-b1,-b2)) #Lightcurve of the GRB in its restframe (observed band x (1+z1), tbreak / (1+z1))
        
        tz2    = tobs/Z2
        nuz2   = nuobs*Z2

        mag2rf = mag0rf -2.5*np.log10(smoothBPL(tz2,tb1/Z1,-b1,-b2)/smoothBPL(trf,tb1/Z1,-b1,-b2)*np.mean(smoothBPL(nuobs*Z2,nuLya,-a1,-a2)/smoothBPL(nuobs*Z1,nuLya,-a1,-a2)))  #Magnitude of the reference measurement when artificially shifting the GRB to another z=z2
        
        magz2  = mag2rf +2.5*np.log10(4.*np.pi*self.lumDist(z2)**2*Z2) -2.5*np.log10(smoothBPL(ts,tb1/Z,-b1,-b2)/smoothBPL(tobs,tb1/Z,-b1,-b2)) #GRB lightcurve in the observed band as if seen at z=z2 in observer frame

        return ts, magz1, magz2
    
    def generateGRBLightCurve4CAGIRE(self, band, z2, ti=0.001, Texp=60., aLya=-30.):
        mag0   = self.tobsmag[band]
        tobs   = self.tobs[band]/24.
        tbreak = self.tbreak
        a1, a2 = self.index, aLya
        b1, b2 = self.alpha['1'], self.alpha['2']
        nuobs  = nu[band]
        
        z1 = self.redshift
        
        Z1 = 1.+z1
        Z2 = 1.+z2
        Z  = Z1/Z2
        
        nub1 = nuLya/Z1
        nub2 = nuLya/Z2
        
        tb1  = tbreak
        tb2  = tbreak/Z
        
        nFrames = int(Texp//TDIT)
        
        ras      = np.zeros(nFrames)
        decs     = np.zeros(nFrames)
        ras[0]  = self.ra
        decs[0] = self.dec
        
        ts = (ti*d2s + np.arange(nFrames)*TDIT)/d2s
        
        magz1 = mag0 -2.5*np.log10(smoothBPL(ts,tb1,-b1,-b2)/smoothBPL(tobs,tb1,-b1,-b2)) #GRB lightcurve in the observed band at z=z1 in observer frame
        
        mag0rf = mag0 -2.5*np.log10(4.*np.pi*self.lumDist(z1)**2*Z1) #Absolute magnitude of the reference measurement in the restframe of the GRB
        trf    = tobs/Z1
        nurf   = nuobs*Z1

        magrf  = mag0rf -2.5*np.log10(smoothBPL(ts,tb1/Z1,-b1,-b2)/smoothBPL(trf,tb1/Z1,-b1,-b2)) #Lightcurve of the GRB in its restframe (observed band x (1+z1), tbreak / (1+z1))
        
        tz2    = tobs/Z2
        nuz2   = nuobs*Z2

        mag2rf = mag0rf -2.5*np.log10(smoothBPL(tz2,tb1/Z1,-b1,-b2)/smoothBPL(trf,tb1/Z1,-b1,-b2)*np.mean(smoothBPL(nuobs*Z2,nuLya,-a1,-a2)/smoothBPL(nuobs*Z1,nuLya,-a1,-a2)))  #Magnitude of the reference measurement when artificially shifting the GRB to another z=z2
        
        magz2  = mag2rf +2.5*np.log10(4.*np.pi*self.lumDist(z2)**2*Z2) -2.5*np.log10(smoothBPL(ts,tb1/Z,-b1,-b2)/smoothBPL(tobs,tb1/Z,-b1,-b2)) #GRB lightcurve in the observed band as if seen at z=z2 in observer frame
        path = './GRBs\
/'+self.name+ '\
/z'+str(z2)+ '\
/'+band+    '/'
        os.makedirs(path, exist_ok=True)
        np.savetxt(path+self.name+'_'+band+'_'+str(round(ti,4))+"T_"+str(nFrames)+"f_z"+str(z2)+".txt", np.asarray([ts*d2s,magz2,ras,decs]))
        
        return 0
    
       
    def lumDist(self, z, H0=72., Om=0.3, Ol=0.7):
        try:
            if z==None:
                z = self.redshift
        except:
            pass
        
        D  = integrator(lumDist2bint, 0, z, args=(Om, Ol))[0]
        
        return  D*1e-3*co.c.value/H0
    
    
def lumDist2bint(z, Om, Ol):
    return 1./np.sqrt(Om*(1+z)**3 + Ol)

def smoothBPL(x, xo, a1, a2, delta=0.01):
    return (x/xo)**-a1*(0.5*(1+(x/xo)**(1./delta)))**(delta*(a1-a2))

def bkn_pow(xvals,breaks,alphas):
    try:
        if len(breaks) != len(alphas) - 1:
            raise ValueError("Dimensional mismatch. There should be one more alpha than there are breaks.")
    except TypeError:
        raise TypeError("Breaks and alphas should be array-like.")
    if any(breaks < np.min(xvals)) or any(breaks > np.max(xvals)):
        raise ValueError("One or more break points fall outside given x bounds.")
    
    breakpoints = [np.min(xvals)] + breaks + [np.max(xvals)] # create a list of all the bounding x-values
    chunks = [np.array([x for x in xvals if x >= breakpoints[i] and x <= breakpoints[i+1]]) for i in range(len(breakpoints)-1)]
    
    all_y = []

    for idx,xchunk in enumerate(chunks):
        yvals = xchunk**alphas[idx]
        all_y.append(yvals) # add this piece to the output
    
    for i in range(1,len(all_y)):
        all_y[i] *= np.abs(all_y[i-1][-1]/all_y[i][0]) # scale the beginning of each piece to the end of the last so it is continuous
    
    return(np.array([y for ychunk in all_y for y in ychunk])) # return flattened list

