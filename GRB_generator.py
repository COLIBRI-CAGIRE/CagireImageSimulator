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

TDIT  = 1.331210   # [s]  / Read frequency of CAGIRE
nuLya = 3.3e15     # [Hz] / Lyman break

nu    = {'H': co.c.value/1.65e-6, 'J': co.c.value/1.25e-6}

d2s   = 24.*3600.

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
        


    def get_lightcurve(self, band, tstart=0.001, ndata=1000):
        mag0   = self.tobsmag[band]
        tobs   = self.tobs[band]
        tbreak = self.tbreak
        
        ts    = tstart + np.arange(ndata)*TDIT/24./3600.
        fact1 = (ts/tobs)**self.alpha['1']
        
        if (True in (ts >= tbreak)):
            fact2 = (ts/(tbreak))**self.alpha['2']
            idx = find_nearest(ts, tbreak)
            mags1 = mag0 - 2.5*np.log10(fact1[:idx])
            mags2 = mags1[-1] - 2.5*np.log10(fact2[idx:])
            mags  = np.concatenate((mags1,mags2))
        else:
            mags = mag0 - np.log10(fact1)
        return ts, mags
    
    def getLightCurve(self, band, z2, ti=0.001, tf=1., ndata=100, aLya=-30.):
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
        
        ts = np.logspace(np.log10(ti), np.log10(tf), ndata)
        
        F0z1 = 1./( (tobs/tb1)**-b1 + (tobs/tb1)**-b2 )
        Fsz1 = 1./( (ts/tb1)**-b1 + (ts/tb1)**-b2 )
        LCz1 = Fsz1/F0z1
        
        F0z2 = 1./( (tobs/tb2)**-b1 + (tobs/tb2)**-b2 )
        Fsz2 = 1./( (ts/tb2)**-b1 + (ts/tb2)**-b2 )
        LCz2 = Fsz2/F0z2*((nuobs/nub1)**-a1 + (nuobs/nub1)**-a2)/((nuobs/nub2)**-a1 + (nuobs/nub2)**-a2)

        D1   = self.lumDist(z1)
        D2   = self.lumDist(z2)

        magz1 = mag0  -2.5*np.log10(LCz1)
        magz2 = magz1 -2.5*np.log10(LCz2/LCz1*(D1*Z1/D2/Z2)**2)

        
        return ts, magz2
    
    def generateCagireLightCurve(self, band, z2, ti=0.001, Texp=60., aLya=-30.):
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
        
        F0z1 = 1./( (tobs/tb1)**-b1 + (tobs/tb1)**-b2 )
        Fsz1 = 1./( (ts/tb1)**-b1 + (ts/tb1)**-b2 )
        LCz1 = Fsz1/F0z1
        
        F0z2 = 1./( (tobs/tb2)**-b1 + (tobs/tb2)**-b2 )
        Fsz2 = 1./( (ts/tb2)**-b1 + (ts/tb2)**-b2 )
        LCz2 = Fsz2/F0z2*((nuobs/nub1)**-a1 + (nuobs/nub1)**-a2)/((nuobs/nub2)**-a1 + (nuobs/nub2)**-a2)

        D1   = self.lumDist(z1)
        D2   = self.lumDist(z2)

        magz1 = mag0  -2.5*np.log10(LCz1)
        magz2 = magz1 -2.5*np.log10(LCz2/LCz1*(D1*Z1/D2/Z2)**2)
        
        print(str(ti))
        outfilename = self.name+'_'+band+'_'+str(round(ti,4))+"T_"+str(nFrames)+"f_z"+str(z2)+".txt"
        outpath     = './GRBs/'+self.name+'/z'+str(z2)+'/'+band+'/'
        if not os.path.exists(outpath):
            os.makedirs(outpath)
            np.savetxt(outpath+outfilename, np.asarray([ts*d2s,magz2,ras,decs]))
        else:
            np.savetxt(outpath+outfilename, np.asarray([ts*d2s,magz2,ras,decs]))
        
        return 0
    
    def export_lightcurve(self, band, tstart=0.001, ndata=1000):
        ts, mags = self.get_lightcurve(band, tstart, ndata)
        ras      = np.zeros(ndata)
        decs     = np.zeros(ndata)
        ts      *= 24.*3600.
        
        ras[0]  = self.ra
        decs[0] = self.dec
        
        np.savetxt('./GRBs/'+self.name+'_'+band+'.txt', np.asarray([ts,mags,ras,decs]))
       
    def lumDist(self, z, H0=72., Om=0.3, Ol=0.7):
        try:
            if z==None:
                z = self.redshift
        except:
            pass
        
        D  = integrator(lumDist2bint, 0, z, args=(Om, Ol))[0]
        
        return  D*1e-3*co.c.value/H0
    
    def specFluxRatio(self, zp, a, b, H0=72., Om=0.3, Ol=0.7):
        LD  = self.lumDist(self.redshift, H0, Om, Ol)
        LDp = self.lumDist(zp, H0, Om, Ol)
        
        return (LD/LDp)**2*( (1.+self.redshift)/(1.+zp) )**( 1. - a + b )
    
def lumDist2bint(z, Om, Ol):
    return 1./np.sqrt(Om*(1+z)**3 + Ol)
