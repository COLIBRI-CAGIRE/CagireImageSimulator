# -*- coding: utf-8 -*-

# Version of the CAGIRE simulator.
version = "1.0"

import numpy as np
import matplotlib.pyplot as plt
import os, sys, datetime, math, shutil, time, gc
import hjson
import scipy
import numexpr as ne
import imp
import pyETC.utils
from scipy.signal import fftconvolve as convolution
from astropy.io import fits
from collections import OrderedDict
from joblib import Parallel, delayed
from utils import PSFUtils

parameters = {'axes.labelsize': 18, 'axes.titlesize': 20, 'xtick.labelsize': 18, 'ytick.labelsize': 18}
plt.rcParams.update(parameters)


""" 
Number of threads to use in parallel. 
	(n_threads = -1) should use all available threads. Expect around 4GB ram usage. Soft-capped around 8 threads, after that the benefits are almost null.
	(n_threads =  1) should be equivalent to single-core computation. Expect <1GB ram usage.
"""
n_threads = 4

simu_dir = imp.util.find_spec('ImSimpyA').origin.rstrip('ImSimPyA.py')

TDIT = 1.331210

class ImageSimulator_UTR():
    """
    Image Simulator for optical/NIR telescope

    """

    def __init__(self, configFile='default_input.hjson', name_telescope='default', seed=None, debug=False,
                 random=False):
        """
        Class Constructor.

        :configFile : name of the cofig file
        :seed: value used to set the random number generator (default: None --> use current time).
        """
        try:
           path = simu_dir+"ImSimpyA"
        except:
            print('path to pyETC can not be found.')
        
        self.path = path
        self.configfile_nopath = configFile
        self.configfile = simu_dir + "config/" + configFile
        self.name_telescope = name_telescope
        

        if seed != None:
            seed = int(seed)
            np.random.seed(seed=seed)  # fix the seed
            print('Setting the random number generator seed to {}'.format(seed))
        else:
            np.random.seed()
            print('Setting the random number generator seed: current time')
        self.debug = bool(debug)
        self.random = bool(random)

        # load instrument model, these values are also stored in the FITS header
        self.information = OrderedDict()
        
        # update settings with defaults. It will be used if they are missing if the config file
        self.information.update(dict(psfoversampling=1.0,
                                     xsize=2048,
                                     ysize=2048,
                                     FWC=125000,
                                     DC=0.001,
                                     RN=54.0,
                                     bias=500.0,
                                     gain=10.0,
                                     zeropoint=23.5,
                                     Nexp=1,
                                     exptime=565.0,
                                     readouttime=5.,
                                     sky_brightness=19.3,
                                     RA=123.0,
                                     DEC=45.0,
                                     mode='same'))

    def readConfigs(self):
        """
        Reads the config file information using configParser.
        """
        # Load the input file in hjson format into a dictionary
        with open(self.configfile, encoding='utf-8') as f:
            self.config = hjson.load(f)

    def etc(self, config_type):
        """ Execute the Exposure Time Calculator to get some information (zeropoint, grb mag,...) """

        try:
            from pyETC.pyETC import etc
            try:
                path     = imp.util.find_spec('pyETC').origin
                path_etc = path.rstrip('__init__.py')
                print("This is ETC path : "+path_etc)
            except:
                print('path to pyETC can not be found.')
        except ValueError:
            print('Package ETC not found, you have to install it')

        # copy config file to ETC config folder
        shutil.copy(self.configfile, path_etc  + self.configfile_nopath)

        if config_type == 'file':
            etc_info = etc(configFile=self.configfile_nopath, config_type=config_type, name_telescope=self.name_telescope)
        elif config_type == 'data':
            etc_info = etc(configFile=self.config, config_type=config_type, name_telescope=self.name_telescope)
        etc_info.sim()
        # Update the config file
        self.config['camera_type'] = etc_info.information['cameras'][etc_info.information['channel']]['camera_type']
        self.config['sensor'] = etc_info.information['cameras'][etc_info.information['channel']]['sensor']
        self.config['RN'] = etc_info.information['cameras'][etc_info.information['channel']]['RN']
        self.config['DC'] = etc_info.information['cameras'][etc_info.information['channel']]['DC']
        self.config['FWC'] = etc_info.information['cameras'][etc_info.information['channel']]['FWC']
        self.config['gain'] = etc_info.information['cameras'][etc_info.information['channel']]['gain']
        self.config['bits'] = etc_info.information['cameras'][etc_info.information['channel']]['bits']
        self.config['xsize'] = etc_info.information['cameras'][etc_info.information['channel']]['Nphotocell_X']
        self.config['ysize'] = etc_info.information['cameras'][etc_info.information['channel']]['Nphotocell_Y']
        self.config['xPixSize'] = etc_info.information['cameras'][etc_info.information['channel']]['Photocell_SizeX'] * \
                                  etc_info.information['binning_X']
        self.config['yPixSize'] = etc_info.information['cameras'][etc_info.information['channel']]['Photocell_SizeY'] * \
                                  etc_info.information['binning_Y']
        self.config['dig_noise'] = etc_info.information['dig_noise']
        self.config['D_M1'] = etc_info.information['D_M1']
        self.config['D_M2'] = etc_info.information['D_M2']
        self.config['M2_factor'] = etc_info.information['M2_factor']
        self.config['FoV_1axis'] = etc_info.information['FoV_axis1']
        self.config['Focal_length'] = etc_info.information['foc_len']
        self.config['filter_folder'] = etc_info.information['filter_folder']
        self.config['Sky_CountRate'] = etc_info.information['Sky_CountRate']
        # self.config['sky_brightness']=etc_info.information['sky_brightness']
        self.config['SB_eff'] = etc_info.information['SB_eff']
        self.config['zeropoint'] = etc_info.information['zeropoint']
        self.config['eff_wvl'] = etc_info.information['effWavelength']
        self.config['pixelScale_X'] = etc_info.information['pixelScale_X']
        self.config['pixelScale_Y'] = etc_info.information['pixelScale_Y']
        self.config['airmass'] = etc_info.information['airmass']
        self.config['seeing'] = etc_info.information['seeing_los_arcsec']
        self.config['camera'] = etc_info.information['channel']
        self.config['sky_site'] = etc_info.information['sky_site']
        self.config['verbose'] = str(etc_info.information['verbose'])

        if self.config['object_type'] == 'grb_sim':
            self.config['grb_mag'] = etc_info.information['mag']
            self.config['SNR'] = etc_info.information['SNR']

    def processConfigs(self):
        """
        Processes configuration information and save the information to a dictionary self.information.

        """
        # update the information dictionary
        self.information.update(self.config)

        # force gain to be float
        self.information['gain'] = float(self.config['gain'])

        #  If resized image is present, need to change image size
        if 'ImageResized' in self.information:
            self.information['xsize'] = self.information['ImageResized'][0]
            self.information['ysize'] = self.information['ImageResized'][1]

        # name of the output file, include CCDs
        # self.information['output'] = '{%s}'.format(self.infor)

        # booleans to control the flow
        if self.config['shotNoise'].lower() == 'yes':
            self.shotNoise = True
        else:
            self.shotNoise = False
        if self.config['addSources'].lower() == 'yes':
            self.Addsources = True
        else:
            self.Addsources = False
        #if self.config['bleeding'].lower() == 'yes':
        #    self.bleeding = True
        #else:
        #    self.bleeding = False
        if self.config['cosmicRays'].lower() == 'yes':
            self.cosmicRays = True
        else:
            self.cosmicRays = False
        if self.config['cosmetics'].lower() == 'yes':
            self.cosmetics = True
        else:
            self.cosmetics = False

        # Ajout Alix
        if self.config['addGRB'].lower() == 'yes':
            self.AddGRB = True
        else:
            self.AddGRB = False

        # these don't need to be in the config file
        try:
            val = self.config['readoutNoise']
            if val.lower() == 'yes':
                self.readoutNoise = True
            else:
                self.readoutNoise = False
        except:
            self.readoutNoise = False
        try:
            val = self.config['digiNoise']
            if val.lower() == 'yes':
                self.digiNoise = True
            else:
                self.digiNoise = False
        except:
            self.digiNoise = False
        try:
            val = self.config['background']
            if val.lower() == 'yes':
                self.background = True
            else:
                self.background = False
        except:
            self.background = False
        try:
            val = self.config['darkCurrent']
            if val.lower() == 'yes':
                self.darkCurrent = True
            else:
                self.darkCurrent = False
        except:
            self.darkCurrent = False
        try:
            val = self.config['shutterOpen']
            if val.lower() == 'yes':
                self.shutterOpen = True
            else:
                self.shutterOpen = False
        except:
            self.shutterOpen = False
        try:
            val = self.config['nonlinearity']
            if val.lower() == 'yes':
                self.nonlinearity = True
            else:
                self.nonlinearity = False
        except:
            self.nonlinearity = False
        try:
            val = self.config['Vignetting']
            if val.lower() == 'yes':
                self.Vignetting = True
            else:
                self.Vignetting = False
        except:
            self.Vignetting = False
        try:
            val = self.config['ADU']
            if val.lower() == 'yes':
                self.ADU = True
            else:
                self.ADU = False
        except:
            self.ADU = False
        try:
            val = self.config['Offset']
            if val.lower() == 'yes':
                self.Offset = True
            else:
                self.Offset = False
        except:
            self.Offset = False
        try:
            val = self.config['intscale']
            if val.lower() == 'yes':
                self.intscale = True
            else:
                self.intscale = False
        except:
            self.intscale = True

        try:
            val = self.config['saturation']
            if val.lower() == 'yes':
                self.saturation = True
            else:
                self.saturation = False
        except:
            self.saturation = False

        try:
            val = self.config['CrossTalk']
            if val.lower() == 'yes':
                self.CrossTalk = True
            else:
                self.CrossTalk = False
        except:
            self.CrossTalk = False
        try:
            val = self.config['FlatField']
            if val.lower() == 'yes':
                self.FlatField = True
            else:
                self.FlatField = False
        except:
            self.FlatField = False

        try:
            val = self.config['CreatePersistance']
            if val.lower() == 'yes':
                self.CreatePersistance = True
            else:
                self.CreatePersistance = False
        except:
            self.CreatePersistance = False

        try:
            val = self.config['Persistance']
            if val.lower() == 'yes':
                self.Persistance = True
            else:
                self.Persistance = False
        except:
            self.Persistance = False

        self.information['variablePSF'] = False

        self.booleans = dict(shotNoiseoise=self.shotNoise,
                             addsources=self.Addsources,
                             #bleeding=self.bleeding,
                             cosmicRays=self.cosmicRays,
                             cosmetics=self.cosmetics,
                             background=self.background,
                             darkCurrent=self.darkCurrent,
                             readoutNoise=self.readoutNoise,
                             digiNoise=self.digiNoise,
                             nonLinearity=self.nonlinearity,
                             Vignetting=self.Vignetting,
                             ADU=self.ADU,
                             Offset=self.Offset,
                             intscale=self.intscale,
                             shutterOpen=self.shutterOpen,
                             saturation=self.saturation,
                             CrossTalk=self.CrossTalk,
                             FlatField=self.FlatField,
                             addGRB=self.AddGRB,
                             CreatePersistance=self.CreatePersistance,
                             Persistance=self.Persistance)

        if self.debug:
            pprint.pprint(self.information)

    def set_fits_header(self):
        """ Save information to save in FITS file header """

        self.fits_header = OrderedDict()
        self.fits_header['xsize'] = self.information['xsize']
        self.fits_header['ysize'] = self.information['ysize']
        self.fits_header['FWC'] = self.information['FWC']
        self.fits_header['DC'] = round(self.information['DC'], 3)
        self.fits_header['RN'] = self.information['RN']
        self.fits_header['gain'] = self.information['gain']
        self.fits_header['ZP'] = round(self.information['zeropoint'], 3)
        self.fits_header['NEXP'] = self.information['Nexp']
        self.fits_header['EXPTIME'] = self.information['exptime']
        self.fits_header['SB'] = self.information['SB_eff']
        self.fits_header['airmass'] = round(self.information['airmass'], 3)
        self.fits_header['seeing'] = round(self.information['seeing'], 3)
        self.fits_header['SKYSITE'] = self.information['sky_site']
        self.fits_header['camera'] = self.information['camera']
        # self.fits_header['Temp_cam'] = self.information['Temp_cam']
        self.fits_header['bits'] = self.information['bits']
        self.fits_header['camera_type'] = self.information['camera_type']
        self.fits_header['sensor'] = self.information['sensor']
        self.fits_header['binning_X'] = self.information['binning_X']
        self.fits_header['binning_Y'] = self.information['binning_Y']
        self.fits_header['filter'] = self.information['filter_folder'] + '-' + self.information['filter_band']
        self.fits_header['D_M1'] = self.information['D_M1']
        self.fits_header['D_M2'] = self.information['D_M2']
        self.fits_header['M2_factor'] = self.information['M2_factor']

        # add WCS to the header
        self.fits_header['WCSAXES'] = 2
        self.fits_header['CRPIX1'] = self.information['ysize'] / 2.
        self.fits_header['CRPIX2'] = self.information['xsize'] / 2.
        self.fits_header['CRVAL1'] = self.information['RA']
        self.fits_header['CRVAL2'] = self.information['DEC']
        self.fits_header['CTYPE1'] = 'RA---TAN'
        self.fits_header['CTYPE2'] = 'DEC--TAN'
        # north is up, east is left
        self.fits_header['CD1_1'] = -self.config['pixelScale_X'] / 3600. #pix size in arc sec / deg
        self.fits_header['CD1_2'] = 0.0
        self.fits_header['CD2_1'] = 0.0
        self.fits_header['CD2_2'] =  self.config['pixelScale_Y'] / 3600.

        self.fits_header['DATE-OBS'] = datetime.datetime.isoformat(datetime.datetime.now())
        self.fits_header['INSTRUME'] = 'ImSimpyA'

        # create a new FITS file, using HDUList instance
        hdu = fits.PrimaryHDU()

        # new image HDU
        # hdu.data=self.image.astype(np.float32)

        # add input keywords to the header
        for key, value in self.fits_header.items():
            # truncate long keys
            if len(key) > 8:
                key = key[:7]
            try:
                hdu.header.set(key.upper(), value)
            except:
                try:
                    hdu.header.set(key.upper(), str(value))
                except:
                    pass

        # write booleans
        for key, value in self.booleans.items():
            # truncate long keys
            if len(key) > 8:
                key = key[:7]
            hdu.header.set(key.upper(), str(value), 'Boolean Flags')

        hdu.header.add_history('If questions, please contact David Corre (david.corre at lam.fr).')
        hdu.header.add_history('Created by ImSimpyA at %s' % datetime.datetime.isoformat(datetime.datetime.now()))
        hdu.header.add_history('Ramps generated by CAGIRE simulator '+version)
        hdu.verify('fix')

        # Create directory if not existing
        os.makedirs(os.path.dirname(simu_dir + '/output/' + self.information['output']), exist_ok=True)

        self.hdu_header = hdu.header

    def _createEmpty(self):
        """
        Creates and empty array of a given x and y size full of zeros.
        """
        self.image = np.zeros((self.information['ysize'], self.information['xsize']), dtype=np.float64)

    def objectOnDetector(self, object):
        """
        Version initiale de la fonction
        Tests if the object falls on the detector area being simulated.

        :param object: object to be placed to the self.image being simulated.
        :type object: list

        :return: whether the object falls on the detector or not
        :rtype: bool
        """
        ny, nx = self.finemap[object[3]].shape
        xt = object[0]  # position de l'objet
        yt = object[1]

        # the bounding box should be in the nominal scale
        fac = 1. / self.information['psfoversampling']

        # Assess the boundary box of the input image
        xlo = (1 - nx) * 0.5 * fac + xt
        xhi = (nx - 1) * 0.5 * fac + xt
        ylo = (1 - ny) * 0.5 * fac + yt
        yhi = (ny - 1) * 0.5 * fac + yt

        i1 = np.floor(xlo + 0.5)
        i2 = np.ceil(xhi + 0.5) + 1
        j1 = np.floor(ylo + 0.5)
        j2 = np.ceil(yhi + 0.5) + 1

        if i2 < 1 or i1 > self.information['xsize']:
            return False

        if j2 < 1 or j1 > self.information['ysize']:
            return False

        return True

    def overlayToCCD(self, data, obj):
        """
        Overlay data from a source object onto the self.image

        :param data: ndarray of data to be overlaid on to self.image
        :type data: ndarray
        :param obj: object information such as x,y position
        :type obj: list
        """
        
        # object centre x and y coordinates (only in full pixels, fractional has been taken into account already)
        xt = np.floor(obj[0]) - 1  # zero indexing  #position de l'objet
        yt = np.floor(obj[1]) - 1  # zero indexing

        # input array size
        nx = data.shape[1]
        ny = data.shape[0]

        # Assess the boundary box of the input image
        xlo = (1 - nx) * 0.5 + xt
        xhi = (nx - 1) * 0.5 + xt + 1
        ylo = (1 - ny) * 0.5 + yt
        yhi = (ny - 1) * 0.5 + yt + 1

        i1 = int(np.floor(xlo + 0.5))
        if i1 < 1:
            i1 = 0

        i2 = int(np.floor(xhi + 0.5))
        if i2 > self.information['xsize']:
            i2 = self.information['xsize']

        j1 = int(np.floor(ylo + 0.5))
        if j1 < 1:
            j1 = 0

        j2 = int(np.floor(yhi + 0.5))
        if j2 > self.information['ysize']:
            j2 = self.information['ysize']

        if i1 > i2 or j1 > j2:
            # print ('Object does not fall on the detector...')
            return

        ni = i2 - i1
        nj = j2 - j1

        # add to the image
        if ni == nx and nj == ny:
            # full frame will fit
            self.image[j1:j2, i1:i2] += data
        elif ni < nx and nj == ny:
            # x dimensions shorter
            if int(np.floor(xlo + 0.5)) < 1:
                # small values, left side
                self.image[j1:j2, i1:i2] += data[:, nx - ni:]
            else:
                # large values, right side
                self.image[j1:j2, i1:i2] += data[:, :ni]
        elif nj < ny and ni == nx:
            # y dimensions shorter
            if int(np.floor(ylo + 0.5)) < 1:
                # small values, bottom
                self.image[j1:j2, i1:i2] += data[ny - nj:, :]
            else:
                # large values, top
                self.image[j1:j2, i1:i2] += data[:nj, :]
        else:
            # both lengths smaller, can be in any of the four corners
            if int(np.floor(xlo + 0.5)) < 1 > int(np.floor(ylo + 0.5)):
                # left lower
                self.image[j1:j2, i1:i2] += data[ny - nj:, nx - ni:]
            elif int(np.floor(xlo + 0.5)) < 1 and int(np.floor(yhi + 0.5)) > self.information['ysize']:
                # left upper
                self.image[j1:j2, i1:i2] += data[:nj, nx - ni:]
            elif int(np.floor(xhi + 0.5)) > self.information['xsize'] and int(np.floor(ylo + 0.5)) < 1:
                # right lower
                self.image[j1:j2, i1:i2] += data[ny - nj:, :ni]
            else:
                # right upper
                self.image[j1:j2, i1:i2] += data[:nj, :ni]

        # ne pas ajouter des étoiles sur pix ref
        ref =fits.getdata(self.path +"/"+ self.information['PixRefFile'])
        im = np.ravel(self.image)
        im[ref] = 0
        
        self.image = np.reshape(im, [2048, 2048])
        
    def writeFITSfile(self, data, filename, unsigned16bit=False):
        """
        Writes out a simple FITS file.

        :param data: data to be written
        :type data: ndarray
        :param filename: name of the output file
        :type filename: str
        :param unsigned16bit: whether to scale the data using bzero=32768
        :type unsigned16bit: bool

        :return: None
        """
        if os.path.isfile(filename):
            os.remove(filename)

        # create a new FITS file, using HDUList instance
        # hdulist = fits.HDUList(fits.PrimaryHDU())
        hdu = fits.PrimaryHDU()

        # new image HDU
        # hdu = fits.ImageHDU(data=data)
        hdu.data = data

        # convert to unsigned 16bit int if requested
        if unsigned16bit:
            hdu.scale('int16', '', bzero=32768)
            hdu.header.add_history('Scaled to unsigned 16bit integer!')

        # add input keywords to the header
        for key, value in self.fits_header.items():
            # truncate long keys
            if len(key) > 8:
                key = key[:7]
            try:
                hdu.header.set(key.upper(), value)
            except:
                try:
                    hdu.header.set(key.upper(), str(value))
                except:
                    pass

        # write booleans
        for key, value in self.booleans.items():
            # truncate long keys
            if len(key) > 8:
                key = key[:7]
            hdu.header.set(key.upper(), str(value), 'Boolean Flags')

        # update and verify the header
        hdu.header.add_history('This is an intermediate data product no the final output!')
        hdu.header.add_history('Created by ImSimpyA at %s' % (datetime.datetime.isoformat(datetime.datetime.now())))
        hdu.verify('fix')

        hdu.writeto(filename, overwrite=True)

    def configure(self, config_type):
        """
        Configures the simulator with input information and creates and empty array to which the final image will
        be build on.
        """
        if config_type == 'file': self.readConfigs()
        self.etc(config_type)
        self.processConfigs()
        self._createEmpty()
        self.set_fits_header()

    def generateObjectList(self):
        """ Generate object to simulate """

        if 'generate' in self.config['SourcesList']:
            from astropy.io import fits

            if 'output' in self.information['SourcesList']['generate']:
                output = self.path + '/catalog/' + self.information['SourcesList']['generate']['output']
            else:
                output = self.path + '/catalog/SourcesCatalog.txt'
            if 'frame' in self.information['SourcesList']['generate']:
                frame = self.information['SourcesList']['generate']['frame']
            else:
                frame = 'icrs'
            if 'band' in self.information['SourcesList']['generate']:
                band = self.information['SourcesList']['generate']['band']
            else:
                band = self.information['filter_band']

            RA = self.information['SourcesList']['generate']['RA']
            DEC = self.information['SourcesList']['generate']['DEC']
            radius = self.information['SourcesList']['generate']['radius']
            # _header=fits.open(self.path+'/images/'+self.information['output'])
            # header=_header['PRIMARY'].header

            if self.information['SourcesList']['generate']['catalog'] == 'Panstarrs':
                print('Downloading objects from Panstarrs catalog')
                from utils.createCatalogue import PanstarrsCatalog
                PanstarrsCatalog(RA, DEC, radius, band, self.config['eff_wvl'], self.hdu_header, frame=frame,
                                 output=output)
            else:
                from utils.createCatalogue import Viziercatalog
                print('Downloading objects from Vizier')
                # print(RA, DEC, radius, band, self.config['eff_wvl'], self.hdu_header, self.information['SourcesList']['generate']['catalog'], frame, output)
                Viziercatalog(RA, DEC, radius, band, self.config['eff_wvl'], self.hdu_header, catalog=self.information['SourcesList']['generate']['catalog'], frame=frame, output=output)
            self.objects = np.loadtxt(output)


        elif "file" in self.config['SourcesList']:
            self.objects = np.loadtxt(self.path + '/catalog/' + self.information['SourcesList']['file'])

    def AddObjectToList(self,k):
        """Alix :  Créer un catalogue ne contenant que le GRB à simuler  """

        if 'gen' in self.config['SourcesToAdd']:
            from astropy.io import fits

            if 'output' in self.information['SourcesToAdd']['gen']:
                output = self.path + '/catalog/' + self.information['SourcesToAdd']['gen']['output']
            else:
                output = self.path + '/catalog/SourcesToAdd.txt'
            
            RA = self.information['SourcesToAdd']['gen']['RA']
            DEC = self.information['SourcesToAdd']['gen']['DEC']
            MAG = self.information['SourcesToAdd']['gen']['listeMAG'][k-1]
            radius = self.information['SourcesList']['generate']['radius']

            from utils.createCatalogue import CreateObject
            CreateObject(RA, DEC, MAG, 2, self.hdu_header, radius, output=output)
            #print(MAG)
            object = np.loadtxt(output)
            object = object[np.newaxis, :]

            self.objects = object

    def readObjectlist(self):
        """
        Reads object list using numpy.loadtxt, determines the number of object types,
        and finds the file that corresponds to a given object type.

        The input catalog is assumed to contain the following columns:

            #. x coordinate
            #. y coordinate
            #. apparent magnitude of the object
            #. type of the object [0=star, number=type defined in the objects.dat]
            #. rotation [0 for stars, [0, 360] for galaxies]

        This method also displaces the object coordinates based on the
        CCD to be simulated.

        .. Note:: If even a single object type does not have a corresponding input then this method
                  forces the program to exit.
        """
        # self.objects = np.loadtxt(self.path+self.information['SourcesList'])
        # Add GRB on the object list
        if self.config['object_type'] == 'grb_sim':
            # If coordinates given in pixels
            if self.config['grb_coord_type'] == 'pixels':
                grb_coord_pix = self.config['grb_coords']
            elif self.config['grb_coord_type'] == 'RADEC':
                from astropy.io import fits
                import astropy.units as u
                import astropy.coordinates as coord
                from astropy.wcs import WCS

                c = coord.SkyCoord(self.config['grb_coords'][0], self.config['grb_coords'][1], unit=(u.deg, u.deg),
                                   frame='icrs')
                # _header=fits.open(self.path + '/images/' +self.information['output'])
                # header=_header['PRIMARY'].header

                w = WCS(self.hdu_header)
                world = np.array([[c.ra.deg, c.dec.deg]])
                # print (world)
                pix = w.all_world2pix(world, 1)
                # print (pix)
                # first transform WCS into pixels
                grb_coord_pix = pix[0]
            self.config['grb_coords_pix_X'] = grb_coord_pix[1]
            self.config['grb_coords_pix_Y'] = grb_coord_pix[0]
            self.objects = np.vstack(
                (self.objects, [grb_coord_pix[0], grb_coord_pix[1], self.config['grb_mag'], 1000, 0]))
            # Add GRB to the object list as a point source
            txt = 'GRB positionned at pixel coordinates (X,Y): {0:.2f},{1:.2f} with mag= {2:.2f}'.format(
                grb_coord_pix[0], grb_coord_pix[1], round(self.config['grb_mag'], 2))
            print(txt)

        # if only a single object in the input, must force it to 2D
        try:
            tmp_ = self.objects.shape[1]
        except:
            self.objects = self.objects[np.newaxis, :]

        # read in object types
        data = open(self.path + '/objects.dat').readlines()

        # only 2D array will have second dimension, so this will trigger the exception if only one input source
        tmp_ = self.objects.shape[1]
        # find all object types
        self.sp = np.asarray(np.unique(self.objects[:, 3]), dtype=int)
        # generate mapping between object type and data
        objectMapping = {}
        for stype in self.sp:

            if stype == 0 or stype == 1000 or stype == 2:
                # delta function
                objectMapping[stype] = 'PSF'
            else:
                for line in data:
                    tmp = line.split()
                    if int(tmp[0]) == stype:
                        # found match
                        if tmp[2].endswith('.fits'):
                            d, header = fits.getdata(self.path + '/' + tmp[2], header=True)
                            # print (type(d),d.shape,d,np.max(d))
                            if 'PIXSIZE1' in header:
                                ratio = float(header['PIXSIZE1'] / (self.information['xPixSize'] * 1e6))
                                print('ratio finemaps: %.2f' % ratio)
                                d2 = scipy.ndimage.zoom(d / d.sum(), ratio, order=3)
                                d2[d2 < 1e-6] = 0
                                if np.sum(d2) != 1: d2 = d2 / np.sum(d2)
                                image_size = d2.shape
                                # Assume same size horizontally and vertically
                                if image_size[0] % 2 == 0:
                                    width = int(image_size[0] / ratio / 4)
                                else:
                                    width = int((image_size[0] / ratio - 1) / 4)

                                # Assume same size horizontally and vertically
                                if np.ceil(image_size[0]) % 2 == 0:
                                    center = int((np.ceil(image_size[0])) / 2)
                                else:
                                    center = int((np.ceil(image_size[0]) - 1) / 2)
                                center = [center, center]
                                # print (ratio)
                                # print (center)
                                # print (width)
                                # print (center[0]-width,center[0]+width,center[1]-width,center[1]+width)
                                d3 = d2[center[0] - width:center[0] + width, center[1] - width:center[1] + width]

                                # print (type(d3),d3.shape,d3,np.max(d3))
                            else:
                                print('No pixel size found in header. Assume the same as the current telescope.')

                                d3 = d
                        else:
                            pass
                        objectMapping[stype] = dict(file=tmp[2], data=d3)
                        break

        self.objectMapping = objectMapping

    def generatePSF(self):
        """ Compute PSF if needed """

        PSF = dict()
        # Load atmosphere and instrument PSF
        if self.information['PSF']['total']['method'] == 'compute':
            for keyword in self.information['PSF']:
                if keyword != "total":
                    if "file" not in self.information['PSF'][keyword]:
                        if self.information['PSF'][keyword]['type'] == 'moffat':
                            if 'beta' in self.information['PSF'][keyword]:
                                beta = self.information['PSF'][keyword]['beta']
                            else:
                                beta = 2
                        else:
                            beta = 2
                        if 'seeing' in self.information['PSF'][keyword]:
                            seeing = self.information['PSF'][keyword]['seeing']
                        else:
                            seeing = self.config['seeing']

                        #  If PSF size bigger than image --> Limit PSF size to image size
                        if self.information['PSF'][keyword]['size'][0] > self.information['xsize']:
                            self.information['PSF'][keyword]['size'][0] = self.information['xsize']
                            print(
                                'PSF size along x axis bigger than image size!\nPSF size limited to image size along x axis now: %d Pixels' % (
                                self.information['xsize']))

                        if self.information['PSF'][keyword]['size'][1] > self.information['ysize']:
                            self.information['PSF'][keyword]['size'][1] = self.information['ysize']
                            print(
                                'PSF size along y axis bigger than image size!\nPSF size limited to image size along y axis now: %d Pixels' % (
                                self.information['ysize']))

                        PSFUtils.createPSF(
                            filename=self.path + '/psf/' + self.information['PSF'][keyword]['output'],
                            PSF_type=self.information['PSF'][keyword]['type'],
                            imsize=self.information['PSF'][keyword]['size'],
                            pixel_size=[self.config['xPixSize'], self.config['yPixSize']],
                            pixel_scale=self.config['pixelScale_X'], eff_wvl=self.config['eff_wvl'], seeing=seeing,
                            DM1=self.config['D_M1'], DM2=self.config['D_M2'], focal_length=self.config['Focal_length'],
                            oversamp=self.config['psfoversampling'], beta=beta, disp=False, unsigned16bit=False)

                        PSF[keyword] = self.path + '/psf/' + self.information['PSF'][keyword]['output']

                    else:
                        # Check pixel size and oversample if needed
                        hdr_ = fits.getheader(
                            self.path + '/psf/' + self.information['PSF'][keyword]['file'] + '.fits')
                        try:
                            if hdr_['XPIXELSZ'] != self.information['cameras'][self.information['channel']][
                                'Photocell_SizeX'] / oversamp or hdr_['YPIXELSZ'] != \
                                self.information['cameras'][self.information['channel']]['Photocell_SizeY'] / oversamp:
                                resampling = [self.information['cameras'][self.information['channel']][
                                                  'Photocell_SizeX'] / oversamp,
                                              self.information['cameras'][self.information['channel']][
                                                  'Photocell_SizeY'] / oversamp]

                                PSFUtils.resize(
                                    filename1=self.path + '/psf/' + self.information['PSF']['keyword']['file'],
                                    filename2=self.path +"/"+self.information['PSF']['keyword']['file'] + '_oversammpled',
                                    type='factor', resampling=resampling, overwrite=True, unsigned16bit=False)

                                PSF[keyword] = self.path + '/psf/' + self.information['PSF'][keyword][
                                    'file'] + '_oversampled'
                            else:
                                PSF[keyword] = self.path + '/psf/' + self.information['PSF'][keyword]['file']
                        except:
                            PSF[keyword] = self.path + '/psf/' + self.information['PSF'][keyword]['file']
            print('PSF convolution')
            # convolve atmosphere and instrument PSF to get the total PSF
            PSFUtils.convolvePSF(filename1=PSF['atmosphere'], filename2=PSF['instrument'],
                                 filename3=self.path + '/psf/' + self.information['PSF']['total']['file'])
            # PSFUtils.convolvePSF(filename1=PSF['instrument'],filename2=PSF['atmosphere'],filename3=self.path+self.information['PSF']['total']['output']+'_oversampled')
            # PSFUtils.resize(filename1=self.path+self.information['PSF']['total']['output']+'_oversampled',filename2=self.path+self.information['PSF']['total']['output'],resampling=32/self.information['psfoversampling'],type='sum')
            # PSFUtils.resize(filename1=self.path+self.information['PSF']['total']['output']+'_oversampled',filename2=self.path+self.information['PSF']['total']['output'],resampling=self.information['psfoversampling']/32,type='zoom')
            print('done')

    def readPSFs(self):
        """
        Reads in a PSF from a FITS file.

        .. Note:: at the moment this method supports only a single PSF file.
        """
        if self.information['variablePSF']:
            # grid of PSFs
            print('Spatially variable PSF not implemented -- exiting')
            sys.exit(-9)
        else:
            # single PSF
            self.PSF = fits.getdata(self.path + '/psf/' + self.information['PSF']['total']['file']).astype(
                np.float64)
            # Normalise if needed
            if np.sum(self.PSF) != 1: self.PSF /= np.sum(self.PSF)
            self.PSFx = self.PSF.shape[1]
            self.PSFy = self.PSF.shape[0]

    def generateFinemaps(self):
        """
        Generates finely sampled images of the input data.
        """
        self.finemap = {}
        self.shapex = {}
        self.shapey = {}
        for k, stype in enumerate(self.sp):
            if stype == 0 or stype == 1000 or stype == 2:
                data = self.PSF.copy().astype(np.float64)
                data /= np.sum(data)
                self.finemap[stype] = data
                self.shapex[stype] = 0
                self.shapey[stype] = 0
            else:

                #  Rescaled to pixel size

                if self.information['psfoversampling'] > 1.0:
                    data = scipy.ndimage.zoom(self.objectMapping[stype]['data'],
                                              self.information['psfoversampling'],
                                              order=0)
                else:
                    data = self.objectMapping[stype]['data']

                data[data < 0.] = 0.0
                if data.sum() != 1: data /= np.sum(data)
                self.finemap[stype] = data

    def generateObject(self,param):
        obj, mag2elec, k = param
        stype = obj[3]
        visible = 0
        if self.objectOnDetector(obj):
            visible += 1
            if stype == 0 or stype == 1000:
                # point source, apply PSF
                                
                data = self.finemap[stype].copy()
                
                # map the data to new grid aligned with the centre of the object and scale
                yind, xind = np.indices(data.shape)
                if self.information['psfoversampling'] != 1.0:
                    yi = yind.astype(float) + self.information['psfoversampling'] * (obj[0] % 1)
                    xi = xind.astype(float) + self.information['psfoversampling'] * (obj[1] % 1)
                else:
                    yi = yind.astype(float) + (obj[0] % 1)
                    xi = xind.astype(float) + (obj[1] % 1)

                data = scipy.ndimage.map_coordinates(data, [yi, xi], order=1, mode='nearest')
                
                if self.information['psfoversampling'] != 1.0:
                    data = scipy.ndimage.zoom(data, 1. / self.information['psfoversampling'], order=1)
                # suppress negative numbers, renormalise and scale with the intscale
                
                data[data < 0.0] = 0.0
                sum = np.sum(data)
                sca = mag2elec / sum

                data = ne.evaluate("data * sca")

                return [data,obj]
                
            elif stype == 2:  # ajout Alix objet variable GRB
                if 'gen' in self.config['SourcesToAdd']:
                    Nframe = self.config['Nframe']
                    objectVariable = self.config['SourcesToAdd']['gen']['listeMAG']
                    mag = objectVariable[k-1]
                    # print('GRB mag [',k,'] =' ,round(mag,2))
                    magelec = 10.0 ** (-0.4 * (mag - self.information['zeropoint'])) * 1.33


                mag2elec = magelec
                #print('mag2elec variable', magelec)
                data = self.finemap[stype].copy()
                # print (data.shape)
                # map the data to new grid aligned with the centre of the object and scale
                yind, xind = np.indices(data.shape)
                if self.information['psfoversampling'] != 1.0:
                    yi = yind.astype(float) + self.information['psfoversampling'] * (obj[0] % 1)
                    xi = xind.astype(float) + self.information['psfoversampling'] * (obj[1] % 1)
                else:
                    yi = yind.astype(float) + (obj[0] % 1)
                    xi = xind.astype(float) + (obj[1] % 1)
                data = scipy.ndimage.map_coordinates(data, [yi, xi], order=1, mode='nearest')
                if self.information['psfoversampling'] != 1.0:
                    data = scipy.ndimage.zoom(data, 1. / self.information['psfoversampling'], order=1)
                # suppress negative numbers, renormalise and scale with the intscale
                data[data < 0.0] = 0.0
                sum = np.sum(data)
                sca = mag2elec / sum

                # sca = intscales[j] / sum
                # data = ne.evaluate("data * sca")
                # sca = mag2elec[j]

                # numexpr apparently faster than numpy for big arrays
                data = ne.evaluate("data * sca")

                # print ('Obj coord: %.2f %.2f  / mag: %.2f   / finemap: min: %f  max: %f  mean: %f' % (obj[0],obj[1],obj[2],np.min(data),np.max(data),np.mean(data)))
                # data[data < 0.0] = 0.0

                # overlay the scaled PSF on the image
                #self.overlayToCCD(data, obj)
                return [data,obj]


            else:
                # extended source, rename finemap
                data = self.finemap[stype].copy()
                # map the data to new grid aligned with the centre of the object
                yind, xind = np.indices(data.shape)
                if self.information['psfoversampling'] != 1.0:
                    yi = yind.astype(float) + self.information['psfoversampling'] * (obj[0] % 1)
                    xi = xind.astype(float) + self.information['psfoversampling'] * (obj[1] % 1)
                else:
                    yi = yind.astype(float) + (obj[0] % 1)
                    xi = xind.astype(float) + (obj[1] % 1)

                # yi = yind.astype(np.float) + (obj[0] % 1)
                # xi = xind.astype(np.float) + (obj[1] % 1)
                data = scipy.ndimage.map_coordinates(data, [yi, xi], order=1, mode='nearest')
                conv = convolution(data, self.PSF, self.information['mode'])
                # suppress negative numbers
                conv[conv < 0.0] = 0.0

                # renormalise and scale to the right magnitude
                sum = np.sum(conv)
                # sca = intscales[j] / sum
                sca = mag2elec / sum
                conv = ne.evaluate("conv * sca")

                # tiny galaxies sometimes end up with completely zero array
                # checking this costs time, so perhaps this could be removed
                #if np.isnan(np.sum(conv)):
                #    continue

                # overlay the convolved image on the image
                #self.overlayToCCD(conv, obj)
                return[data,obj]
        else:
            # not on the screen
            # print ('Object %i is outside the detector area' % (j + 1))
            pass
    
    
    
    def addObjects(self,A,k):
        """
        Add objects from the object list to the CCD image (self.image).

        Scale the object's brightness in electrons and size using the input catalog magnitude.

        """
        # total number of objects in the input catalogue and counter for visible objects
        n_objects = self.objects.shape[0]
        visible = 0

        #print('Total number of objects in the input catalog = %i' % n_objects)
        
        # calculate the scaling factors from the magnitudes
        # intscales = 10.0**(-0.4 * self.objects[:, 2])*self.information['magzero']) * self.information['exptime']
        # calculate the number of electrons from the magnitudes
        mag2elec = 10.0 ** (-0.4 * (self.objects[:, 2] - self.information['zeropoint'])) * self.information['exptime']

        intscales = mag2elec
        if ~self.random:
            # Using a fixed size-magnitude relation (equation B1 from Miller et al. 2012 (1210.8201v1).

            # testin mode will bypass the small random scaling in the size-mag relation
            # loop over exposures
            t1 = time.perf_counter()
            for i in range(self.information['Nexp']):
                # loop over the number of objects
                
                
                """
                    Below is an attempt to parallelize the generating of objects on the detector. Decreases runtime of object generation from ~13s to ~10s (n_jobs=4) for ~2400 objects.
                """
                
                params = []
                for j in range(len(self.objects)):
                    params.append((self.objects[j],mag2elec[j],k))
                data_objects = Parallel(n_jobs=n_threads)(delayed(self.generateObject)(param) for param in params)

                for res in data_objects:
                    self.overlayToCCD(res[0],res[1])
                del(params)
                
                """
                    Below is working cleaner code, but not passing info to visible.
                """
                #for j in range(len(self.objects)):
                    #self.generateObject(self.objects[j], visible,mag2elec[j])
                
            #print("Add Objects took :"+str(round(time.perf_counter()-t1,1))+" s.")
            #print('normal', self.information['exptime'], self.objects[13, 2], mag2elec[13])
            #print('ZERO POINT', self.information['zeropoint'])
            #print('%i/%i objects were placed on the detector' % (visible, n_objects))



    def addCosmetics(self,A):
        """ Add the cosmetics effects """
        deadPixs = fits.getdata(self.path + '/Cosmetics/' + self.information['DeadPixFile'])
        # HotPixs=fits.getdata(self.path+'/Cosmetics/'+self.information['HotPixFile'])

        A *= deadPixs

        return(A)

    def applyCreatePersistance(self,A,k):
        """ Creation de la carte de persistance à enregistrer pour l'exposition suivante en mV"""
        Nfin = int(np.round(self.information['Nfin']))

        actif = fits.getdata(self.path +"/"+self.information['PixActifFile'])


        if k == Nfin-1:
            nom = self.information['nom'] + '_persistance.fits'
            path = simu_dir+'/ImSimpyA/Persistance/'
            satu = np.ravel(fits.getdata(self.path + '/' + self.information['SaturationFile']))

            indPers = np.intersect1d(np.argwhere(np.ndarray.flatten(A) > satu), actif)
            #print('indpers',len(indPers))
            carte = np.zeros(2048*2048)
            carte[indPers]=1

            primary = fits.PrimaryHDU()
            image_hdu = fits.ImageHDU(carte)
            hdul = fits.HDUList([primary, image_hdu])
            hdul.writeto(path + nom, overwrite=True)


    def applyPersistance(self,k,A):
        """ Application de la carte de persistance à appliquer (exposition précédente) déjà en electrons"""


        path = simu_dir+'/ImSimpyA/Persistance/'
        nom = self.information['nomPersistance']
        Treset = self.information['Treset']  # 5*60  # temps depuis le premier reset
        
        
        PixPeristants = fits.open(path + nom)
        PixPeristants = PixPeristants[1].data


        indPers = np.argwhere(np.ravel(PixPeristants) > 0)

        Texp = self.information['exptime']*k + Treset
        
        """ Just replaced fits.open with .getdata. (FF)"""
        conv = fits.getdata(self.path + '/' + self.information['PersistanceConv'])
        # Replace 0's with 1's
        conv += 1
        
        
        amp = fits.open(self.path + '/' + self.information['PersistanceAmp'])
        amp = amp[0].data
        amp_1 = conv * amp[0]
        amp_2 = conv * amp[1]
        amp_3 = conv * amp[2]
        
        tau = fits.open(self.path + '/' + self.information['PersistanceTau'])
        tau = tau[0].data
        tau_1 = conv * tau[0]
        tau_2 = conv * tau[1]
        tau_3 = conv * tau[2]
        
        
        switch = np.min([k-1,1])
        Tprec = (Texp - 1.33)*switch + Treset*(1-switch)
        p0 = amp_1 * (1 - np.exp(-Texp / tau_1)) + amp_2 * (1 - np.exp(-Texp / tau_2)) + amp_3 * ( 1 - np.exp(-Texp / tau_3))
        p1 = (amp_1 * (1 - np.exp((-Tprec ) / tau_1)) + amp_2 * ( 1 - np.exp((-Tprec) / tau_2)) + amp_3 * (1 - np.exp((-Tprec) / tau_3)))
        
        dp = p0-p1
        
        persistance = np.ravel(dp)
        persistance[np.isnan(persistance)] = 0

                
        map = np.zeros(2048 * 2048)
        map[indPers] = 1
        pers = persistance * map
        pers[np.isnan(pers)] = 0
        persistance = np.reshape(pers, [2048, 2048])


        A = A + persistance
        
        
        return(A)



    def addCosmicRays(self,k,A):

        """ Add cosmic rays """
        Texp =  k* 1.33  # temps de l'impact
        Nframe = k
        cosmic = fits.open(self.path + '/Cosmics/' + self.information['CosmicsFile'])
        pos = (cosmic[1].data).astype(int)
        energie = cosmic[2].data
        temps = (cosmic[3].data).astype(int)
        nbCos = len(pos)

        imageCos = np.zeros(2048 * 2048)

        
        for i in range(nbCos):
            if Texp >= temps[i] and Texp < temps[i]+1.33:
                # if Texp*Nframe <= temps[i] and Texp*(Nframe+1)>= temps[i] :
                imageCos[pos[i]] = energie[i] * 10


        A = A+ np.reshape(imageCos, [2048, 2048])

        return(A)

    def applyVignetting(self):
        """ Add vignetting  """
        vignetting = fits.getdata(self.path + '/Vignetting/' + self.information['VignettingFile'])
        self.image *= vignetting

    def applyFlatField(self):
        """ Add FlatField  """

        # FlatField calculé sur le det ALFA
        FlatField = fits.getdata(self.path + '/' + self.information['FlatFieldFile'])


        ref = fits.getdata(self.path +"/"+self.information['PixRefFile'])
        FlatField = np.ndarray.flatten(FlatField)  # sinon pb lors de la correction par les pixels de ref, leur non linéarité ne peux pas être calibrée par illumination : pas sensible light
        FlatField[ref] = 1
        FlatField[np.argwhere(FlatField <= 0)] = 1
        FlatField = np.reshape(FlatField, [2048, 2048])

        self.image *= FlatField

        

    def applyNonLinearity(self,k,A):
        """ Add non linearity  """

        NL = fits.getdata(self.path + '/NonLinearity/' + self.information['NonLinearityFile'])


        ref = fits.getdata(self.path +"/"+self.information['PixRefFile'])
        NL = np.ravel(NL)  # sinon pb lors de la correction par les pixels de ref, leur non linéarité ne peux pas être calibrée par illumination : pas sensible light
        NL[ref] = 0

        offset = np.ravel(fits.getdata(self.path + '/Offset/' + self.information['OffsetFile']))
        satu = np.ravel(fits.getdata(self.path + '/' + self.information['SaturationFile']))
        sat = (satu - offset) * 10  # pour mettre en electrons
        im = np.ndarray.flatten(A)
        a = np.argwhere(im*k > sat)
        NL[a] = 0

        #av = np.copy(A)
        #print(np.median(NL))
        #print(np.median(A))
        NL = np.reshape(NL, [2048, 2048])
        A = A + A*A*NL*k      # on met un + car NL déjà négatif # alpha doit être calulé sur des cartes non corrigées des pix ref


        return(A)
        

    def applyDarkCurrent(self):
        """
        Apply dark current. Scales the dark with the exposure time.

        """
        #self.image = 2255 * np.ones([2048, 2048]) * self.information['exptime']  # test d'une illumination constante.

        DC = np.reshape(fits.getdata(self.path + '/DarkCurrent/' + self.information['DarkFile']), [2048, 2048])


        ref = fits.getdata(self.path +"/"+self.information['PixRefFile'])
        dc = np.ndarray.flatten(DC)
        dc[ref] = np.median(dc)

        av = np.copy(self.image)

        # Alix : on multiplie le  DC du pixel  ( en e-/s) par le temps d'exposition : prend en compte DC et pixels chauds.
        Texp = self.information['exptime']
        DC = np.reshape(dc,[2048,2048])
        self.image += DC * Texp


    def applySkyBackground(self):
        """
        Apply dark the sky background. Scales the background with the exposure time.

        Additionally saves the image without noise to a FITS file.
        """
        av = np.copy(self.image)


        sky_back_el = self.information['Sky_CountRate']

        bcgr = self.information['exptime'] * sky_back_el


        # sky_image = np.random.poisson(bcgr,size=self.image.shape).astype(np.float64)
        sky_image = np.ones([2048, 2048]) * bcgr  # on considère le fond de ciel constant sur le détecteur : ligne Alix

        # ne pas ajouter le bruit de font de ciel aux pix de reférence

        ref = fits.getdata(self.path +"/"+ self.information['PixRefFile'])
        sky_f = np.ravel( sky_image)  # sinon pb lors de la correction par les pixels de ref, leur non linéarité ne peux pas être calibrée par illumination : pas sensible light
        sky_f[ref] = 0
        sky_image = np.reshape(sky_f, [2048, 2048])

        self.image += sky_image


    def applyShotNoise(self,A):
        """
        Add Poisson noise to the image.
        """
        
        im = np.ravel(A)

        im[np.argwhere(im < 0)] = 0

        actif = fits.getdata(self.path +'/'+ self.information['PixActifFile'])
        image = im[actif]

        im[actif] = np.random.poisson(image).astype(np.float64)
        
        A = np.reshape(im,[2048,2048])

        return(A)

    def applyReadoutNoise(self,A):
        """
        Applies readout noise to the image being constructed.

        The noise is drawn from a Normal (Gaussian) distribution with average=0.0 and std=readout noise.
        """

        noise = np.random.normal(loc=0.0, scale=self.information['RN'], size=A.shape)
        #print('readout',self.information['RN'])

        # av = np.copy(A)

        # add to the image
        output = (noise/10)

        return(A+output)


    def electrons2ADU(self,A):
        """
        Convert from electrons to ADUs using the value read from the configuration file.
        """
        gain_map = fits.getdata(self.path + '/GainMap/' + self.information['GainMapFile'])

        A /= gain_map

        return(A)

    def addOffset(self,F):
        """
        Add the offset (bias) in ADU
        """
        offset = fits.getdata(self.path + '/Offset/' + self.information['OffsetFile'])
        offset = np.reshape(offset, [2048, 2048])
        #print('max offset', np.max(offset))
        F += offset

        F[np.isnan(F)] = 0
        
        return(F)


    def writeOutputs(self,F, overwrite=False):
        """
        Writes out a FITS file using PyFITS and converts the image array to 16bit unsigned integer as
        appropriate for VIS.

        Updates header with the input values and flags used during simulation.
        """
        T0 = datetime.datetime.now()
        DELTA_DIT = datetime.timedelta(seconds = TDIT)
        self.hdu_header['DATE-OBS'] = datetime.datetime.isoformat(T0)[:-2]
        self.hdu_header['HIERARCH ESO DET SEQ UTC'] = (datetime.datetime.isoformat(T0)[:-2],"Sequencer Start Time")
        
        F=F.astype(np.uint16)
        
        primary_header = self.hdu_header.copy()
        primary_header['ORIGIN'] = "SIMUL_RAMP"
        primary_header['ORIGFILE'] = (self.information['output'].split('/')[1],"Original File Name")

        primary = fits.PrimaryHDU(header=primary_header)
        
        hdul = fits.HDUList([primary])
        
        n = np.shape(F)[0]
        
        if n != 1:
            for i in range(0, n):
                frame_header = self.hdu_header.copy()
                frame_header['HIERARCH ESO DET EXP UTC'] = (datetime.datetime.isoformat(T0 + i*DELTA_DIT)[:-2],"Time Recv Frame")
                frame_name = "CHIP1.DIT"+str(i+1)
                hdul.append(fits.ImageHDU(F[i], header=frame_header, name=frame_name))

        hdul.writeto(simu_dir + '/output/' + self.information['output'], overwrite=overwrite)

    def applySaturation(self,F):
        """
        Met au maximum de remplissage des puits les pixels de l'image (self.image) qui dépassent le niveau de saturation
        défini par l'argument max.

        :param max: maximum value the the integer array may contain [default 120 000 électrons ce qui donne 350000
        en prenant en compte l'offset qui est
        :type max: float

        :return: None
        """
        # max=self.information['FWC']
        # satu = fits.getdata('/home/alix/Documents/Programmes/Cartes_det4_Pixel_Reset/CarteSaturation_Det4.fits')
        satu = fits.open(self.path + '/' + self.information['SaturationFile'])
        maxi = np.ravel(satu[1].data)
        # max=120000

        # cut of the values larger than max
        im = np.ravel(F)
        a = np.argwhere(im >= maxi)

        im[a] = maxi[a]


        F = np.reshape(im, [2048, 2048])

        return(F)


    def applyCrossTalk(self,A):
        """
        appliquer l'impact du cross talk aux images
        :param ct: matrice 3*3 des coeficients de cross talk
        :type ct: array
        :return: None
        """
        crosstalk = fits.open(self.path + '/' + self.information['CrossTalkFile'])
        ct = np.zeros([3, 3])
        ct[0] = crosstalk[1].data
        ct[1] = crosstalk[2].data
        ct[2] = crosstalk[3].data
        # print('ct',np.shape(ct),ct)
        # ct = np.array([[0.0002, 0.002, 0.0002], [0.002, 1-(0.002*4+0.0002*4), 0.002], [0.0002, 0.002, 0.0002]])

        # on coupe les valeurs au dela de la saturation pour ne pas générer un cross talk enorme lorsque l'on dépasse de beaucoup la saturation
        offset = np.ravel(fits.getdata(self.path + '/Offset/' + self.information['OffsetFile']))
        #satu = np.ravel(fits.getdata(simu_dir+'/ImSimpyA/saturation_1V.fits'))
        satu = fits.open(self.path + '/' + self.information['SaturationFile'])
        satu = np.ravel(satu[1].data)

        sat = (satu - offset) * 10
        im = np.ravel(A)

        actif = fits.getdata(self.path +"/"+ self.information['PixActifFile'])

        a = np.intersect1d(np.argwhere(im > 1.1 * sat), actif)
        
        im[a] = 1.1 * sat[a]
        av = im
        im = np.reshape(im, [2048, 2048])


        image = np.ravel(scipy.signal.convolve2d(im, ct, mode='same', boundary='fill'))

        refct = np.loadtxt(simu_dir+'ImSimpyA/ref_ct.txt').astype(int)
        image[refct] = np.ndarray.flatten(im)[refct]

        A = np.reshape(image, [2048, 2048])

        
        return(A)
    
    
    def buildFrame(self, params):
        
        dF, k = params

        

        if self.Persistance:
            dF=self.applyPersistance(k,dF)

        if self.shotNoise:
            dF=self.applyShotNoise(dF)

        if self.cosmicRays:
            dF=self.addCosmicRays(k,dF)

        if self.CrossTalk:
            dF=self.applyCrossTalk(dF)

        if self.nonlinearity:
            dF=self.applyNonLinearity(k,dF)

        if self.cosmetics:
            dF=self.addCosmetics(dF)

        if self.ADU:
            dF=self.electrons2ADU(dF)

        return dF
        
    
    
    def readDetector(self, params):
        F, k = params
        
        if self.readoutNoise:
            F = self.applyReadoutNoise(F)

        if self.Offset:
            F = self.addOffset(F)

        if self.CreatePersistance:
            F = self.applyCreatePersistance(F,k)

        if self.saturation:
            F = self.applySaturation(F)

        return F

    
    def simulate(self, config_type='file', plot=False):
        """
        Create a single simulated image defined by the configuration file.
        Will do all steps defined in the config file sequentially.

        :return: None
        """
        # if self.config['verbose'] == 'True': print ("Read config file and execute ETC")
        print("Read config file and execute ETC")
        self.configure(config_type)
        print("Building image: %s:" % self.information['output'])
        # print (self.information)

        Nfin = int(np.round(self.information['Nfin']))
        nf =  np.arange(1,Nfin+1)

        F=np.zeros([len(nf),2048,2048])

        if self.Addsources:
            print("\tGENERATE OBJECTS CATALOG")
            self.generateObjectList()
            self.readObjectlist()

            print("\tGENERATE PSF")
            self.generatePSF()
            self.readPSFs()
            self.generateFinemaps()

            print("\tADD OBJECTS")
            self.addObjects(self.image,1)

            '''plt.figure('init')
            plt.imshow(self.image, vmin= 0, vmax = 50) #np.quantile(self.image, 0.1), vmax=np.quantile(self.image, 0.8))
            plt.colorbar()
            plt.show()'''

        if self.background:
            print("\tAdd Sky background")
            self.applySkyBackground()

            '''plt.figure('back')
            plt.imshow(self.image, vmin = 180, vmax = 205) # vmin=np.quantile(self.image, 0.1), vmax=np.quantile(self.image, 0.8))
            plt.colorbar()
            plt.show()'''

        if self.FlatField:
            print("\tAdd FlatField")
            self.applyFlatField()

            '''plt.figure('flatfield')
            plt.imshow(self.image, vmin = 180, vmax=205) #vmin=np.quantile(self.image, 0.05), vmax=np.quantile(self.image, 0.95))
            plt.colorbar()
            plt.show()'''

        if self.darkCurrent:
            print("\tAdd dark current")
            self.applyDarkCurrent()
        
        print("\n")
        print("Generating frames... ", end="\r")
        image = np.copy(self.image)
        #plt.figure('A')
        #plt.imshow(self.image, vmin=np.quantile(self.image, 0.1), vmax=np.quantile(self.image, 0.9))
        #plt.colorbar()
        
        """ This is an attempt at parallelizing the generation of frames. (FF) """
        
        begin = time.perf_counter()
        params = []
        for k in range(len(nf)):
            params.append([image,k])

        F = Parallel(n_jobs=n_threads, backend='loky')(delayed(self.buildFrame)(param) for param in params)
        del(params)
        print("Generating frames... Done.\n")
        
        
        if self.AddGRB:
            print("Adding GRB to frames... ", end="\r")
            for k in nf:
                A = np.copy(self.image)
                self.image = np.zeros((self.information['ysize'], self.information['xsize']), dtype=np.float64)
                self.AddObjectToList(k)  # ajouter un objet fixe à la liste: fonction crée Alix
                self.readObjectlist()
                self.generatePSF()
                self.readPSFs()
                self.generateFinemaps()
                self.addObjects(A,k) #a copier, modifier et faire pour cas GRB : juste if grb sur A et pas self.image
                F[k-1] = self.image + F[k-1]
        
            print("Adding GRB to frames...Done.\n")
        for k in range(len(nf)-1):
            F[k+1] += F[k]
        F = np.array(F)
        end = time.perf_counter()
        print("\nTotal time to generate the ramp: "+str(round(end-begin,2))+" s.\n")
        
        
        """ Parallel version of the code below """
        
        print("Reading detector... ", end="\r")
        params = []
        for k in range(len(nf)):
            params.append([F[k],k])
        
        F = Parallel(n_jobs=n_threads)(delayed(self.readDetector)(param) for param in params)
        end = time.perf_counter()
        F = np.array(F)
        print("Reading detector...Done.\n")
        
        
        """ Original block. Same execution time as the parallel version. Not enough jobs (~45) or cores (4) to warrant the parallel computing (FF) """
        """
        for k in range(0,len(nf)):

            if self.readoutNoise:
                # if self.config['verbose'] == 'True': print ("Add Readout Noise")
                #print("\tAdd Readout Noise")
                F[k]= self.applyReadoutNoise(F[k])

            if self.Offset:
                # if self.config['verbose'] == 'True': print ("Add offset")
                # Nframe = int(np.round(self.information['Nframe']))
                # if Nframe == 1:
                #print("\tAdd offset")
                F[k]=self.addOffset(F[k])

            if self.CreatePersistance:
                #print("\tCreate Persistance")
                self.applyCreatePersistance(F[k],k)


            if self.saturation:
                # if self.config['verbose'] == 'True': print ("Apply Saturation")
                #print("\tApply Saturation")
                F[k] =self.applySaturation(F[k])

                #plt.figure('SAT')
                #plt.imshow(F[k], vmin=np.quantile(F[k], 0.05), vmax=np.quantile(F[k], 0.95))
                #plt.show()
	"""
        
        
        satu = fits.open(self.path + '/' + self.information['SaturationFile'])
        satu = np.ravel(satu[1].data)
        if plot:
            plt.figure('rampe')
            # plt.plot(np.median(F,axis=(1,2)))
            plt.plot(F[:, 1000, 1000])
            plt.plot(F[:, 1200, 1200])
            plt.plot(F[:, 2000, 1400])
            #plt.plot(F[:, 735, 267])
            plt.plot(F[:, 736, 267])
            #plt.plot(np.repeat(satu[267+2048* 735],len(nf)),'--')
            plt.plot(np.repeat(satu[1507595],len(nf)),'--')

            plt.figure('image')
            plt.imshow(F[len(nf)-2], vmin=np.quantile(F[len(nf)-2], 0.01), vmax=np.quantile(F[2], 0.99))
            plt.colorbar()
            
            plt.show()

        # if self.config['verbose'] == 'True': print ("Write outputs")
        print("Writing outputs... ", end="\r")
        self.writeOutputs(F, overwrite=True)
        print("Writing outputs... Done.\n")
        del(F)
        print("===========================\n\tJOB'S DONE.\n===========================")
        
