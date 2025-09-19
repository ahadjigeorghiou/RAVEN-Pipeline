#significant input and copied functions from T. Morton's VESPA code (all mistakes are my own)

#coords		--	RA and DEC of target in degrees. Needed for GAIA querying.
#        		Degrees, 0-360 and -90 to +90. List format [RA,DEC].

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy import stats
import astropy.constants as const
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactic
from astropy_healpix import HEALPix
import subprocess as sp
import os, re
import time
AU = const.au.cgs.value
RSUN = const.R_sun.cgs.value
REARTH = const.R_earth.cgs.value
MSUN = const.M_sun.cgs.value
DAY = 86400 #seconds
G = const.G.cgs.value
import logging

import scipy.integrate as integrate
    


def calculate_universal_priors(detectionfactor, periodrange = [], radiusrange = [], primarymass=1.0):
    '''
    detectionfactor - prior representing fraction of that scenario which are detected by search algorithm. E.g. BTPs are abundant but only a tiny fraction of a realistic distribution will lead to detected candidates. That comes in here.

    periodrange - min and max periods considered in this set. Used for prior occurrence integration limits
    radiusrange - as periodrange for radius. May need to consider as depth instead? Or possible can be ignored? (i.e. just a minimum sensible planet radius, assume all stars detectable)
    '''
   
    f_multiple = 0.4
    f_binary = 0.3
    
    planet_occ = planet_prior(periodrange, radiusrange)
    cb_occ = close_binary_prior(periodrange, primarymass)
    priors = {'Planet': planet_occ * detectionfactor.loc['PLA', 'adj_detection'],
              'EB':  cb_occ * detectionfactor.loc['EB', 'adj_detection'],
              'HEB': cb_occ * f_multiple * detectionfactor.loc['TRIPLE', 'adj_detection'],
              'BEB': cb_occ * detectionfactor.loc['BEB', 'adj_detection'],
              'BTP': planet_occ * detectionfactor.loc['BTP', 'adj_detection'],
              'HTP': planet_occ * f_binary * 0.5 * detectionfactor.loc['PIB', 'adj_detection'],
              'NTP': planet_occ * detectionfactor.loc['NPLA', 'adj_detection'],
              'NEB': cb_occ * detectionfactor.loc['NEB', 'adj_detection'],
              'NHEB': cb_occ * f_multiple * detectionfactor.loc['NTRIPLE', 'adj_detection']
    }

    return priors
    

def calculate_candidate_priors(prob_source_target, universal_priors, maglim_beb, maglim_btp, bg_skyrad, ra, dec, densitymaps):
    '''
    location - star location (target star may be set to zero but other sources for same candidate will not be)
    centroiddat - centroid mean location (0,1) and sigma on that axis (2,3) Sigma should be in arcsec for beb density calculation. Will need to check compatibility with location units later.
    universal_priors - output of calculate_universal_priors
    maglim - magnitude limit for considering background stars (should be different for BTPs vs BEBs? BTP would be much lower, depends on eclipse depth)
    '''
    
    if maglim_beb is None:
        prob_background_beb = 0
    else:
        beb_density = trilegal_density(ra,dec,kind='interp',maglim=maglim_beb,densitymaps=densitymaps,nside=16) 
        prob_background_beb = np.pi*(bg_skyrad/3600.)**2 * beb_density #assumes skyrad in arcsec
    
    if maglim_btp is None:
        prob_background_btp = 0
    else:
        btp_density = trilegal_density(ra,dec,kind='interp',maglim=maglim_btp,densitymaps=densitymaps,nside=16) 
        prob_background_btp = np.pi*(bg_skyrad/3600.)**2 * btp_density #assumes skyrad in arcsec
    
    candidate_priors = {'Planet': universal_priors['Planet'] * prob_source_target,
                        'EB':   universal_priors['EB'] * prob_source_target,
                        'HEB':  universal_priors['HEB'] * prob_source_target,
                        'BEB':  universal_priors['BEB'] * prob_background_beb * prob_source_target,
                        'BTP':  universal_priors['BTP'] * prob_background_btp * prob_source_target,
                        'HTP':  universal_priors['HTP'] * prob_source_target,
                        'NTP':  universal_priors['NTP'] * (1-prob_source_target),
                        'NEB':  universal_priors['NEB'] * (1-prob_source_target),
                        'NHEB':  universal_priors['NHEB'] * (1-prob_source_target)                   
    }
        
    return candidate_priors
    

def semimajor(P,mtotal=1.):
    """
    Returns semimajor axis in AU given P in days, total mass in solar masses.
    """
    return np.power(np.power(P*DAY/2/np.pi,2)*G*mtotal*MSUN,(1./3.))/AU
    
    
def eclipse_probability(R2, P, R1, M1, M2):
    return (R1 + R2) *RSUN / (semimajor(P , M1 + M2)*AU)
                
def EBoccrate_02logP1(logP, logmass):
    #0.2<logP<1
    f_logPlt1 =  (0.02 + 0.04*logmass + 0.07*logmass**2)
    return f_logPlt1 

def EBoccrate_1logP2(logP, logmass):
    #1<=logP<2
    f_logPlt1 =  (0.02 + 0.04*logmass + 0.07*logmass**2)
    f_logPe27 = (0.039 + 0.07*logmass + 0.01*logmass**2)
    return (f_logPlt1 + (logP - 1)*(f_logPe27 - f_logPlt1 - 0.018*0.7) ) 

def EBoccrate_2logP34(logP, logmass):    
    #2<logP<3.4
    f_logPe27 = (0.039 + 0.07*logmass + 0.01*logmass**2)
    return (f_logPe27 + 0.018*(logP-2.7))

def EBoccrate_34logP55(logP, logmass):    
    #3.4<logP<5.5
    f_logPe27 = (0.039 + 0.07*logmass + 0.01*logmass**2)
    f_logPe55 = (0.078 - 0.05*logmass + 0.04*logmass**2)
    
    return (f_logPe27 + 0.018*0.7 + ((logP - 2.7 - 0.7)/(2.8-0.7))*(f_logPe55 - f_logPe27 - 0.018*0.7))

def EBoccrate_55logP8(logP, logmass):
    #5.5<logP<3.4
    f_logPe55 = (0.078 - 0.05*logmass + 0.04*logmass**2)
    
    return (f_logPe55 * np.exp(-0.3*(logP-5.5)))

        
def close_binary_prior(periodrange=[1.58,27.], primarymass=1.0):
    '''
    primary mass in solar
    periodmax in days - upper limit of considered range
    centroid pdf at source location

    eclipse prob - stars of same radius maximises eclipse prob. Using low M2 minimises a, maximising eclipse prob
    works for defined source EBs too, just use appropriate centroid pdf value.
    
    MULTIPLY BY 1.3 TO INCLUDE 0.1<q<0.3 COMPANIONS
    
    1.58 IS TECHNICAL LOWER LIMIT OF M+dS EQUATION.
    CAN ARGUE LOWER WOULD BE CONTACT BINARIES AND HENCE NOT RELEVANT AS FP SCENARIOS
    
    '''
    import scipy.integrate as integrate
    
    logmass = np.log10(primarymass)
    logpmin = np.log10(periodrange[0])
    logpmax = np.log10(periodrange[1])
    
    if logpmin < 1.:
        if logpmax < 1.:
            result = integrate.quad(EBoccrate_02logP1, logpmin, logpmax, args=(logmass))[0]
        elif logpmax < 2.:
            result = integrate.quad(EBoccrate_02logP1, logpmin, 1.0, args=(logmass))[0] \
            		+ integrate.quad(EBoccrate_1logP2, 1.0, logpmax, args=(logmass))[0]
        elif logpmax < 3.4:
            result = integrate.quad(EBoccrate_02logP1, logpmin, 1.0, args=(logmass))[0] \
            		+ integrate.quad(EBoccrate_1logP2, 1.0, 2.0, args=(logmass))[0] \
            		+ integrate.quad(EBoccrate_2logP34, 2.0, logpmax, args=(logmass))[0]
        else:
            print('Upper limit of period range too high')
    
    elif logpmin < 2:
        if logpmax < 2.:
            result = integrate.quad(EBoccrate_1logP2, logpmin, logpmax, args=(logmass))[0]
        elif logpmax < 3.4:
            result = integrate.quad(EBoccrate_1logP2, logpmin, 2.0, args=(logmass))[0] \
            		+ integrate.quad(EBoccrate_2logP34, 2.0, logpmax, args=(logmass))[0]
        else:
            print('Upper limit of period range too high')
        
    elif logpmin < 3.4:
        if logpmax < 3.4:
            result = integrate.quad(EBoccrate_2logP34, logpmin, logpmax, args=(logmass))[0]
        else:
            print('Upper limit of period range too high')
    else:
        print('Lower limit of period range too high')
    
    return 1.3*result
    
       
def planet_prior(periodrange=[1.58,27.],radiusrange=[2.0,12.0],r1=1.0, m1=1.0, mp=0.0):
    hsu19_dat = np.genfromtxt(os.path.join(os.path.dirname(__file__),'hsu19_occrates.txt'),delimiter=',')
    hsu19_radbins = np.array([0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,4.0,6.0,8.0,12.0,16.0])
    hsu19_perbins = np.array([0.5,1.0,2.0,4.0,8.0,16.0,32.0,64.0,128.0,256.0,500.0])
    
    radwidth = np.diff(np.log(hsu19_radbins))
    perwidth = np.diff(np.log(hsu19_perbins))
    binarea = radwidth[:,  None] * perwidth
    occrates = hsu19_dat / binarea
   
    prior = integrate.dblquad(planet_prior_integral, np.log(periodrange[0]), np.log(periodrange[1]), np.log(radiusrange[0]), np.log(radiusrange[1]), args=(occrates, np.log(hsu19_radbins[1:]), np.log(hsu19_perbins[1:]), r1, m1, mp))[0]
    return prior
       

def planet_prior_integral(R, P, occrates, radbins, perbins, r1=1.0, m1=1.0, mp=0.0):
    radidx = np.digitize(R, radbins)
    peridx = np.digitize(P, perbins)
    rate = occrates[radidx, peridx]
    return rate 


def trilegal_density(ra,dec,kind='target',maglim=21.75,area=1.0,densitymaps=None,nside=16):
    if kind=='interp' and densitymaps is None:
        print('HEALPIX maps must be passed')
        return 0
    if kind not in ['target','interp']:
        print('kind not recognised. Setting kind=target')
        kind = 'target'
        
    if kind=='target':
        
        basefilename = 'trilegal_'+str(ra)+'_'+str(dec)
        h5filename = basefilename + '.h5'
        if not os.path.exists(h5filename):
            get_trilegal(basefilename,ra,dec,maglim=maglim,area=area)
        else:
            print('Using cached trilegal file. Sky area may be different.')

        if os.path.exists(h5filename):
        
            stars = pd.read_hdf(h5filename,'df')
            with pd.HDFStore(h5filename) as store:
                trilegal_args = store.get_storer('df').attrs.trilegal_args

            if trilegal_args['maglim'] < maglim:
                print('Re-calling trilegal with extended magnitude range')
                get_trilegal(basefilename,ra,dec,maglim=maglim,area=area)
                stars = pd.read_hdf(h5filename,'df')
            
            stars = stars[stars['TESS_mag'] < maglim]  #in case reading from file

            area = trilegal_args['area']*(u.deg)**2
            density = len(stars)/area
            return density.value
        else:
            return 0
    else:
        #interpolate pre-calculated densities
        coord = SkyCoord(ra,dec,unit='deg')
        if np.abs(coord.galactic.b.value)<5:
            print('Near galactic plane, Trilegal density may be inaccurate.')

        if maglim < 9: #current limit of interp maps
            print('Warning, maglim outside interp range, setting to 9: ',maglim)
            maglim = 9 
        if maglim > 21.75:  #current limit of interp maps
            print('Warning, maglim outside interp range, setting to 21.75: ',maglim)
            maglim = 21.75  
        
        mapkey = np.ceil(maglim*4) * 0.25
        hp = HEALPix(nside=16, order='RING', frame=Galactic())
        density = hp.interpolate_bilinear_skycoord(coord, densitymaps[mapkey])
        
        return density
    
#maglim of 21 used following sullivan 2015

def get_trilegal(filename,ra,dec,folder='.', galactic=False,
                 filterset='TESS_2mass_kepler',area=1,maglim=21,binaries=False,
                 trilegal_version='1.6',sigma_AV=0.1,convert_h5=True):
    """Runs get_trilegal perl script; optionally saves output into .h5 file
    Depends on a perl script provided by L. Girardi; calls the
    web form simulation, downloads the file, and (optionally) converts
    to HDF format.
    Uses A_V at infinity from :func:`utils.get_AV_infinity`.
    .. note::
        Would be desirable to re-write the get_trilegal script
        all in python.
    :param filename:
        Desired output filename.  If extension not provided, it will
        be added.
    :param ra,dec:
        Coordinates (ecliptic) for line-of-sight simulation.
    :param folder: (optional)
        Folder to which to save file.  *Acknowledged, file control
        in this function is a bit wonky.*
    :param filterset: (optional)
        Filter set for which to call TRILEGAL.
    :param area: (optional)
        Area of TRILEGAL simulation [sq. deg]
    :param maglim: (optional)
        Limiting magnitude in first mag (by default will be Kepler band)
        If want to limit in different band, then you have to
        got directly to the ``get_trilegal`` perl script.
    :param binaries: (optional)
        Whether to have TRILEGAL include binary stars.  Default ``False``.
    :param trilegal_version: (optional)
        Default ``'1.6'``.
    :param sigma_AV: (optional)
        Fractional spread in A_V along the line of sight.
    :param convert_h5: (optional)
        If true, text file downloaded from TRILEGAL will be converted
        into a ``pandas.DataFrame`` stored in an HDF file, with ``'df'``
        path.
    """
    if galactic:
        l, b = ra, dec
    else:
        try:
            c = SkyCoord(ra,dec)
        except:
            c = SkyCoord(ra,dec,unit='deg')
        l,b = (c.galactic.l.value,c.galactic.b.value)

    if os.path.isabs(filename):
        folder = ''

    if not re.search('\.dat$',filename):
        outfile = '{}/{}.dat'.format(folder,filename)
    else:
        outfile = '{}/{}'.format(folder,filename)
        
    NONMAG_COLS = ['Gc','logAge', '[M/H]', 'm_ini', 'logL', 'logTe', 'logg',
               'm-M0', 'Av', 'm2/m1', 'mbol', 'Mact'] #all the rest are mags

    AV = get_AV_infinity(l,b,frame='galactic')
    print(AV)
    if AV is not None:
      if AV<=1.5:
        trilegal_webcall(trilegal_version,l,b,area,binaries,AV,sigma_AV,filterset,maglim,outfile)

        if convert_h5 and os.path.exists(outfile):
            df = pd.read_table(outfile, sep='\s+', skipfooter=1, engine='python')
            df = df.rename(columns={'#Gc':'Gc'})
            for col in df.columns:
                if col not in NONMAG_COLS:
                    df.rename(columns={col:'{}_mag'.format(col)},inplace=True)
            if not re.search('\.h5$', filename):
                h5file = '{}/{}.h5'.format(folder,filename)
            else:
                h5file = '{}/{}'.format(folder,filename)
            df.to_hdf(h5file,'df')
            with pd.HDFStore(h5file) as store:
                attrs = store.get_storer('df').attrs
                attrs.trilegal_args = {'version':trilegal_version,
                                   'ra':ra, 'dec':dec,
                                   'l':l,'b':b,'area':area,
                                   'AV':AV, 'sigma_AV':sigma_AV,
                                   'filterset':filterset,
                                   'maglim':maglim,
                                   'binaries':binaries}
            os.remove(outfile)
    else:
        print('Skipping, AV > 10 or not found')

def trilegal_webcall(trilegal_version,l,b,area,binaries,AV,sigma_AV,filterset,maglim,
					 outfile):
    """Calls TRILEGAL webserver and downloads results file.
    :param trilegal_version:
        Version of trilegal (only tested on 1.6).
    :param l,b:
        Coordinates (galactic) for line-of-sight simulation.
    :param area:
        Area of TRILEGAL simulation [sq. deg]
    :param binaries:
        Whether to have TRILEGAL include binary stars.  Default ``False``.
    :param AV:
    	Extinction along the line of sight.
    :param sigma_AV:
        Fractional spread in A_V along the line of sight.
    :param filterset: (optional)
        Filter set for which to call TRILEGAL.
    :param maglim:
        Limiting magnitude in mag (by default will be 1st band of filterset)
        If want to limit in different band, then you have to
        change function directly.
    :param outfile:
        Desired output filename.
    """
    webserver = 'http://stev.oapd.inaf.it'
    args = [l,b,area,AV,sigma_AV,filterset,maglim,1,binaries]
    mainparams = ('imf_file=tab_imf%2Fimf_chabrier_lognormal.dat&binary_frac=0.3&'
    			  'binary_mrinf=0.7&binary_mrsup=1&extinction_h_r=100000&extinction_h_z='
    			  '110&extinction_kind=2&extinction_rho_sun=0.00015&extinction_infty={}&'
    			  'extinction_sigma={}&r_sun=8700&z_sun=24.2&thindisk_h_r=2800&'
    			  'thindisk_r_min=0&thindisk_r_max=15000&thindisk_kind=3&thindisk_h_z0='
    			  '95&thindisk_hz_tau0=4400000000&thindisk_hz_alpha=1.6666&'
    			  'thindisk_rho_sun=59&thindisk_file=tab_sfr%2Ffile_sfr_thindisk_mod.dat&'
    			  'thindisk_a=0.8&thindisk_b=0&thickdisk_kind=0&thickdisk_h_r=2800&'
    			  'thickdisk_r_min=0&thickdisk_r_max=15000&thickdisk_h_z=800&'
    			  'thickdisk_rho_sun=0.0015&thickdisk_file=tab_sfr%2Ffile_sfr_thickdisk.dat&'
    			  'thickdisk_a=1&thickdisk_b=0&halo_kind=2&halo_r_eff=2800&halo_q=0.65&'
    			  'halo_rho_sun=0.00015&halo_file=tab_sfr%2Ffile_sfr_halo.dat&halo_a=1&'
    			  'halo_b=0&bulge_kind=2&bulge_am=2500&bulge_a0=95&bulge_eta=0.68&'
    			  'bulge_csi=0.31&bulge_phi0=15&bulge_rho_central=406.0&'
    			  'bulge_cutoffmass=0.01&bulge_file=tab_sfr%2Ffile_sfr_bulge_zoccali_p03.dat&'
    			  'bulge_a=1&bulge_b=-2.0e9&object_kind=0&object_mass=1280&object_dist=1658&'
    			  'object_av=1.504&object_avkind=1&object_cutoffmass=0.8&'
    			  'object_file=tab_sfr%2Ffile_sfr_m4.dat&object_a=1&object_b=0&'
    			  'output_kind=1').format(AV,sigma_AV)
    cmdargs = [trilegal_version,l,b,area,filterset,1,maglim,binaries,mainparams,
    		   webserver,trilegal_version]
    cmd = ("wget -o lixo -Otmpfile --post-data='submit_form=Submit&trilegal_version={}"
    	   "&gal_coord=1&gc_l={}&gc_b={}&eq_alpha=0&eq_delta=0&field={}&photsys_file="
    	   "tab_mag_odfnew%2Ftab_mag_{}.dat&icm_lim={}&mag_lim={}&mag_res=0.1&"
    	   "binary_kind={}&{}' {}/cgi-bin/trilegal_{}").format(*cmdargs)
    complete = False
    while not complete:
        notconnected = True
        busy = True
        print("TRILEGAL is being called with \n l={} deg, b={} deg, area={} sqrdeg\n "
        "Av={} with {} fractional r.m.s. spread \n in the {} system, complete down to "
        "mag={} in its {}th filter, use_binaries set to {}.".format(*args))
        sp.Popen(cmd,shell=True).wait()
        if os.path.exists('tmpfile') and os.path.getsize('tmpfile')>0:
            notconnected = False
        else:
            print("No communication with {}, will retry in 2 min".format(webserver))
            time.sleep(120)
        if not notconnected:
            with open('tmpfile','r') as f:
                lines = f.readlines()
            for line in lines:
                if 'The results will be available after about 2 minutes' in line:
                    busy = False
                    break
            sp.Popen('rm -f lixo tmpfile',shell=True)
            if not busy:
                filenameidx = line.find('<a href=../tmp/') +15
                fileendidx = line[filenameidx:].find('.dat')
                filename = line[filenameidx:filenameidx+fileendidx+4]
                print("retrieving data from {} ...".format(filename))
                count = 0
                while not complete:
                    time.sleep(120)
                    modcmd = 'wget -o lixo -O{} {}/tmp/{}'.format(filename,webserver,filename)
                    modcall = sp.Popen(modcmd,shell=True).wait()
                    if os.path.getsize(filename)>0:
                        with open(filename,'r') as f:
                            lastline = f.readlines()[-1]
                        if 'normally' in lastline:
                            complete = True
                            print('model downloaded!..')
                    if not complete:
                        print('still running...')        
                        count += 1
                        if count >= 5:
                            print('Waited 10 minutes, moving on..')
                            break
            else:
                print('Server busy, trying again in 2 minutes')
                time.sleep(120)
            if count >= 5 and not complete:
                break
    if complete:
        sp.Popen('mv {} {}'.format(filename,outfile),shell=True).wait()
        print('results copied to {}'.format(outfile))
       
   


        
def get_AV_infinity(ra,dec,frame='icrs'):
    """
    Gets the A_V exctinction at infinity for a given line of sight.
    Queries the NED database using ``curl``.
    .. note::
        It would be desirable to rewrite this to avoid dependence
        on ``curl``.
    :param ra,dec:
        Desired coordinates, in degrees.
    :param frame: (optional)
        Frame of input coordinates (e.g., ``'icrs', 'galactic'``)
    """
    coords = SkyCoord(ra,dec,unit='deg',frame=frame).transform_to('icrs')

    rah,ram,ras = coords.ra.hms
    decd,decm,decs = coords.dec.dms
    if decd > 0:
        decsign = '%2B'
    else:
        decsign = '%2D'
    url = 'http://ned.ipac.caltech.edu/cgi-bin/nph-calc?in_csys=Equatorial&in_equinox=J2000.0&obs_epoch=2010&lon='+'%i' % rah + \
        '%3A'+'%i' % ram + '%3A' + '%05.2f' % ras + '&lat=%s' % decsign + '%i' % abs(decd) + '%3A' + '%i' % abs(decm) + '%3A' + '%05.2f' % abs(decs) + \
        '&pa=0.0&out_csys=Equatorial&out_equinox=J2000.0'

    tmpfile = '/tmp/nedsearch%s%s.html' % (ra,dec)
    cmd = 'curl -s \'%s\' -o %s' % (url,tmpfile)
    sp.Popen(cmd,shell=True).wait()
    AV = None
    try:
        with open(tmpfile, 'r') as f:
            for line in f:
                m = re.search('V \(0.54\)\s+(\S+)',line)
                if m:
                    AV = float(m.group(1))
        os.remove(tmpfile)
    except:
        logging.warning('Error accessing NED, url={}'.format(url))

    return AV