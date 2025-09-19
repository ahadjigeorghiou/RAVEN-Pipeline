import numpy as np
from scipy import optimize
import batman
from Features import utils

def Trapezoidmodel(phase_data, t0_phase, t23, t14, depth):
    centrediffs = np.abs(phase_data - t0_phase)
    model = np.ones(len(phase_data))
    model[centrediffs<t23/2.] = 1-depth
    in_gress = (centrediffs>=t23/2.)&(centrediffs<t14/2.)   
    model[in_gress] = (1-depth) + (centrediffs[in_gress]-t23/2.)/(t14/2.-t23/2.)*depth
    if t23>t14:
        model = np.ones(len(phase_data))*1e8
    return model
    
def Trapezoidmodel_fixephem(phase_data, t23, t14, depth):
    centrediffs = np.abs(phase_data - 0.5)
    model = np.ones(len(phase_data))
    model[centrediffs<t23/2.] = 1-depth
    in_gress = (centrediffs>=t23/2.)&(centrediffs<t14/2.)   
    model[in_gress] = (1-depth) + (centrediffs[in_gress]-t23/2.)/(t14/2.-t23/2.)*depth
    if t23>t14:
        model = np.ones(len(phase_data))*1e8
    return model

def Trapezoidfitfunc(fitparams,y_data,y_err,x_phase_data, init_tdur):
    t0 = fitparams[0]
    t23 = fitparams[1]
    t14 = fitparams[2]
    depth = fitparams[3]
        
    if (t0<0.2) or (t0>0.8) or (t23 < 0) or (t14 < 0) or (t14 < t23) or (depth < 0):
        return np.ones(len(x_phase_data))*1e8
    if (np.abs(t14-init_tdur)/init_tdur) > 0.05:
        return np.ones(len(x_phase_data))*1e8
    
    model = Trapezoidmodel(x_phase_data, t0,t23,t14,depth)
    return (y_data - model)/y_err


class TransitFit(object):

    def __init__(self,lc,initialguess,fittype='model', bounds=None, 
                 fixper=None,fixt0=None, exp_time=None, sfactor=None):
        
        self.lc = lc
        self.init = initialguess
        self.exp_time = exp_time
        self.sfactor = sfactor
        self.bounds = bounds
        self.fixper = fixper
        self.fixt0 = fixt0        
        if fittype == 'model':
            self.params,self.cov = self.FitTransitModel()
            self.errors,self.chisq = self.GetErrors()
        elif fittype == 'trap':
            self.params,self.cov = self.FitTrapezoid()
            if not np.isnan(self.cov).all():
                self.errors = np.sqrt(np.diag(self.cov))
            else:
                self.errors = np.full_like(self.params, -10)
                
            if self.fixt0 is None:  #then convert t0 back to time
                self.params[0] = (self.params[0]-0.5)*self.fixper + self.initial_t0
                self.errors[0] *= self.fixper 
    
    def FitTrapezoid(self):
        if self.fixt0 is None:
            self.initial_t0 = self.init[0]
            phase = utils.phasefold(self.lc['time'],self.fixper,
            					    self.initial_t0+self.fixper*0.5)  #transit at phase 0.5
            initialguess = self.init.copy()
            initialguess[0] = 0.5

            if self.bounds is None:
                self.bounds =  [(0.45, 0.0001, 0.0001, 0),(0.55, 0.35, 0.35, 1)]
                
            try:
                fit = optimize.curve_fit(Trapezoidmodel, phase, self.lc['flux'], 
                                        p0=initialguess, sigma=self.lc['error'],
                                        bounds=self.bounds, absolute_sigma=False, loss='huber',
                                        xtol=1e-6, ftol=1e-6, gtol=1e-6)
            except (RuntimeError, ValueError):
                nan_arr = np.array([np.nan, np.nan, np.nan, np.nan])
                fit = (nan_arr, np.nan)
        else:
            self.initial_t0 = self.fixt0
            phase = utils.phasefold(self.lc['time'],self.fixper,
            				        self.initial_t0+self.fixper*0.5)  #transit at phase 0.5
            initialguess = self.init.copy()

            if self.bounds is None:
                self.bounds =  [(0.0001, 0.0001, 0),(0.35, 0.35, 1)]
            try:
                fit = optimize.curve_fit(Trapezoidmodel_fixephem, phase, self.lc['flux'], 
                                        p0=initialguess, sigma=self.lc['error'],
                                        bounds=self.bounds, absolute_sigma=False, loss='huber',
                                        xtol=1e-6, ftol=1e-6, gtol=1e-6)
            except (RuntimeError, ValueError):
                nan_arr = np.array([np.nan, np.nan, np.nan])
                fit = (nan_arr, np.nan)
        return fit[0], fit[1]

    def FitTransitModel(self):
        self.bparams = batman.TransitParams()    #object to store transit parameters
        self.bparams.rp = self.init[3]           #planet radius (in units of stellar radii)
        self.bparams.a = self.init[2]            #semi-major axis (in units of stellar radii)
        self.bparams.inc = 90.
        self.bparams.ecc = 0.                 #eccentricity
        self.bparams.w = 90.                 #longitude of periastron (in degrees)
        self.bparams.limb_dark = 'quadratic'           #limb darkening model
        self.bparams.u = [0.1,0.3]                 #limb darkening coefficients
        if self.fixper is None:
            self.bparams.t0 = self.init[1]       #time of inferior conjunction
            self.bparams.per = self.init[0]      #orbital period
            self.bmodel = batman.TransitModel(self.bparams, self.lc['time'],exp_time=self.exp_time,
            						supersample_factor=self.sfactor)
            
            initialguess = self.init.copy()
            if self.bounds is None:
                self.bounds =  [(self.bparams.per*0.99, self.bparams.t0*0.99, 1.5, 0),(self.bparams.per*1.01, self.bparams.t0*1.01, np.inf, np.inf)]
            
            try:
                fit = optimize.curve_fit(self.Transitfit_model, self.lc['time'], self.lc['flux'],
                                        p0=initialguess, sigma=self.lc['error'],
                                        bounds=self.bounds, absolute_sigma=False, loss='huber',
                                        xtol=1e-6, ftol=1e-6, gtol=1e-6)
            except (RuntimeError, ValueError):
                nan_arr = np.array([np.nan, np.nan, np.nan, np.nan])
                fit = (nan_arr, np.nan)
        else:
            self.bparams.t0 = self.fixt0
            self.bparams.per = self.fixper
            self.bmodel = batman.TransitModel(self.bparams, self.lc['time'],exp_time=self.exp_time,
            						supersample_factor=self.sfactor)
            initialguess = self.init[2:]
            if self.bounds is None:
                self.bounds =  [(1.5, 0),(np.inf, np.inf)]
            try:
                fit = optimize.curve_fit(self.Transitfit_model_fixephem, self.lc['time'], self.lc['flux'],
                                        p0=initialguess, sigma=self.lc['error'],
                                        bounds=self.bounds, absolute_sigma=False, loss='huber',
                                        xtol=1e-6, ftol=1e-6, gtol=1e-6)   
            except (RuntimeError, ValueError):
                nan_arr = np.array([np.nan, np.nan])
                fit = (nan_arr, np.nan)       
        return fit[0], fit[1]

    def Transitfit_model(self, time, per, t0, arstar, rprstar):

        self.bparams.t0 = t0                        #time of inferior conjunction
        self.bparams.per = per                      #orbital period
        self.bparams.rp = rprstar                   #planet radius (in units of stellar radii)
        self.bparams.a = arstar                     #semi-major axis (in units of stellar radii)
        model_flux = self.bmodel.light_curve(self.bparams)
        return model_flux


    def Transitfit_model_fixephem(self, time, arstar, rprstar):
        self.bparams.rp = rprstar                      #planet radius (in units of stellar radii)
        self.bparams.a = arstar                        #semi-major axis (in units of stellar radii)
        model_flux = self.bmodel.light_curve(self.bparams)
        return model_flux

    def GetErrors(self):
        self.bparams = batman.TransitParams()    #object to store transit parameters
        self.bparams.inc = 90.
        self.bparams.ecc = 0.                    #eccentricity
        self.bparams.w = 90.                     #longitude of periastron (in degrees)
        self.bparams.limb_dark = 'quadratic'     #limb darkening model
        self.bparams.u = [0.1,0.3]               #limb darkening coefficients
        if self.fixper is None:
            self.bparams.t0 = self.params[1]     #time of inferior conjunction
            self.bparams.per = self.params[0]    #orbital period
            self.bparams.rp = self.params[3]     #planet radius (in units of stellar radii)
            self.bparams.a = self.params[2]      #semi-major axis (in units of stellar radii)

        else:
            self.bparams.t0 = self.fixt0
            self.bparams.per = self.fixper
            self.bparams.rp = self.params[1]     #planet radius (in units of stellar radii)
            self.bparams.a = self.params[0]      #semi-major axis (in units of stellar radii)

        model_flux = self.bmodel.light_curve(self.bparams)
        if self.fixper is None:
            if self.cov is not None and not np.isnan(self.cov).all():
                err = np.sqrt(np.diag(self.cov))
            else:
                #print('Fit did not give covariance, error based features will not be meaningful')
                err = np.ones(4)*-10
            chisq = 1./len(self.lc['flux']-4) * np.sum(np.power((self.lc['flux'] - model_flux)/self.lc['error'],2))
        else:
            if self.cov is not None and not np.isnan(self.cov).all():
                err = np.sqrt(np.diag(self.cov))
            else:
                #print('Fit did not give covariance, error based features will not be meaningful')
                err = np.ones(2)*-10
            chisq = 1./len(self.lc['flux']-2) * np.sum(np.power((self.lc['flux'] - model_flux)/self.lc['error'],2))   
        return err,chisq
