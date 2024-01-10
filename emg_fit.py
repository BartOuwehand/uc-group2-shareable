from scipy.optimize import curve_fit
import scipy.special as sse
import numpy as np

def EMG(x, mu_x, sigma_x, E, x_0, B):
    ''' Exponential modified gaussian function;
        x: independent variable
        mu_x: location of the apparent source relative to the ship center
        sigma_x: standard deviation of the gaussian
        E: amplitude of the gaussian
        x_0: is the e-folding distance that represents the length scale of the NO2 decay
        B: background level of NO2 
        Formula from Lange et al. 2022
    '''

    return E/2 * np.exp( mu_x/x_0 + sigma_x**2/(2*x_0**2) - x/x_0) * sse.erfc( (mu_x+sigma_x**2/x_0-x)/(np.sqrt(2)*sigma_x) ) + B

def EMG_fit(x, y, yerr, p0, absolute_sigma=True):
    ''' Fit the exponential modified gaussian function to data
    '''
    
    try:
        popt, pcov = curve_fit(EMG, x, y, p0=p0, sigma=yerr, absolute_sigma=absolute_sigma, bounds=([0,0,0,0,-np.inf],[10,np.inf,np.inf,np.inf,np.inf]))
    except (RuntimeError, ValueError) as e:
        if 'Optimal parameters not found' in str(e):
            print("Fit did not converge")
            return None, None
        elif 'array must not contain infs or NaNs' in str(e):
            print("Overflow encountered in the EMG function while fitting")
            return None, None
        else:
            print(x, y, yerr, p0)
            print(x.shape, y.shape, yerr.shape)
            raise e
    
    return popt, pcov

def PlumeParams(popt, pcov, w, gamma=1.32):
    ''' Calculate plume parameters from the fitted EMG function.
        Formulas found in Lange et al. 2022
        w: ship speed
        gamma: ratio of NOx to NO2
    '''
    mu_x, sigma_x, E, x_0, B = popt
    mu_x_err, sigma_x_err, E_err, x_0_err, B_err = np.sqrt(np.diag(pcov))
    
    tau_EMG = x_0 / w # plume lifetime; we should expect a few hours. Units depend on x_0 and w units (e-folding distance and velocity)
    tau_err_EMG = x_0_err / w 
    E_EMG = E*w # NO2 emission of the plume. Units depend on E and w units (amplitude and velocity)
    E_err_EMG = E_err*w
    
    return (tau_EMG, tau_err_EMG), (E_EMG, E_err_EMG)