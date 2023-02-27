import sys
#sys.path.insert(0, '/Volumes/ligangwork/project1/work_dir')
import numpy as np

from multiprocessing import Pool
from fit_splittings import *
from stretched_echelle_diagram_functions import *
from plot_optimized_stretched_echelle_diagram import *

from power_spectrum_simulation import implicit_asymp_relation_for_mixed_modes, make_splitting_grid, bin_mode_function, MCMC_optimize_grid
import glob, os
from models import harvey_three, harvey_two, harvey_one
import syd
from get_peaks_no_MCMC import *
from bin_data import bin_data_function
import scipy
from astropy.timeseries import LombScargle
from sklearn.cluster import KMeans
import time as tm
import seaborn as sns
from astropy.modeling.models import Lorentz1D
from scipy.optimize import minimize, rosen, rosen_der

from tqdm import tqdm
import emcee
from astropy.io import ascii
import runpy
import pandas as pd
from scipy.signal import argrelextrema
from fit_package import *

def ligang_gaussian(x, ampl, center, sigma):
    return ampl * np.exp(-(x-center)**2/2/(sigma)**2)

def smooth(freq, PSD, width, skip, mode):
    print('Start smooth')
    freq_step = np.median( freq[1:]-freq[0:-1] )
    index_width = width / freq_step
    smooth_PSD = []
    smooth_freq = []
    for counter in  range(0, len(freq), skip) :
        smooth_freq.append(freq[counter])
        min_index = np.max( [int( counter - 0.5*index_width ), 0]    )
        max_index = np.min(  [len(freq)-1, int( counter + 0.5*index_width ) ]   )  
        if mode == 'MAX':
            smooth_PSD.append(np.max( PSD[min_index:max_index] ) )
        else:
            smooth_PSD.append(np.mean( PSD[min_index:max_index] ) )

    smooth_PSD=np.array(smooth_PSD)
    smooth_freq=np.array(smooth_freq)
    return smooth_freq, smooth_PSD



def fit_two_lorentz(freq, PSD):
    #smooth
    width=10#microHz
    skip = 20 
    smooth_freq, smooth_PSD = smooth(freq, PSD, width, skip, 'MEAN')

    #plt.figure()
    #plt.plot(freq, PSD)
    #plt.plot(smooth_freq, smooth_PSD)
    #plt.title('Smoothed')
    #plt.show()
    #a=input()

    return freq, PSD/np.interp(freq, smooth_freq, smooth_PSD)





def cal_PSD(time, flux):
    ls=LombScargle(time, flux)
    
    freq, power=ls.autopower(nyquist_factor=1,samples_per_peak=10,  normalization='psd')

    negative_index=np.where(power<0)[0]
    power[negative_index]=np.median(power)

    freq, power=freq, (  2*(power/len(time))**0.5   )**2     #amplitude
    PSD = power * (max(time)-min(time))*365.2422*24*60*60*10**6
    freq = freq *10**6/86400
    plt.figure()
    plt.title('raw PSD')
    ax=plt.subplot(111)
    ax.plot(freq, PSD)
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    #plt.show()
    return freq, PSD



def psd_yaguang(x, y, oversampling=1, freqMin=None, freqMax=None, freq=None, return_val="psd_old"):
    """
    Calculate the power spectrum density for a discrete time series.
    https://en.wikipedia.org/wiki/Spectral_density


    Input:
    x: array-like[N,]
        The time array.

    y: array-like[N,]
        The flux array.


    Optional input:
    oversampling: float, default: 1
        The oversampling factor to control the frequency grid.
        The larger the number, the denser the grid.

    freqMin: float, default: frequency resolution

    freqMax: float, default: nyquist frequency


    Output:
    freq: np.array
        The frequency, in unit of [x]^-1.

    psd: np.array
        The power spectrum density, in unit of [y]^2/[x].
        https://en.wikipedia.org/wiki/Spectral_density


    Examples:
    >>> ts = np.load("flux.npy")
    >>> t = ts["time_d"]   # the time in day
    >>> f = ts["flux_mf"]   # the relative flux fluctuated around 1
    >>> f = (f-1)*1e6   # units, from 1 to parts per million (ppm)

    >>> freq, psd = se.psd(t, f, return_val="psd_new")
    >>> freq = freq/(24*3600)*1e6   # c/d to muHz
    >>> psd = psd*(24*3600)*1e-6   # ppm^2/(c/d) to ppm^2/muHz

    """

    if not (return_val in ["psd_old", "periodogram", "power", "amplitude", "psd_new", "window"]):
        raise ValueError("return_val should be one of ['psd_old', 'periodogram', 'power', 'amplitude', 'psd_new', 'window'] ")

    Nx = len(x)
    dx = np.median(x[1:]-x[:-1]) 
    fs = 1.0/dx
    Tobs = dx*len(x)
    fnyq = 0.5*fs
    dfreq = fs/Nx

    if freqMin is None: freqMin = dfreq
    if freqMax is None: freqMax = fnyq

    if freq is None: freq = np.arange(freqMin, freqMax, dfreq/oversampling)
    
    if return_val == "psd_old":
        p = LombScargle(x, y).power(freq, normalization='psd')*dx*4.
    if return_val == "periodogram":
        p = LombScargle(x, y).power(freq, normalization='psd')
    if return_val == "power":
        p = LombScargle(x, y).power(freq, normalization='psd')/Nx*4.
    if return_val == "amplitude":
        p = np.sqrt(LombScargle(x, y).power(freq, normalization='psd')/Nx*4.)
    if return_val == "psd_new":
        nu = 0.5*(freqMin+freqMax)
        freq_window = np.arange(freqMin, freqMax, dfreq/10)
        power_window = LombScargle(x, np.sin(2*np.pi*nu*x)).power(freq_window, normalization="psd")/Nx*4.
        Tobs = 1.0/np.sum(np.median(freq_window[1:]-freq_window[:-1])*power_window)
        p = (LombScargle(x, y).power(freq, normalization='psd')/Nx*4.)*Tobs
    if return_val == "window":
        nu = 0.5*(freqMin+freqMax)
        freq_window = np.arange(freqMin, freqMax, dfreq/10)
        power_window = LombScargle(x, np.sin(2*np.pi*nu*x)).power(freq_window, normalization="psd")/Nx*4.
        freq, p = freq_window-nu, power_window

    return freq/(24*3600)*1e6, p*10**12*(24*3600)*1e-6 #microHz, ppm^2/microHz



def plot_echelle_diagram(freq, PSD, MOD, region):
    print('MOD, region for echelle diagram', MOD, region)
    sample_rate = np.median( freq[1:]-freq[0:-1] )
    smooth_freq, smooth_flat_PSD = smooth(freq, PSD, sample_rate*10, 10, 'mean')
    #get the boundary of each row
    freq_boundary = [0]
    for i in range(1, int(  max(freq)/MOD  )+1  ):
        freq_boundary.append(i*MOD)
        if freq_boundary[-1]>max(freq):
            break
    #print('freq_boundary',freq_boundary)
    #account the number of row
    for i in range(len(freq_boundary)-1):
        if freq_boundary[i] < min( region ) < freq_boundary[i+1]:
                lower_i=i
        if freq_boundary[i] < max( region ) < freq_boundary[i+1]:
                upper_i=i
                
    if 'upper_i' not in locals():#upper_i reference before assignment
        upper_i = len(freq_boundary)-1-1


    how_many_row = upper_i-lower_i+1
    fig = plt.figure(1, figsize = (9, 14))
    ax_list = []
    for i in range(how_many_row):
        local_ax = plt.subplot(how_many_row, 1, how_many_row-i)#plot from down to up
        ax_list.append(local_ax)
    
    counter=0
    for one_ax in ax_list:
        one_ax.plot(freq, PSD, c='k')
        one_ax.plot(smooth_freq, smooth_flat_PSD, c='orange')
        within_index = np.where( (freq > freq_boundary[lower_i+counter]) &  (freq < freq_boundary[lower_i+counter+1])   )[0]
        one_ax.set_ylim([0, max(PSD[within_index])])
        one_ax.set_xlim(freq_boundary[lower_i+counter], freq_boundary[lower_i+counter+1])
        counter+=1   
        one_ax.set_ylabel('S/N')
    plt.title('Freq, $\mu$Hz, with $\Delta \\nu=$'+str(MOD)[0:7]+' $\mu$Hz')
    plt.tight_layout() 
    #plt.show()

    #plt.close()
    return fig, ax_list, freq_boundary

def autocorrelation(freq, flat_PSD, initial_Delta_nu):

    freq_step = np.median(  freq[1:]-freq[0:-1]  )
    Delta_nu_fraction = 0.6 #search best delta nu in initial_Delta_nu +- Delta_nu_fraction/2*initial_Delta_nu
    shift_array = np.arange(0, initial_Delta_nu*Delta_nu_fraction, 0.05)+ initial_Delta_nu - initial_Delta_nu*Delta_nu_fraction/2#microHz
    correlation = shift_array * 0

    #plt.ion()
    
    counter = -1
    for shift in shift_array:
        counter += 1
        shifted_index = int(  shift / freq_step  )
        shifted_PSD = flat_PSD * 0
        #print(len(flat_PSD[0:len(flat_PSD)-shifted_index]), len(shifted_PSD[shifted_index:]))
        shifted_PSD[shifted_index:] = flat_PSD[0:len(flat_PSD)-shifted_index]
        #correlation[counter] = np.correlate(flat_PSD, shifted_PSD)
        correlation[counter] = np.sum(flat_PSD * shifted_PSD)
        #plt.figure()
        #plt.plot(freq, shifted_PSD)
        #a = input()
    #plt.figure()
    #plt.title('Autocorrelation')
    #plt.plot(shift_array, correlation)
    #plt.title('Autocorrelation')
    #plt.show()
    #a=input('autocorrelation')
    best_index = np.argmax(correlation)
    return shift_array[best_index]

def my_gaussian(x, sigma, centre, A):
    return A * np.exp(-(x-centre)**2/2/sigma**2)

def gaussian_likelihood(theta, x, y):
    model = my_gaussian(x, *theta)
    return -0.5*np.sum(  (model-y)**2+np.log(2*np.pi)  )

def from_numax_to_deltanu_and_envolope(numax):
    deltanu = 0.28*numax**0.75
    envolope = 0.59*numax** 0.9
    print('numax, deltanu, envolope', numax, deltanu, envolope, 'printed in the function from_numax_to_deltanu_and_envolope')
    return deltanu, envolope

def find_nu_max(freq, flat_PSD, if_run_correlation_to_find_nu_max, search_range):
    index = np.where( (freq > 20) & (freq<2000))[0]
    cut_freq, cut_flat_PSD = freq[index], flat_PSD[index]
    PSD_median, PSD_std = np.median(cut_flat_PSD), np.std(cut_flat_PSD)
    index = np.where( cut_flat_PSD-PSD_median > 8*PSD_std)[0]
    freq_to_fit = cut_freq[index]
    flat_PSD_to_fit = cut_flat_PSD[index]


    estimated_numax = np.median( freq_to_fit )
    estimated_deltanu, estimated_envolope = from_numax_to_deltanu_and_envolope(estimated_numax)

    #if_run_correlation_to_find_nu_max = True
    if if_run_correlation_to_find_nu_max:
        #sigma = 0.5*estimated_envolope
        if search_range == 9999:
            freq_grid = np.arange(estimated_numax-0.5*estimated_envolope, estimated_numax+0.5*estimated_envolope, 2*(freq[1]-freq[0]) )
        else:
            freq_grid = np.arange(min(search_range), max(search_range), 5*(freq[1]-freq[0]) )

        #used_index = np.where( () & () )[0]
        correlation_product = np.zeros(len(freq_grid))
        counter = -1
        for central_freq in tqdm(freq_grid):
            counter+=1
            area_friction = max(freq)-(central_freq-3*0.5*estimated_envolope)/(6*0.5*estimated_envolope)
            correlation_product[counter] = 1/area_friction * np.sum(flat_PSD * ligang_gaussian(freq, 1, central_freq, estimated_envolope/4))#def ligang_gaussian(x, ampl, center, sigma):#return ampl * np.exp(-(x-center)**2/2/(sigma)**2)
            #plt.figure()
            #plt.plot(freq, flat_PSD)
            #plt.plot(freq_grid, np.zeros(len(freq_grid)), 'ro')
            #plt.plot(freq, 60*ligang_gaussian(freq, 1, central_freq, estimated_envolope/4))
            #a=input()
        #plt.figure()
        #plt.plot(freq_grid, correlation_product)
        #plt.show()
        #a=input()
        max_product_index = np.argmax(correlation_product)
        estimated_numax = freq_grid[max_product_index]
        estimated_deltanu, estimated_envolope = from_numax_to_deltanu_and_envolope(estimated_numax)

    return estimated_numax, estimated_deltanu, estimated_envolope


def run_syd_pipline(time, flux, freq, power, initial_numax, STAR_NAME):
    #try sydney pipeline
    syd_pip_dir = glob.glob('/Users/*gang*/Library/Mobile Documents/com~apple~CloudDocs/solarlikecodes/SYDpy-master')[0]
    syd_pip_output_dir = syd_pip_dir+'/Files/data'
    pure_KIC_number = STAR_NAME.replace('KIC00', '').replace('KIC0', '')
    syd_pip_result_dir = syd_pip_dir+'/Files/results/'+pure_KIC_number

    print('pure KIC number', pure_KIC_number)
    #generate light curve file
    output_array_for_flux = np.zeros([len(time), 2])
    output_array_for_flux[:,0], output_array_for_flux[:,1] = time-min(time), flux
    np.savetxt(syd_pip_output_dir+'/'+pure_KIC_number+'_LC.txt', output_array_for_flux)

    #generate power spectrum file
    if '2997455' in STAR_NAME:
        index = np.where((freq < 9) | (freq > 12.0))[0]
        freq, power = freq[index], power[index]
        index = np.where((freq < 19) | (freq > 22))[0]
        freq, power = freq[index], power[index]

    if '11968514' in STAR_NAME:
        index = np.where((freq < 10.5) | (freq > 12))[0]
        freq, power = freq[index], power[index]

    if '5652071' in STAR_NAME:
        index = np.where((freq < 22) | (freq > 23))[0]
        freq, power = freq[index], power[index]

    #if '8747222' in STAR_NAME:
    #    index = np.where((freq < 95))[0]
    #    freq, power = freq[index], power[index]
        #index = np.where((freq < 70) )[0]
        #freq, power = freq[index], power[index]
    #plt.figure()
    #plt.plot(freq, power)
    #plt.title('PSD to sydpipeline')
    #a=input()
    #plt.show()
    output_array_for_PS = np.zeros([len(freq), 2])
    output_array_for_PS[:,0], output_array_for_PS[:,1] = freq, power
    np.savetxt(syd_pip_output_dir+'/'+pure_KIC_number+'_PS.txt', output_array_for_PS)

    #generate initial guess file
    write_only_mode = False
    if write_only_mode:
        star_info_file_location = glob.glob('/Users/*gang*/Library/Mobile Documents/com~apple~CloudDocs/solarlikecodes/SYDpy-master/Files/')[0]+'star_info.csv'
    else:
        star_info_file_location = glob.glob('/Users/*gang*/Library/Mobile Documents/com~apple~CloudDocs/solarlikecodes/SYDpy-master/Files/')[0]+'star_info_useless.csv'

    if write_only_mode:
        fid=open(star_info_file_location, 'r')
        lines=fid.readlines()
        fid.close()
        if len(lines)<2:
            fid=open(star_info_file_location, 'w')
            fid.write('targets,rad,teff,logg,numax,lowerx,upperx,lowerb,upperb,seed\n')
            fid.close()

        fid=open(star_info_file_location, 'a')
        fid.write(pure_KIC_number+',,,,'+str(initial_numax)+',,,,,\n')
        fid.close()
    else:
        fid=open(star_info_file_location, 'w')
        fid.write('targets,rad,teff,logg,numax,lowerx,upperx,lowerb,upperb,seed\n')
    #if '8702921' in STAR_NAME:
    #    initial_numax = 194.98419851657502
        fid.write(pure_KIC_number+',,,,'+str(initial_numax)+',,,,,')
        fid.close()

    if write_only_mode:
        return 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999

    if len(glob.glob(syd_pip_result_dir+'/background.csv'))==0:
        current_work_dir = os.getcwd()
        os.chdir(syd_pip_dir)
        os.system('python3 syd.py -v -show -target '+pure_KIC_number+'  -mciter 100')
        os.chdir(current_work_dir)

    #read results
    if glob.glob(syd_pip_result_dir+'/background.csv')==0:#there is no result, something wrong happened in sydpipeline
        return 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999

    results = pd.read_csv(syd_pip_result_dir+'/background.csv')
    nu_max, nu_max_err = results['value'][0],results['uncertainty'][0]
    if '5179609' in STAR_NAME:
        nu_max, nu_max_err = 321.84, 1.00#from P. Gaulme 2016
    Delta_nu, Delta_nu_err = results['value'][5],results['uncertainty'][5]
    envolope, envolope_nu_err = results['value'][4],results['uncertainty'][4]
    amplitude, amplitude_err = results['value'][3],results['uncertainty'][3]
    print('from sydpip, nu_max, Delta_nu, envolope, amplitude', nu_max, Delta_nu, envolope, amplitude)

    #judge how many harvey model the code use
    data_length = len(results['value'])
    if data_length == 13:
        harvey_background = harvey_three(freq, results['value'][7], results['value'][8], results['value'][9], results['value'][10], results['value'][11], results['value'][12], results['value'][6])
    if data_length == 11:
        harvey_background = harvey_two(freq, results['value'][7], results['value'][8], results['value'][9], results['value'][10], results['value'][6])
    if data_length == 9:
        harvey_background = harvey_one(freq, results['value'][7], results['value'][8], results['value'][6] )
    #plt.figure()
    #plt.plot(freq, power)
    #plt.plot(freq, harvey_background, label = 'Harvey profile')
    #plt.plot(freq, 8*harvey_background, label = '8*Harvey profile')
    #plt.title('Harvey profile')
    #plt.plot(freq, ligang_gaussian(freq, amplitude, nu_max, envolope))
    #plt.show()
    #a=input()
    return nu_max, nu_max_err, Delta_nu, Delta_nu_err, envolope, envolope_nu_err, amplitude, amplitude_err, freq, harvey_background
    #exit()



def fit_l_equal_0_2_mode_automatically(Delta_nu, nu_max, freq, flat_PSD, echelle_fig, ax_list, freq_boundary, asymptotic_deviation):
    '''
        find l=0 and 2 mode and fit it automatically
        (n+l/2+epsilon+alpha/2*(n-n_max)**2)*Delta_nu - l*(l+1)*D
        alpha=0
        epsilon = 0.02177618*Delta_nu-0.00259194, with scatter 0.033
        D=0.02076451*Delta_nu+0.00879749, with scatter 0.014
        #this fitting results comes from 422 manually-selected stars of Gehan sample   
        small separation: delta_nu_02=0.122*Delta_nu+0.05 citation: Bedding 2010     
    '''

    delta_nu_02=0.122*Delta_nu+0.05

    central_n_p = int( nu_max/Delta_nu )
    search_radial_order_list = [central_n_p, central_n_p+1, central_n_p-1, central_n_p+2, central_n_p-2, central_n_p+3, central_n_p-3]
    search_l_list = [0, 2]

    epsilon = 0.02177618*Delta_nu-0.00259194
    D=0.02076451*Delta_nu+0.00879749
    alpha = 0
#first step, search in large range to detect frequency deviation using collapsed PSD
    counter=-1
    for one_n in search_radial_order_list:
        counter+=1
        print('Now search n=', one_n)
        freq_location = asymptotic_relation_pure_p_mode(one_n, 0, nu_max, Delta_nu, epsilon, alpha, D)+asymptotic_deviation
        print('predicted freq for n=', one_n, 'l=0: ', freq_location)
        #search area
        if counter == 0 :
            search_index = np.where( (freq>freq_location-2*delta_nu_02) & (freq<freq_location+2*delta_nu_02) )[0]
            search_length = len(search_index)
            freq_range = freq[search_index]
            collapse_PSD=flat_PSD[search_index]
            #plt.figure()
            #plt.plot(freq_range, collapse_PSD)
        if counter > 0: 
            new_begin_freq = min(freq_range)+Delta_nu*(one_n-search_radial_order_list[0])
            #print(new_begin_freq)
            begin_index = np.argmin(np.abs(freq-new_begin_freq))
            search_index = np.arange(begin_index, begin_index+search_length)
            if begin_index > len(flat_PSD)-1:
                continue
            elif max(search_index)> len(flat_PSD)-1:
                collapse_PSD = collapse_PSD+np.array( list(flat_PSD[begin_index:])+list( np.zeros(search_length-(len(flat_PSD)-begin_index)) ) )
                continue
            else:
                collapse_PSD = collapse_PSD+flat_PSD[search_index]
                continue
            #plt.plot(freq_range, flat_PSD[search_index])
            

    #do correlation twice, the first time it finds the first peak, the second time it finds the second peak
    #first time
    cross_correlation=[]
    for central_freq in freq_range:
        eta = delta_nu_02/10
        a=2*eta
        template = peak_lorentz(freq_range, a, central_freq, eta, 0)
        cross_correlation.append( np.sum(template*collapse_PSD) )
    first_cross_correlation = cross_correlation
    first_max_index = np.argmax(cross_correlation)
    first_max_freq = freq_range[first_max_index]
    print('first_max_freq', first_max_freq)

    #fit the first peak to remove it
    fit_index = np.where(  (freq_range>first_max_freq-0.5*delta_nu_02) & (freq_range<first_max_freq+0.5*delta_nu_02)  )[0]
    fit_freq, fit_freq_err, eta, all_para, all_para_std, _, _ = fit_a_lorentz_on_one_peak(freq_range[fit_index], collapse_PSD[fit_index], 10, 0.06 )
    fit_lorentz=peak_lorentz(freq_range, *all_para)    
    #plt.figure()
    #plt.plot(freq_range, collapse_PSD)
    #plt.plot(freq_range, fit_lorentz)
    #plt.figure()
    #plt.plot(freq_range,collapse_PSD/fit_lorentz )
    #plt.show()
    #a=input()

    #second time
    cross_correlation=[]
    for central_freq in freq_range:
        eta = delta_nu_02/10
        a=2*eta
        template = peak_lorentz(freq_range, a, central_freq, eta, 0) 
        #plt.figure()
        #plt.plot(freq_range, template)
        #plt.show()
        #a=input()
        cross_correlation.append( np.sum(template*collapse_PSD/fit_lorentz) )
    second_cross_correlation = cross_correlation
    second_max_index = np.argmax(cross_correlation)
    second_max_freq = freq_range[second_max_index]
    print('second_max_freq', second_max_freq)
    print('measured small separation', np.abs( first_max_freq-second_max_freq ), 'predicted value', delta_nu_02)


    #############PLOT############
    if_plot_collapse_echelle=False
    if if_plot_collapse_echelle:
        plt.figure()
        plt.plot(freq_range, collapse_PSD, '-')
        plt.title('collapse PSD')
        plt.figure()
        plt.plot(freq_range, first_cross_correlation)
        plt.plot(freq_range, cross_correlation)
        plt.title('cross correlation')
        for one_max_index in [first_max_index, second_max_index]:
            plt.plot(freq_range[one_max_index], first_cross_correlation[one_max_index], 'ro')
        plt.show()
        a=input()
    #############PLOT############



    if np.abs(np.abs( first_max_freq-second_max_freq ) - delta_nu_02)/ delta_nu_02 >0.2:
        print('WARNING! The measured small separation has large deviation from the predicted one!')
        freq_correction = first_max_freq - (asymptotic_relation_pure_p_mode(search_radial_order_list[0], 0, nu_max, Delta_nu, epsilon, alpha, D) +asymptotic_deviation)
        return [8888], freq_correction, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999   
    l_0_freq = max([first_max_freq, second_max_freq])
    print('l_0_freq', l_0_freq)

    #CORRECTION OF THE ASYMPTOTIC RELATION
    freq_correction = l_0_freq - ( asymptotic_relation_pure_p_mode(search_radial_order_list[0], 0, nu_max, Delta_nu, epsilon, alpha, D) )
    print('freq_correction', freq_correction)

    initial_l_0_freq_list = []
    initial_l_0_radial_order_list = []
    initial_l_2_freq_list = []
    initial_l_2_radial_order_list = []

#cross correlation for each mode
    for one_l in search_l_list:
        for one_n in search_radial_order_list:
            print('Now searching l=', one_l, ', n=', one_n)
            freq_location = asymptotic_relation_pure_p_mode(one_n, one_l, nu_max, Delta_nu, epsilon, alpha, D)+freq_correction
            print('The predicted frequency is', freq_location, 'microHz')
            if freq_location > max( freq ):
                continue
            search_index = np.where( (freq>freq_location-0.7*delta_nu_02) & (freq<freq_location+0.7*delta_nu_02) )[0]
            freq_range = freq[search_index]
            PSD_freq_range = flat_PSD[search_index]
            cross_correlation=[]
            for one_central_freq in freq_range:#do the cross correlation
                eta = delta_nu_02/20
                a=2*eta
                template = peak_lorentz(freq_range, a, one_central_freq, eta, 0)
                cross_correlation.append(  np.sum(template*PSD_freq_range)  )
            best_freq = freq_range[  np.argmax(cross_correlation)  ]
            cross_correlation_noise=np.median(  np.sort(cross_correlation)[0:int(0.5*len(cross_correlation))]  )
            cross_correlation_S_N = max(cross_correlation)/cross_correlation_noise
            print('cross_correlation_S_N', cross_correlation_S_N)
            #plt.figure()
            #plt.plot(freq_range, PSD_freq_range)
            #plt.title('PSD')
            #plt.figure()
            #plt.plot(freq_range, cross_correlation)
            #plt.title('cross correlation')
            #a=input()
            if (one_l == 0 and cross_correlation_S_N<2.7) or (one_l == 2 and cross_correlation_S_N<2.5):#there is no mode
                continue
            if one_l == 0:
                initial_l_0_freq_list.append(best_freq)
                initial_l_0_radial_order_list.append(one_n)
            if one_l == 2:
                initial_l_2_freq_list.append(best_freq)
                initial_l_2_radial_order_list.append(one_n)

    if len(initial_l_0_freq_list)<=2 or len(initial_l_2_freq_list)<1:
        print('WARNING! TOO FEW MODES DETECTED!')
        return [9999], 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999        


#fit a lorentz on them!
    list_l_0_freq = []
    list_l_0_freq_err = []
    list_l_2_freq = []
    list_l_2_freq_err = []

    l_0_all_para =[]
    l_0_all_para_err = []

    l_2_all_para =[]
    l_2_all_para_err = []

    for one_l in search_l_list:
        if one_l ==0:
            freq_to_be_fit = initial_l_0_freq_list
        else:
            freq_to_be_fit = initial_l_2_freq_list

        for one_freq in freq_to_be_fit:
            index_to_be_fit = np.where( (freq>one_freq-0.4*delta_nu_02) & (freq<one_freq+0.4*delta_nu_02) )[0]
            background_guess=1
            eta_guess=0.06
            fit_freq, fit_freq_err, eta, all_para, all_para_std, _, _ = fit_a_lorentz_on_one_peak(freq[index_to_be_fit], flat_PSD[index_to_be_fit], background_guess, eta_guess )
            print('eta',eta)
            if one_l==0:
                list_l_0_freq.append(fit_freq)
                list_l_0_freq_err.append(fit_freq_err)
                l_0_all_para.append(all_para)
                l_0_all_para_err.append(all_para_std)
            if one_l==2:
                list_l_2_freq.append(fit_freq)
                list_l_2_freq_err.append(fit_freq_err)
                l_2_all_para.append(all_para)
                l_2_all_para_err.append(all_para_std)
    initial_l_0_radial_order_list=np.array(initial_l_0_radial_order_list)
    list_l_0_freq=np.array(list_l_0_freq)
    list_l_0_freq_err=np.array(list_l_0_freq_err)

    initial_l_2_radial_order_list=np.array(initial_l_2_radial_order_list)
    list_l_2_freq=np.array(list_l_2_freq)
    list_l_2_freq_err=np.array(list_l_2_freq_err)

    asymp_para, asymp_para_err = fit_asymptotic_using_l_0_and_2(initial_l_0_radial_order_list, list_l_0_freq, np.zeros(len(list_l_0_freq))+0.01, initial_l_2_radial_order_list, list_l_2_freq, np.zeros(len(list_l_2_freq))+0.01, nu_max)
    list_fit_l_0_freq = asymptotic_relation_pure_p_mode(initial_l_0_radial_order_list, 0, nu_max, *asymp_para)
    list_fit_l_2_freq = asymptotic_relation_pure_p_mode(initial_l_2_radial_order_list, 2, nu_max, *asymp_para)


    #plot initial result
    for echelle_ax in ax_list:
        ylim = echelle_ax.get_ylim()
        for one_freq in initial_l_0_freq_list:
            echelle_ax.plot( [one_freq, one_freq],  ylim, 'r-')
            x_range = [one_freq-0.4*delta_nu_02, one_freq+0.4*delta_nu_02]
            echelle_ax.fill_between(x_range, min(ylim), max(ylim), color ='r', alpha = 0.3)
        for one_freq in initial_l_2_freq_list:
            echelle_ax.plot( [one_freq, one_freq],  ylim, 'b-')
            x_range = [one_freq-0.4*delta_nu_02, one_freq+0.4*delta_nu_02]
            echelle_ax.fill_between(x_range, min(ylim), max(ylim), color ='b', alpha = 0.3)

    #plot fit result
    for echelle_ax in ax_list:
        ylim = echelle_ax.get_ylim()
        for fit_l_0_freq in list_fit_l_0_freq:
            echelle_ax.plot( [fit_l_0_freq,fit_l_0_freq],  ylim, 'r:')
        for fit_l_2_freq in list_fit_l_2_freq:
            echelle_ax.plot( [fit_l_2_freq,fit_l_2_freq],  ylim, 'b:')
    #print('pause to see the echelle diagram')
    #a=input()
            #plt.figure()
            #plt.plot(freq_range, PSD_freq_range)
            #plt.plot([best_freq], [0], 'ro')
            #plt.title('PSD range')
            #plt.figure()
            #plt.plot(freq_range, cross_correlation)
            #plt.plot([min(freq_range), max(freq_range)], [cross_correlation_noise, cross_correlation_noise])
            #plt.title('cross correlation')
            #plt.show()
            #a=input()
    return asymp_para, asymp_para_err, list_l_0_freq, list_l_0_freq_err, initial_l_0_radial_order_list, list_l_2_freq, list_l_2_freq_err, initial_l_2_radial_order_list, l_0_all_para, l_0_all_para_err, l_2_all_para, l_2_all_para_err



def fit_l_equal_0_2_mode(Delta_nu, nu_max, freq, flat_PSD, echelle_fig, ax_list, freq_boundary):
    '''
        find l=0 and l=2 mode, and fit them 
    '''

    width = 0.1*Delta_nu

    if_exit = False
    list_l_0_lower=[]#lower boundary of one l=0 mode frequency
    list_l_0_upper=[]
    list_l_0_freq=[]
    list_l_0_freq_err = []
    list_l_0_eta = []

    list_l_2_lower=[]#lower boundary of one l=0 mode frequency
    list_l_2_upper=[]
    list_l_2_freq = []
    list_l_2_freq_err = []
    list_l_2_eta = []

    default_l = 0
    while if_exit != '9999':
        if_exit=input('click on the figure to select lower boundary of l='+str(default_l)+' mode, 8888 to change l, 7777 to delete the last selection, 9999 to exit>>>')
        if if_exit == '9999':
            break
        if if_exit == '8888':
            if default_l == 0:
                default_l = 2
            elif default_l == 2:
                default_l = 0
            print('Now we select the lower boundary of a l='+str(default_l)+' mode>>>')
        if if_exit == '7777':
            if default_l==0:
                print('Before delete',list_l_0_lower, list_l_0_lower)
                print('Delete the last selection of l=0 mode')
                list_l_0_lower.pop(-1)
                list_l_0_upper.pop(-1)
                print('after delete',list_l_0_lower, list_l_0_lower)
            if default_l==2:
                print('Delete the last selection of l=2 mode')
                print('Before delete',list_l_2_lower, list_l_2_lower)
                list_l_2_lower.pop(-1)
                list_l_2_upper.pop(-1)
                print('after delete',list_l_2_lower, list_l_2_lower)
            continue
#for l=0 modes

        if default_l == 0:
            coords=plt.ginput(1)
            list_l_0_lower.append(coords[0][0]-0.5*width)
            #if_exit=input('click on the figure to select upper boundary of l='+str(default_l)+' mode>>>')
            #if if_exit == '9999':
            #   break
            #coords=plt.ginput(1)
            list_l_0_upper.append(list_l_0_lower[-1]+width)
            print('Now the boundary is', list_l_0_lower[-1], list_l_0_upper[-1])
            if list_l_0_lower[-1]>=list_l_0_upper[-1]:#for click the upper boundary first
                list_l_0_lower[-1], list_l_0_upper[-1] = list_l_0_upper[-1], list_l_0_lower[-1]
        #plot shaded area
            for echelle_ax in ax_list:
                y_lim = echelle_ax.get_ylim()
                x_range = [list_l_0_lower[-1], list_l_0_upper[-1]]
                echelle_ax.fill_between(x_range, min(y_lim), max(y_lim), color ='r', alpha = 0.3)


#for l=2 modes

        if default_l==2:
            coords=plt.ginput(1)
            list_l_2_lower.append(coords[0][0]-0.5*width)

            #if_exit=input('click on the figure to select upper boundary of l='+str(default_l)+' mode>>>')
            #if if_exit == '9999':
            #    break
            #coords=plt.ginput(1)
            list_l_2_upper.append(list_l_2_lower[-1]+width)

            print('Now the boundary is', list_l_2_lower[-1], list_l_2_upper[-1])
            if list_l_2_lower[-1]>=list_l_2_upper[-1]:#for click the upper boundary first
                list_l_2_lower[-1], list_l_2_upper[-1] = list_l_2_upper[-1], list_l_2_lower[-1]
            for echelle_ax in ax_list:
                y_lim = echelle_ax.get_ylim()
                x_range = [list_l_2_lower[-1], list_l_2_upper[-1]]
                echelle_ax.fill_between(x_range, min(y_lim), max(y_lim), color ='b', alpha = 0.3)


######delete if there are multiple click on one mode
######Now the code can only delete one point
######a complete code is needed to be finished
    multiple_click_criterion = width 
#for l = 0
    l_0_delete_item = []
    print('The original list is', list_l_0_lower)
    for i in sorted(range(len(list_l_0_lower)), reverse = True  ):
        #print('i', i)
        for j in range(len(list_l_0_lower)):
            #print('j',  j)
            if (np.abs(list_l_0_lower[i]-list_l_0_lower[j]) < multiple_click_criterion) and (i != j) and (i not in l_0_delete_item):
                print('find a multi-click mode in l=0 mode list!')
                l_0_delete_item.append(i)
    print('l_0_delete_item', l_0_delete_item)
    if len(l_0_delete_item)>=1:
        for one_delete_item in l_0_delete_item[0:-1]:
            list_l_0_lower.pop(one_delete_item)
            list_l_0_upper.pop(one_delete_item)
        print('The deleted list is', list_l_0_lower)


#for l = 2
    l_2_delete_item = []
    for i in sorted(range(len(list_l_2_lower)), reverse = True  ):
        #print('i', i)
        for j in range(len(list_l_2_lower)):
            #print('j',  j)
            if np.abs(list_l_2_lower[i]-list_l_2_lower[j]) < multiple_click_criterion and i != j and i not in l_2_delete_item:
                print('find a multi-click mode in l=2 mode list!')
                l_2_delete_item.append(i)
    print('l_2_delete_item', l_2_delete_item)
    if len(l_2_delete_item)>=1:
        for one_delete_item in l_2_delete_item[0:-1]:
            list_l_2_lower.pop(one_delete_item)
            list_l_2_upper.pop(one_delete_item)
        print('The deleted list is', list_l_2_lower) 

####################
##########
##########
#fit l=0 mode
    l_0_all_para, l_0_all_para_err = [], []
    for i in range(len(list_l_0_lower)):
        index_to_be_fit = np.where(  (freq > list_l_0_lower[i])  &  (freq < list_l_0_upper[i])  )[0]
        print('Now we fit a lorentz on that')
        if len(list_l_0_eta)==0:
            eta_guess = 0.05
        else:
            eta_guess = list_l_0_eta[-1]
        background_guess = np.median(flat_PSD)
        l_0_freq, l_0_freq_err, l_0_eta, all_para, all_para_std, _, _ = fit_a_lorentz_on_one_peak(freq[index_to_be_fit], flat_PSD[index_to_be_fit], background_guess, eta_guess )
        list_l_0_freq.append(l_0_freq)
        list_l_0_freq_err.append(l_0_freq_err)
        list_l_0_eta.append(l_0_eta)
        l_0_all_para.append(all_para)
        l_0_all_para_err.append(all_para_std)

        #plt.figure(1)
        for echelle_ax in ax_list:
            #print(echelle_ax)
            ylim = echelle_ax.get_ylim()
            echelle_ax.plot( [list_l_0_freq[-1],list_l_0_freq[-1]],  ylim, 'r')

#fit l=2 mode 
    l_2_all_para, l_2_all_para_err = [], []
    for i in range(len(list_l_2_lower)):
        index_to_be_fit = np.where(  (freq > list_l_2_lower[i])  &  (freq < list_l_2_upper[i])  )[0]
        print('Now we fit a lorentz on that')
        if len(list_l_2_eta)==0 and len(list_l_0_eta)!=0:#use l=0 mode as an estimate
            eta_guess = np.max(list_l_0_eta)
        elif len(list_l_2_eta)==0 and len(list_l_0_eta)==0:
            eta_guess == 0.05
        else:
            eta_guess = list_l_2_eta[-1]
        background_guess = np.median(flat_PSD)
        l_2_freq, l_2_freq_err, l_2_eta, all_para, all_para_std, _, _ = fit_a_lorentz_on_one_peak(freq[index_to_be_fit], flat_PSD[index_to_be_fit], background_guess, eta_guess )
        list_l_2_freq.append(l_2_freq)
        list_l_2_freq_err.append(l_2_freq_err)
        list_l_2_eta.append(l_2_eta)
        l_2_all_para.append(all_para)
        l_2_all_para_err.append(all_para_std)
        #plt.figure(1)
        for echelle_ax in ax_list:
            #print(echelle_ax)
            ylim = echelle_ax.get_ylim()
            echelle_ax.plot( [list_l_2_freq[-1],list_l_2_freq[-1]],  ylim, 'b')

    #sort frequencies
    index = np.argsort(list_l_0_freq)
    list_l_0_freq = np.array(list_l_0_freq)[index]
    list_l_0_freq_err = np.array(list_l_0_freq_err)[index]
    list_l_0_eta = np.array(list_l_0_eta)[index]

    index = np.argsort(list_l_2_freq)
    list_l_2_freq = np.array(list_l_2_freq)[index]
    list_l_2_freq_err = np.array(list_l_2_freq_err)[index]
    list_l_2_eta = np.array(list_l_2_eta)[index]




    #arrange radial order
    list_l_0_n = np.array([])
    for i in range(len(list_l_0_freq)):
        if i == 0:#first one
            list_l_0_n = np.append(list_l_0_n, np.round(list_l_0_freq[i]/Delta_nu)  )
        else:
            list_l_0_n = np.append(list_l_0_n, list_l_0_n[0]+  np.round((list_l_0_freq[i]-list_l_0_freq[0])/Delta_nu)  )
    print('Their radial orders for l=0 modes are', list_l_0_n)
    print('l=0 frequencies are', list_l_0_freq, 'microHz')
    print('l=0 frequency errors are', list_l_0_freq_err, 'microHz')
    list_l_2_n = np.array([])
    #because the radial modes are not necessary to be continuous
    artificial_l_0_n = np.arange(min(list_l_0_n)-2, max(list_l_0_n)+3 )
    #print('artificial_l_0_n', artificial_l_0_n)
    artificial_l_0_freq = min(list_l_0_freq)+( artificial_l_0_n-min(list_l_0_n) )*Delta_nu
    #print('artificial_l_0_freq', artificial_l_0_freq)
    #print('artificial_l_0_n, artificial_l_0_freq', artificial_l_0_n, artificial_l_0_freq)
    for l_2_freq in list_l_2_freq:
        #print('l_2_freq', l_2_freq)
        #print('artificial_l_0_freq', artificial_l_0_freq)
        close_index = np.argmin(np.abs(l_2_freq - artificial_l_0_freq))
        #print(close_index)
        list_l_2_n = np.append(  list_l_2_n, artificial_l_0_n[close_index]-1) #not the closest one, it should have the same n as the l=0 mode on the left.
        
    print('Their radial orders for l=2 modes are', list_l_2_n)
    print('l=2 modes are', list_l_2_freq, 'microHz')
    print('l=2 frequency errors are', list_l_2_freq_err, 'microHz')
    #a=input('pause')

    asymp_para, asymp_para_err = fit_asymptotic_using_l_0_and_2(list_l_0_n, list_l_0_freq, list_l_0_freq_err, list_l_2_n, list_l_2_freq, list_l_2_freq_err, nu_max)
    list_fit_l_0_freq = asymptotic_relation_pure_p_mode(list_l_0_n, 0, nu_max, *asymp_para)
    list_fit_l_2_freq = asymptotic_relation_pure_p_mode(list_l_2_n, 2, nu_max, *asymp_para)
    list_fit_l_1_freq = asymptotic_relation_pure_p_mode(list_l_2_n, 1, nu_max, *asymp_para)+0.6

    for echelle_ax in ax_list:
        ylim = echelle_ax.get_ylim()
        for fit_l_0_freq in list_fit_l_0_freq:
            echelle_ax.plot( [fit_l_0_freq,fit_l_0_freq],  ylim, 'r:')
        for fit_l_2_freq in list_fit_l_2_freq:
            echelle_ax.plot( [fit_l_2_freq,fit_l_2_freq],  ylim, 'b:')

    a = input('pause to look at the figure')
    print('The MCMC result for asymptotic relation fit are', asymp_para)
    return asymp_para, asymp_para_err, list_l_0_freq, list_l_0_freq_err, list_l_0_n, list_l_2_freq, list_l_2_freq_err, list_l_2_n, l_0_all_para, l_0_all_para_err, l_2_all_para, l_2_all_para_err


###############




    #################



    #plt.figure()
    #plt.plot(tau, flat_PSD)
    #plt.xlabel('$\\tau$, s')
    #index = np.where(flat_PSD>20)[0]
    #print('Delta_Pi', Delta_Pi)
    #plt.figure()
    #ax = plt.subplot(111)
    #ax.scatter(tau[index] % Delta_Pi, freq[index], s = flat_PSD[index])
    #ax.scatter(tau[index] % Delta_Pi+Delta_Pi, freq[index], s = flat_PSD[index])
    #y_lim = ax.get_ylim()
    #ax.plot([Delta_Pi, Delta_Pi], y_lim, ':')

    #a = input('pause to look at the stretched echelle diagram>>>')
    #plt.show()


def stretched_echelle_diagram_template(theta, input_x, MOD):#input_x should be a list, or array
    #'''
    #    The stretched echelle diagram is rotated by 90 degrees
    #    so the x-axis is the frequency, while y-axis is the tau mod Delta Pi
    #    theta is slope, intercept, slope, intercept,....
    #'''

    x_total = []
    y_total = []

    for counter in range(0, len(theta), 2):
        this_slope, this_intercept = theta[counter], theta[counter+1]
        output_y = ( this_slope * input_x + this_intercept ) % MOD
        y_total = y_total + list(output_y)
        x_total = x_total + list(input_x)
    #plt.figure()
    #plt.plot(x_total, y_total, '.')
    #plt.show()
    #a=input('pause to see the template')

    return np.array(x_total), np.array( y_total )


def strechted_echelle_diagram_auto_likelihood(theta, x, y, point_amplitude, MOD, x_range):
    model_x, model_y = stretched_echelle_diagram_template(theta, np.array(x), MOD)

    #plt.figure()
    #plt.plot(model_x, model_y, 'bo',  alpha = 0.5)
    #plt.plot(x, y, 'k.')



    how_many_x = len(x)
    how_many_ridges = int( len(theta)/2 )
    distance_sum = 0
    distance_number = 0
    for i in range(len(x)):
        distance_list = np.zeros(how_many_ridges)
        #print('distance_list', distance_list)
        for one_ridge in range(how_many_ridges):
            #print('one_ridge', one_ridge)
            #print('model x', x[i+one_ridge*how_many_x])
            distance_list[one_ridge] = (  y[i] -model_y[int(i+one_ridge*how_many_x)]   )**2

        #print(distance_list)
        min_distance_index = np.argmin(distance_list)
        min_distance=min(distance_list)

        recalculate = False
        if min_distance > 0.2 * MOD and y[i] < 0.1*MOD:#at the lower boundary of the figure, mirror up
            new_y = y[i] + MOD
            recalculate = True
        elif min_distance > 0.2 * MOD and y[i] > 0.9 * MOD:#at the upper boundary of the figure, mirror down
            new_y = y[i]-MOD
            recalculate = True

        if recalculate:
            distance_list = np.zeros(how_many_ridges)
            for one_ridge in range(how_many_ridges):
                distance_list[one_ridge] = (new_y - model_y[int(i + one_ridge * how_many_x)]      )**2
            #continue
        min_distance_index = np.argmin(distance_list)
        min_distance = min(distance_list)

        if min_distance > 0.2 * MOD:#in the middle
            min_distance = MOD


        #distance_number += 1
        distance_sum=distance_sum + min_distance

        #plt.plot(x[i], y[i], 'r.')
        #plt.plot(model_x[i], model_y[ int(i + min_distance_index * how_many_x) ], 'ro', alpha = 0.5)
        #a=input()


    return -1*distance_sum#/distance_number

def strechted_echelle_diagram_auto_prior(theta, MOD, x_range):
    max_slope = MOD / (max(x_range) - min(x_range))
    #check slope
    for i in range(0, len(theta), 2):
        if np.abs(theta[i])>2*max_slope:
            return -np.inf
    #check intercept
    for i in range(1, len(theta), 2):
        if not 0 < theta[i] < MOD:
            return -np.inf
    return 0


def strechted_echelle_diagram_auto_probability(theta, x, y, point_amplitude, MOD, x_range):
    #for i in range(1, len(theta), 2):
    #    theta[i] = theta[i] % MOD
    auto_prior = strechted_echelle_diagram_auto_prior(theta, MOD, x_range)
    if np.isfinite(auto_prior):
        return strechted_echelle_diagram_auto_likelihood(theta, x, y, point_amplitude, MOD, x_range)
    else:
        return -np.inf

def select_mode_from_stretched_echelle_diagram_automatically(x_auto, y_auto, point_amplitude, MOD, x_range, how_many_mode):
    #generate initial guess
    ndim = int(2*how_many_mode)
    nwalkers = int(3*ndim)
    initial_paramter = np.zeros(ndim)

    #add pertubations
    pos = np.zeros( (nwalkers, ndim) )
    #for slope
    max_slope = MOD/(max(x_range) - min(x_range))
    for i in range(0, ndim, 2):
        pos[:, i] = 0.5 * max_slope * np.random.randn(nwalkers)
    #for intercept
    for i in range(1, ndim, 2):
        pos[:, i] = 0.5*MOD + np.abs( 0.1 * MOD * np.random.randn(nwalkers) )

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, strechted_echelle_diagram_auto_probability, args=(x_auto, y_auto, point_amplitude, MOD, x_range))
        sampler.run_mcmc(pos, 3000, progress=True)
        flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    #intercept mod Delta Pi
    print('np.shape(flat_samples)', np.shape(flat_samples))
    for i in range(1, how_many_mode, 2):
        flat_samples[:,i] = flat_samples[:,i]%MOD
    #find the largest likelihood
    likelihood_list = []
    for i in range(len(flat_samples[:, 1])):
        theta_here = flat_samples[i, :]
        likelihood_list.append(strechted_echelle_diagram_auto_likelihood(theta_here, x_auto, y_auto, point_amplitude, MOD, x_range))
    max_index = np.argmax(likelihood_list)
    best_parameters = flat_samples[max_index, :]
    print('best_parameters', best_parameters)
    corner_fig = corner.corner(flat_samples)

    best_fit_x, best_fit_y = stretched_echelle_diagram_template(best_parameters, np.linspace(min(x_auto), max(x_auto), 100), MOD)
    #show fit result
    plt.figure()
    plt.plot(x_auto, y_auto, 'k.')
    plt.plot(best_fit_x, best_fit_y,'bo', alpha = 0.5)
    plt.title('best fit result')
    plt.show()
    a=input('pause to see the best fit result')


    return



def select_ridges_using_collapsed_echelle_diagram(x, y, within_index, Delta_Pi):

    collapse_number = []
    one_Delta_Pi_list = np.linspace(Delta_Pi*0.9, Delta_Pi*1.1, 100)
    for one_Delta_Pi in one_Delta_Pi_list:
        x_MOD = x % one_Delta_Pi
        given_x_mod = x_MOD[within_index]
        x_boundary = [min(given_x_mod), max(given_x_mod)]

        within_index = np.where((x_MOD>x_boundary[0]) & (x_MOD<x_boundary[1]))[0]
        collapse_number.append( len(within_index) )

    new_Delta_Pi = one_Delta_Pi_list[  np.argmax( collapse_number ) ]

    return new_Delta_Pi

def shell_for_stretched_echelle_diagram(freq, flat_PSD, asymp_para, nu_max, envolope, Delta_Pi, q, dipole_freq_shift, echelle_fig, ax_list):

    initial_Delta_Pi=Delta_Pi
    if initial_Delta_Pi>150:
        Delta_Pi_increase = 1
    else:
        Delta_Pi_increase = 0.1
    coefficient=1
    judge=0
    theta_p = 9999

    #Delta_pi_list = np.arange(0.01,60, 0.1)

    Delta_pi_list = np.array(  list(np.arange(60, 110, 0.2)))#change here to modify the searching range
    #Delta_pi_list = np.array(  list(np.arange(60, 100, 0.2)) + list(np.arange(100, 200, 1)) + list(np.arange(200, 350, 3)) )#change here to modify the searching range



    histogram_S_N=np.array([])
    print('Now searching for the best stretched echelle diagram')
    stretched_echelle_significance_file = one_dir + '/' + get_star_name(one_dir)+'_stretched_echelle_significance.dat'
    first_time_for_stretched_echelle_diagram = True
    for Delta_Pi in tqdm(Delta_pi_list):
        if len(glob.glob(stretched_echelle_significance_file))>0:
            first_time_for_stretched_echelle_diagram=False
            break
        tau, stretched_freq, stretched_flat_PSD, theta_p, zeta = calculate_stretched_echelle_diagram(freq, flat_PSD, asymp_para, nu_max, envolope, Delta_Pi, q, dipole_freq_shift, theta_p)
        peak_index = np.where(stretched_flat_PSD>10)[0]        

        if q == 0.15: #RGB
            this_bins = 15
        elif q == 0.3: #RC
            this_bins = 8
        histogram, bin_edges = np.histogram(tau[peak_index] % Delta_Pi, bins=this_bins)
        #histogram_S_N = np.append( histogram_S_N, max(histogram)/np.median(np.sort(histogram)[0:-3]) )#sort the histogram ascend, and remove the largest three points
        histogram_S_N = np.append( histogram_S_N, max(histogram) )#sort the histogram ascend, and remove the largest three points
    
    if first_time_for_stretched_echelle_diagram:
        ascii.write([Delta_pi_list, histogram_S_N], stretched_echelle_significance_file, names = ['Delta Pi in s', 'significance'], overwrite = True)
        return 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999
    
    if not first_time_for_stretched_echelle_diagram:
        significance = np.loadtxt(stretched_echelle_significance_file, skiprows=1)
        Delta_pi_list, histogram_S_N = significance[:, 0], significance[:, 1]



    #manual mode
    #manual mode
    #manual mode
    #manual mode
    #manual mode
    #manual mode
    #manual mode
    #manual mode

    histogram_with_Delta_Pi_figure = plt.figure()
    plt.plot(Delta_pi_list, histogram_S_N)
    plt.xlabel('Delta Pi in seconds')
    plt.ylabel('Significance')
    plt.show()
    a=input('pause to see histogram_with_Delta_Pi_figure, enter to select the Delta Pi>>>')
    #click to select the best result
    best_Delta_Pi = []
    while 1 < 2:
        coords = plt.ginput(1)
        click_Pi = coords[0][0]
        best_Delta_Pi.append(Delta_pi_list[np.argmin(np.abs(click_Pi-Delta_pi_list))])
        judge_Delta_Pi = input('enter to next, 9999 to exit, 8888 to delete the last click>>>')

        if judge_Delta_Pi != '9999' and judge_Delta_Pi != '8888':
            plt.plot([best_Delta_Pi,best_Delta_Pi], [min(histogram_S_N), max(histogram_S_N)], ':')

        if judge_Delta_Pi == '8888':
            best_Delta_Pi.pop(-1)
        if judge_Delta_Pi == '9999':
            break
    #select the best Delta_Pi
    Delta_Pi = np.mean(np.array(best_Delta_Pi))




    selected_mode_list = []
    selected_mode_Delta_tau=[]
    counter = -1
    while judge != '9999' or judge != '8888':
        counter += 1
        print('Delta_Pi', Delta_Pi)
        tau, stretched_freq, stretched_flat_PSD, theta_p, zeta = calculate_stretched_echelle_diagram(freq, flat_PSD, asymp_para, nu_max, envolope, Delta_Pi, q, dipole_freq_shift, theta_p)

        
        peak_index = np.where(stretched_flat_PSD>10)[0]

        tau_module = tau[peak_index] % Delta_Pi
        tau_module = np.array(  list(tau_module)+list(tau_module+Delta_Pi)  )
        used_freq = stretched_freq[peak_index]
        used_freq = np.array( list(used_freq) + list(used_freq)    )
        if counter == 0:#first time, the figure will be saved

            best_stretched_freq = stretched_freq
            best_stretched_flat_PSD = stretched_flat_PSD

            #used for automatic selection
            x_auto, y_auto = stretched_freq[peak_index], tau[peak_index] % Delta_Pi
            point_amplitude = stretched_flat_PSD[peak_index]
            MOD = Delta_Pi
            ####

            first_stretch_echelle_figure = plt.figure(figsize=(5, 7))
            first_stretch_echelle_ax = plt.subplot(111)
            first_stretch_echelle_ax.scatter(tau[peak_index] % Delta_Pi, stretched_freq[peak_index], s = stretched_flat_PSD[peak_index]/5, c = 'k')
            first_stretch_echelle_ax.scatter(tau[peak_index] % Delta_Pi + Delta_Pi, stretched_freq[peak_index], s = stretched_flat_PSD[peak_index]/5, c = 'k')
            y_lim = first_stretch_echelle_ax.get_ylim()
            first_stretch_echelle_ax.set_ylim(y_lim)
            first_stretch_echelle_ax.set_xlim(0, 2*Delta_Pi)
            first_stretch_echelle_ax.plot([Delta_Pi, Delta_Pi], y_lim, 'k:')
            first_stretch_echelle_ax.set_xlabel('$\\tau$ mod '+str(np.round(Delta_Pi, 6))+' s')
            first_stretch_echelle_ax.set_ylabel('Freq, microHz')
            y_lim = first_stretch_echelle_ax.get_ylim()
            x_range = y_lim
        else:#other times, the figure will not be saved
            stretch_echelle_figure = plt.figure(figsize=(5, 7))
            stretch_echelle_ax = plt.subplot(111)
            #used for automatic selection
            x_auto, y_auto = stretched_freq[peak_index], tau[peak_index] % Delta_Pi
            point_amplitude = stretched_flat_PSD[peak_index]
            MOD = Delta_Pi
            x_range = y_lim
            ####
            stretch_echelle_ax.scatter(tau[peak_index] % Delta_Pi, stretched_freq[peak_index], s = stretched_flat_PSD[peak_index]/5, c = 'k')
            stretch_echelle_ax.scatter(tau[peak_index] % Delta_Pi + Delta_Pi, stretched_freq[peak_index], s = stretched_flat_PSD[peak_index]/5, c = 'k')
            y_lim = stretch_echelle_ax.get_ylim()
            stretch_echelle_ax.set_ylim(y_lim)
            stretch_echelle_ax.set_xlim(0, 2*Delta_Pi)
            stretch_echelle_ax.plot([Delta_Pi, Delta_Pi], y_lim, 'k:')
            stretch_echelle_ax.set_xlabel('$\\tau$ mod '+str(np.round(Delta_Pi, 6))+' s')
            stretch_echelle_ax.set_ylabel('Freq, microHz')


        judge = input('enter to continue, \n9999 to exit, \n8888 to exit without a good result, \n7777 to save this Delta Pi, \n6666 to re-plot the figure, \n5555 to select the vertical boundary, \n4444 for RC mode, \n3333 to input Delta Pi, \n2222 to select automatically, \ninput + to plus Delta Delta Pi, input - to minus it>>>')
        if judge == '7777':#this is a good Delta Pi for one vertical ridge
            best_Delta_Pi.append(Delta_Pi)
            print('the Delta Pi that show vertical ridge', best_Delta_Pi)
        if judge == '6666':
            best_Delta_Pi.pop(0)#the first data comes from the S/N diagram, which may not be accurate
            print('the Delta Pi that show vertical ridge', best_Delta_Pi)
            Delta_Pi = np.mean(best_Delta_Pi) #use the mean value to calculate the Delta Pi
            counter = -1
        
        if judge == '5555':
            best_Delta_Pi.append(Delta_Pi)
            print('the Delta Pi that show vertical ridge', best_Delta_Pi)
            print('Now make sure the ridge is vertial')
            select_mode_judge = 0
            tau_module_boundary = []
            while select_mode_judge != '9999':
                print('Now select the boundary')
                coords=plt.ginput(1)
                tau_module_boundary.append(coords[0][0])
                if counter == 0:
                    first_stretch_echelle_ax.plot([tau_module_boundary[-1], tau_module_boundary[-1]],y_lim, 'b:' )
                else:
                    stretch_echelle_ax.plot([tau_module_boundary[-1], tau_module_boundary[-1]],y_lim, 'b:' )

                select_mode_judge = input('Is it ok? enter to the next, 1 to detete, 9999 to exit')
                if select_mode_judge == '1':
                    tau_module_boundary.pop(-1)
            selected_mode_Delta_tau.append(Delta_Pi)
            tau_index = np.where((tau_module>min(tau_module_boundary)) & (tau_module<max(tau_module_boundary)) )[0]
            selected_freq_for_one_mode = used_freq[tau_index]
            #print('selected_freq_for_one_mode', selected_freq_for_one_mode)
            selected_mode_list.append(selected_freq_for_one_mode)

            #print('selected_mode_list', selected_mode_list)
            #print('length of selected_mode_list', len( selected_mode_list ) )
            #if counter == 0:
            #    first_stretch_echelle_ax.plot(tau_module[tau_index], selected_freq_for_one_mode, 'bs' )
            #else:
            #    stretch_echelle_ax.plot(tau_module[tau_index], selected_freq_for_one_mode, 'bs' )

        if judge == '4444':#RC mode
            m_list = []
            selected_mode_list = []
            selected_mode_Delta_tau = []
            best_Delta_Pi = []#should be the same as selected_mode_Delta_tau
            while judge != '9999':
                input_m = input('now input m value you want to select, 9999 to exit, note that negative m is prograde>>>')
                if input_m =='9999':
                    judge == '9999'
                    print('For RC mode:')
                    print('Now Delta_Pi is', Delta_Pi, 'm_list is', m_list)
                    selected_mode_Delta_tau = Delta_Pi + np.array(m_list)*0.1
                    best_Delta_Pi = selected_mode_Delta_tau
                    print('best_Delta_Pi is', best_Delta_Pi)
                    break
                m_list.append(int(input_m))
                this_freq_list = []
                if input_m == '1': this_color, this_marker = 'g', '+'
                if input_m == '0': this_color, this_marker = 'b', 'o'
                if input_m == '-1': this_color, this_marker = 'r', '_'
                click_exit = '1'
                while click_exit != '9999':
                    print('now click the echelle diagram for selecting the m='+input_m+' mode...')
                    coords = echelle_fig.ginput(1)
                    this_freq, this_PSD = coords[0][0], coords[0][1]
                    #this_freq_list.append( this_freq )
                    #width = 0.5
                    #width_index = np.where( (freq > this_freq -0.5*width ) & (freq < this_freq +0.5*width) )[0]
                    close_index = np.argmin(  (this_freq - freq)**2 )
                    this_freq = freq[close_index]
                    this_freq_list.append(this_freq)
                    #plot
                    for one_ax in ax_list:
                        #freq, flat_PSD
                        one_ax.plot(freq[close_index], flat_PSD[close_index], this_color + this_marker, markersize=10, markeredgewidth=2)
                    click_exit = input('enter to select the next, 9999 to switch to the next m value>>>')
                    if click_exit == '9999':
                        selected_mode_list.append(this_freq_list)

        if judge == '3333':
            Delta_Pi = float( input('now input Delta Pi you want>>>') )

        if judge == '2222':
            if_exit = 0
            while if_exit != '9999':
                print('click on the stretched echelle diagram to selece one ridge>>>')
                coords=plt.ginput(1)#first click
                print('click again')
                coords2 = plt.ginput(1)#second click
                print('coords', coords)
                print('coords2', coords2)

                x_boundary = np.sort(  [coords[0][0], coords2[0][0]]  )
                y_boundary = np.sort(  [coords[0][1], coords2[0][1]]  )

                if min( x_boundary ) > Delta_Pi:
                    x_boundary = np.array(x_boundary) - Delta_Pi


                y=stretched_freq[peak_index]
                x = tau[peak_index]%Delta_Pi
                within_index = np.where((x > x_boundary[0]) & (x < x_boundary[1]) & (y>y_boundary[0]) & (y<y_boundary[1]) )[0]
                print('within_index', within_index)
                if counter == 0 :
                    first_stretch_echelle_ax.plot(x[within_index], y[within_index], 'rx')
                else:
                    stretch_echelle_ax.plot(x[within_index], y[within_index], 'rx')

                one_of_the_best_Delta_Pi = select_ridges_using_collapsed_echelle_diagram(tau[peak_index], stretched_freq[peak_index], within_index, Delta_Pi)
                print('one_of_the_best_Delta_Pi', one_of_the_best_Delta_Pi)


        if judge == '+':
            coefficient = 1
        if judge == '-':
            coefficient = -1
        if judge != '6666' and judge != '7777' and judge != '5555' and judge != '9999' and judge != '3333':
            Delta_Pi = Delta_Pi + coefficient * Delta_Pi_increase
        if counter > 0:
            plt.close(stretch_echelle_figure)



        if judge == '8888':#no good discovery
            Delta_Pi = 9999
            best_Delta_Pi = [9999]
            break
    
        if judge == '9999':
            break
    
    return Delta_Pi, best_Delta_Pi, tau, best_stretched_freq, best_stretched_flat_PSD, first_stretch_echelle_figure, selected_mode_list, selected_mode_Delta_tau

def click_on_echelle_diagram_to_select_dipole_multiplets():
    while judge != '9999':
        judge = input('input 9999 to exit>>>')
    return judge

if __name__ == '__main__':

####################################################################################
####################################################################################
####################################################################################
####################################################################################
#The structure of the code is written here
#1, load light curve, smooth, and calculate PSD
#2, use the sydpipeline to get delta nu. (STOP POINT 1), but now the sydpipeline is abandoned
#3, automatically select l=0 and 2 modes to fit asymptotic relation (STOP POINT 2)
#4, automatically calculate the significance of the Delta Pi (STOP POINT 3)
#5, manually select l=1 m=1 -1 0 modes(STOP POINT 4)
#6, calculate grid (STOP POINT 5)
#7, manually select splittings (STOP POINT 6)
#8, fit splittings and splitting-zeta linear relation (STOP POINT 7)
####################################################################################
####################################################################################
####################################################################################
####################################################################################

    YuJie_sample = pd.read_csv('./YuJie_sample_all.dat', sep=';')


    record_file_name = './last_KIC_record.dat'

    dir_list = ['./KIC008684542_MCMC']

#star type
    STAR_TYPE = 'RGB' #'RC' or 'RGB'
    #STAR_TYPE = 'RC'  # 'RC' or 'RGB'

    dir_list=sorted(dir_list, key=lambda name: int(name[name.index('KIC')+3:name.index('KIC')+12]))

    if len(glob.glob(record_file_name))>0:
        last_KIC = open(record_file_name).read().replace('\n', '')
        for start_index in range(len(dir_list)):
            if last_KIC in dir_list[start_index]:
                break
    else:
        for start_index in range(len(dir_list)):
            if '4360072' in dir_list[start_index]:
                break


    start_index = 0
    for one_dir in tqdm(dir_list[start_index:start_index+1]):
        plt.close('all')
        record_file = open(record_file_name, 'w')
        print('\n\n\n\n')
        print('***********************************************')
        print(one_dir)
        print('***********************************************')

        STAR_NAME = get_star_name(one_dir)



        record_file.write(STAR_NAME+'\n')
        record_file.close()

        #pre processing
        if len(glob.glob(one_dir+'/*light_curves.dat'))==0:#no file found
            continue


        if len(glob.glob(one_dir+'/*light_curves_after_process.dat')) == 0:
            exit_for_solar_like=True
            time, flux = MAIN_KEPLER_PROGRAM_no_emcee(one_dir, STAR_NAME, exit_for_solar_like)
        else:
            print('Loading data')
            data = np.loadtxt(glob.glob(one_dir+'/*light_curves_after_process.dat')[0])
            time, flux = data[:,0], data[:,1]


        #continue
        print('mean time step', np.median(np.diff(time)) )
        #a=input('pause')
        if np.median( time[1:]-time[0:-1] )<1/60/24 and False:
            print('bin data')
            time, flux = bin_data_function(time, flux, 5/60/24)
        #plt.figure()
        #plt.plot(time, flux, '.')
        #plt.title('binned flux')





        PSD_filename = one_dir+'/'+STAR_NAME+'_PSD.dat'
        if len(  glob.glob(PSD_filename)  ) > -1:#calculate it everytime
            #freq, PSD = cal_PSD(time, flux)
            freq, PSD = psd_yaguang(time, flux, oversampling=2, freqMin=None, freqMax=None, freq=None, return_val="psd_old")
            ascii.write([freq, PSD], PSD_filename, names=['#Frequency in microHz', 'PSD'], overwrite = True )
        else:
            tmp_data = np.loadtxt(PSD_filename, skiprows = 1)
            freq, PSD = tmp_data[:,0], tmp_data[:, 1]
        #plt.figure(figsize=(10, 5))
        #plt.plot(freq, PSD)
        #plt.title('Raw PSD')
        #plt.show()


        #find background
        flat_PSD_filename = one_dir+'/'+STAR_NAME+'_PSD_flat.dat'
        if len(glob.glob(flat_PSD_filename)) <= 1:#do it everytime
            freq, flat_PSD = fit_two_lorentz(freq, PSD)
            ascii.write([freq, flat_PSD], flat_PSD_filename, names=['#Frequency in microHz', 'PSD'], overwrite = True )
        else:
            tmp_data = np.loadtxt(flat_PSD_filename, skiprows = 1)
            freq, flat_PSD = tmp_data[:,0], tmp_data[:, 1]
        plt.ion()
        plt.figure(figsize=(10, 5))
        ax=plt.subplot(111)
        ax.plot(freq, flat_PSD)
        ax.set_xlabel('Freq, microHz')
        ax.set_ylabel('S/N')
        plt.title('Flattened PSD, initial result')
        #find nu max
        initial_nu_max, initial_Delta_nu, envolope = find_nu_max( freq, flat_PSD, False, 9999 )
        print('Initial nu max without correlation method is', initial_nu_max)
        initial_envolope = envolope
        #confirm that the code find the correct numax
        ylim=ax.get_ylim()
        ax.set_ylim([0, max(ylim)])
        ax.plot([initial_nu_max, initial_nu_max], [0, max(ylim)])
        plt.savefig(one_dir+'/'+STAR_NAME+'_PSD_flat_with_numax.png', dpi=100)

#########
#pause here to see the PSD
#########
        if len(glob.glob(one_dir+'/*initial_nu_max.dat'))==1:
            judge = 1
            nu_max = np.loadtxt(glob.glob(one_dir+'/*initial_nu_max.dat')[0], skiprows=1)
            print('Loaded nu_max is', nu_max)
        else:
            #judge=input('Is the numax correct? Input 1 to continue, 9999 to next star, input anything else to click on the figure>>>')
            judge = '1'
            if judge == '9999':
                continue
            if judge != '1':
                while 2>1:
                    coords=plt.ginput(1)
                    exit_judge = input('1 to exit, else to click again>>>')
                    if exit_judge == '1':
                        break
                nu_max = coords[0][0]
                print('The initial nu max is', nu_max)
            else:
                nu_max = initial_nu_max
            nu_max_fid = open(one_dir+'/'+STAR_NAME+'_initial_nu_max.dat', 'w')
            nu_max_fid.write('#nu max in microHz\n')
            nu_max_fid.write(str(nu_max))
        plt.close()
        #continue

        Delta_nu, initial_envolope = from_numax_to_deltanu_and_envolope(nu_max)
#fit a harvey profile, and update flat_PSD
        do_you_want_to_use_your_own_harvey_fitting=True
        if do_you_want_to_use_your_own_harvey_fitting:
            my_harvey_file = one_dir + '/'+STAR_NAME+'_myself_harvey.dat'
            if len(glob.glob(my_harvey_file))==0:
                harvey_freq, harvey_background = fit_harvey_profile(freq, PSD, nu_max, initial_envolope)
                ascii.write([harvey_freq, harvey_background], my_harvey_file, names=['#freq in microHz', 'harvey background in PSD'], overwrite=True)
            else:
                harvey_data = np.loadtxt(my_harvey_file, skiprows=1)
                harvey_freq, harvey_background = harvey_data[:, 0], harvey_data[:, 1]
        print('interp harvey profile and generate flat_PSD')
        flat_PSD = PSD / np.interp(freq, harvey_freq, harvey_background)

        #calculate numax using correlation again
        if_run_numax_correlation = True
        if if_run_numax_correlation:
            print('Run correlation method to calculate nu max')
            YuJie_index = np.where(YuJie_sample['KIC'] == int( STAR_NAME.replace('KIC', '')  )  )[0]
            if len(YuJie_index) != 0  and nu_max > 190: #when numax is large, my method leads to large deviation
                print('WARNING! numax is too large, use YuJie value')
                initial_nu_max = YuJie_sample['numax'][YuJie_index[0]]
                print('Updated nu max is',initial_nu_max)
                nu_max = initial_nu_max
                #if nu_max > 200: #a problem when obtaining nu_max
                nu_max_fid = open(one_dir + '/' + STAR_NAME + '_nu_max_by_YuJie_because_numax_large.dat', 'w')
                nu_max_fid.write('#nu max in microHz\n')
                nu_max_fid.write(str(nu_max))
                nu_max_fid.close()
            else:
                initial_nu_max, initial_Delta_nu, envolope = find_nu_max(freq, flat_PSD, True, 9999)
                print('Updated nu max is',initial_nu_max)
                nu_max = initial_nu_max
                #if nu_max > 200: #a problem when obtaining nu_max
                nu_max_fid = open(one_dir + '/' + STAR_NAME + '_nu_max_by_me.dat', 'w')
                nu_max_fid.write('#nu max in microHz\n')
                nu_max_fid.write(str(nu_max))
                nu_max_fid.close()
        else:
            if len(glob.glob(one_dir + '/' + STAR_NAME + '_nu_max_by_YuJie_because_numax_large.dat'))>0:
                nu_max = np.loadtxt(one_dir + '/' + STAR_NAME + '_nu_max_by_YuJie_because_numax_large.dat')
            else:
                nu_max = np.loadtxt(one_dir + '/' + STAR_NAME + '_nu_max_by_me.dat')
            Delta_nu, initial_envolope = from_numax_to_deltanu_and_envolope(nu_max)

        #continue

        #calculate Delta_nu and envolope again 
        print('vvvvvvvvvvvvvvvvvvvvvv')
        print('by myself nu_max, Delta_nu, envolope', nu_max, Delta_nu, initial_envolope)
        my_nu_max = nu_max
        #autocorrelation for Delta_nu
        index = np.where((freq>nu_max-initial_envolope) & (freq<nu_max+initial_envolope))[0]
        Delta_nu = autocorrelation(freq[index], flat_PSD[index], Delta_nu)
        my_Delta_nu = Delta_nu
        print('The best delta nu by autocorrelation is', Delta_nu)
        print('^^^^^^^^^^^^^^^^^^^^^^')

        #use the syd pipeline to calculate 
        if not do_you_want_to_use_your_own_harvey_fitting:
            if_run_syd_pipline = True
            if if_run_syd_pipline:
                nu_max, nu_max_err, Delta_nu, Delta_nu_err, envolope, envolope_nu_err, amplitude, amplitude_err, harvey_freq, harvey_background = run_syd_pipline(time, flux, freq, PSD, nu_max, STAR_NAME)
            if nu_max == 9999:  continue#run syd pipeline later
            #save the results from syd pipeline
            syd_outputfile = open(one_dir+'/'+STAR_NAME+'_sydpipeline_result.dat', 'w')
            syd_outputfile.write('nu_max in microHz, nu_max_err, Delta_nu in microHz, Delta_nu_err, envolope in microHz (looks half), envolope_nu_err, amplitude, amplitude_err\n')
            for element in [nu_max, nu_max_err, Delta_nu, Delta_nu_err, envolope, envolope_nu_err, amplitude, amplitude_err]:
                syd_outputfile.write(str(element)+' ')
            syd_outputfile.close()
            ascii.write([harvey_freq, harvey_background], one_dir+'/'+STAR_NAME+'_sydpipeline_harvey.dat', names=['#freq in microHz', 'ppm^2/microHz'], overwrite=True)
            flat_PSD = PSD/np.interp(freq, harvey_freq, harvey_background )
            envolope = 2 * envolope
            #if envolope smaller than 0.6*initial_envolope, still use initial envolope
            if envolope < 0.6*initial_envolope:
                print('WARNING! The envolope given by syd pipeline is wrong!')
                envolope = initial_envolope
            if np.abs( Delta_nu-my_Delta_nu )/my_Delta_nu > 0.1:
                print('WARNING! The Delta_nu given by syd pipeline is wrong!')
                Delta_nu = my_Delta_nu   

            if np.abs(nu_max-my_nu_max)/nu_max > 0.1:
                print('WARNING! The Delta_nu given by syd pipeline is wrong!')
                nu_max = my_nu_max 
#I still want to use my own envolope value
        envolope = initial_envolope


        #is p-mode asymptotic spacing is applied
        if len(glob.glob(one_dir+'/*asymptotic_relation_fit.dat'))==1:
            asymp_para = np.loadtxt( glob.glob(one_dir+'/*asymptotic_relation_fit.dat')[0], skiprows=1 )
            Delta_nu=asymp_para[0, 0]

        if '4774298' in one_dir:
            Delta_nu = 14.63
        #if '3531436' in one_dir:
        #    Delta_nu = 13.5772

        plt.ioff()
        plt.figure(figsize=(10, 5))
        plt.plot(freq, PSD)
        plt.plot(harvey_freq, harvey_background)
        plt.plot([nu_max, nu_max], [0, max(PSD)], ':')
        plt.xlabel('Freq, $\mu$Hz')
        plt.ylabel('PSD, ppm^2/$\mu$Hz')
        plt.savefig(  PSD_filename.replace('.dat', '.png')  )


        plt.figure( figsize=(10, 5) )
        plt.plot(freq, flat_PSD )
        plt.xlabel('Freq, $\mu$Hz')
        plt.ylabel('S/N')
        plt.savefig(  PSD_filename.replace('.dat', '_S_N.png')  )

        plt.close('all')
        #plt.show()


        Delta_nu_step = 0.1

        keyboard_input = ''
        coeff = 1
        MOD = Delta_nu
        if '12833657' in one_dir:
            Delta_nu = 256.7-237.61
            MOD = Delta_nu





        plt.ion()
        print('Now delta nu is', MOD)
        print('Now envolope is', envolope)
        echelle_lower_boundary = max([0, nu_max-0.8*envolope])
        echelle_upper_boundary = min([max(freq)-0.05*Delta_nu, nu_max+0.8*envolope])
        echelle_fig, ax_list, freq_boundary = plot_echelle_diagram(freq, flat_PSD, MOD, [echelle_lower_boundary, echelle_upper_boundary])
        echelle_fig.savefig(one_dir+'/'+STAR_NAME+'_echelle.png', dpi=300)


#################
#################
#################
        #continue here if you only want to run syd pipeline
        #continue
#################
#################
#################

        #try stretched echelle diagram
        Delta_Pi = 76.8  #seconds for 8564976
        epsilon_g = 0.9
        q = 0.15
        dipole_freq_shift = 0.6 #microHz
        Delta_Pi_increase = 0.1
        #fit l=0 and 2 mode

        if len(glob.glob(one_dir+'/*asymptotic_relation_fit.dat'))==0:
            asymptotic_deviation = 0
            if_manual_mode = False
            if not if_manual_mode:
                asymp_para, asymp_para_err, list_l_0_freq, list_l_0_freq_err, list_l_0_n, list_l_2_freq, list_l_2_freq_err, list_l_2_n, l_0_all_para, l_0_all_para_err, l_2_all_para, l_2_all_para_err = fit_l_equal_0_2_mode_automatically(Delta_nu, nu_max, freq, flat_PSD, echelle_fig, ax_list, freq_boundary, asymptotic_deviation)
                if asymp_para[0] == 9999 or asymp_para[0] == 8888:
                    continue
            else:
                asymp_para, asymp_para_err, list_l_0_freq, list_l_0_freq_err, list_l_0_n, list_l_2_freq, list_l_2_freq_err, list_l_2_n, l_0_all_para, l_0_all_para_err, l_2_all_para, l_2_all_para_err = fit_l_equal_0_2_mode(Delta_nu, nu_max, freq, flat_PSD, echelle_fig, ax_list, freq_boundary)

            echelle_fig.savefig(one_dir+'/'+STAR_NAME+'_echelle_with_radial_quadrupole_modes.png', dpi=300)
            ascii.write([asymp_para, asymp_para_err], one_dir+'/'+STAR_NAME+'_asymptotic_relation_fit.dat', names = ['#values for Delta_nu microHz, epsilon, alpha, D', 'error'], overwrite = True)
            sort_index = np.argsort(list_l_0_n)
            ascii.write([list_l_0_freq[sort_index], list_l_0_freq_err[sort_index], list_l_0_n[sort_index]], one_dir+'/'+STAR_NAME+'_splitting_oscillation_l0.dat', names = ['##oscillation frequency in microHz', ' frequency error', 'radial order'], overwrite = True)
            sort_index = np.argsort(list_l_2_n)
            ascii.write([list_l_2_freq[sort_index], list_l_2_freq_err[sort_index], list_l_2_n[sort_index] ], one_dir+'/'+STAR_NAME+'_splitting_oscillation_l2.dat', names = ['##oscillation frequency in microHz', ' frequency error', 'radial order'], overwrite = True)
            l_list_here = [0, 2]
            counter=-1
            for radial_order_list in [list_l_0_n, list_l_2_n]:
                counter+=1
                l_here = l_list_here[counter]
                for index in range( len(radial_order_list) ):
                    output_array=np.zeros((2, 4))
                    if l_here == 0:
                        output_array[0, :] = l_0_all_para[index]
                        output_array[1, :] = l_0_all_para_err[index]
                    else:  
                        output_array[0, :] = l_2_all_para[index]
                        output_array[1, :] = l_2_all_para_err[index]
                    np.savetxt(one_dir+'/'+STAR_NAME+'_l_'+str(l_here)+'_n_'+str(radial_order_list[index])+'_fitting_detail.dat',  output_array, header = '#a_0 in S/N, centre in microHz, eta in microHz, background in S/N')
            plt.close('all')
            #exit
            #continue
        else:
            print('Asymptotic relation for p mode has been fitted, load directly')
            asymp_para = np.loadtxt(glob.glob(one_dir+'/*asymptotic_relation_fit.dat')[0], skiprows = 1)[:, 0]
            #print('asymp_para', asymp_para)

################
################
################
        #continue
#continue here if you only want to fit asymptotic relation to p modes
################
################
################

#try auto-correlation for best p-mode-frequency-shift
#if try_autocorrelation is False, will not run it here
        try_autocorrelation=True
        q=0.15
        if STAR_TYPE == 'RGB':
            q=0.15
        if STAR_TYPE == 'RC':
            q = 0.3
        if try_autocorrelation == True and len(glob.glob(one_dir + '/*_points_for_l_1_m_*dat'))>0:
            #print(os.getcwd())
            points_list =glob.glob(one_dir + '/'+STAR_NAME+'_points_for_l_1_m*dat')
            bin_points_list =glob.glob(one_dir + '/'+STAR_NAME+'_bin_points_for_l_1_m*dat')
            if len(bin_points_list) != 0:#used bin mode
                points_list = bin_points_list

            #print('points_list', points_list)
            m_list = []
            selected_mode_list=[]
            for one_points_file in points_list:
                tmp_data = np.loadtxt(one_points_file, ndmin=1)

                selected_mode_list.append(tmp_data)
                one_m = int(one_points_file[ one_points_file.index('m_')+2: one_points_file.index('.dat')   ])
                m_list.append( one_m )
            
            mode_colour_array, mode_shape_array = [], []
            for one_m in m_list:
                if one_m == -1:
                    mode_colour_array.append('r')
                    mode_shape_array.append('_')
                if one_m == 0:
                    mode_colour_array.append('b')
                    mode_shape_array.append('o')                    
                if one_m == 1:
                    mode_colour_array.append('g')
                    mode_shape_array.append('+')
            #plot on the echelle diagram
            central_n_p = round(nu_max/asymp_para[0])
            n_p = np.arange(central_n_p-3, central_n_p+4)

            #DELETE OUTLIERS IN THE 'selected_mode_list'
            for one_m_index in range(len(m_list)):
                for mode_index in range(len(selected_mode_list[one_m_index])-1, -1, -1):
                    if selected_mode_list[one_m_index][mode_index]<asymptotic_relation_pure_p_mode(min(n_p), 0, nu_max, *asymp_para) or selected_mode_list[one_m_index][mode_index]>asymptotic_relation_pure_p_mode(max(n_p), 0, nu_max, *asymp_para):
                        selected_mode_list[one_m_index] = np.delete(selected_mode_list[one_m_index], mode_index)
                        #print('delete a outlier')

            bin_mode_list = bin_mode_function(selected_mode_list)
            if len(glob.glob(one_dir+'/*bin_points*dat'))==0:
                for bin_mode_index in range(len(bin_mode_list)):#first time, save the bin result
                    bin_mode_filename = one_dir+'/'+STAR_NAME+'_bin_points_for_l_1_m_'+str(int(m_list[bin_mode_index]))+'.dat'
                    np.savetxt(bin_mode_filename, bin_mode_list[bin_mode_index], header='# points in the power spectrum, not mode frequencies, in unit of microHz')
            else:
                bin_mode_list=[]
                for m_list_index in range(len(m_list)):#second time, load the bin result, ignore the before
                    bin_mode_filename = one_dir+'/'+STAR_NAME+'_bin_points_for_l_1_m_'+str(int(m_list[m_list_index]))+'.dat'
                    bin_mode_list.append(  np.loadtxt(bin_mode_filename, ndmin=1)  )




            for i in range(len(bin_mode_list)):#plot on the echelle diagram
                index_in_echelle_diagram = []
                for mode_index in range(len(bin_mode_list[i])):
                    index_in_echelle_diagram.append(np.argmin(np.abs(bin_mode_list[i][mode_index]-freq)))
                for one_echelle_ax in ax_list:
                    one_echelle_ax.plot(freq[index_in_echelle_diagram], flat_PSD[index_in_echelle_diagram],mode_colour_array[i]+mode_shape_array[i], markersize=10, markeredgewidth=2, alpha=0.5)

            #write radial orders on the echelle diagram
            for one_radial_order in range(int(nu_max/asymp_para[0])-3, int(nu_max/asymp_para[0])+4):
                l_0_mode_here = asymptotic_relation_pure_p_mode(one_radial_order, 0, nu_max, *asymp_para)
                for one_echelle_ax in ax_list:
                    y_lim_here = one_echelle_ax.get_ylim()
                    one_echelle_ax.text(l_0_mode_here, 0.9*max(y_lim_here), 'n='+str(one_radial_order))
            
            
            
            #load Delta Pi
            Delta_Pi = np.loadtxt(glob.glob(one_dir+'/'+STAR_NAME+'_best_Delta_P.dat')[0])
            #load core rotation
            if len(glob.glob(one_dir+'/'+STAR_NAME+'_core_rotation.dat'))>0:
                core_cotations_Omega = np.loadtxt(glob.glob(one_dir+'/'+STAR_NAME+'_core_rotation.dat')[0])
            else:
                core_cotations_Omega = 0

            #do the grid here
            best_grid_output_file = one_dir+'/'+STAR_NAME+'_grid_best_parameters.dat'
            Omega_p = 0
            ########################
            ########################grid, not optimisation
            ########################
            if len(glob.glob(best_grid_output_file))==0:
                ########################
                ########################
                ########################
                best_parameters = make_splitting_grid(freq, flat_PSD, selected_mode_list, m_list, n_p, nu_max, asymp_para, Delta_Pi, core_cotations_Omega, Omega_p, STAR_TYPE, one_dir)
                Delta_Pi, q, epsilon_g, one_p_shift, Omega_g, Omega_p = best_parameters
                print('The grid gives the results of Delta_Pi, q, epsilon_g, one_p_shift, Omega_g, Omega_p', Delta_Pi, q, epsilon_g, one_p_shift, Omega_g, Omega_p)
                best_grid_output_file_fid = open(best_grid_output_file, 'w')
                best_grid_output_file_fid.write('#Delta Pi in seconds, mode coupling q, epsilon_g, one_p_shift in microHz, Omega_g, Omega_p in radians microHz\n')
                best_grid_output_file_fid.write(str(Delta_Pi)+' '+str(q)+' '+str(epsilon_g)+' '+str(one_p_shift)+' '+str(Omega_g)+' '+str(Omega_p))
                best_grid_output_file_fid.close()
                plt.close('all')
                #continue
            ###############
            ###############
            ###############
            ###############
            else:
                best_parameters=np.loadtxt(best_grid_output_file)
                if len(best_parameters) == 5:#old code do not consider Omega_p
                    best_parameters = np.append(best_parameters, 0)
                Delta_Pi, q, epsilon_g, one_p_shift, Omega_g, Omega_p = best_parameters
                #Delta_Pi, q, epsilon_g, one_p_shift, Omega_g = 79.0, 0.15, 0.1, 0.5, 0.8*2*np.pi#from sebastien


            print('from grid, Delta_Pi, q, epsilon_g, one_p_shift, Omega_g, Omega_p', Delta_Pi, q, epsilon_g, one_p_shift, Omega_g, Omega_p)
            #a = input('pause to see the autocorrelation result')

            #run a MCMC to optimize the grid
            optimized_grid_output_file = one_dir+'/'+STAR_NAME+'_grid_optimized_parameters.dat'
            ########################
            ########################optimisation
            ########################
            if len(glob.glob(optimized_grid_output_file))==0:########################
                ########################
                ########################
                ########################
                if len(glob.glob(optimized_grid_output_file)) > 0:#use the last fit
                    best_parameters=np.loadtxt(optimized_grid_output_file)[0,:]

                #best_parameters[2] = 0.3

                bin_mode_list = bin_mode_function(selected_mode_list)
                optimized_parameters, optimized_parameters_err = MCMC_optimize_grid(bin_mode_list, m_list, asymp_para, best_parameters, n_p, nu_max, one_dir, STAR_TYPE, STAR_NAME)
                Delta_Pi, q, epsilon_g, one_p_shift, Omega_g, Omega_p = optimized_parameters
                Delta_Pi_err, q_err, epsilon_g_err, one_p_shift_err, Omega_g_err, Omega_p_err = optimized_parameters_err
                optimized_grid_output_file_fid = open(optimized_grid_output_file, 'w')
                optimized_grid_output_file_fid.write('#Optimized Delta Pi in seconds, mode coupling q, epsilon_g, one_p_shift in microHz, Omega_g, Omega_p in radians microHz\n')
                optimized_grid_output_file_fid.write(str(Delta_Pi)+' '+str(q)+' '+str(epsilon_g)+' '+str(one_p_shift)+' '+str(Omega_g)+' '+str(Omega_p)+'\n')
                optimized_grid_output_file_fid.write(str(Delta_Pi_err)+' '+str(q_err)+' '+str(epsilon_g_err)+' '+str(one_p_shift_err)+' '+str(Omega_g_err)+' '+str(Omega_p_err)+'\n')
                optimized_grid_output_file_fid.close()
                plt.close('all')
                #continue
            else:
                best_parameters=np.loadtxt(optimized_grid_output_file)
                Delta_Pi, q, epsilon_g, one_p_shift, Omega_g, Omega_p = best_parameters[0, :]

            #best_parameters=np.loadtxt(best_grid_output_file)
            #Delta_Pi, q, epsilon_g, one_p_shift, Omega_g, Omega_p = best_parameters

            print('from optimized grid, Delta_Pi, q, epsilon_g, one_p_shift, Omega_g, Omega_p', Delta_Pi, q, epsilon_g, one_p_shift, Omega_g, Omega_p)


            #recalculate tau and zeta
            tau, stretched_freq, stretched_flat_PSD, theta_p, zeta = calculate_stretched_echelle_diagram(freq, flat_PSD, asymp_para, nu_max, envolope, Delta_Pi, q, one_p_shift, 9999)
            ascii.write([tau, stretched_freq, stretched_flat_PSD, zeta], one_dir+'/'+STAR_NAME+'_stretched_PSD_updated.dat', names = ['tau in s', 'freq in microHz', 'PSD in S/N', 'zeta'], overwrite=True)

            mixed_mode_frequencies, their_zeta = implicit_asymp_relation_for_mixed_modes(n_p, nu_max, asymp_para, Delta_Pi, epsilon_g, q, one_p_shift)
            #assume the surface rotation is zero, plot splittings on the echelle diagram
            mixed_mode_splittings = their_zeta * Omega_g/2/2/np.pi + Omega_p/2/np.pi * (1-their_zeta)

            #calculate residual of the grid
            #bin modes, one peak may have many points, bin them
            bin_mode_list = bin_mode_function(selected_mode_list)
            residual_of_the_grid = 0
            resitual_list = []
            #calculate grid residuals
            for one_m_index in range(len(m_list)):
                one_m = m_list[one_m_index]
                obs_freq = bin_mode_list[one_m_index]
                cal_freq = mixed_mode_frequencies-one_m*mixed_mode_splittings
                for one_obs_freq in obs_freq:
                    this_residual = np.min(np.abs(one_obs_freq-cal_freq))
                    residual_of_the_grid=residual_of_the_grid+this_residual
                    resitual_list.append(this_residual)
                    if this_residual>0.5:
                        print('warning, a large deviation, m=', one_m, ', obs freq', one_obs_freq)
            print('The total residual of the grid is', residual_of_the_grid)
            print('The median of the deviation between the grid and the observed frequencies is', np.median(resitual_list) )
            ascii.write([mixed_mode_frequencies, their_zeta, mixed_mode_splittings], one_dir + '/'+STAR_NAME+'_best_fit_dipole_freq_from_grid.dat', names=['l=1 freq in microHz','zeta','splittings in microHz'], overwrite=True)
            ascii.write([resitual_list], one_dir + '/'+STAR_NAME+'_deviations_between_grid_and_obs_freq_list.dat', names=['#freq difference in microHz'], overwrite=True)




            for i in range(len(mixed_mode_frequencies)):
                for one_echelle_ax in ax_list:
                    y_lim = one_echelle_ax.get_ylim()
                    x_lim = one_echelle_ax.get_xlim()
                    y_location = (max(y_lim)-0.5*min(y_lim))/(max(x_lim)-min(x_lim))*(mixed_mode_frequencies[i] -min(x_lim))+ 0.5*min(y_lim)
                    one_echelle_ax.plot( [mixed_mode_frequencies[i]-mixed_mode_splittings[i], mixed_mode_frequencies[i], mixed_mode_frequencies[i]+mixed_mode_splittings[i]], [y_location, y_location, y_location], 'r.-'   )

            plt.ioff()
            echelle_fig.savefig(one_dir+'/'+STAR_NAME+'_echelle_with_grid_splittings.png', dpi=300)

            #a=input('pause')

#####################################################
#####################################################
            #a=input('pause to see the bin result')
            #continue#continue here if you do not want to run splitting fit
#####################################################
#####################################################

            #a=input('pause to see the echelle diagram>>>')
            #if len(glob.glob(best_grid_output_file))==0:
            #    continue
                #continue here because selecting splittings will be done in the next step


            splitting_work_dir = one_dir+'/solar_like_splittings'

            manually_select_mode = False
            if manually_select_mode:
                if len(glob.glob(splitting_work_dir+'/splitting_boundary.dat')) == 0:
                    First_time_splitting = True
                else:
                    First_time_splitting = False 

                if First_time_splitting:
                    plt.ion()
                    manually_select_splitting_shell(freq, flat_PSD, echelle_fig, ax_list, one_dir)
                    fit_splitting_EMCEE_shell(splitting_work_dir)
                    #continue
            else:
                #if len( glob.glob(splitting_work_dir) )>0:
                #    os.system('rm -rf '+splitting_work_dir)

                if len(glob.glob(splitting_work_dir)) == 0:
                    os.mkdir(splitting_work_dir)

                #bin_mode_list = bin_mode_function(selected_mode_list)

                try:
                    useless = fit_multi_splittings_automatically_shell(freq, flat_PSD, bin_mode_list, m_list, mixed_mode_frequencies, their_zeta, mixed_mode_splittings, nu_max, asymp_para, Delta_Pi, tau, stretched_freq, stretched_flat_PSD, splitting_work_dir, STAR_TYPE)
                except:
                    print('something wrong when fitting the splittings')
            #try:
            if len(m_list) > 1 and len(glob.glob(splitting_work_dir+'/fit_result_No*.dat'))>0:
                fit_splitting_vs_zeta(one_dir)
            elif len(m_list) ==1:
                print('no splittings, no need to fit rotations')
            elif len(glob.glob(splitting_work_dir+'/fit_result_No*.dat'))==0:
                print('no splittings in the directory')
            #except:
            #    print('something wrong when fitting spliting-zeta relation')
            #a=input('pause to see fitting result')
            plt.close('all')
            plot_stretched_echelle_diagram(one_dir)
            #print('finish one star, sleep 10 seconds')
            #tm.sleep(10)
            continue





































#stretched echelle diagram
#Delta_Pi_list gives the Delta Pi which has the largest S/N in the collapsed stretched echelle diagram. We can use Delta_Pi_list to calculate the core rotation
        if STAR_TYPE == 'RGB':
            q = 0.15
        if STAR_TYPE == 'RC':
            q = 0.3
        Delta_Pi, Delta_Pi_list, tau, stretched_freq, stretched_flat_PSD, stretch_echelle_figure, selected_mode_list, selected_mode_Delta_tau= shell_for_stretched_echelle_diagram(freq, flat_PSD, asymp_para, nu_max, envolope, Delta_Pi, q, dipole_freq_shift, echelle_fig, ax_list)
        if Delta_Pi == 9999:#only calculate the histogram significance
            continue
        fid=open(one_dir+'/'+STAR_NAME+'_best_Delta_P.dat', 'w')
        fid.write('#The best Delta Pi in seconds \n')
        fid.write(str(Delta_Pi))
        fid.close()
        np.savetxt(one_dir+'/'+STAR_NAME+'_best_Delta_P_list.dat', Delta_Pi_list, header='#These numbers give the delta tau in seconds which can be used to calculate the splittings')
        stretch_echelle_figure.savefig(one_dir+'/'+STAR_NAME+'_stretched_echelle_diagram.png', dpi = 300)
        ascii.write([tau, stretched_freq, stretched_flat_PSD], one_dir+'/'+STAR_NAME+'_stretched_PSD.dat', names=['#tau in s', 'freq in microHz', 'power in S/N'], overwrite = True)
        #plt.close('all')


#calculate splittings and core rotations
        Delta_Pi_list = sorted(Delta_Pi_list)
        if len(Delta_Pi_list) == 2:
            m_list_for_delta_Pi = np.array([-1, 1])
        if len(Delta_Pi_list) == 3:
            Delta_Pi_list.pop(1)#m=0 mode doesn't contribute
            m_list_for_delta_Pi = np.array([-1, 1])
        if len(Delta_Pi_list)>1:
            mixed_mode_density = ( Delta_nu*10**(-6) )/Delta_Pi/( nu_max*10**(-6) )**2
            splittings_nu= np.mean(  (np.array(Delta_Pi_list)/np.mean(Delta_Pi_list)-1)/2/np.array(m_list_for_delta_Pi)/mixed_mode_density*(mixed_mode_density+1)*nu_max  )#in unit of microHz
            core_cotations_Omega = 2*splittings_nu*2*np.pi # in unit of radians microHz
            fid = open(one_dir+'/'+STAR_NAME+'_core_rotation.dat', 'w')
            fid.write('#mean core rotation in radians microHz\n')
            fid.write(str(core_cotations_Omega))
            fid.close()
        
        if len(Delta_Pi_list)==1:
            print('No splitting identified')
            core_cotations_Omega = 0
        else:
            print('Core rotation from the stretched echelle diagram', core_cotations_Omega, 'radians microHz')

#plot the selected modes on the echelle diagram
#negative m is prograde!!!
#negative m is prograde!!!
#negative m is prograde!!!
#negative m is prograde!!!
#negative m is prograde!!!
#negative m is prograde!!!
        sorted_index = np.argsort(selected_mode_Delta_tau)#Delta tau from small to large, azimuthal order m from -1 to 0 to 1
        #selected_mode_list = selected_mode_list[sorted_index]
        m_list = list(selected_mode_Delta_tau)
        mode_colour_array=list(selected_mode_Delta_tau)
        mode_shape_array=list(selected_mode_Delta_tau)
        i_counter = -1
        for i in sorted_index:
            i_counter += 1
            if len(sorted_index) == 1:
                mode_colour_array[i] = 'b'
                mode_shape_array[i] = 'o'
                m_list[i] = 0
            elif i_counter == 0:
                mode_colour_array[i] = 'r'
                mode_shape_array[i] = '_'
                m_list[i]=-1
            elif i_counter == 1 and len(sorted_index)==2:
                mode_colour_array[i] = 'g'
                mode_shape_array[i] = '+'
                m_list[i] = 1
            elif i_counter == 1 and len(sorted_index)==3:
                mode_colour_array[i] = 'b'
                mode_shape_array[i] = 'o'
                m_list[i] = 0
            elif i_counter == 2:
                mode_colour_array[i] = 'g'
                mode_shape_array[i] = '+'
                m_list[i] = 1

        print('m_list', m_list)
        print('selected_mode_Delta_tau',selected_mode_Delta_tau)
        print('sorted_index of Delta tau', sorted_index)
        #save mixed mode frequencies
        for i in range(len(selected_mode_list)):
            np.savetxt(one_dir+'/'+STAR_NAME+'_points_for_l_1_m_'+str(m_list[i])+'.dat', selected_mode_list[i], header='points in the power spectrum, not mode frequencies, in unit of microHz')



        for i in range(len(selected_mode_list)):
            index_in_echelle_diagram = []
            for mode_index in range(len(selected_mode_list[i])):
                index_in_echelle_diagram.append(np.argmin(np.abs(selected_mode_list[i][mode_index]-freq)))
            for one_echelle_ax in ax_list:
                one_echelle_ax.plot(freq[index_in_echelle_diagram], flat_PSD[index_in_echelle_diagram],mode_colour_array[i]+mode_shape_array[i], markersize=10, markeredgewidth=2)


        a=input('pause to see the echelle diagram, do you see the markers?>>>')
        continue
        #continue here because the next fitting will be doing later



        #useless below
        #useless below
        #useless below
        #useless below
        #useless below
        #useless below

        print('Now calculate the asymptotic spacings and splittings...')
        central_n_p = round(nu_max/asymp_para[0])
        print('central_n_p', central_n_p)
        n_p = np.arange(central_n_p-2, central_n_p+3)
        epsilon_g, q, p_mode_freq_shift = 0.25, 0.15, 0.6
        #optimize the parameters here
        Omega_p = 0
        best_parameters = make_splitting_grid(selected_mode_list, m_list, n_p, nu_max, asymp_para, Delta_Pi, core_cotations_Omega, Omega_p)
        optimized_Delta_Pi, q, epsilon_g, one_p_shift, optimized_Omega_g = best_parameters
        #optimized_Delta_Pi, q, epsilon_g, one_p_shift, optimized_Omega_g = 80.9, 0.14, 0.4, 0.6, 0.8*2*np.pi

        mixed_mode_frequencies, their_zeta = implicit_asymp_relation_for_mixed_modes(n_p, nu_max, asymp_para, optimized_Delta_Pi, epsilon_g, q, one_p_shift)

        #assume the surface rotation is zero, plot splittings on the echelle diagram
        mixed_mode_splittings = their_zeta * optimized_Omega_g/2/2/np.pi
        for i in range(len(mixed_mode_frequencies)):
            for one_echelle_ax in ax_list:
                y_lim = one_echelle_ax.get_ylim()
                x_lim = one_echelle_ax.get_xlim()
                y_location = (max(y_lim)-0.5*min(y_lim))/(max(x_lim)-min(x_lim))*(mixed_mode_frequencies[i] -min(x_lim))+ 0.5*min(y_lim)
                one_echelle_ax.plot( [mixed_mode_frequencies[i]-mixed_mode_splittings[i], mixed_mode_frequencies[i], mixed_mode_frequencies[i]+mixed_mode_splittings[i]], [y_location, y_location, y_location], 'r.-'   )
        a=input('pause to see the echelle diagram>>>')


#use stretched echelle diagram to do mode identification
        #mode_identification_stretched_echelle_diagram(freq, flat_PSD, asymp_para, nu_max, envolope, Delta_Pi, q, dipole_freq_shift, theta_p)