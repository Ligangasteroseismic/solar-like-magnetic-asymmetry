import numpy as np, matplotlib.pyplot as plt
import emcee, corner
from multiprocessing import Pool
import os, glob, sys
from astropy.io import ascii
from multiprocessing import Pool

#########################
#########################
#########################
#########################
#########################
#fit a single lorentz peak
def zeta_function(nu, q, theta_p, Delta_nu, Delta_Pi):
    zeta = 1+1/(  q*(np.cos(theta_p))**2+1/q*(np.sin(theta_p))**2     )*(nu*10**(-6))**2*Delta_Pi/(Delta_nu*10**(-6))
    zeta = 1/zeta
    return zeta


def peak_lorentz(freq, a_0, centre_freq, eta, background): #JCD notebook Page 19
    '''
        This equation comes from page 19 of JCD's notebook
    '''
    return 0.25*a_0**2/( (freq-centre_freq)**2 + eta**2) + background

def log_likelihood(theta, freq, PSD):
    '''
        this likelihood comes from Li Yaguang 2020
    '''
    Model = peak_lorentz(freq, *theta)
    return -np.sum( np.log(Model) + PSD/Model   ) 


def log_prior(theta, freq, PSD):
    a_0, centre_freq, eta, background = theta
    #print('a_0, centre_freq, eta, background', a_0, centre_freq, eta, background)
    #print('a_0**2/4/eta**2', a_0**2/4/eta**2)
    #if eta<0:
    #    return -np.inf
    if a_0<=0:
        #print('a_0 out of range') 
        return -np.inf 
    if a_0**2/4/eta**2>max(PSD):
        return -np.inf
    if eta<=0 or eta > 0.2:
        #print('eta out of range')
        return -np.inf 
    if background<=0 or background > max(PSD):
        #print('background out of range')
        return -np.inf
    if np.abs(centre_freq-freq[np.argmax(PSD)])>0.1:
        #print('centre_freq out of range')
        return -np.inf 
    return 0

def log_probability(theta, freq, PSD):
    prior = log_prior(theta, freq, PSD)
    if prior == -np.inf:
        #print('some parameters are out of the range', theta)
        return -np.inf
    prob = prior + 2*log_likelihood(theta, freq, PSD)
    return prob

def fit_a_lorentz_on_one_peak(freq, PSD, input_background, eta_guess):

    #find initial value
    initial_eta = np.std(freq[  np.where(PSD>4*input_background)[0]   ])
    if initial_eta == 0:
        initial_eta = 0.2*(max(freq)-min(freq))
    if eta_guess != 0.1:
        initial_eta = eta_guess
    initial_a_0 = (  4*0.6*np.max(PSD)*initial_eta**2  )**0.5
    initial_centre_freq = np.median(freq)#freq[ np.argmax(PSD) ]
    initial_background = input_background

    print('initial_eta, initial_a_0, initial_centre_freq, initial_background', initial_eta, initial_a_0, initial_centre_freq, initial_background)
    #a=input('pause>>>')


    #initial guess
    initial_theta = np.array([initial_a_0, initial_centre_freq, initial_eta, initial_background])

    #generate pos
    number_of_parallel_chains=30
    pos = np.zeros((number_of_parallel_chains, 4))
    i=0
    while i <number_of_parallel_chains:
        #print(i)
        this_eta = initial_eta+ (0.03)* np.random.normal(loc=0, scale=1, size = 1)
        pos[i, 2]=this_eta
        this_a_0 = initial_a_0 + 0.1*initial_a_0*np.random.normal(loc=0, scale=1, size = 1)
        this_background = initial_background + 0.1*initial_background*np.random.normal(loc=0, scale=1, size = 1)
        this_centre_freq = initial_centre_freq+0.1*np.random.normal(loc=0, scale=1, size = 1)
        pos[i, 0], pos[i, 1], pos[i, 2], pos[i, 3] = this_a_0, this_centre_freq, this_eta, this_background
        if log_prior(pos[i, :], freq, PSD) == 0:
            i = i+1
        #else:
            #print('Generated data not suitable, try again...')


   # print(pos[:,3])
    #a=input('look at the pos')
    #print('pos shape', np.shape(pos))
    nwalkers, ndim = pos.shape
    with Pool() as pool:
        sampler = emcee.EnsembleSampler( nwalkers, ndim, log_probability, args=(freq, PSD) )
        sampler.run_mcmc(pos, 2000, progress=True)
    #tau = sampler.get_autocorr_time()
    flat_samples = sampler.get_chain(discard=100, thin=100, flat=True)
    MCMC_result = np.median( flat_samples , axis = 0)
    MCMC_std = np.std(flat_samples , axis = 0)
    print('MCMC result of a_0, centre, eta, background are', MCMC_result)
    print('And their errors are', MCMC_std)
    corner_fig = corner.corner( flat_samples, labels=['a_0', 'center freq', 'eta', 'wn'] , quantiles=[0.16, 0.5, 0.84], show_titles=True, truths=initial_theta   )
    #plt.show()
    #a=input('pause to see the corner')

    fit_result_fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(freq, PSD)
    ax.plot(freq, peak_lorentz(freq,initial_a_0, initial_centre_freq, initial_eta, initial_background), label = 'initial guess')
    ax.plot(freq, peak_lorentz(freq, *MCMC_result), label = 'MCMC result')
    plt.legend()
    #plt.show()
    #a=input()
    #plt.close(corner_fig)
    #plt.close(fit_result_fig)
    return MCMC_result[1], MCMC_std[1], MCMC_result[2], MCMC_result, MCMC_std, corner_fig, fit_result_fig



################################
################################
################################
################################
################################
################################
################################
################################
#fit asymptotic relation for p modes
def asymptotic_relation_pure_p_mode(n, l, nu_max, Delta_nu, epsilon, alpha, D):
    n_max = nu_max/Delta_nu-epsilon
    #print(n_max)
    return (n+l/2+epsilon+alpha/2*(n-n_max)**2)*Delta_nu - l*(l+1)*D

def log_likelihood_for_fitting_asymptotic(theta, list_l_0_n, list_l_0_freq, list_l_0_freq_err, list_l_2_n, list_l_2_freq, list_l_2_freq_err, nu_max):
    #we still prefer a uniform-error fit
    #use the mean value
    list_l_0_freq_err = np.zeros(len(list_l_0_freq_err)) + np.mean(list(list_l_0_freq_err)+list(list_l_2_freq_err))
    list_l_2_freq_err = np.zeros(len(list_l_2_freq_err)) + np.mean(list_l_0_freq_err)
    
    #l=0 first
    l_0_theo_freq = asymptotic_relation_pure_p_mode(list_l_0_n, 0, nu_max, *theta)
    chi_square=-0.5*np.sum(  (l_0_theo_freq-list_l_0_freq)**2/list_l_0_freq_err**2  )

    #l=2 then
    l_2_theo_freq = asymptotic_relation_pure_p_mode(list_l_2_n, 2, nu_max,  *theta)
    chi_square= chi_square -0.5*np.sum(  (l_2_theo_freq-list_l_2_freq)**2/list_l_2_freq_err**2  ) 
    return chi_square

def prior_for_fitting_asymptotic(theta):
    Delta_nu, epsilon, alpha, D = theta
    #if 6*D>Delta_nu:return -np.inf 
    return 0

def log_prob_for_fitting_asymptotic(theta, list_l_0_n, list_l_0_freq, list_l_0_freq_err, list_l_2_n, list_l_2_freq, list_l_2_freq_err, nu_max):

    prior = prior_for_fitting_asymptotic(theta)
    if prior == -np.inf:
        return -np.inf
    else:
        return prior+log_likelihood_for_fitting_asymptotic(theta, list_l_0_n, list_l_0_freq, list_l_0_freq_err, list_l_2_n, list_l_2_freq, list_l_2_freq_err, nu_max)




def fit_asymptotic_using_l_0_and_2(list_l_0_n, list_l_0_freq, list_l_0_freq_err, list_l_2_n, list_l_2_freq, list_l_2_freq_err, nu_max):
    '''
        fit Delta nu, second order term, and small separation
        nu = [ n+l/2+epsilon+alpha/2*(n-n_max)**2 ]*Delta_nu-l(l+1)*D
        n_max = nu_max/Delta_nu - epsilon
    '''
    #guess Delta nu
    para = np.polyfit(list_l_0_n, list_l_0_freq, 1)
    initial_Delta_nu = para[0]
    #guess small saperation D
    l_0_index = -1
    for l_0_n in list_l_0_n:
        l_0_index+=1
        l_2_index = np.where(l_0_n == list_l_2_n)[0]
        if len( l_2_index )!=0:
            break
    #with the same n, nu_{l=2}-nu_{l=0}=Delta_nu - 6*D
    print('initial_Delta_nu')
    print('list_l_2_freq', list_l_2_freq)
    print('l_2_index', l_2_index)
    initial_D = ( initial_Delta_nu-list_l_2_freq[l_2_index]+list_l_0_freq[l_0_index]  )/6
    print('initial_D', initial_D)
    #a=input('pause to see l_2_index')
    try:
        initial_D = initial_D[0]
    except:
        print('in fit_package.py, initial_D = initial_D[0] goes wrong!!!')
        initial_D = initial_D

    print('initial_Delta_nu, initial_D', initial_Delta_nu, initial_D )
    pos = np.array([initial_Delta_nu, 0.15, 0.1, initial_D]) + 1e-4 * np.array([initial_Delta_nu, 0.5, 0.1, initial_D]) * np.random.randn(32, 4)
    nwalkers, ndim = pos.shape
    with Pool() as pool:  # multiprocessing
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_for_fitting_asymptotic, args=(list_l_0_n, list_l_0_freq, list_l_0_freq_err, list_l_2_n, list_l_2_freq, list_l_2_freq_err, nu_max), pool=pool)
        sampler.run_mcmc(pos, 2000, progress=True)

    flat_samples = sampler.get_chain(discard=400, thin=15, flat=True)
    MCMC_result = np.median(flat_samples, axis = 0)
    MCMC_std = np.std(flat_samples, axis = 0)

    fig = corner.corner(flat_samples, labels = ['Delta nu', 'epsilon', 'alpha', 'D'])

    tmp_fig = plt.figure()
    #l=0 first
    l_0_counter = -1
    for l_0_freq in list_l_0_freq:
        l_0_counter+=1
        plt.plot([l_0_freq, l_0_freq],  [0, 1], 'r')
        fitted_l_0_freq = asymptotic_relation_pure_p_mode(list_l_0_n[l_0_counter], 0, nu_max, *MCMC_result)
        plt.plot([fitted_l_0_freq, fitted_l_0_freq],  [0, 1], 'r:')

    l_2_counter = -1
    for l_2_freq in list_l_2_freq:
        l_2_counter+=1
        plt.plot([l_2_freq, l_2_freq],  [0, 1], 'b')
        fitted_l_2_freq = asymptotic_relation_pure_p_mode(list_l_2_n[l_2_counter], 2, nu_max, *MCMC_result)
        plt.plot([fitted_l_2_freq, fitted_l_2_freq],  [0, 1], 'b:')
    #plt.show()
    #a = input('enter to continue>>>')
    plt.close(fig)
    plt.close(tmp_fig)
    
    return MCMC_result, MCMC_std



###############################
#####g modes and stretched echelle diagram######
###############################
###############################


def theta_p_function(nu, nu_p):
    fast_algorithm = False

    if not fast_algorithm:#slow algorithm
        theta_p = np.zeros(len(nu))
        for i in range(len(nu)):
            close_index = np.argmin( np.abs(nu[i]-nu_p) )
            if nu[i] < nu_p[close_index]:
                local_delta_nu = nu_p[close_index] - nu_p[close_index-1]
            else:
                if close_index+1 == len(nu_p):#reach boundary
                    local_delta_nu = nu_p[close_index] - nu_p[close_index-1]
                else:
                    local_delta_nu = nu_p[close_index+1] - nu_p[close_index]
            #print('close_index', close_index)
            theta_p[i] = np.pi/local_delta_nu*(nu[i]-nu_p[close_index])
            #print(theta_p[i], nu[i]-nu_p[close_index])
    else:#fast algorithm
        #print('nu', min(nu), max(nu))
        #print('nu_p', nu_p)
        theta_p = []
        #find boundary
        for i in range(0, len(nu_p)):
            if nu_p[i]< min(nu) and nu_p[i+1]> min(nu):
                lower_i = i
            if nu_p[i] < max(nu) and nu_p[i+1]> max(nu):
                upper_i = i+1
                break
        #print('lower_i, upper_i', lower_i, upper_i)
        for i in np.arange(lower_i, upper_i):
            this_nu_p = nu_p[i]
            last_nu_p = nu_p[i-1]
            next_nu_p = nu_p[i+1]
            last_delta_nu = this_nu_p - last_nu_p
            next_delta_nu = next_nu_p - this_nu_p
            #print('last_nu_p, this_nu_p,next_nu_p', last_nu_p, this_nu_p,next_nu_p)
            index = np.where((nu>this_nu_p - 0.5*last_delta_nu) & (nu <= this_nu_p))[0]
            if len(index)!=0:
                theta_p_segment = np.pi*(  nu[index]-this_nu_p  )/last_delta_nu
                theta_p = theta_p+list(theta_p_segment)
            index = np.where( (nu > this_nu_p) & (nu<=this_nu_p + 0.5 * next_delta_nu) )[0]
            if len(index)!=0:
                theta_p_segment = np.pi*(  nu[index]-this_nu_p  )/next_delta_nu
                theta_p = theta_p+list(theta_p_segment)
        theta_p=np.array(theta_p)

    #plt.figure()
    #plt.plot(nu, theta_p, '.')
    #plt.xlabel('freq')
    #plt.ylabel('theta_p')
    #plt.show()
    #a=input('pause to see theta_p')

    return theta_p




def calculate_stretched_echelle_diagram(freq, flat_PSD, asymp_para, nu_max, envolope, Delta_Pi, q, dipole_freq_shift, theta_p):
    pure_p_mode_n = np.arange(0, 100)
    pure_p_mode = asymptotic_relation_pure_p_mode(pure_p_mode_n, 1, nu_max, *asymp_para)+dipole_freq_shift
    radial_mode = asymptotic_relation_pure_p_mode(pure_p_mode_n, 0, nu_max, *asymp_para)
    quadrupole_mode = asymptotic_relation_pure_p_mode(pure_p_mode_n, 2, nu_max, *asymp_para)

    index = np.where((freq>nu_max-envolope*1.2) & (freq<nu_max+envolope*1.2))[0]
    freq, flat_PSD = freq[index], flat_PSD[index]

    Delta_nu = asymp_para[0]
    #calculate theta_p
    if type(theta_p) == int:
        #print('first time, so calculate theta_p')
        theta_p = theta_p_function(freq, pure_p_mode)
    zeta = zeta_function(freq, q, theta_p, Delta_nu, Delta_Pi)
    tau= np.zeros(len(freq))
    tau[0]=1/(freq[0]*10**(-6))
    for i in range(1, len(tau)):
        tau[i] =  tau[i-1]+(freq[i]-freq[i-1])*10**(-6)/(freq[i]*10**(-6))**2/zeta[i]
    #print('tau', tau)
    #get rid of the radial and quadrupole mode
    for i in range(len(freq)):
        if np.min(np.abs(freq[i] - radial_mode))<0.05*Delta_nu:
            #print('find a radial mode')
            flat_PSD[i] = 0
        if np.min(np.abs(freq[i] - quadrupole_mode))<0.1*Delta_nu:
            #print('find a quadrupole mode')
            flat_PSD[i] = 0

    return tau, freq, flat_PSD, theta_p, zeta



################################
################################
################################
################################
################################
################################
################################
################################
#fit splittings and inclinations
def visibility(i, l, m):#i is incliation in degrees, l angular degree, m azimuthal order
    #Gizon et al. 2003
    i = np.deg2rad(i)
    if l == 1 and m == 0:
        return np.cos(i)**2
    if l == 1 and np.abs(m)==1:
        return 0.5*np.sin(i)**2
    if l == 2 and m == 0:
        return 0.25*( 3*np.cos(i)**2-1 )**2
    if l == 2 and np.abs(m) == 1:
        return 3/8*sin(2*i)**2
    if l == 2 and np.abs(m) == 2:
        return 3/8*np.sin(i)**4

def one_splitting(freq, l, i, a_0, centre_freq, splitting, eta, background):

    splitting_model = np.zeros(len(freq))
    for m in range(-l, l+1):
        #print(m)
        splitting_model = splitting_model + visibility(i, l, m)*(peak_lorentz(freq, a_0, centre_freq+m*splitting, eta, background)-background)
    splitting_model = splitting_model + background
    return splitting_model

def one_splitting_ln_likelihood(theta, freq, flat_PSD):
    i, a_0, centre_freq, splitting, eta, background = theta

    l = 1

    model_splitting = one_splitting(freq, l, i, a_0, centre_freq, splitting, eta, background)
    ln_likelihood = - np.sum(  np.log(model_splitting) + flat_PSD / model_splitting        )

    return ln_likelihood

def one_splitting_log_prior(theta, freq, flat_PSD, initial_guess):
    initial_centre_freq = initial_guess[2]
    initial_splitting = initial_guess[3]

    i, a_0, centre_freq, splitting, eta, background = theta
    l=1
    model_in_prior=one_splitting(freq, l, i, a_0, centre_freq, splitting, eta, background)
    if  (0 <= i <= 90) and (0 < eta < 0.2) and background>0 and max(model_in_prior) < 2*max(flat_PSD) and np.abs(centre_freq-initial_centre_freq)<0.1 and np.abs(splitting-initial_splitting)<0.1*initial_splitting:
        return 0.0
    else:
        return -np.inf

def one_splitting_log_prob(theta, freq, flat_PSD, initial_guess):
    #print('theta', theta)
    #a=input()
    one_splitting_prior = one_splitting_log_prior(theta, freq, flat_PSD, initial_guess)
    #print('one_splitting_prior', one_splitting_prior)
    if not np.isfinite(one_splitting_prior):
        return -np.inf
    return one_splitting_prior + one_splitting_ln_likelihood(theta, freq, flat_PSD)

def fit_splittings(freq, flat_PSD, initial_guess):
    #initial_guess = np.array([initial_i, initial_amplitude, initial_centre_freq, initial_splitting, initial_eta, initial_background])
    print('initial_guess', initial_guess)

    #plt.figure()
    #plt.plot(freq, flat_PSD)
    #plt.plot(freq, one_splitting(freq, 1, *initial_guess))
    #plt.show()
    #a=input('pause to see the initial guess')
    n_walkers = 32
    number_of_parameter = len(initial_guess)
    pos = initial_guess + 1e-1 * initial_guess * np.random.randn(n_walkers, number_of_parameter)
    pos[:,0] = np.random.uniform(low=0.0, high=90.0, size=n_walkers)
    pos[:,2] = initial_centre_freq+np.random.randn(n_walkers)
    pos[:,3] = initial_splitting+10**(-1)*np.random.randn(n_walkers)
    sampler = emcee.EnsembleSampler(   n_walkers, number_of_parameter, one_splitting_log_prob, args=(freq, flat_PSD, initial_guess)   )
    sampler.run_mcmc(pos, 2000, progress=True)

################################
################################
################################
################################
################################
################################
################################
################################
#fit splitting--zeta linear function
def splitting_zeta_model(zeta, core_Omega, surface_Omega):#Mosser 2015, Omega is angular frequency
    splitting = core_Omega/2/2/np.pi*zeta + surface_Omega/2/np.pi*(1-zeta)
    return splitting


def splitting_zeta_linear_likelihood(theta, zeta, splitting, splitting_err):
    core_Omega, surface_Omega = theta
    model_splitting = splitting_zeta_model(zeta, core_Omega, surface_Omega)
    #plt.figure()
    #plt.plot(zeta, splitting, 'r.')
    #plt.plot(zeta, model_splitting, 'g.')
    #plt.show()
    #print('splitting_err', splitting_err)
    #a=input()
    ln_likelihood = -2*np.sum(  (splitting-model_splitting)**2/splitting_err**2 )# + np.log(2*np.pi*splitting_err**2)     )
    #print(ln_likelihood)
    return ln_likelihood

def splitting_zeta_prior(theta, initial_core_Omega, initial_surface_Omega):
    #print(theta)
    #print(initial_core_Omega, initial_surface_Omega)
    #print('-------')
    #a=input()
    if np.abs(theta[0])>10*initial_core_Omega:# or theta[1]<0:
        return -np.inf
    #if theta[1]<0:
    #    return -np.inf
    return 0.0

def splitting_zeta_prob(theta, zeta, splitting, splitting_err, initial_core_Omega, initial_surface_Omega):
    lp = splitting_zeta_prior(theta, initial_core_Omega, initial_surface_Omega)
    return lp + splitting_zeta_linear_likelihood(theta, zeta, splitting, splitting_err)

################################
################################
################################
################################
################################
################################
################################
################################
#fit harvey profile
def harvey_one(frequency, a1, b1, white_noise):
    #print('in harvey one a1, b1, white_noise', a1, b1, white_noise)
    """The first Harvey component.

    Parameters
    ----------
    frequency : np.ndarray
        the frequency of the power spectrum
    a1 : float
        TODO: Write description
    b1 : float
        TODO: Write description
    white_noise : float
        the white noise component

    Returns
    -------
    model : np.ndarray
        the first Harvey component
    """

    model = np.zeros_like(frequency)

    model += a1/(1.0+(b1*frequency)**2.0+(b1*frequency)**4.0)
    model += white_noise

    return model

def harvey_two(frequency, a1, b1, a2, b2, white_noise):
    """The second Harvey component.

    Parameters
    ----------
    frequency : np.ndarray
        the frequency of the power spectrum
    a1 : float
        TODO: Write description
    b1 : float
        TODO: Write description
    a2 : float
        TODO: Write description
    b2 : float
        TODO: Write description
    white_noise : float
        the white noise component

    Returns
    -------
    model : np.ndarray
        the second Harvey component
    """

    model = np.zeros_like(frequency)

    model += a1/(1.0+(b1*frequency)**2.0+(b1*frequency)**4.0)
    model += a2/(1.0+(b2*frequency)**2.0+(b2*frequency)**4.0)
    model += white_noise

    return model

def harvey_three(frequency, a1, b1, a2, b2, a3, b3, white_noise):
    """The third Harvey component.

    Parameters
    ----------
    frequency : np.ndarray
        the frequency of the power spectrum
    a1 : float
        TODO: Write description
    b1 : float
        TODO: Write description
    a2 : float
        TODO: Write description
    b2 : float
        TODO: Write description
    a3 : float
        TODO: Write description
    b3 : float
        TODO: Write description
    white_noise : float
        the white noise component

    Returns
    -------
    model : np.ndarray
        the third Harvey component
    """

    model = np.zeros_like(frequency)

    model += a1/(1.0+(b1*frequency)**2.0+(b1*frequency)**4.0)
    model += a2/(1.0+(b2*frequency)**2.0+(b2*frequency)**4.0)
    model += a3/(1.0+(b3*frequency)**2.0+(b3*frequency)**4.0)
    model += white_noise

    return model

def harvey_profile_likelihood(theta, freq, PSD, used_model):
    if len(theta) == 7:
        MODEL = harvey_three(freq, *theta)
    if len(theta) == 5:
        MODEL = harvey_two(freq, *theta)
    if len(theta) == 3:
        MODEL = harvey_one(freq, *theta)

    ln_likelihood = -np.sum(np.log(MODEL)+PSD/MODEL)
    return ln_likelihood

def harvey_profile_prior(theta, top_amplitude):
    for one_element in theta:
        if one_element<=0:
            return -np.inf
    amplitude_sum = 0
    for i in range(0, len(theta)-1, 2):
        amplitude_sum += theta[i]

    if amplitude_sum > 5 * top_amplitude:
        return -np.inf
    return 0.0
            
def harvey_probability(theta, freq, PSD, used_model, top_amplitude):
    lp = harvey_profile_prior(theta, top_amplitude)
    if  np.isfinite(lp):
        return lp + harvey_profile_likelihood(theta, freq, PSD, used_model)
    else:
        return -np.inf

def fit_harvey_profile(freq, PSD, nu_max, envolope):
    initial_freq = freq * 1.0
    initial_PSD = PSD * 1.0
    print('nu_max, envolope', nu_max, envolope)
    index = np.where(freq>10)[0]#cut off the low freq area
    freq, PSD = freq[index], PSD[index]

    index = np.where( (freq<nu_max-0.7*envolope) | (freq>nu_max+0.7*envolope) )[0]#cut off the pulsations
    freq, PSD = freq[index], PSD[index]


    used_model_list = [1, 2]

    initial_guess = [[], [] ,[] ]

    best_parameters_list = []

    best_likelihood_list = []

    BIC_list = []

    for used_model in used_model_list:
        print('#####################')
        print('Now calculate harvey', used_model)
        print('#####################')
        background = np.median(   PSD[ np.argsort(freq)[-1000:-1] ]   )#the last 1000 points as the PSD background
        a1=np.mean(    PSD[0:100]    )#the first 100 points as the amplitude
        top_amplitude = a1 * 1.0
        b1=0.01#microHz
        if used_model == 1:
            initial_guess[used_model-1] = np.array([a1, b1, background])
        if used_model == 2:
            initial_guess[used_model-1] = np.zeros(2*used_model+1)
            initial_guess[used_model-1][0:2] = best_parameters[0:2]
            initial_guess[used_model-1][2:4] = best_parameters[0:2]/4
            initial_guess[used_model-1][4] = background
        if used_model == 3:
            initial_guess[used_model-1] = np.zeros(2*used_model+1)
            initial_guess[used_model-1][0:4] = best_parameters[0:4]
            initial_guess[used_model-1][4:6] = best_parameters[2:4]
            initial_guess[used_model-1][6] = background

        print('initial guess', initial_guess[used_model-1])

        #plt.figure()
        #plt.plot(freq, PSD)
        #plt.plot(freq, harvey_one(freq, *initial_guess[used_model-1]), 'r-')
        #plt.show() 
        #a=input()


        parameters = initial_guess[used_model-1]+ initial_guess[used_model-1]/2 * np.random.randn(3*(used_model*2+1), len(initial_guess[used_model-1]))
        nwalkers, ndim = parameters.shape
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, harvey_probability, args=(freq, PSD, used_model, top_amplitude), pool = pool)
            steps = int( (1000+(used_model-1)*1000)/3 )
            sampler.run_mcmc(parameters, steps, progress=True);

        flat_samples = sampler.get_chain(discard=int(steps/2), thin=15, flat=True)
        best_parameters = np.median(flat_samples, axis=0)
        best_parameters_list.append(best_parameters)
        best_likelihood_list.append(harvey_profile_likelihood(best_parameters, freq, PSD, used_model))
        BIC_list.append(-2*best_likelihood_list[-1]+(2*used_model+1)*np.log(len(freq))  )
        print('best_parameters', best_parameters)
        print('BIC', BIC_list)
        if_plot_mode = False
        if if_plot_mode:
            plt.figure()
            ax1 = plt.subplot(211)
            ax1.plot(freq, PSD)
            if used_model == 1:
                model = harvey_one(freq, *best_parameters )
            if used_model == 2:
                model = harvey_two(freq, *best_parameters )
            if used_model == 3:
                model = harvey_three(freq, *best_parameters )

            ax1.plot(freq, model, 'r-')
            ax2=plt.subplot(212)
            ax2.plot(freq, PSD/model)
            plt.title('model '+str(used_model))

            fig = corner.corner(flat_samples)
            plt.show() 
            print('BEST LIKELIHOOD', best_likelihood_list)
            a=input()

    #find best model
    best_model_index = np.argmin(BIC_list)
    best_model = used_model_list[best_model_index]
    print('Best model is', best_model )
    if best_model == 1:
        harvey = harvey_one(initial_freq, *best_parameters_list[best_model_index])
    if best_model == 2:
        harvey = harvey_two(initial_freq, *best_parameters_list[best_model_index])
    if best_model == 3:
        harvey = harvey_three(initial_freq, *best_parameters_list[best_model_index])
        #a=input()
    return initial_freq, harvey
########################
########################
########################
########################

def multi_splittings(freq, theta):#theta is: central frequency, splitting, amplitude, eta, central frequency, splitting, amplitude, eta, ..... , inclination, background
    l=1
    inclination = theta[-2]
    background = theta[-1]
    #background = 1
    model = np.zeros(len(freq))
    for i in range(0, len(theta)-4, 4):
        a_0 = theta[i+2]
        centre_freq = theta[i]
        splitting = theta[i+1]
        eta = theta[i+3]
        model = model + one_splitting(freq, l, inclination, a_0, centre_freq, splitting, eta, 0)
    model = model + background

    return model

def multi_splittings_likelihood(theta, freq, flat_PSD):
    model = multi_splittings(freq, theta)
    ln_p = - np.sum(np.log(model)+flat_PSD/model)
    return ln_p

def multi_splittings_prior(theta, initial_theta, bin_mode_list, m_list):
    for one_element in theta:
        if one_element <=0: 
            return -np.inf
    #amplitude larger than 0
    for i in range(0, len(theta)-2, 4):
        #print(i)
        one_freq = np.array( [theta[i]-theta[i+1], theta[i], theta[i]+theta[i+1]] )
        max_value = one_splitting(one_freq, 1, theta[-2], theta[i+2], theta[i], theta[i+1], theta[i+3], theta[-1])
        if max(max_value)<6: #must have some amplitude!
            return -np.inf
        if max( max_value ) > 500:#too large
            return -np.inf
    
    #frequency, splitting, eta cannot be to deviated
    for i in range(0, len(theta)-2, 4):
        if np.abs( theta[i]-initial_theta[i] )>0.02:#frequency
            return -np.inf
        if np.abs( theta[i+1]-initial_theta[i+1] )>0.02:#splitting
            return -np.inf 
        if theta[i+3]>0.07:#eta
            return -np.inf 
    #inclination between 0 and 90
    if theta[-2]<0 or theta[-2]>90:
        return -np.inf
    
    if np.abs(theta[-1]-1)>1:#background
        return -np.inf

    distance_prior = 1
    for theta_index in range(0, (len(theta)-2), 4):
        #print(theta_index, len(theta))
        m_counter = -1
        for m in m_list:
            m_counter += 1
            model_frequency = theta[theta_index] - m * theta[theta_index+1]
            if np.min(np.abs(model_frequency-bin_mode_list[m_counter]))<0.05:
                min_index = np.argmin(np.abs(model_frequency-bin_mode_list[m_counter]))
                #print('m', m, 'model_frequency', model_frequency, 'obs freq', bin_mode_list[m_counter][min_index], 'min distance', np.min(np.abs(model_frequency-bin_mode_list[m_counter])))
                HMFW=0.02
                distance_prior = distance_prior * HMFW**2/4/(  np.min(  np.abs(model_frequency-bin_mode_list[m_counter])  )**2+HMFW**2/4  )

    return np.log(  distance_prior  )

def multa_splittings_prob(theta, freq, flat_PSD, initial_theta, bin_mode_list, m_list):
    prior = multi_splittings_prior(theta, initial_theta, bin_mode_list, m_list)
    if not np.isfinite(prior):
        return -np.inf
    else:
        return prior + multi_splittings_likelihood(theta, freq, flat_PSD)



def fit_p_mode_asymptotic_relation_only(one_dir):
    STAR_NAME = one_dir[one_dir.index('KIC'):one_dir.index('_MCMC')]
    #backup old result
    old_file = glob.glob(one_dir+'/*asymptotic_relation_fit.dat')[0]
    #os.system('cp '+old_file+' '+old_file.replace('.dat', '_backup.dat'))
    #print(STAR_NAME)
    #load numax
    if len(glob.glob(one_dir + '/' + STAR_NAME + '_nu_max_by_YuJie_because_numax_large.dat')) > 0:
        nu_max = np.loadtxt(one_dir + '/' + STAR_NAME + '_nu_max_by_YuJie_because_numax_large.dat')
    else:
        nu_max = np.loadtxt(one_dir + '/' + STAR_NAME + '_nu_max_by_me.dat')
    #print('nu_max', nu_max)
    #load p mode data
    #for l=0 modes
    l_0_radial_order = []
    l_0_freq = []
    file_list = glob.glob(one_dir+'/*l_0_n*fitting_detail.dat')
    if len(file_list) == 0:
        print('WARNING! NO FILE!')
        exit
    for one_file in file_list:
        data = np.loadtxt(one_file)
        l_0_freq.append(data[0, 1])
        n = int( float(one_file[one_file.index('l_0_n_')+6: one_file.index('_fitting_detail')  ] ) )
        l_0_radial_order.append(  n  )
    l_0_freq = np.array(l_0_freq)
    l_0_radial_order = np.array(l_0_radial_order)

    #sort
    print('l_0_freq', l_0_freq)
    print('l_0_radial_order', l_0_radial_order)
    sort_index = np.argsort(l_0_radial_order)
    l_0_radial_order = l_0_radial_order[sort_index]
    l_0_freq = l_0_freq[sort_index]
    print('after sort')
    print('l_0_freq', l_0_freq)
    print('l_0_radial_order', l_0_radial_order)
    #for l=2 modes
    l_2_radial_order = []
    l_2_freq = []
    file_list = glob.glob(one_dir+'/*l_2_n*fitting_detail.dat')
    if len(file_list) == 0:
        print('WARNING! NO FILE!')
        exit
    for one_file in file_list:
        data = np.loadtxt(one_file)
        l_2_freq.append(data[0, 1])
        n = int( float(  one_file[one_file.index('l_2_n_')+6: one_file.index('_fitting_detail')  ] )  )
        l_2_radial_order.append(  n  )
    l_2_radial_order = np.array(l_2_radial_order)
    l_2_freq = np.array(l_2_freq)
    #sort
    print('l_2_freq', l_2_freq)
    print('l_2_radial_order', l_2_radial_order)
    sort_index = np.argsort(l_2_radial_order)
    l_2_radial_order = l_2_radial_order[sort_index]
    l_2_freq = l_2_freq[sort_index]
    print('after sort')
    print('l_2_freq', l_2_freq)
    print('l_2_radial_order', l_2_radial_order)

    asymp_para, asymp_para_err = fit_asymptotic_using_l_0_and_2(l_0_radial_order, l_0_freq, np.zeros(len(l_0_freq))+0.01, l_2_radial_order, l_2_freq, np.zeros(len(l_2_freq))+0.01, nu_max)
    ascii.write([asymp_para, asymp_para_err], one_dir + '/' + STAR_NAME + '_asymptotic_relation_fit.dat',names=['#values for Delta_nu microHz, epsilon, alpha, D', 'error'], overwrite=True)



if __name__ == '__main__':
    dir_list = glob.glob('/Volumes/ligangwork/solar_like_oscillators/Gehan_sample/*_MCMC') + glob.glob('/Volumes/ligangwork/solar_like_oscillators/Yujie_RGB/*_MCMC') + glob.glob('/Volumes/ligangwork/solar_like_oscillators/Gehan_sample/only_delta_pi/*MCMC') + glob.glob('/Volumes/ligangwork/solar_like_oscillators/Yujie_RGB/only_delta_pi/*_MCMC')
    #dir_list = glob.glob( '/Volumes/ligangwork/solar_like_oscillators/YuJie_RGB/KIC008681963_MCMC' )
    dir_list=sorted(dir_list, key=lambda name: int(name[name.index('KIC')+3:name.index('KIC')+12]))

    wanted = '2855642'
    #wanted = '12885196'
    counter = -1
    for one_dir in dir_list:
        counter+=1
        if wanted in one_dir:
            break


    for one_dir in dir_list[counter:counter+1]:
        print('\n\n\n****************')
        print(one_dir)
        fit_p_mode_asymptotic_relation_only(one_dir)
