import numpy as np
import matplotlib.pyplot as plt
import glob, os
from tqdm import tqdm
import emcee, corner
#from solar_like_lightcurves import *
from fit_package import asymptotic_relation_pure_p_mode, zeta_function, theta_p_function
#from solar_like_lightcurves import *
from multiprocessing import Pool
from sklearn.cluster import KMeans


def this_lorentz(x, a_0, x_0, eta):#page 19 of JCD notebook
    return 0.25*a_0**2/((x-x_0)**2+eta**2)



########MCMC to fit splittings##########

def bin_mode_function(selected_mode_list):
    bin_width = 0.25 #microHz
    bin_mode_list = []
    for i in range(len(selected_mode_list)):
        new_mode=[]
        freq_list = selected_mode_list[i]
        markers = np.zeros(len(freq_list))#1 for the freq has been used
        for one_freq_index in range(len(freq_list)):
            if markers[one_freq_index] == 1:
                continue
            index = np.where(np.abs(freq_list[one_freq_index] - freq_list)<bin_width)[0]
            markers[index]=1
            new_mode.append(np.median(freq_list[index]))
        new_mode=np.array(new_mode)
        bin_mode_list.append(new_mode)
    
    return bin_mode_list


def read_frequency_uncertainty(one_dir, bin_mode_list, m_list, if_consider_magnetism):
    if len(m_list) == 1:
        fit_freq_uncertainty_list=[  np.zeros(len(bin_mode_list[0]))+0.02  ]
        if len(glob.glob(one_dir+'/lorentzian_fit_for_only_delta_pi/No*best_fit_result.dat'))==0:
            print('no splitting detected, use 0.02 microHz as uncertainties')
            #fit_freq_uncertainty_list[0] = np.zeros(len(bin_mode_list[0]))+0.02
            #print('bin_mode_list, fit_freq_uncertainty_list', bin_mode_list, fit_freq_uncertainty_list)
            return bin_mode_list, fit_freq_uncertainty_list
        else:
            file_list = glob.glob(one_dir+'/lorentzian_fit_for_only_delta_pi/No*best_fit_result.dat')
            for one_file in file_list:
                data=np.loadtxt(one_file)
                min_freq_difference = np.min( np.abs(data[0, 1]-bin_mode_list[0]) )
                close_index = np.argmin( np.abs(data[0, 1]-bin_mode_list[0]) )
                amplitude = 0.25*data[0, 0]**2/data[0, 2]**2+data[0, 3]
                if min_freq_difference<0.03 and data[1, 2]<0.06 and data[1, 1]<0.03 and amplitude>3:#freq_difference < 0.03, eta err < 0.06, freq_err<0.03, amplitude>3
                    fit_freq_uncertainty_list[0][close_index] = data[1, 1]
                else:
                    print('cannot find a good fit result, use 0.02 microHz as uncertainty')

            return bin_mode_list, fit_freq_uncertainty_list



    if len(m_list) == 2:
        fit_freq_list = [[], []]
        fit_freq_uncertainty_list = [[], []]
    elif len(m_list) == 3:
        fit_freq_list = [[], [], []]
        fit_freq_uncertainty_list = [[], [], []]

    if not if_consider_magnetism:
        print('prepare to load uncertainty from solar_like_splittings')
        splitting_fit_dir = one_dir+'/solar_like_splittings'
    else:
        print('prepare to load uncertainty from asymmetry')
        splitting_fit_dir = one_dir+'/asymmetry'

    splitting_fit_file_list = glob.glob(splitting_fit_dir+'/*fit_result_No_*dat')
    if_force_uniform_uncertainty=False#true to turn off the real uncertainty
    if len(splitting_fit_file_list)<3 or if_force_uniform_uncertainty:
        print('No fit results found, use 0.02 microHz as uncertainty')
        for i in range(len(fit_freq_uncertainty_list)):
            fit_freq_uncertainty_list[i] = np.zeros(len(bin_mode_list[i]))+0.02

        return bin_mode_list, fit_freq_uncertainty_list

    print('Read fit result uncertainty!')
    match_threshold=0.1
    fit_centre_freq_list = []
    fit_centre_freq_uncertainty_list = []
    fit_splitting_list = []
    fit_splitting_uncertainty_list = []
    central_freq_shift_list = []
    central_freq_shift_uncertainty_list = []
    for one_file in splitting_fit_file_list:
        data = np.loadtxt(one_file, skiprows=1)
        fit_centre_freq_list.append(data[0,2])
        fit_centre_freq_uncertainty_list.append( data[1,2] )
        fit_splitting_list.append(data[0,3])
        fit_splitting_uncertainty_list.append(data[1,3])
        if if_consider_magnetism:
            central_freq_shift_list.append(data[0,5])
            central_freq_shift_uncertainty_list.append(data[1, 5])
        else:
            central_freq_shift_list.append( 0 )
            central_freq_shift_uncertainty_list.append(0)

    fit_centre_freq_list=np.array(fit_centre_freq_list)
    fit_centre_freq_uncertainty_list=np.array(fit_centre_freq_uncertainty_list)
    fit_splitting_list=np.array(fit_splitting_list)
    central_freq_shift_list = np.array(central_freq_shift_list)
    m_counter=-1
    for one_m in m_list:
        m_counter+=1
        freq_counter=-1
        if one_m != 0:
            this_m_fit_freq = fit_centre_freq_list-one_m*fit_splitting_list
        else:
            this_m_fit_freq = fit_centre_freq_list+central_freq_shift_list

        one_freq_counter=-1
        for one_freq in bin_mode_list[m_counter]:
            one_freq_counter+=1
            #print('one_m, m_counter, one_freq', one_m, m_counter, one_freq)
            #print('difference', np.min(np.abs(one_freq-this_m_fit_freq)))
            if np.min(np.abs(one_freq-this_m_fit_freq))<match_threshold:
                index=np.argmin(np.abs(one_freq-this_m_fit_freq))
                if one_m == 1 or one_m == -1:
                    fit_freq_uncertainty_list[m_counter].append(  (fit_centre_freq_uncertainty_list[index]**2+ fit_splitting_uncertainty_list[index]**2)**0.5    )
                elif one_m == 0:
                    fit_freq_uncertainty_list[m_counter].append(  (fit_centre_freq_uncertainty_list[index]**2+ central_freq_shift_uncertainty_list[index]**2)**0.5    )

                bin_mode_list[m_counter][one_freq_counter]= this_m_fit_freq[index]
                #print('best fit freq',this_m_fit_freq[index], 'error', fit_freq_uncertainty_list[m_counter][-1]  )
            else:
                fit_freq_uncertainty_list[m_counter].append(  max(fit_centre_freq_uncertainty_list)*1.414  )
                #print('cannot find best fit freq')

        #print(fit_freq_uncertainty_list)
    #print('bin_mode_list',bin_mode_list)
    #print('fit_freq_uncertainty_list',fit_freq_uncertainty_list)
    for i in range(len(fit_freq_uncertainty_list)):
        fit_freq_uncertainty_list[i]  = np.array(fit_freq_uncertainty_list[i])
    #print('fit_freq_uncertainty_list',fit_freq_uncertainty_list)

    #exit()
    return bin_mode_list, fit_freq_uncertainty_list




def make_splitting_grid(freq, flat_PSD, selected_mode_list, m_list, n_p, nu_max, asymp_para, Delta_Pi, Omega_g, Omega_p, STAR_TYPE, one_dir):

    # merge the selected_mode_list
    bin_mode_list = bin_mode_function(selected_mode_list)
    #plt.figure()
    #plt.plot(selected_mode_list[0], np.zeros(len(selected_mode_list[0])), 'k.')
    #plt.plot(bin_mode_list[0], np.zeros(len(bin_mode_list[0])), 'b.')
    #plt.show()
    #a=input('pause to look at the bined mode')

    print('Now the star is assumed to be a '+STAR_TYPE)


    Delta_Pi_list = [Delta_Pi]#np.linspace(-1, +1, 5)*0.1*Delta_Pi+Delta_Pi
    if STAR_TYPE == 'RGB':
        q_list = [0.1, 0.15, 0.2]#np.arange(0.15, 0.25, 0.5)
    elif STAR_TYPE == 'RC':
        q_list = np.linspace(0.1, 0.6, 5)#np.arange(0.15, 0.25, 0.5)

    epsilon_g_list = np.arange(0, 1.0, 0.1)#np.linspace(-0.1, 0.1, 5) +0.25
    if STAR_TYPE == 'RGB':
        p_mode_freq_shift_list = np.linspace(0.5, 0.8, 10)
    elif STAR_TYPE == 'RC':
        #p_mode_freq_shift_list = np.linspace(0.1, 0.5, 10)
        p_mode_freq_shift_list = np.linspace(-2.6, 0.8, 10)#convert from Deheuvels 2015 to my formula
    if STAR_TYPE == 'RGB':
        Omega_g_list = np.linspace(0, 3*2*np.pi, 30)#from Gehan's paper, max value is about 2*2*np.pi
    elif STAR_TYPE == 'RC':
        Omega_g_list = np.linspace(0, 0.4*2*np.pi, 30)#from Deheuvel's 2015 paper, max value is about 0.2*2*np.pi

    if STAR_TYPE == 'RGB':
        Omega_p_list = [0]#now fixed to be zero
    elif STAR_TYPE == 'RC':
        Omega_p_list = np.linspace(0, max(Omega_g_list), 10)

    if len(m_list)==1:#no splitting detected
        Omega_p_list = [0]#no fixed to be zero



    if len(m_list)==0: Omega_g_list=[0]

    Omega_p = 0
    K_list = []#the smaller the K, the better the match
    cross_correlation_list = []
    parameter_list = []
    for one_Omega_g in tqdm(Omega_g_list):
        #print('one_Omega_g', one_Omega_g)
        for one_Delta_Pi in Delta_Pi_list:
            for one_q in q_list:
                for one_epsilon_g in epsilon_g_list:
                    for one_p_shift in p_mode_freq_shift_list:
                        for one_Omega_p in Omega_p_list:
                            #print(one_Omega_g)
                            parameter_list.append([one_Delta_Pi, one_q, one_epsilon_g, one_p_shift, one_Omega_g, one_Omega_p])
                            mixed_mode_frequencies, their_zeta = implicit_asymp_relation_for_mixed_modes(n_p, nu_max, asymp_para, one_Delta_Pi, one_epsilon_g, one_q, one_p_shift)
                            mixed_mode_splittings = their_zeta * one_Omega_g/2/2/np.pi + one_Omega_p/2/np.pi * (1-their_zeta)
                        
                            #plt.figure()
                            #plt.plot(mixed_mode_frequencies % asymp_para[0], mixed_mode_splittings, '.')
                            #a=input('pause to see the splittings')

                            #calculate residuals
                            if_mode_residual_method = True
                            if if_mode_residual_method:
                                auto_width = 0.1 #microHz
                                K=0#the larger K, the better match
                                colour_list = ['r', 'b', 'g']
                                for m_index in range(len(m_list)):
                                    one_m = m_list[m_index]
                                    for one_freq in bin_mode_list[m_index]:
                                        #if one_freq < 90 or one_freq > 108: continue
                                        close_distance = np.min(np.abs(one_freq-(mixed_mode_frequencies-one_m*mixed_mode_splittings)    ))#prograde is negative m
                                        #if close_distance < auto_width:
                                        if close_distance < auto_width:#deviation not too large
                                            K = K + close_distance
                                        else:
                                            #print('find a large deviation')
                                            K = K+asymp_para[0]#if deviation is too large, add a large separation

                                K_list.append(K)
                            else:#cross correlation method, no matter the mode identification
                                cross_correlation = 0
                                mixed_counter=-1
                                for one_mixed_freq in mixed_mode_frequencies:
                                    mixed_counter+=1
                                    for central_freq in [one_mixed_freq-mixed_mode_splittings[mixed_counter], one_mixed_freq, one_mixed_freq+mixed_mode_splittings[mixed_counter]]:
                                        eta = 0.1 #microHz
                                        a_0=2*eta
                                        template = this_lorentz(freq, a_0, central_freq, eta)
                                        cross_correlation+=cross_correlation + np.sum(template * flat_PSD)
                                cross_correlation_list.append(cross_correlation)
    if if_mode_residual_method:
        best_index = np.argmin(K_list)
    else:
        best_index = np.argmax(cross_correlation_list)

    plt.figure()
    plt.hist(K_list, bins=20)
    plt.figure()
    tmp_parameter_list =np.array(parameter_list)
    if if_mode_residual_method:
        plt.plot(tmp_parameter_list[:, 4], K_list, '.')
    else:
        plt.plot(tmp_parameter_list[:, 4], cross_correlation_list, '.')

    plt.show()
    #a=input()
    plt.close()
    print('(in the function) The best parameters for Delta_Pi, q, epsilon_g, one_p_shift, Omega_g, Omega_p are\n', parameter_list[best_index] )
    return parameter_list[best_index]


########MCMC to fit splittings##########




def implicit_asymp_relation_for_mixed_modes(n_p, nu_max, asymp_para_for_p_mode, Delta_Pi_1, epsilon_g, q, p_mode_freq_shift):

    pure_p_freq = asymptotic_relation_pure_p_mode(n_p, 1, nu_max, *asymp_para_for_p_mode)+p_mode_freq_shift
    pure_p_freq_larger_range = asymptotic_relation_pure_p_mode(np.arange(min(n_p)-2, max(n_p)+3), 1, nu_max, *asymp_para_for_p_mode)+p_mode_freq_shift


    Delta_nu = asymp_para_for_p_mode[0]

    diff_freq_list = []
    diff_list = []
    zeta_list = []
    p_mode_counter = -1
    for one_pure_p_freq in pure_p_freq:
        p_mode_counter +=1
        previous_pure_p_freq = asymptotic_relation_pure_p_mode(n_p[p_mode_counter]-1, 1, nu_max, *asymp_para_for_p_mode)+p_mode_freq_shift
        next_pure_p_freq = asymptotic_relation_pure_p_mode(n_p[p_mode_counter]+1, 1, nu_max, *asymp_para_for_p_mode)+p_mode_freq_shift
        previous_p_separation = one_pure_p_freq - previous_pure_p_freq
        next_pure_p_separation = next_pure_p_freq - one_pure_p_freq
        #print('three continuous p freq', previous_pure_p_freq, one_pure_p_freq, next_pure_p_freq)
        #print('previous_p_separation, next_pure_p_separation',previous_p_separation, next_pure_p_separation, previous_p_separation/Delta_nu, next_pure_p_separation/Delta_nu)

        #print('one_pure_p_freq', one_pure_p_freq)
        freq_list = np.arange(one_pure_p_freq-0.5*previous_p_separation, one_pure_p_freq+0.5*next_pure_p_separation, Delta_nu/500)
        #for one_freq in freq_list:
        #    theta_p = np.pi*(one_freq-one_pure_p_freq)/Delta_nu
        #    theta_g = np.pi/(Delta_Pi_1)/(one_freq*10**(-6))-np.pi*epsilon_g

        #    diff = np.tan(theta_p)-q*np.tan(theta_g)
        #    diff_freq_list.append(one_freq)
        #    diff_list.append(diff)
        #    zeta_list.append( zeta_function(one_freq, q, theta_p, Delta_nu, Delta_Pi_1) )

        if_new_theta_p=False
        if not if_new_theta_p:
            theta_p =  np.pi*(freq_list-one_pure_p_freq)/( 0.5*(previous_p_separation+next_pure_p_separation) )
        else:
            theta_p = theta_p_function(freq_list, pure_p_freq_larger_range)

        theta_g = np.pi / (Delta_Pi_1) / (freq_list * 10 ** (-6)) - np.pi * epsilon_g
        diff_freq_list = diff_freq_list + list(freq_list)
        diff_list = diff_list + list( np.tan(theta_p)-q*np.tan(theta_g) )
        zeta_list = zeta_list + list( zeta_function(freq_list, q, theta_p, Delta_nu, Delta_Pi_1) )

    diff_freq_list = np.array(diff_freq_list)
    diff_list = np.array(diff_list)
    zeta_list = np.array(zeta_list)


    if_plot = False
    if if_plot:
        plt.figure()
        plt.plot(diff_freq_list, diff_list, 'k.')
        plt.plot(diff_freq_list, diff_list, '-', c= 'gray')
        plt.plot(diff_freq_list, zeta_list, 'r.')
        for one_pure_p_freq in pure_p_freq:
            plt.plot([one_pure_p_freq,one_pure_p_freq], [-10, 10], 'r:')
        plt.xlabel('Freq, microHz')
        plt.ylabel('$\\tan \\theta_p -q\\tan \\theta_g$ or $\zeta$')
        plt.ylim([-10, 10])

    #search for the solutions
    #search for the solutions around the pure p mode frequencies
    solution_freq_list = []
    solution_zeta_list = []


    #find solution
    solution_index = np.where( (diff_list[0:-1]*diff_list[1:]<0) )[0]
    for one_index in solution_index:
        if diff_list[one_index]<diff_list[one_index+1]:
            x1, x2 = diff_freq_list[one_index], diff_freq_list[one_index+1]
            y1, y2 = diff_list[one_index], diff_list[one_index+1]
            zeta1, zeta2 = zeta_list[one_index], zeta_list[one_index+1]
            solution_freq_list.append(-y2/(y1-y2)*(x1-x2)+x2)
            solution_zeta_list.append(   (solution_freq_list[-1]-x2)/(x1-x2)*(zeta1-zeta2)+zeta2    )
    solution_freq_list = np.array(solution_freq_list)
    solution_zeta_list = np.array(solution_zeta_list)
    if if_plot:
        plt.plot(solution_freq_list, np.zeros(len(solution_freq_list)), 'ro')
        plt.plot(solution_freq_list, solution_zeta_list, 'ro')

        print('solution_freq_list', solution_freq_list)
        plt.show()
        a=input()

    return solution_freq_list, solution_zeta_list



    for one_pure_p_freq in pure_p_freq:
        #print('one_pure_p_freq', one_pure_p_freq)
        number_of_solution_you_want = 7
        solution_counter = 0
        shift_index = 0
        while solution_counter < number_of_solution_you_want:
            if solution_counter == 0:
                start_index = np.argmin( np.abs( diff_freq_list-one_pure_p_freq ) )
            if start_index+shift_index+1 <= len(diff_list)-1 and diff_list[start_index+shift_index]*diff_list[start_index+shift_index+1]<0 and np.abs(diff_list[start_index+shift_index]-diff_list[start_index+shift_index+1])<1 and diff_list[start_index+shift_index]<diff_list[start_index+shift_index+1]:#search from the pure p mode freq to the right
                x1, x2 = diff_freq_list[start_index+shift_index], diff_freq_list[start_index+shift_index+1]
                y1, y2 = diff_list[start_index+shift_index], diff_list[start_index+shift_index+1]
                new_solution = -y2/(y1-y2)*(x1-x2)+x2
                if new_solution not in solution_freq_list:
                    solution_freq_list.append( new_solution )
                    solution_zeta_list.append(zeta_list[start_index+shift_index])
                solution_counter+=1
                #print('find a solution at the right')
            if start_index-shift_index-1 >=0 and diff_list[start_index-shift_index]*diff_list[start_index-shift_index-1]<0 and np.abs(diff_list[start_index-shift_index]-diff_list[start_index-shift_index-1])<1 and diff_list[start_index-shift_index]>diff_list[start_index-shift_index-1]:#search from the pure p mode freq to the left
                x1, x2 = diff_freq_list[start_index-shift_index], diff_freq_list[start_index-shift_index-1]
                y1, y2 = diff_list[start_index-shift_index], diff_list[start_index-shift_index-1]
                new_solution = -y2/(y1-y2)*(x1-x2)+x2
                if new_solution not in solution_freq_list:
                    solution_freq_list.append( new_solution )
                    solution_zeta_list.append(zeta_list[start_index-shift_index])                
                solution_counter+=1
                #print('find a solution at the left')

            shift_index += 1    


    #check repeated solutions

    solution_freq_list = np.array( solution_freq_list )
    solution_zeta_list = np.array( solution_zeta_list )
    sort_index = np.argsort(solution_freq_list)
    solution_freq_list = solution_freq_list[sort_index]
    solution_zeta_list = solution_zeta_list[sort_index]
    if if_plot:
        plt.plot(solution_freq_list, np.zeros(len(solution_freq_list)), 'ro')
        print('solution_freq_list', solution_freq_list)
        plt.show()
        a=input()

    return solution_freq_list, solution_zeta_list


############################
############################
############################
############################
#MCMC method to optimize the grid
############################
############################
############################
############################
def grid_ln_likelihood(theta, bin_mode_list, m_list, asymp_para, n_p, nu_max, fit_freq_uncertainty_list):

    Delta_Pi, q, epsilon_g, one_p_shift, Omega_g, Omega_p = theta
    #epsilon_g = epsilon_g % 1
    mixed_mode_frequencies, their_zeta = implicit_asymp_relation_for_mixed_modes(n_p, nu_max, asymp_para, Delta_Pi,epsilon_g, q, one_p_shift)
    mixed_mode_splittings = their_zeta * Omega_g / 2 / 2 / np.pi + Omega_p / 2 / np.pi * (1 - their_zeta)


    auto_width = 0.2  # microHz
    K = 0  # the smaller K, the better match
    colour_list = ['r', 'b', 'g']
    for m_index in range(len(m_list)):
        one_m = m_list[m_index]
        freq_counter=-1
        for one_freq in bin_mode_list[m_index]:
            freq_counter+=1
            calculated_freq = mixed_mode_frequencies - one_m * mixed_mode_splittings
            close_distance = np.min( ( one_freq - calculated_freq )**2 )  # prograde is negative m
            #print(one_m, one_freq, fit_freq_uncertainty_list[m_index][freq_counter])
            K = K + close_distance/fit_freq_uncertainty_list[m_index][freq_counter]**2 + np.log(2*np.pi*fit_freq_uncertainty_list[m_index][freq_counter]**2)#0.01 is the typical error for a frequency
            #if close_distance < auto_width**2:  # deviation not too large
            #    K = K + close_distance
            #else:
            #    K = K + asymp_para[0]  # if deviation is too large, add a large separation

    return -0.5*K

def grid_prior(theta, initial_parameters, m_list, STAR_TYPE):
    Delta_Pi, q, epsilon_g, one_p_shift, Omega_g, Omega_p = theta
    initial_Delta_Pi, initial_q, initial_epsilon_g, initial_one_p_shift, initial_Omega_g, initial_Omega_p = initial_parameters
    #print('initial_Delta_Pi',initial_Delta_Pi)
    if np.abs(Delta_Pi - initial_Delta_Pi)/initial_Delta_Pi>2:
        #print('Delta Pi out')
        return -np.inf
    if epsilon_g < -0.1 or epsilon_g > 1.1:
        #print('epsilon g out')
        return -np.inf
    if STAR_TYPE == 'RGB':
        q_range = [0.08, 0.24]
        p_mode_shift = [0.3, 0.9]
    if STAR_TYPE == 'RC':
        q_range = [0.1, 0.6]
        p_mode_shift = [-2.6, 0.8]

    if len(m_list) ==1:#no splitting
        Omega_g_range = [-0.0001, 0.0001]
        Omega_p_range = [-0.0001, 0.0001]
    else:
        Omega_g_range = [0, 20]
        Omega_p_range = [-1, 5]

    if q<min(q_range) or q>max(q_range):
        #print('q out')
        return -np.inf
    if one_p_shift<min(p_mode_shift) or one_p_shift > max(p_mode_shift):
        #print('p shift out')
        return -np.inf
    if Omega_g < min(Omega_g_range) or Omega_g > max(Omega_g_range):
        #print('Omega g out')
        return -np.inf
    if Omega_p < min(Omega_p_range) or Omega_p > max(Omega_p_range):
        #print('Omega p out')
        return -np.inf
    #if len(m_list) == 1:#no rotation information
    #    if np.abs( Omega_g ) > 0.1 or np.abs( Omega_p ) > 0.1:
    #        return -np.inf


    return 0.0

def grid_probability(theta, bin_mode_list, m_list, asymp_para, initial_parameters, n_p, nu_max, STAR_TYPE, fit_freq_uncertainty_list, asymp_para_error):
    #for epsilon g
    #theta[2] = theta[2] % 1
    grid_pr = grid_prior(theta, initial_parameters, m_list, STAR_TYPE)

    asymp_para_error[2] = 0

    if np.random.uniform(0, 1, 1)<-1:
        asymp_para_perturbed = asymp_para + 1 * np.random.randn(4)*asymp_para_error
    else:
        asymp_para_perturbed = asymp_para + 0 * np.random.randn(4)*asymp_para_error

    #print('1', asymp_para)
    #print('2', asymp_para_error)
    #print('3', asymp_para_perturbed)

    if np.isfinite(grid_pr):
        return grid_ln_likelihood(theta, bin_mode_list, m_list, asymp_para_perturbed, n_p, nu_max, fit_freq_uncertainty_list)
    else:
        return grid_pr


def MCMC_optimize_grid(bin_mode_list, m_list, asymp_para, best_grid_parameters, n_p, nu_max, one_dir, STAR_TYPE, STAR_NAME):


    #load p mode parameter uncertainty
    asymp_para_error = np.loadtxt(glob.glob(one_dir+'/*asymptotic_relation_fit.dat')[0], skiprows=1)[:, 1]
    #asymp_para = asymp_para + np.random.randn(4)*asymp_para_error

    #try_asymp_para_time = 10
    #plt.figure(figsize=(20, 5))
    #for try_counter in range(try_asymp_para_time):
    #    asymp_para_new = asymp_para + np.random.randn(4)*asymp_para_error
    #    try_p_freq = asymptotic_relation_pure_p_mode(np.arange(int(nu_max/asymp_para[0])-2,int(nu_max/asymp_para[0])+3), 1, nu_max, *asymp_para_new )+best_grid_parameters[3]
    #    for one_try_p_freq in try_p_freq:
    #        plt.plot([one_try_p_freq,one_try_p_freq], [0, 1], '-', c = 'gray', alpha=0.1)

    #plt.show()
    #a=input('pause')



    initial_parameters = best_grid_parameters
    if initial_parameters[4] == 0:#zero rotation
        initial_parameters[4] = 0.1
    if initial_parameters[2]>1 or initial_parameters[2]<0:
        initial_parameters[2] = initial_parameters[2] % 1
    if len(m_list) == 1: #no splitting
        initial_parameters[-2] = 0#rotation is zero
        initial_parameters[-1] = 0#rotation is zero
        parameter_sigma = np.array([0.2, 0.01, 0.1, 0.05, 0.01, 0.01  ])
    else:
        parameter_sigma = np.array([0.2, 0.01, 0.1, 0.05, 0.1, 0.1  ])#np.array([1, 0.05, 0.05, 0.05, 0.5, 0.5  ])#

    bin_mode_list, fit_freq_uncertainty_list = read_frequency_uncertainty(one_dir, bin_mode_list, m_list, False)
    print('fit_freq_uncertainty_list before adding 0.011 error', fit_freq_uncertainty_list)
    for error_counter in range(len(fit_freq_uncertainty_list)):
        fit_freq_uncertainty_list[error_counter] = (fit_freq_uncertainty_list[error_counter]**2+0.011**2)**0.5
    print('fit_freq_uncertainty_list after adding 0.011 error', fit_freq_uncertainty_list)


    nwalkers = len(best_grid_parameters)*2+2
    ndim = len(best_grid_parameters)
    pos = np.zeros((nwalkers, ndim))
    for ndim_counter in range(ndim):
        pos[:, ndim_counter] = initial_parameters[ndim_counter] + parameter_sigma[ndim_counter] * np.random.normal(0, 1, nwalkers)

    #for epsilon g
    #pos[:, 2] = pos[:, 2] % 1

    #check if out of the prior
    nwalkers_counter = 0
    while nwalkers_counter < nwalkers:
        this_parameter = pos[nwalkers_counter, :]
        this_prior = grid_prior(this_parameter, initial_parameters, m_list, STAR_TYPE)
        if not np.isfinite(this_prior):#out of the prior, re-sample
            #print('yes there is one', this_parameter)
            pos[nwalkers_counter, :] = initial_parameters + parameter_sigma * np.random.normal(0, 1, ndim)
        else:
            nwalkers_counter += 1


    #for ndim_counter in range(ndim):
    #    plt.figure()
    #    plt.hist(pos[:, ndim_counter], bins=15)
    #    plt.title(str(ndim_counter))
    #    plt.show()
    #    a=input()
    #    plt.close()



    with Pool() as pool:#multiprocessing
        sampler = emcee.EnsembleSampler(nwalkers, ndim, grid_probability, args=(bin_mode_list, m_list, asymp_para, initial_parameters, n_p, nu_max, STAR_TYPE, fit_freq_uncertainty_list, asymp_para_error), pool=pool)
        chain_length = 5000
        print('Now optimize the grid...')
        sampler.run_mcmc(pos, chain_length, progress=True)

    #burn_in_tau = sampler.get_autocorr_time()
    #print('burn_in_tau',burn_in_tau)
    #mean_burn_in = np.mean(burn_in_tau)

    #flat_samples = sampler.get_chain(discard=int( mean_burn_in*3 ), thin=int(mean_burn_in/2), flat=True)
    flat_samples = sampler.get_chain(discard=int( chain_length*0.5 ), thin=40, flat=True)
    #save flat_samples
    np.savetxt(one_dir+'/'+STAR_NAME+'_grid_optimized_chains.dat', flat_samples, header='#delta_Pi q epsilon_g p_shift Omega_g Omega_p')

    #for counter in range(len(flat_samples[:,0])-1, -1, -1):
    #    this_prior = grid_prior( flat_samples[counter,:] )

    #get rid of three sigma points
    for aim_cluster_index in [0, 2, 3]:
        this_median_value = np.median( flat_samples[:,aim_cluster_index] )

        axis_array = np.sort(flat_samples[:,aim_cluster_index])
        axis_length = len(axis_array)
        this_std = np.min(  [np.abs(axis_array[int(0.16*axis_length)]-axis_array[int(0.5*axis_length)]), np.abs(axis_array[int(0.5*axis_length)]-axis_array[int(0.84*axis_length)])   ])
        in_index = np.where(np.abs(flat_samples[:,aim_cluster_index]-this_median_value)<3*this_std)[0]
        flat_samples=flat_samples[in_index, :]

    #cluster for p-mode-shift (n=3) and epsilon_g (n=2)
    if_clustering = True
    for aim_cluster_index in [3, 2]:
        if not if_clustering:
            break
        if aim_cluster_index == 3:
            print('cluster p_mode_shift')

        if aim_cluster_index == 2:
            if np.std(flat_samples[:,aim_cluster_index])<0.2:
                print('no need to cluster epsilon g')
                continue
            print('cluster epsilon g')
        kmeans = KMeans(n_clusters=3, n_init=10, max_iter=3000, random_state=0, algorithm = 'elkan').fit( flat_samples[:,aim_cluster_index].reshape(-1, 1) )
        #print('kmeans.labels_', kmeans.labels_)
        print('kmeans.cluster_centers_', kmeans.cluster_centers_)
        #judge which cluster is more
        first_cluster_index = np.where(kmeans.labels_==0)[0]
        second_cluster_index = np.where(kmeans.labels_==1)[0]
        third_cluster_index = np.where(kmeans.labels_==2)[0]
        cluster_index_list = [first_cluster_index, second_cluster_index, third_cluster_index]
        cluster_length_list = [len(first_cluster_index), len(second_cluster_index), len(third_cluster_index)]
        length_sort = np.argsort(cluster_length_list)
        max_cluster_index = np.argmax(  [len(first_cluster_index), len(second_cluster_index), len(third_cluster_index)]  )
        #print('length_sort', length_sort)
        #print(cluster_index_list)
        #print(length_sort[-1])
        #print(cluster_index_list[length_sort[-1]])

        cluster_index = list( cluster_index_list[length_sort[-1]] )+list( cluster_index_list[length_sort[-2]] )

        #for p-mode-shift, if cluster center distance larger than two sigma of the cluster that have the most points, then cluster, else, no
        if aim_cluster_index == 3:
            p_centers=kmeans.cluster_centers_
            p_center_distance = [np.abs(p_centers[0]-p_centers[1]), np.abs(p_centers[0]-p_centers[2]), np.abs(p_centers[1]-p_centers[2])]
            print('p_center_distance', p_center_distance)
            print('p_center_distance std', np.std(p_center_distance))
            print('std of p-shift', np.std(flat_samples[:, aim_cluster_index]))
            if np.std(p_center_distance) < np.std(flat_samples[:, aim_cluster_index]):
                print('no need to cluster p-center, continue')
                continue

        #plt.figure()
        #plt.hist(flat_samples[first_cluster_index,3], bins=20, color='r', alpha=0.5)
        #plt.hist(flat_samples[second_cluster_index,3], bins=20, color='g', alpha=0.5)
        #plt.show()
        #a=input('pause to see the histogram')
        flat_samples = flat_samples[cluster_index,:]

    #get rid of three sigma points again
    for aim_cluster_index in [0, 2, 3]:
        this_median_value = np.median( flat_samples[:,aim_cluster_index] )

        axis_array = np.sort(flat_samples[:,aim_cluster_index])
        axis_length = len(axis_array)
        this_std = np.min(  [np.abs(axis_array[int(0.16*axis_length)]-axis_array[int(0.5*axis_length)]), np.abs(axis_array[int(0.5*axis_length)]-axis_array[int(0.84*axis_length)])   ])
        in_index = np.where(np.abs(flat_samples[:,aim_cluster_index]-this_median_value)<3*this_std)[0]
        flat_samples=flat_samples[in_index, :]

    #fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    #for i in range(ndim):
    #    ax = axes[i]
    #    ax.plot(flat_samples[:, i], "k", alpha=0.3)
    #    ax.set_xlim(0, len(flat_samples[:, i]))
    #    #ax.set_ylabel(labels[i])
    #    ax.yaxis.set_label_coords(-0.1, 0.5)
    #axes[-1].set_xlabel("step number");


    fig = corner.corner(flat_samples, labels=['$\Delta \Pi$', 'q', '$\epsilon_g$', 'p_shift', '$\Omega_g$','$\Omega_p$'], quantiles=[0.16, 0.5, 0.84], truths=initial_parameters);
    #a=input('pause to see the corner')
    fig.savefig(one_dir+'/optimize_grid_corner.png', dpi = 200)
    plt.close()
    optimized_grid_parameters = np.median(flat_samples, axis = 0)
    likelihood_array=[]
    for i in range(len(flat_samples[:,0])):
        likelihood_array.append(  grid_ln_likelihood(flat_samples[i,:], bin_mode_list, m_list, asymp_para, n_p, nu_max, fit_freq_uncertainty_list)  )
    best_likelihood_index = np.argmax(likelihood_array)
    optimized_grid_parameters = flat_samples[best_likelihood_index, :]
    optimized_grid_parameters_err = np.std(flat_samples, axis = 0)
    #plt.show()
    #a=input('in the fitting result')

    return optimized_grid_parameters, optimized_grid_parameters_err


if __name__ == '__main__':
    work_dir = '/Volumes/ligangwork/solar_like_oscillators/binaries_sample/long_cadence/KIC010491544_MCMC'
    #load pure-p-mode parameters
    asymp_para_for_p_mode = np.loadtxt(glob.glob(work_dir+'/*_asymptotic_relation_fit.dat')[0], skiprows = 1)[:, 0]
    print('Delta_nu', asymp_para_for_p_mode[0])
    nu_max = (asymp_para_for_p_mode[0]/0.28)**(1/0.75)   #deltanu = 0.28*numax**0.75
    envolope = 0.59*nu_max**0.9
    print('nu_max, envolope',nu_max, envolope)
    #load Delta_Pi
    Delta_Pi_1 = np.loadtxt(glob.glob(work_dir + '/*best_Delta_P.dat')[0], skiprows = 1)
    
    central_n_p = round(nu_max/asymp_para_for_p_mode[0])
    print('central_n_p', central_n_p)
    n_p = np.arange(central_n_p-2, central_n_p+3)
    print('n_p', n_p)
    epsilon_g = 0.25
    q = 0.15
    p_mode_freq_shift = 0.6
    mixed_mode_frequencies, their_zeta = implicit_asymp_relation_for_mixed_modes(n_p, nu_max, asymp_para_for_p_mode, Delta_Pi_1, epsilon_g, q, p_mode_freq_shift)

    plt.figure()
    plt.plot(   mixed_mode_frequencies[0:-1], -np.diff(1/(mixed_mode_frequencies*10**(-6)) ) ,'k.' , label='simulated' )
    plt.plot(   mixed_mode_frequencies, Delta_Pi_1*their_zeta ,'r.' , label = 'by $\\zeta$' )
    plt.xlabel('Freq, microHz')
    plt.ylabel('$\Delta \\Pi_1$')
    plt.legend()
    plt.ylim([min(Delta_Pi_1*their_zeta), Delta_Pi_1*1.1])
    plt.show()

#generate simulated power spectrum
    observation_span = 4#years
    sample_rate = 30 #minutes
    surface_rotation = 26.65 #days
    core_rotation = 26.65 #days
    arti_freq = np.arange(0, 1/(sample_rate/60/24)/2, 1/(observation_span*365.2422)  )/0.0864 # frequencies in unit of microHz
    arti_PSD = np.zeros(len(arti_freq))
    for i in range( len(mixed_mode_frequencies) ):
        splitting = 0.5*(1/(core_rotation*86400))*10**(6) * their_zeta[i] + (1/(surface_rotation*86400))*10**(6) * (1 - their_zeta[i])  #microHz
        print('splitting', splitting)
        for m in [-1, 1]:   
            close_index = np.argmin(np.abs(arti_freq-(mixed_mode_frequencies[i]+m*splitting)))
            arti_PSD[close_index] = 100
    


    plt.ion()
    Delta_Pi, best_Delta_Pi, tau, best_stretched_freq, best_stretched_flat_PSD, first_stretch_echelle_figure = shell_for_stretched_echelle_diagram(arti_freq, arti_PSD, asymp_para_for_p_mode, nu_max, envolope, Delta_Pi_1, 0.15, 0.6)