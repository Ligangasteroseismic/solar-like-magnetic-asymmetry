import os, glob, sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, '/Users/ligang/Library/Mobile Documents/com~apple~CloudDocs/gammaDorcodes')
sys.path.insert(0, '/Users/ligang/Library/Mobile Documents/com~apple~CloudDocs/solarlikecodes/SYDpy-master')
sys.path.insert(0, '/Users/gangli/Library/Mobile Documents/com~apple~CloudDocs/gammaDorcodes')
sys.path.insert(0, '/Users/gangli/Library/Mobile Documents/com~apple~CloudDocs/solarlikecodes/SYDpy-master')
sys.path.insert(0, '/Users/gang/Library/Mobile Documents/com~apple~CloudDocs/gammaDorcodes')


from get_peaks_no_MCMC import *
from power_spectrum_simulation import bin_mode_function, read_frequency_uncertainty
from fit_package import *
from fit_splittings import *
from tqdm import tqdm
import emcee, corner
#from solar_like_lightcurves import *
from fit_package import asymptotic_relation_pure_p_mode, zeta_function
#from solar_like_lightcurves import *
from multiprocessing import Pool
from sklearn.cluster import KMeans
from solar_like_lightcurves import plot_echelle_diagram
from stretched_echelle_diagram_functions import *
from plot_optimized_stretched_echelle_diagram import plot_magnetic_stretched_echelle_diagram
from astropy.io import ascii
from astropy.table import Table
import time as tm


def calcaulate_magnetic_grid(bin_mode_list, m_list, asymp_para, n_p, n_g, nu_max, Delta_Pi_1, fit_freq_uncertainty_list):
    #magnetic_grid_ln_likelihood(theta, bin_mode_list, m_list, asymp_para, n_p, n_g, nu_max)

    Delta_Pi_1_list = [Delta_Pi_1]
    epsilon_g_list = [0.28]#[0.28-0.08, 0.28, 0.28+0.08]
    q_list = [0.15]#[0.12, 0.15, 0.17]
    p_mode_shift_list = [0.67]#[0.6, 0.7, 0.8]
    magnetic_a_list = np.arange(-0.5, 1, 0.1)
    magnetic_delta_nu_list = np.linspace(0, 1, 10)
    core_Omega_list = np.linspace(0, 15, 30)
    surface_Omega_list = [0]

    K_list = []
    parameter_list = []
    for one_core_Omega in tqdm(core_Omega_list):
        for one_Delta_Pi_1 in Delta_Pi_1_list:
            for one_epsilon_g in epsilon_g_list:
                for one_q in q_list:
                    for one_p_mode_shift in p_mode_shift_list:
                        for one_magnetic_a in magnetic_a_list:
                            #print('one_magnetic_a', one_magnetic_a)
                            for one_magnetic_delta_nu in magnetic_delta_nu_list:
                                for one_surface_Omega in surface_Omega_list:
                                    this_parameter = np.array([one_Delta_Pi_1, one_q, one_epsilon_g, one_p_mode_shift, one_core_Omega, one_surface_Omega, one_magnetic_a, one_magnetic_delta_nu])
                                    parameter_list.append(this_parameter)
                                    #print('this_parameter', this_parameter)
                                    #print('parameter_list', parameter_list)
                                    this_K = magnetic_grid_ln_likelihood(this_parameter, bin_mode_list, m_list, asymp_para, n_p, n_g, nu_max, fit_freq_uncertainty_list)
                                    K_list.append(this_K)

    max_index = np.argmax(K_list)
    print('max index',max_index )
    print('best parameters are',parameter_list[max_index] )
    return parameter_list[max_index]

def calculate_perturbated_g_period(n_g, Delta_Pi_1, epsilon_g, given_m, magnetic_a, magnetic_delta_nu, nu_max, core_Omega):
    pure_g_period_before_perturbation = Delta_Pi_1*(n_g+epsilon_g)#seconds
    if not np.isscalar(pure_g_period_before_perturbation) and 0 in pure_g_period_before_perturbation:
        print('zero in pure_g_period_before_perturbation')
        print('pure_g_period_before_perturbation', pure_g_period_before_perturbation)
        print('parameters', n_g, Delta_Pi_1, epsilon_g, given_m, magnetic_a, magnetic_delta_nu, nu_max, core_Omega)
        return [-np.inf]
    if np.isscalar(pure_g_period_before_perturbation) and pure_g_period_before_perturbation == 0:
        print('zero in pure_g_period_before_perturbation second condition')
        print('pure_g_period_before_perturbation', pure_g_period_before_perturbation)
        print('parameters', n_g, Delta_Pi_1, epsilon_g, given_m, magnetic_a, magnetic_delta_nu, nu_max, core_Omega)
        return -np.inf
    #print('pure_g_period_before_perturbation', pure_g_period_before_perturbation)
    pure_g_freq_before_perturbation = 1/pure_g_period_before_perturbation*10**6 #microHz
    if given_m == 1 or given_m == -1:
        pure_g_freq_after_perturbation = pure_g_freq_before_perturbation + (1+0.5*magnetic_a)*magnetic_delta_nu*(nu_max/pure_g_freq_before_perturbation)**3 - given_m * core_Omega/2/2/np.pi
    if given_m == 0:
        pure_g_freq_after_perturbation = pure_g_freq_before_perturbation + (1-magnetic_a)*magnetic_delta_nu*(nu_max/pure_g_freq_before_perturbation)**3
    pure_g_period_after_perturbation = 1/(pure_g_freq_after_perturbation*10**(-6))#microHz to seconds

    #plt.figure()
    #plt.plot(pure_g_period_after_perturbation[:-1], np.diff(pure_g_period_after_perturbation), '.')
    #plt.xlabel('period, s')
    #plt.ylabel('period spacing, s')
    #plt.show()
    #a=input('enter to see the period spacing diagram>>>')
    return pure_g_period_after_perturbation

def get_perturbated_g_period_within_certain_range(Delta_Pi_1, epsilon_g, given_m, magnetic_a, magnetic_delta_nu, nu_max, core_Omega, lower_period, upper_period):

    n_g = 0
    if_exit = False
    returned_g_period = []
    start_to_record = False
    #plt.ion()
    #plt.figure()
    #plt.plot([lower_period,lower_period], [0, 1], 'r-')
    #plt.plot([upper_period,upper_period], [0, 1], 'r-')

    while not if_exit:
        n_g += 1
        this_pertubated_g_period = calculate_perturbated_g_period(n_g, Delta_Pi_1, epsilon_g, given_m, magnetic_a, magnetic_delta_nu, nu_max, core_Omega)
        next_pertubated_g_period = calculate_perturbated_g_period(n_g+1, Delta_Pi_1, epsilon_g, given_m, magnetic_a, magnetic_delta_nu, nu_max, core_Omega)
        #print('this_pertubated_g_period, next_pertubated_g_period', this_pertubated_g_period, next_pertubated_g_period)
        #plt.plot([this_pertubated_g_period, this_pertubated_g_period], [0, 1], 'b-')
        #a=input('enter to next frame...')
        if this_pertubated_g_period < lower_period and next_pertubated_g_period >= lower_period:
            start_to_record = True
        if start_to_record:
            returned_g_period.append(this_pertubated_g_period)
        if this_pertubated_g_period > upper_period:
            if_exit = True

        if n_g > 2000:
            return [-np.inf]

    return returned_g_period



def implicit_asymp_relation_for_megnetic_perturbated_mixed_modes_given_m(n_p, n_g, given_m, nu_max, asymp_para_for_p_mode, Delta_Pi_1, epsilon_g, q, p_mode_freq_shift, magnetic_a, magnetic_delta_nu, core_Omega, surface_Omega):
    #print('surface Omega, core Omega', surface_Omega, core_Omega)
    pure_p_freq = asymptotic_relation_pure_p_mode(n_p, 1, nu_max, *asymp_para_for_p_mode) + p_mode_freq_shift - given_m * surface_Omega/2/np.pi

    pure_p_freq_larger_range = asymptotic_relation_pure_p_mode(np.arange(min(n_p)-1, max(n_p)+2), 1, nu_max, *asymp_para_for_p_mode) + p_mode_freq_shift - given_m * surface_Omega/2/np.pi
    #print('pure_p_freq', pure_p_freq)
    #print('pure_p_freq_larger_range', pure_p_freq_larger_range)
    #exit()
    #pure_g_period_after_perturbation = calculate_perturbated_g_period(n_g, Delta_Pi_1, epsilon_g, given_m, magnetic_a, magnetic_delta_nu, nu_max, core_Omega)
    p_freq_range_lower, p_freq_range_upper = min(pure_p_freq_larger_range), max(pure_p_freq_larger_range)#(min(pure_p_freq_larger_range)+min(pure_p_freq))/2, (max(pure_p_freq_larger_range)+max(pure_p_freq))/2
    #print('p_freq_range_lower, p_freq_range_upper', p_freq_range_lower, p_freq_range_upper)
    pure_g_period_after_perturbation = get_perturbated_g_period_within_certain_range(Delta_Pi_1, epsilon_g, given_m, magnetic_a, magnetic_delta_nu, nu_max, core_Omega, 1/(p_freq_range_upper*10**(-6)) , 1/(p_freq_range_lower*10**(-6))  )
    pure_g_period_after_perturbation = np.array(pure_g_period_after_perturbation)
    if not np.isfinite(pure_g_period_after_perturbation[0]):
        print('get_perturbated_g_period_within_certain_range got infinite')
        #print('parameters', Delta_Pi_1, epsilon_g, given_m, magnetic_a, magnetic_delta_nu, nu_max, core_Omega, min( 1/(pure_p_freq_larger_range*10**(-6)) ), max( 1/(pure_p_freq_larger_range*10**(-6)) ))
        return -np.inf, -np.inf, -np.inf
    #print('pure_g_period_after_perturbation', pure_g_period_after_perturbation)
    #exit()

    #plt.figure()
    #for one_pure_p_freq_larger_range in pure_p_freq_larger_range:
    #    plt.plot([one_pure_p_freq_larger_range, one_pure_p_freq_larger_range], [0, 1], 'r-', linewidth=10)
    #for one_pure_g_period_after_perturbation in pure_g_period_after_perturbation:
    #    plt.plot([1/one_pure_g_period_after_perturbation*10**6, 1/one_pure_g_period_after_perturbation*10**6], [0, 1], 'b-')
    #plt.xlabel('Freq, microHz')
    #plt.show()
    #a=input('>>>')

    Delta_nu = asymp_para_for_p_mode[0]

    diff_freq_list = []
    diff_list = []
    zeta_list = []
    theta_g_list = []
    theta_p_list = []
    pure_p_freq_counter = -1
    for one_pure_p_freq in pure_p_freq:
        pure_p_freq_counter +=1
        previous_p_freq = pure_p_freq_larger_range[pure_p_freq_counter]
        next_p_freq = pure_p_freq_larger_range[pure_p_freq_counter+2]
        previous_delta_nu = one_pure_p_freq - previous_p_freq
        next_delta_nu = next_p_freq - one_pure_p_freq
        #print('previous_delta_nu/Delta_nu', 'next_delta_nu/Delta_nu',  previous_delta_nu/Delta_nu, next_delta_nu/Delta_nu)
        # print('one_pure_p_freq', one_pure_p_freq)

        ##########
        ##########
        ##########
        #non-equally-spaced freq_list!!!!!
        ##########
        ##########
        ##########
        #freq_list = np.arange(one_pure_p_freq - 0.5 * previous_delta_nu, one_pure_p_freq + 0.5 * next_delta_nu, Delta_nu / 500)
        freq_list = []
        how_many_in_each_g_mode_spacing = 150
        pure_g_freq_after_perturbation = np.sort( 1/pure_g_period_after_perturbation*10**6 )
        #print('freq range', one_pure_p_freq - 0.5 * previous_delta_nu, one_pure_p_freq + 0.5 * next_delta_nu)
        index = np.where((pure_g_freq_after_perturbation>one_pure_p_freq - 0.5 * previous_delta_nu) & (pure_g_freq_after_perturbation<one_pure_p_freq + 0.5 * next_delta_nu))[0]
        #index = np.where((pure_g_freq_after_perturbation > one_pure_p_freq) & (pure_g_freq_after_perturbation < one_pure_p_freq + next_delta_nu))[0]

        for g_freq_counter in range(len(index)+1):
            if g_freq_counter < len(index):
                current_g_freq = pure_g_freq_after_perturbation[index[g_freq_counter]]
                last_g_freq = pure_g_freq_after_perturbation[index[g_freq_counter]-1]
            elif g_freq_counter == len(index):
                current_g_freq = pure_g_freq_after_perturbation[max(index)+1]
                last_g_freq = pure_g_freq_after_perturbation[max(index)]
            local_g_freq_separation = current_g_freq - last_g_freq
            if g_freq_counter == 0:
                sample_number = round( ( current_g_freq - (one_pure_p_freq - 0.5 * previous_delta_nu) )/local_g_freq_separation *how_many_in_each_g_mode_spacing ) + 2
                freq_segment = np.linspace(one_pure_p_freq - 0.5 * previous_delta_nu, current_g_freq,  sample_number)[0:-1]
            elif g_freq_counter == len(index):
                sample_number = round( (one_pure_p_freq + 0.5 * next_delta_nu - last_g_freq )/local_g_freq_separation *how_many_in_each_g_mode_spacing ) + 2
                freq_segment = np.linspace(last_g_freq, one_pure_p_freq + 0.5 * previous_delta_nu,  sample_number)[0:-1]
            else:
                freq_segment = np.linspace(last_g_freq, current_g_freq, how_many_in_each_g_mode_spacing+1)[0:-1]
            freq_list = freq_list+list(freq_segment)
            #plt.figure()
            #plt.plot([one_pure_p_freq - 0.5 * previous_delta_nu, one_pure_p_freq - 0.5 * previous_delta_nu], [0, 1], 'r-', linewidth = 10)
            #plt.plot([one_pure_p_freq + 0.5 * next_delta_nu, one_pure_p_freq + 0.5 * next_delta_nu], [0, 1], 'r-', linewidth = 10)
            #for one_g_freq in pure_g_freq_after_perturbation:
            #    plt.plot([one_g_freq,one_g_freq],[0, 1], 'b-')
            #for one_freq_list in freq_list:
            #    plt.plot([one_freq_list,one_freq_list],[0, 1], 'g:', alpha = 0.5)
            #plt.xlabel('freq, microHz')
            #plt.show()
            #a=input()

        freq_list = np.array(freq_list)
        theta_p = np.pi * (freq_list - one_pure_p_freq) / ((previous_delta_nu+next_delta_nu)/2)#Delta_nu
        theta_g = []# np.pi / (Delta_Pi_1) / (freq_list * 10 ** (-6)) - np.pi * epsilon_g
        ##############################
        method_slow = False
        if method_slow:#slow
            for one_freq in freq_list:
                one_period = 1/(one_freq*10**(-6))
                theta_g_index = max(  np.where( one_period - pure_g_period_after_perturbation>0 )[0]  )
                local_delta_pi = pure_g_period_after_perturbation[theta_g_index+1]-pure_g_period_after_perturbation[theta_g_index]
                theta_g.append(   np.pi/local_delta_pi*(one_period-pure_g_period_after_perturbation[theta_g_index])   )
        ##############################
        else:#fast
            period_list = 1/(freq_list*10**(-6))#seconds
            pure_g_period_after_perturbation_used_index = np.where(  (pure_g_period_after_perturbation>min(period_list)) & (pure_g_period_after_perturbation<max(period_list))  )[0]
            if len(pure_g_period_after_perturbation_used_index) == 0:
                print('pure_g_period_after_perturbation_used_index is zero')
                return -np.inf, -np.inf, -np.inf
            pure_g_period_index = [min(pure_g_period_after_perturbation_used_index)-1] + list( pure_g_period_after_perturbation_used_index )

            #pure_g_period_after_perturbation = get_perturbated_g_period_within_certain_range(Delta_Pi_1, epsilon_g, given_m, magnetic_a, magnetic_delta_nu, nu_max, core_Omega, min(period_list), max(period_list))

            #plt.figure()
            #plt.plot([min(period_list), min(period_list)], [0, 1], 'r-')
            #plt.plot([max(period_list), max(period_list)], [0, 1], 'r-')
            #for one_plot_period in pure_g_period_after_perturbation:
            #    plt.plot([one_plot_period, one_plot_period], [0, 1], 'b-')
            #plt.show()


            for one_pure_g_period_index in pure_g_period_index[::-1]:
            #for one_pure_g_period_index in np.arange(len(pure_g_period_after_perturbation)-1)[::-1]:
                #print('one_pure_g_period_index', one_pure_g_period_index)
                g_period_lower_boundary = pure_g_period_after_perturbation[one_pure_g_period_index]
                try:
                    g_period_upper_boundary = pure_g_period_after_perturbation[one_pure_g_period_index + 1]
                except:
                    print('\n\n\n\n\n\n\n')
                    print('pure_g_period_after_perturbation', pure_g_period_after_perturbation)
                    print('min(period_list), min(period_list)',min(period_list), max(period_list))
                    print('g_period_upper_boundary = pure_g_period_after_perturbation[one_pure_g_period_index + 1] exceed boundary')
                    print('one_pure_g_period_index',one_pure_g_period_index)
                    print('pure_g_period_index', pure_g_period_index)
                    print('len(pure_g_period_after_perturbation)', len(pure_g_period_after_perturbation))
                    print('len(n_g)', len(n_g))
                    print('\n\n\n\n\n\n\n\n')
                    plt.figure()
                    for i_p in pure_p_freq:
                        plt.plot([i_p,i_p], [0, 1], 'r-', linewidth=10)
                    for i_g in pure_g_period_after_perturbation:
                        plt.plot([1/i_g*10**6, 1/i_g*10**6], [0, 1], 'b-', alpha=0.5)
                    plt.show()
                    a=input('pause to see the figure')
                    exit()
                    return -np.inf, -np.inf, -np.inf
                    #g_period_upper_boundary = g_period_lower_boundary + pure_g_period_after_perturbation[one_pure_g_period_index] - pure_g_period_after_perturbation[one_pure_g_period_index-1]


                #try:
                #    g_period_upper_boundary = pure_g_period_after_perturbation[one_pure_g_period_index+1]
                #except:
                #    g_index = -1
                #    for one_g_period in pure_g_period_after_perturbation[0:-1]:
                #        g_index+=1
                #        print('g_index, one_g_period, period_spacing', g_index, one_g_period, pure_g_period_after_perturbation[g_index+1]-one_g_period)
                #    exit()
                local_g_spacing = g_period_upper_boundary-g_period_lower_boundary
                used_index = np.where((period_list>=g_period_lower_boundary) & (period_list<g_period_upper_boundary) )[0]#[::-1]
                theta_g_segment= np.pi/local_g_spacing*(period_list[used_index]-g_period_lower_boundary)
                theta_g =  theta_g +  list( theta_g_segment )
                #plt.figure()
                #plt.plot(period_list[used_index], theta_g_segment, '.-')
                #plt.plot([g_period_lower_boundary, g_period_lower_boundary], [0, np.pi], ':')
                #plt.plot([g_period_upper_boundary, g_period_upper_boundary], [0, np.pi], ':')
                #plt.show()
        ###############################
            #a=input('pause')
        #exit()

        #plt.figure()
        #plt.plot(period_list, theta_g)
        #plt.xlabel('period, seconds')
        #plt.ylabel('theta g')
        #plt.title('here')
        #plt.show()


        theta_g_list = theta_g_list + list(theta_g)
        theta_p_list = theta_p_list + list(theta_p)
        diff_freq_list = diff_freq_list + list(freq_list)
        diff_list = diff_list + list(np.tan(theta_p) - q * np.tan(theta_g))
        zeta_list = zeta_list + list(zeta_function(freq_list, q, theta_p, Delta_nu, Delta_Pi_1))

    theta_g_list = np.array(theta_g_list)
    diff_freq_list = np.array(diff_freq_list)
    diff_list = np.array(diff_list)
    zeta_list = np.array(zeta_list)

    if_check_theta_g = False
    if if_check_theta_g:
        plt.figure()
        plt.plot(1/(diff_freq_list*10**(-6)), theta_g_list, '.-')
        for one_period in pure_g_period_after_perturbation:
            plt.plot([one_period,one_period], [0, np.pi], ':')
        plt.xlabel('period')
        plt.ylabel('theta_g')
        plt.show()
        a = input()




    if_plot = False
    if if_plot:
        plt.figure()
        plt.plot(diff_freq_list, theta_p_list, 'r.-', label = '$\\theta_p$')
        plt.plot(diff_freq_list, theta_g_list, 'g.-', label = '$\\theta_g$')
        for one_pure_p_freq in pure_p_freq:
            plt.plot([one_pure_p_freq, one_pure_p_freq], [-5, 5], 'r:')
        for one_perturbated_g_period in pure_g_period_after_perturbation:
            plt.plot([1/one_perturbated_g_period*10**6, 1/one_perturbated_g_period*10**6], [-5, 5], 'b:')
        plt.xlabel('Freq, microHz')
        plt.ylabel('$\\theta_p$ or $\\theta_g$')


        plt.figure()
        plt.title('m '+str(given_m)+'core Omega '+str(core_Omega)[0:6])
        plt.plot(diff_freq_list, diff_list, 'k.')
        plt.plot(diff_freq_list, diff_list, '-', c='gray')
        plt.plot(diff_freq_list, zeta_list, 'r.')
        for one_pure_p_freq in pure_p_freq:
            plt.plot([one_pure_p_freq, one_pure_p_freq], [-10, 10], 'r:')
        for one_perturbated_g_period in pure_g_period_after_perturbation:
            plt.plot([1/one_perturbated_g_period*10**6, 1/one_perturbated_g_period*10**6], [-10, 10], 'b:')
        plt.xlabel('Freq, microHz')
        plt.ylabel('$\\tan \\theta_p -q\\tan \\theta_g$ or $\zeta$')
        plt.ylim([-10, 10])

    # search for the solutions
    # search for the solutions around the pure p mode frequencies
    solution_freq_list = []
    solution_zeta_list = []

    # find solution
    solution_index = np.where((diff_list[0:-1] * diff_list[1:] < 0))[0]
    for one_index in solution_index:
        if diff_list[one_index] < diff_list[one_index + 1] and diff_list[one_index + 1]-diff_list[one_index]<10:
            x1, x2 = diff_freq_list[one_index], diff_freq_list[one_index + 1]
            y1, y2 = diff_list[one_index], diff_list[one_index + 1]
            zeta1, zeta2 = zeta_list[one_index], zeta_list[one_index + 1]
            solution_freq_list.append(-y2 / (y1 - y2) * (x1 - x2) + x2)
            solution_zeta_list.append((solution_freq_list[-1] - x2) / (x1 - x2) * (zeta1 - zeta2) + zeta2)
    solution_freq_list = np.array(solution_freq_list)
    solution_zeta_list = np.array(solution_zeta_list)

    if_output_related_data = False
    if if_output_related_data:
        output_dir = '/Volumes/ligangwork2/solar_like_oscillators/YuJie_RGB/only_delta_pi/KIC008560280_MCMC/magnetism'
        ascii.write([diff_freq_list, theta_p_list, theta_g_list, diff_list], output_dir+'/theta_p_theta_g_data.dat', names = ['freq in microHz', 'theta_p', 'theta_g', 'tan_theta_p - q tan_theta_g'], overwrite = True)

    if if_plot:
        plt.plot(solution_freq_list, np.zeros(len(solution_freq_list)), 'ro')
        plt.plot(solution_freq_list, solution_zeta_list, 'ro')

        print('solution_freq_list', solution_freq_list)
        plt.show()
        a = input()

    return solution_freq_list, solution_zeta_list, given_m

def all_m_implicit_asymp_relation_for_megnetic_perturbated_mixed_modes(n_p, n_g, m_list, nu_max, asymp_para_for_p_mode, Delta_Pi_1, epsilon_g, q, p_mode_freq_shift, magnetic_a, magnetic_delta_nu, core_Omega, surface_Omega):
    calculated_mixed_mode_list = []
    calculated_zeta_list = []
    for given_m in m_list:
        #print('given_m', given_m)
        solution_freq_list, solution_zeta_list, _ = implicit_asymp_relation_for_megnetic_perturbated_mixed_modes_given_m(n_p, n_g, given_m, nu_max, asymp_para_for_p_mode, Delta_Pi_1, epsilon_g, q, p_mode_freq_shift, magnetic_a, magnetic_delta_nu, core_Omega, surface_Omega)
        #print('solution_freq_list', solution_freq_list)
        calculated_mixed_mode_list.append(solution_freq_list)
        calculated_zeta_list.append(solution_zeta_list)

    return m_list, calculated_mixed_mode_list, calculated_zeta_list


def magnetic_grid_ln_likelihood(theta, bin_mode_list, m_list, asymp_para, n_p, n_g, nu_max, fit_freq_uncertainty_list):

    min_grid_freq, max_grid_freq = min( asymptotic_relation_pure_p_mode(n_p, 0, nu_max, *asymp_para) ), max( asymptotic_relation_pure_p_mode(n_p, 0, nu_max, *asymp_para) )+asymp_para[0] #l=0 modes as boundary

    Delta_Pi_1, q, epsilon_g, one_p_shift, Omega_g, Omega_p, magnetic_a, magnetic_delta_nu = theta
    auto_width = 0.2  # microHz
    freq_uncertainty = 0.02 #frequency uncertainty
    K = 0  # the smaller K, the better match
    colour_list = ['r', 'b', 'g']
    for m_index in range(len(m_list)):
        one_m = m_list[m_index]
        given_m = one_m
        calculated_freq, _, _ = implicit_asymp_relation_for_megnetic_perturbated_mixed_modes_given_m(n_p, n_g, given_m, nu_max,asymp_para, Delta_Pi_1,epsilon_g, q, one_p_shift,magnetic_a, magnetic_delta_nu, Omega_g, Omega_p)
        occupy_index = np.zeros(len(calculated_freq)) # 1 means the freq is used
        if not np.isfinite(calculated_freq).any():
            print('in the likelihood, calculate freq infinite')
            return -np.inf

        if_plot = False
        if if_plot:
            plt.figure()
            for one_cal_freq in calculated_freq:
                plt.plot([one_cal_freq, one_cal_freq], [0, 1], 'r-')
            for one_obs_freq in bin_mode_list[m_index]:
                plt.plot([one_obs_freq, one_obs_freq], [0, 1], 'b-')
            plt.xlabel('Freq, microHz')
            plt.title('m='+str(int(given_m)))
            a=input('see the likelihood diagram')

        if_gaussian_distribution_method = False

        one_freq_counter=-1
        for one_freq in bin_mode_list[m_index]:
            one_freq_counter+=1
            freq_uncertainty=fit_freq_uncertainty_list[m_index][one_freq_counter]
            #print(freq_uncertainty)
            if not if_gaussian_distribution_method:
                #if one_freq < min_grid_freq or one_freq > max_grid_freq:
                #    continue
                #calculated_freq = mixed_mode_frequencies - one_m * mixed_mode_splittings
                #print('calculated_freq', calculated_freq)
                #print('occupy_index', occupy_index)
                used_index = np.where(occupy_index == 0)[0]
                close_distance = np.min( ( one_freq - calculated_freq[used_index] )**2 )

                close_distance_index = np.argmin(  ( one_freq - calculated_freq[used_index] )**2  )

                #if close_distance > auto_width**2:
                #    close_distance = asymp_para[0]**2
                best_cal_freq_index = np.argmin(np.abs(calculated_freq[used_index][close_distance_index]-calculated_freq))
                occupy_index[best_cal_freq_index] = 1
                #if close_distance > 0.1**2:#typical is 0.011
                #    continue
                K = K + close_distance/freq_uncertainty**2+np.log(2*np.pi*freq_uncertainty**2)#0.01 is the typical error for a frequency
                #print('m, obs_freq, uncertainty', one_m, one_freq, freq_uncertainty)
            else:
                K = K + 1/(2*np.pi)**0.5*np.sum(-0.5*(one_freq-calculated_freq)**2/freq_uncertainty**2-np.log(freq_uncertainty))

    if if_gaussian_distribution_method:
        K = K /(-0.5)

    return -0.5*K



def magnetic_grid_prior(theta, asymp_para, initial_parameters, m_list, STAR_TYPE, n_g, n_p, nu_max, if_consider_magnetism):
    #print('in the prior, theta', theta)
    Delta_Pi, q, epsilon_g, one_p_shift, Omega_g, Omega_p, magnetic_a, magnetic_delta_nu = theta
    #print('in the prior, parameters', Delta_Pi, q, epsilon_g, one_p_shift, Omega_g, Omega_p, magnetic_a, magnetic_delta_nu)
    initial_Delta_Pi, initial_q, initial_epsilon_g, initial_one_p_shift, initial_Omega_g, initial_Omega_p, initial_magnetic_a, initial_magnetic_delta_nu = initial_parameters

    #print('Delta_Pi, initial_Delta_Pi, minus', Delta_Pi, initial_Delta_Pi, np.abs(Delta_Pi - initial_Delta_Pi))
    if np.abs(Delta_Pi - initial_Delta_Pi)>0.5:#/initial_Delta_Pi>0.02:
        #print('Delta Pi out')
        return -np.inf
    #if epsilon_g < 0.28-0.08 or epsilon_g > 0.28+0.08:
    if epsilon_g < -0.1 or epsilon_g > 1.1:
        #print('epsilon g out')
        return -np.inf
    if STAR_TYPE == 'RGB':
        q_range = [0.08, 0.25]
        p_mode_shift = [0.3, 1.0]
    if STAR_TYPE == 'RC':
        q_range = [0.1, 0.6]
        p_mode_shift = [-2.6, 0.8]
    if len(m_list) == 1:#slow rotations
        Omega_g_range = [0, 0.0001]
        Omega_p_range = [0, 0.0001]
        a_range = [-0.0001, 0.0001]
    else:
        Omega_g_range = [0, 20]
        Omega_p_range = [-1, 1]
        a_range=[-0.5, 1]
    magnetic_delta_nu_range = [0,  0.2]

    if not if_consider_magnetism:
        a_range = [-10**(-4), 10**(-4)]
        magnetic_delta_nu_range = [0,  10**(-4)]


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

    if magnetic_a < min(a_range) or magnetic_a > max(a_range):
        return -np.inf


    if magnetic_delta_nu < min(magnetic_delta_nu_range) or magnetic_delta_nu > max(magnetic_delta_nu_range):
        return -np.inf

    for one_m in m_list:
        #break
        pure_p_freq_larger_range = asymptotic_relation_pure_p_mode(np.arange(min(n_p) - 1, max(n_p) + 2), 1, nu_max, *asymp_para) + one_p_shift - one_m * Omega_p / 2 / np.pi
        p_freq_range_lower, p_freq_range_upper = min(pure_p_freq_larger_range), max(pure_p_freq_larger_range)  # (min(pure_p_freq_larger_range)+min(pure_p_freq))/2, (max(pure_p_freq_larger_range)+max(pure_p_freq))/2
        # print('p_freq_range_lower, p_freq_range_upper', p_freq_range_lower, p_freq_range_upper)
        perturbated_g_modes = get_perturbated_g_period_within_certain_range(Delta_Pi, epsilon_g, one_m, magnetic_a, magnetic_delta_nu, nu_max, Omega_g, 1 / (p_freq_range_upper * 10**(-6)), 1/(p_freq_range_lower*10**(-6))  )
        if len(perturbated_g_modes) == 1 and not np.isfinite(perturbated_g_modes[0]):
            print('cannot get perturbated_g_modes in the prior')
            print('one_m',one_m)
            print('parameters', Delta_Pi, q, epsilon_g, one_p_shift, Omega_g, Omega_p, magnetic_a, magnetic_delta_nu )
            return -np.inf
        g_modes_spacing = np.diff(perturbated_g_modes)
        #plt.figure()
        #plt.plot(perturbated_g_modes[:-1], g_modes_spacing, '.-')
        #plt.title('Delta Pi 1 = '+str(Delta_Pi)+' s')
        #plt.show()
        if len(np.where(g_modes_spacing<0)[0])!=0:
            print('negative g mode spacing')
            return -np.inf

    #return 0.0
    epsilon_g_standard_deviation = 0.08
    #return -0.5*((epsilon_g-0.28)/epsilon_g_standard_deviation)**2-np.log(epsilon_g_standard_deviation*(2*np.pi)**0.5)  #np.exp(-0.5*(epsilon_g-0.28)**2/0.08**2)/0.08/(2*np.pi)**0.5
    return 0

def magnetic_grid_probability(theta, bin_mode_list, m_list, asymp_para, initial_parameters, n_p, n_g, nu_max, STAR_TYPE, if_consider_magnetism, fit_freq_uncertainty_list):
    #theta[2] = theta[2] % 1#for epsilon_g
    #for epsilon g
    #theta[2] = theta[2] % 1
    grid_pr = magnetic_grid_prior(theta, asymp_para, initial_parameters, m_list, STAR_TYPE, n_g, n_p, nu_max, if_consider_magnetism)
    if np.isfinite(grid_pr):
        return  grid_pr + magnetic_grid_ln_likelihood(theta, bin_mode_list, m_list, asymp_para, n_p, n_g, nu_max, fit_freq_uncertainty_list)
    else:
        return grid_pr


def run_magnetic_MCMC(best_parameters, m_list, bin_mode_list, asymp_para, n_p, n_g, nu_max, STAR_TYPE, MCMC_chain_length, if_consider_magnetism, one_dir):

    #plt.figure()
    #m_counter=-1
    #this_color = ['r', 'g', 'b']
    #for one_m in m_list:
    #    m_counter+=1
    #    for one_freq in bin_mode_list[m_counter]:
    #        plt.plot([one_freq, one_freq], [0,1], this_color[m_counter]+':')


    #old_bim_mode_list = bin_mode_list
    if len(glob.glob(one_dir+'/freq*from_Sebastien.txt'))==0:
        bin_mode_list, fit_freq_uncertainty_list = read_frequency_uncertainty(one_dir, bin_mode_list, m_list, if_consider_magnetism)
        print('fit_freq_uncertainty_list before adding 0.011 uncertainty', fit_freq_uncertainty_list)
        for error_counter in range(len(fit_freq_uncertainty_list)):
            fit_freq_uncertainty_list[error_counter] = (fit_freq_uncertainty_list[error_counter]**2+0.011**2)**0.5
        print('fit_freq_uncertainty_list after adding 0.011 uncertainty', fit_freq_uncertainty_list)


    else:
        fit_freq_uncertainty_list = [np.loadtxt(glob.glob(one_dir+'/freq*from_Sebastien.txt')[0])[:,1]]
        #fit_freq_uncertainty_list[0] = (fit_freq_uncertainty_list[0]**2-0.02**2)**0.5

        #for error_counter in range(len(fit_freq_uncertainty_list)):
        #    if fit_freq_uncertainty_list[error_counter]=='nan':
        #        fit_freq_uncertainty_list[error_counter] = max(fit_freq_uncertainty_list)
        print('fit_freq_uncertainty_list', fit_freq_uncertainty_list)
    #m_counter=-1
    #for one_m in m_list:
    #    m_counter+=1
    #    one_freq_counter=-1
    #    for one_freq in bin_mode_list[m_counter]:
    #        one_freq_counter+=1
    #        plt.plot([one_freq, one_freq], [0,1], this_color[m_counter]+'-')
    #        plt.plot([one_freq-fit_freq_uncertainty_list[m_counter][one_freq_counter], one_freq-fit_freq_uncertainty_list[m_counter][one_freq_counter]], [0,1], this_color[m_counter]+'-')
    #        plt.plot([one_freq+fit_freq_uncertainty_list[m_counter][one_freq_counter], one_freq+fit_freq_uncertainty_list[m_counter][one_freq_counter]], [0,1], this_color[m_counter]+'-')

    #plt.show()
    #a=input('pause to see the figure')



    if_given_delta_Pi = False
    if if_given_delta_Pi:
        print('************')
        print('************')
        print('************')
        print('************')
        print('WARNING! Delta Pi is changed')
        print('************')
        print('************')
        print('************')
        print('************')

        best_parameters[0] = 79.9000
    initial_Delta_Pi = best_parameters[0]


    ndim = 8
    nwalkers = ndim*2+2

    initial_prior = -np.inf
    if_first_run_MCMC = False
    if len(best_parameters)==6:
        if_first_run_MCMC = True
        initial_parameters = np.array(  list( best_parameters )+[0, 0.2]  )
    elif len(best_parameters)==8:
        initial_parameters = np.array(best_parameters)
    print('Now Delta Pi is', initial_Delta_Pi, 'and initial_parameters[0] is', initial_parameters[0] )
    print('initial_parameters are', initial_parameters)
    pos = np.zeros((nwalkers, ndim))
    pos_counter = 0
    # initial guess for the parameters
    while pos_counter<nwalkers:
        #Delta_Pi, q, epsilon_g, one_p_shift, Omega_g, Omega_p, magnetic_a, magnetic_delta_nu = theta
        #initial Delta_Pi
        initial_Delta_Pi_list = initial_Delta_Pi + 0.1 * np.random.normal(0, 1, 1)
        #initial_Delta_Pi_list = np.random.uniform(initial_Delta_Pi-2, initial_Delta_Pi+2, 1)
        #initial q
        if if_first_run_MCMC:
            initial_q_list = 0.15+0.08*np.random.normal(0, 1, 1)
        else:
            initial_q_list = best_parameters[1]+0.02*np.random.normal(0, 1, 1) #)%1

        #initial epsilon_g
        if if_first_run_MCMC:
            initial_epsilon_g_list = 0.28+0.08*np.random.normal(0, 1, 1)
        else:
            initial_epsilon_g_list = best_parameters[2]+0.02*np.random.normal(0, 1, 1)
            #initial_epsilon_g_list =  0.28+0.08*np.random.normal(0, 1, 1)#best_parameters[2]+0.02*np.random.normal(0, 1, 1)

        #initial p shift
        if if_first_run_MCMC:
            p_shift_list = np.random.uniform(0.6, 0.8, 1)#0.65+0.1*np.random.normal(0, 1, 1)
        else:
            #p_shift_list = np.random.uniform(0.6, 0.8, 1)#best_parameters[3]+0.05*np.random.normal(0, 1, 1)
            p_shift_list = best_parameters[3]+0.1*np.random.normal(0, 1, 1)

        if len(m_list) == 1:#no splittings
            #initial omega g
            Omega_g = np.random.uniform(0, 0.0001)#best_parameters[-2]+0.2*np.random.normal(0, 1, 1)
            #initial omega p
            Omega_p = np.random.uniform(0, 0.0001)
        else:#splittings

           #initial omega g
            if if_first_run_MCMC:
                Omega_g = best_parameters[4]+5*np.random.normal(0, 1, 1)
            else:
                Omega_g = best_parameters[4] + 0.1 * np.random.normal(0, 1, 1)
            #initial omega p
            Omega_p = 0+0.1*np.random.normal(0, 1, 1)
        #initial magnetic a
        if len(m_list) == 1 or not if_consider_magnetism:#slow rotations or no magnetism
            magnetic_a = np.random.uniform(-10**(-6), 10**(-6))
        else:
            if if_first_run_MCMC:
                magnetic_a = np.random.uniform(-0.5, 1, 1)
            else:
                magnetic_a = best_parameters[6]+0.02*np.random.normal(0, 1, 1)
        #initial magnetic shift
        if if_first_run_MCMC and if_consider_magnetism: #consider magnetism
            magnetic_delta_nu = np.random.uniform(0, 1, 1)
        elif not if_first_run_MCMC and if_consider_magnetism: #consider magnetism
            magnetic_delta_nu = best_parameters[7]+0.1*np.random.normal(0, 1, 1)
        elif not if_consider_magnetism:
            magnetic_delta_nu = np.random.uniform(0, 10**(-6), 1)

        try_theta = [initial_Delta_Pi_list, initial_q_list, initial_epsilon_g_list, p_shift_list, Omega_g, Omega_p, magnetic_a, magnetic_delta_nu]
        #print('try_theta',try_theta)
        this_prior = magnetic_grid_prior(try_theta, asymp_para, initial_parameters, m_list, STAR_TYPE, n_g, n_p, nu_max, if_consider_magnetism)
        if np.isfinite(this_prior):
            pos[pos_counter, :] = try_theta
            pos_counter+=1
        else:
            continue


    if_plot_pos_histogram = False
    if if_plot_pos_histogram:
        label = ['delta Pi', 'q', 'epsilon_g', 'p_shift', 'Omega_g', 'Omega_p', 'a', 'magnetic shift']
        plt.figure()
        for parameter_index in range(8):
            ax=plt.subplot(2,4, parameter_index+1)
            ax.hist(pos[:, parameter_index])
            ax.set_xlabel(label[parameter_index])
        plt.show()
        a=input('pause to see pos histogram')

    #for pos_i in range( len(pos[:,0]) ):
    #    print('pos[pos_i, :]', pos[pos_i, :])

    #exit()

    if_parallel = True
    if if_parallel:
        with Pool() as pool:#multiprocessing
            sampler = emcee.EnsembleSampler(nwalkers, ndim, magnetic_grid_probability, args=(bin_mode_list, m_list, asymp_para, initial_parameters, n_p, n_g, nu_max, STAR_TYPE, if_consider_magnetism, fit_freq_uncertainty_list), pool=pool)
            chain_length = MCMC_chain_length
            print('Now optimize the grid with magntism perturbation...')
            #for pos_i in range( len(pos[0,:]) ):
            #    plt.figure()
            #    plt.hist(pos[:, pos_i], bins=20)
            #    plt.show()
            #    print(pos[:, pos_i])
            #    a=input('pause to see the histogram')

            sampler.run_mcmc(pos, chain_length, progress=True)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, magnetic_grid_probability, args=(bin_mode_list, m_list, asymp_para, initial_parameters, n_p, n_g, nu_max, STAR_TYPE, if_consider_magnetism))
        chain_length = MCMC_chain_length
        print('Now optimize the grid with magntism perturbation...')
        sampler.run_mcmc(pos, chain_length, progress=True)



    flat_samples = sampler.get_chain(discard=int(chain_length * 0.7), thin=40, flat=True)
    delete_index = []
    #remove outliers
    for index in range(len(flat_samples[:, 0])):
        this_prior = magnetic_grid_prior(flat_samples[index,:], asymp_para, initial_parameters, m_list, STAR_TYPE, n_g, n_p, nu_max, if_consider_magnetism)
        if not np.isfinite(this_prior):
            delete_index.append(index)
    flat_samples = np.delete(  flat_samples, delete_index, axis=0 )
    #largest likelihood
    if if_first_run_MCMC:
        print('first time run MCMC, use max likelihood as result')
        K_list = []
        for index in range(len(flat_samples[:,0])):
            K = magnetic_grid_ln_likelihood(flat_samples[index, :], bin_mode_list, m_list, asymp_para, n_p, n_g, nu_max, fit_freq_uncertainty_list)
            K_list.append(K)
        best_index = np.argmax(K_list)
        best_magnetic_parameters = flat_samples[best_index, :]
    else:
        print('second time run MCMC, use medians as result')
        best_magnetic_parameters = np.median( flat_samples, axis=0 )
        small_uncertainties = np.zeros(len(best_magnetic_parameters))
        for i in range(ndim):
            this_parameter_array = np.sort(flat_samples[:, i])
            length = len(this_parameter_array)
            upper = this_parameter_array[int(0.84 * length)] - this_parameter_array[int(0.5 * length)]
            lower = this_parameter_array[int(0.5 * length)] - this_parameter_array[int(0.16 * length)]
            small_uncertainties[i] = min([lower, upper])
        #remove outliers
        print('remove 3-sigma outliers')
        delete_index = []
        for index in range(len(flat_samples[:, 0])):
            for parameter_index in range(len(flat_samples[index, :])):
                if np.abs( flat_samples[index, parameter_index] - best_magnetic_parameters[parameter_index] ) > 3 * small_uncertainties[parameter_index]:
                    delete_index.append(index)
                    break
        flat_samples = np.delete(  flat_samples , delete_index, axis=0 )
        best_magnetic_parameters = np.median(flat_samples, axis=0)#get best parameters again

    print('************************')
    print('best_magnetic_parameters', best_magnetic_parameters)
    upper_uncertainties = np.zeros(len(best_magnetic_parameters))
    lower_uncertainties = np.zeros(len(best_magnetic_parameters))
    for i in range(ndim):
        this_parameter_array = np.sort( flat_samples[:, i] )
        length = len(this_parameter_array)
        upper_uncertainties[i] = this_parameter_array[int(0.84*length)] - this_parameter_array[int(0.5*length)]
        lower_uncertainties[i] = this_parameter_array[int(0.5*length)] - this_parameter_array[int(0.16*length)]

    return best_magnetic_parameters, upper_uncertainties, lower_uncertainties, flat_samples



def fit_magnetic_shift(one_dir, if_consider_magnetism):

    STAR_TYPE = 'RGB'
    STAR_NAME = get_star_name(one_dir)
    if if_consider_magnetism:
        magnetism_output_dir = one_dir+'/magnetism'
    else:
        magnetism_output_dir = one_dir+'/non_magnetism'



    if len(glob.glob(magnetism_output_dir)) == 0:
        os.system('mkdir '+magnetism_output_dir)
    #load PSD
    print(one_dir+'/'+STAR_NAME+'_PSD.dat')
    PSD_data = np.loadtxt(one_dir+'/'+STAR_NAME+'_PSD.dat', skiprows=1)
    freq, PSD = PSD_data[:,0], PSD_data[:, 1]
    harvey = np.loadtxt(one_dir+'/'+STAR_NAME+'_myself_harvey.dat', skiprows=1)
    flat_PSD = PSD_data[:, 1]/harvey[:,1]
    #load p-mode fit
    asymp_para = np.loadtxt(glob.glob(one_dir+'/*asymptotic_relation_fit.dat')[0], skiprows=1)
    asymp_para = asymp_para[:, 0]
    delta_nu = asymp_para[0]
    if len(glob.glob(one_dir + '/' + STAR_NAME + '_nu_max_by_YuJie_because_numax_large.dat')) > 0:
        nu_max = np.loadtxt(one_dir + '/' + STAR_NAME + '_nu_max_by_YuJie_because_numax_large.dat')
    else:
        nu_max = np.loadtxt(one_dir + '/' + STAR_NAME + '_nu_max_by_me.dat')

    n_p_centre = round(nu_max/delta_nu)

    #load g-mode fit
    optimized_parameters = np.loadtxt(glob.glob(one_dir+'/*grid_optimized_parameters.dat')[0])
    best_parameters = np.loadtxt(glob.glob(one_dir+'/*grid_best_parameters.dat')[0])
    initial_Delta_Pi = best_parameters[0]
    #load observed peaks
    if len(glob.glob(one_dir+'/*bin_points_for_l_1_m_*.dat'))==3:
        m_1_list = np.loadtxt(glob.glob(one_dir+'/*bin_points_for_l_1_m_1.dat')[0], skiprows=1, ndmin=1)
        m_0_list = np.loadtxt(glob.glob(one_dir+'/*bin_points_for_l_1_m_0.dat')[0], skiprows=1, ndmin=1)
        m_negative_1_list = np.loadtxt(glob.glob(one_dir+'/*bin_points_for_l_1_m_-1.dat')[0], skiprows=1, ndmin=1)
        min_obs_freq = min( [min(m_1_list), min(m_0_list), min(m_negative_1_list)]  )
        max_obs_freq = max([max(m_1_list), max(m_0_list), max(m_negative_1_list)])
        m_list = [1, 0, -1]
        selected_mode_list = [m_1_list, m_0_list, m_negative_1_list]
        mode_colour_array=['g', 'b', 'r']
        mode_shape_array=['+', 'o', '_']
        bin_mode_list = selected_mode_list
    if len(glob.glob(one_dir+'/*bin_points_for_l_1_m_*.dat'))==2:
        m_1_list = np.loadtxt(glob.glob(one_dir+'/*bin_points_for_l_1_m_1.dat')[0], skiprows=1, ndmin=1)
        m_negative_1_list = np.loadtxt(glob.glob(one_dir+'/*bin_points_for_l_1_m_-1.dat')[0], skiprows=1, ndmin=1)
        min_obs_freq = min( [min(m_1_list), min(m_negative_1_list)]  )
        max_obs_freq = max([max(m_1_list), max(m_negative_1_list)])
        m_list = [1, -1]
        selected_mode_list = [m_1_list, m_negative_1_list]
        mode_colour_array=['g', 'r']
        mode_shape_array=['+', '_']
        bin_mode_list = selected_mode_list
    if len(glob.glob(one_dir+'/*bin_points_for_l_1_m_*.dat'))==1:
        m_0_list = np.loadtxt(glob.glob(one_dir+'/*bin_points_for_l_1_m_0.dat')[0], skiprows=1)
        print('m_0_list', m_0_list)
        min_obs_freq = min( m_0_list  )
        max_obs_freq = max(m_0_list)
        m_list = [0]
        selected_mode_list = [m_0_list]
        mode_colour_array=['b']
        mode_shape_array=['o']
        bin_mode_list = selected_mode_list

    #determine n_p range
    for try_n_p in range(0, n_p_centre+100):
        this_l_0_freq = asymptotic_relation_pure_p_mode(try_n_p, 0, nu_max, *asymp_para)
        next_l_0_freq = asymptotic_relation_pure_p_mode(try_n_p+1, 0, nu_max, *asymp_para)
        #if this_l_0_freq > min_obs_freq:#
        if this_l_0_freq < min_obs_freq and next_l_0_freq > min_obs_freq:
            min_grid_freq = this_l_0_freq
            min_n_p = try_n_p
            break
    for try_n_p in range(min_n_p, n_p_centre+100):
        this_l_0_freq = asymptotic_relation_pure_p_mode(try_n_p, 0, nu_max, *asymp_para)
        next_l_0_freq = asymptotic_relation_pure_p_mode(try_n_p+1, 0, nu_max, *asymp_para)
        if this_l_0_freq < max_obs_freq and next_l_0_freq > max_obs_freq:
            max_grid_freq = next_l_0_freq
            max_n_p = try_n_p
            break

    n_p = np.arange(min_n_p, max_n_p+1)
    print('min_grid_freq, max_grid_freq', min_grid_freq, max_grid_freq)
    #use a larger range
    min_grid_freq, max_grid_freq = min_grid_freq-delta_nu, max_grid_freq+delta_nu
    min_grid_period, max_grid_period = 1/(max_grid_freq*10**(-6)), 1/(min_grid_freq*10**(-6))#seconds
    print('min_grid_period, max_grid_period', min_grid_period, max_grid_period)
    n_g = np.arange(max([1, round(min_grid_period/initial_Delta_Pi)-10]), round(max_grid_period/initial_Delta_Pi)+10)
    #n_g = np.arange(min(n_g), 2000)
    #print('n_g', n_g)




    #############################################################################################################################
    #run grid
    #initial_parameters = calcaulate_magnetic_grid(bin_mode_list, m_list, asymp_para, n_p, n_g, nu_max, best_parameters[0])
    #exit()

    #############################################################################################################################
    magnetic_result_file = magnetism_output_dir+'/'+STAR_NAME+'_magnetic_parameters_result.dat'
    if len(glob.glob(magnetic_result_file))>0:
        last_fit_best_magnetic_parameters = np.loadtxt(magnetic_result_file)[0, :]
    if_run_MCMC = True
    if len(glob.glob(magnetic_result_file))==0 and if_consider_magnetism:
        MCMC_chain_length = 2000
        print('*********************')
        print('FIRST time to run the magnetic MCMC')
        best_magnetic_parameters, upper_uncertainties, lower_uncertainties, flat_samples = run_magnetic_MCMC(best_parameters, m_list, bin_mode_list, asymp_para, n_p, n_g, nu_max, STAR_TYPE, MCMC_chain_length, if_consider_magnetism, one_dir)
        #save results
        save_array = np.zeros((3, len(best_magnetic_parameters)))
        save_array[0,:]=best_magnetic_parameters
        save_array[1,:]=upper_uncertainties
        save_array[2,:]=lower_uncertainties
        np.savetxt(magnetic_result_file, save_array, header='#first line is the best result for Delta_Pi (s), q, epsilon_g, one_p_shift (microHz), Omega_g (radius microHz), Omega_p (radius microHz), magnetic_a, magnetic_delta_nu (microHz), second line is upper uncertainties, third line is lower uncertainties')
    print('*********************')
    if if_run_MCMC:
        print('SECOND time to run the magnetic MCMC')
        MCMC_chain_length = 4000
        if if_consider_magnetism:
            print('load result by last time magnetic fit')
            best_magnetic_parameters = np.loadtxt(magnetic_result_file)[0, :]
        else:
            best_magnetic_parameters = np.array( list(optimized_parameters[0, :])+[0, 0] )
        #best_magnetic_parameters = np.array( list(best_parameters)+[0, 2] )
        best_magnetic_parameters, upper_uncertainties, lower_uncertainties, flat_samples = run_magnetic_MCMC(best_magnetic_parameters, m_list, bin_mode_list, asymp_para, n_p, n_g, nu_max, STAR_TYPE, MCMC_chain_length, if_consider_magnetism, one_dir)
        #save results
        save_array = np.zeros((3, len(best_magnetic_parameters)))
        save_array[0,:]=best_magnetic_parameters
        save_array[1,:]=upper_uncertainties
        save_array[2,:]=lower_uncertainties
        np.savetxt(magnetic_result_file, save_array, header='#first line is the best result for Delta_Pi (s), q, epsilon_g, one_p_shift (microHz), Omega_g (radius microHz), Omega_p (radius microHz), magnetic_a, magnetic_delta_nu (microHz), second line is upper uncertainties, third line is lower uncertainties')
    else:
        print('***WARNING***')
        print('Do not run the magnetic MCMC')

        if not if_consider_magnetism:
            fake_output_array = np.zeros((3, 8))
            fake_output_array[0,:] = np.array( list(optimized_parameters[0, :])+[0, 0] ).reshape(1, -1)
            np.savetxt(magnetic_result_file,fake_output_array , header='#first line is the best result for Delta_Pi (s), q, epsilon_g, one_p_shift (microHz), Omega_g (radius microHz), Omega_p (radius microHz), magnetic_a, magnetic_delta_nu (microHz), second line is upper uncertainties, third line is lower uncertainties')

        #for magnetic field



    if not if_run_MCMC:
        print('do not run MCMC, use default parameters')
        if if_consider_magnetism:
            best_magnetic_parameters = last_fit_best_magnetic_parameters
            best_magnetic_parameters = np.array([85.55, 0.15, 0.3, 0.7, 0, 0, 0, 2.2])  # for 8684542
            flat_samples = np.zeros((1000, len(best_magnetic_parameters)))
            for flat_samples_counter in range(1000):
                flat_samples[flat_samples_counter, :] = best_magnetic_parameters
        else:
            best_magnetic_parameters = np.array( list(optimized_parameters[0,:])+[0, 0] )

    #get best fitting results
    best_magnetic_freq_list = []
    best_magnetic_zonal_zeta = []

    #best_magnetic_parameters = [7.13077836e+01, 1.38861567e-01, 2.28250759e-01, 6.34702920e-01, 4.62409168e-04, 3.55681551e-04, -6.04660142e-02,4.44469718e+00]

    #print('Now best_magnetic_parameters is fixed as', best_magnetic_parameters)

    #Delta_Pi_1, epsilon_g, q, p_mode_freq_shift, magnetic_a, magnetic_delta_nu, core_Omega, surface_Omega
    _, best_magnetic_freq_list, best_magnetic_zonal_zeta_for_all_m = all_m_implicit_asymp_relation_for_megnetic_perturbated_mixed_modes(n_p, n_g, m_list, nu_max, asymp_para, best_magnetic_parameters[0],best_magnetic_parameters[2], best_magnetic_parameters[1], best_magnetic_parameters[3], best_magnetic_parameters[6], best_magnetic_parameters[7], best_magnetic_parameters[4], best_magnetic_parameters[5])
    #if not if_run_MCMC:
        #exit()
    for m_counter in range(len(m_list)):#write down frequencies
        output_file_name = magnetism_output_dir+'/'+STAR_NAME+'_best_fitting_magnetic_frequency_m_'+str(int(m_list[m_counter]))+'.dat'
        #print('m', m_list[m_counter])
        #print('best_magnetic_freq_list[m_counter]', best_magnetic_freq_list[m_counter])
        np.savetxt(output_file_name, best_magnetic_freq_list[m_counter], header='#frequency in microHz')
    output_file_name=magnetism_output_dir+'/'+STAR_NAME+'_magnetism_best_zeta.dat'
    #calculate zeta and stretched echelle diagram
    tau, stretched_freq, stretched_flat_PSD, theta_p, zeta = calculate_stretched_echelle_diagram(freq, flat_PSD, asymp_para, nu_max, 3*delta_nu, best_magnetic_parameters[0], best_magnetic_parameters[1],best_magnetic_parameters[3], 9999)
    ascii.write([tau, stretched_freq, stretched_flat_PSD, zeta], output_file_name, names=['tau in s', 'freq in microHz', 'PSD in S/N', 'zeta'], overwrite=True)

    #plt.figure()
    echelle_fig, ax_list, freq_boundary = plot_echelle_diagram(freq, flat_PSD, delta_nu, [nu_max-3*delta_nu, nu_max+3*delta_nu])
    for i in range(len(bin_mode_list)):  # plot on the echelle diagram
        index_in_echelle_diagram = []
        for mode_index in range(len(bin_mode_list[i])):
            index_in_echelle_diagram.append(np.argmin(np.abs(bin_mode_list[i][mode_index] - freq)))
        for one_echelle_ax in ax_list:
            one_echelle_ax.plot(freq[index_in_echelle_diagram], flat_PSD[index_in_echelle_diagram],mode_colour_array[i] + mode_shape_array[i], markersize=10, markeredgewidth=2, alpha=0.5)

    # write radial orders on the echelle diagram
    for one_radial_order in range(int(nu_max / asymp_para[0]) - 3, int(nu_max / asymp_para[0]) + 4):
        l_0_mode_here = asymptotic_relation_pure_p_mode(one_radial_order, 0, nu_max, *asymp_para)
        for one_echelle_ax in ax_list:
            y_lim_here = one_echelle_ax.get_ylim()
            one_echelle_ax.text(l_0_mode_here, 0.9 * max(y_lim_here), 'n=' + str(one_radial_order))

    #plot fitted peaks on the echelle diagram
    for one_ax in ax_list:
        y_lim = one_ax.get_ylim()
        x_lim = one_ax.get_xlim()
        for m_counter in range(len(m_list)):
            one_ax.plot(best_magnetic_freq_list[m_counter], np.zeros(len(best_magnetic_freq_list[m_counter]))+0.9*max(y_lim),  'k'+ mode_shape_array[m_counter]   )
        #y_location = (max(y_lim) - 0.5 * min(y_lim)) / (max(x_lim) - min(x_lim)) * (mixed_mode_frequencies[i] - min(x_lim)) + 0.5 * min(y_lim)
    echelle_fig.savefig(magnetism_output_dir+'/'+STAR_NAME+'_magnetism_echelle.png', dpi=200)


    if len(m_list) > 1: #with splittings
        chain_file_name = magnetism_output_dir+'/'+STAR_NAME+'_chain.dat'
        if if_run_MCMC:
            np.savetxt(chain_file_name, flat_samples, header='#Delta Pi in s, q, epsilon_g, p_shift in microHz, Omega_g in radians microHz, Omega_p in radians microHz, a, delta_nu_mag in microHz' )
            if if_consider_magnetism:
                plt.figure()
                fig = corner.corner(flat_samples, labels=['$\Delta \Pi_1$', '$q$', '$\epsilon_g$', '$f_\mathrm{shift}$', '$\Omega_\mathrm{core}$','$\Omega_\mathrm{env}$', '$a$', '$\delta \\nu_\mathrm{B}$'], quantiles=[0.16, 0.5, 0.84], truths=best_magnetic_parameters);
                plt.savefig(magnetism_output_dir+'/'+STAR_NAME+'_magnetism_corner.png', dpi=200)
            else:
                plt.figure()
                fig = corner.corner(flat_samples[:,0:6], labels=['$\Delta \Pi_1$', '$q$', '$\epsilon_g$', '$f_\mathrm{shift}$', '$\Omega_\mathrm{core}$','$\Omega_\mathrm{env}$'], quantiles=[0.16, 0.5, 0.84], truths=best_magnetic_parameters[0:6]);
                plt.savefig(magnetism_output_dir+'/'+STAR_NAME+'_magnetism_corner.png', dpi=200)
    else:#no splittings

        flat_samples_without_rotations = flat_samples[:, [0, 1, 2, 3, 6, 7]]
        flat_samples_without_rotations = np.zeros( (len(flat_samples[:,0]), 5) )
        flat_samples_without_rotations[:,0] = flat_samples[:, 0]
        flat_samples_without_rotations[:,1] = flat_samples[:, 1]
        flat_samples_without_rotations[:,2] = flat_samples[:, 2]
        flat_samples_without_rotations[:,3] = flat_samples[:, 3]
        flat_samples_without_rotations[:,4] = (1-flat_samples[:, 6])*flat_samples[:, 7]
        chain_file_name = magnetism_output_dir+'/'+STAR_NAME+'_chain.dat'
        if if_run_MCMC:
            np.savetxt(chain_file_name, flat_samples_without_rotations, header='#Delta Pi in s, q, epsilon_g, p_shift in microHz,(1-a)*delta_nu_mag in microHz' )
            if if_consider_magnetism:
                plt.figure()
                fig = corner.corner(flat_samples_without_rotations, labels=['$\Delta \Pi_1$', '$q$', '$\epsilon_g$', '$f_\mathrm{shift}$', '$(1-a)\delta\\nu_\mathrm{B}$'], quantiles=[0.16, 0.5, 0.84], truths=[best_magnetic_parameters[0],  best_magnetic_parameters[1], best_magnetic_parameters[2], best_magnetic_parameters[3], (1-best_magnetic_parameters[6])*best_magnetic_parameters[7]   ]);#best_magnetic_parameters[[0, 1, 2, 3, 6, 7]]
                plt.savefig(magnetism_output_dir+'/'+STAR_NAME+'_magnetism_corner.png', dpi=200)
            else:
                plt.figure()
                fig = corner.corner(flat_samples_without_rotations[:, 0:4], labels=['$\Delta \Pi_1$', '$q$', '$\epsilon_g$', '$f_\mathrm{shift}$'], quantiles=[0.16, 0.5, 0.84], truths=[best_magnetic_parameters[0],  best_magnetic_parameters[1], best_magnetic_parameters[2], best_magnetic_parameters[3]   ]);#best_magnetic_parameters[[0, 1, 2, 3, 6, 7]]
                plt.savefig(magnetism_output_dir+'/'+STAR_NAME+'_magnetism_corner.png', dpi=200)
    #plt.figure()
    #plt.hist(flat_samples[:,0], bins =200)

    plt.close('all')

    plot_magnetic_stretched_echelle_diagram(one_dir, if_consider_magnetism)
    #plt.show()


def try_perturbated_g_period():
    Delta_Pi_1 = 87.9
    epsilon_g = 0.28
    magnetic_a = 0.4
    magnetic_delta_nu = 5.3
    nu_max = 246.629
    core_Omega = 5.2
    lower_period = 1/(281.7*10**(-6))
    upper_period = 1/(174.3*10**(-6))

    plt.figure(figsize=(10, 5))
    ax=plt.subplot(111)
    m_counter=-1
    color_array=['r','b', 'g']
    for one_m in [-1, 0, 1]:
        m_counter+=1
        perturbated_g_period = get_perturbated_g_period_within_certain_range(Delta_Pi_1, epsilon_g, one_m, magnetic_a, magnetic_delta_nu, nu_max, core_Omega, lower_period, upper_period)
        perturbated_g_period_spacing = np.diff(perturbated_g_period)
        print('one_m', one_m)
        print('perturbated_g_period', perturbated_g_period)
        #ax=plt.subplot(1, 3, m_counter+1)
        ax.plot(perturbated_g_period[0:-1], perturbated_g_period_spacing, color_array[m_counter]+'.', label = 'm='+str(int(one_m)) )
    ax.set_xlabel('period, s')
    ax.set_ylabel('period spacing, s')
    #ax.set_title('m='+str(int(one_m)))
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_corner_again(one_dir, if_magnetism):
    import matplotlib.pylab as pylab
    #params = {'legend.fontsize': 'x-large','figure.figsize': (15, 5),'axes.labelsize': 'x-Large','axes.titlesize': 'x-large'}
    params = {'fontsize': 'xx-large'}

    #pylab.rcParams.update(params)

    STAR_NAME = get_star_name(one_dir)
    if if_magnetism:
        chain_data = np.loadtxt(glob.glob(one_dir+'/magnetism/KIC*chain.dat')[0])
        x_labels = ['$\Delta \Pi_1$\n(s)', 'q', '$\epsilon_g$', '$f_\mathrm{shift}$\n($\mu$Hz)', '$\Omega_\mathrm{core}$\n($\mu$Hz)', '$\Omega_\mathrm{env}$\n($\mu$Hz)', '$a$', '$\delta \\nu_\mathrm{B}$\n($\mu$Hz)']
        y_labels = ['(s)\n$\Delta \Pi_1$', 'q', '$\epsilon_g$', '($\mu$Hz)\n$f_\mathrm{shift}$', '($\mu$Hz)\n$\Omega_\mathrm{core}$', '($\mu$Hz)\n$\Omega_\mathrm{env}$', '$a$', '($\mu$Hz)\n$\delta \\nu_\mathrm{B}$']
    else:
        chain_data = np.loadtxt(glob.glob(one_dir+'/non_magnetism/KIC*chain.dat')[0])
        chain_data = chain_data[:, 0:6]
        x_labels = ['$\Delta \Pi_1$\n(s)', 'q', '$\epsilon_g$', '$f_\mathrm{shift}$\n($\mu$Hz)', '$\Omega_\mathrm{core}$\n($\mu$Hz)', '$\Omega_\mathrm{env}$\n($\mu$Hz)']
        y_labels = ['(s)\n$\Delta \Pi_1$', 'q', '$\epsilon_g$', '($\mu$Hz)\n$f_\mathrm{shift}$', '($\mu$Hz)\n$\Omega_\mathrm{core}$', '($\mu$Hz)\n$\Omega_\mathrm{env}$']



    fig = my_own_corner(chain_data, x_labels, y_labels)


    #fig = plt.figure(figsize=np.array((15, 15))*0.8 )

    #corner.corner(chain_data, labels=['$\Delta \Pi_1$', 'q', '$\epsilon_g$', '$f_\mathrm{shift}$', '$\Omega_\mathrm{core}$', '$\Omega_\mathrm{env}$', '$a$', '$\delta \\nu_\mathrm{B}$'], quantiles=[0.16, 0.5, 0.84], fig = fig, max_n_ticks=3, label_kwargs = params)
    #plt.rc('axes', labelsize=20)
    #plt.rc('axes', titlesize=20)

    #plt.tight_layout()
    if if_magnetism:
        fig.savefig(one_dir+'/magnetism/' + STAR_NAME + '_magnetism_corner.png', dpi=200)
    else:
        fig.savefig(one_dir+'/non_magnetism/' + STAR_NAME + '_magnetism_corner.png', dpi=200)
    plt.show()
    #a=input()

def my_own_corner(data, x_labels, y_labels):
    #params = {'fontsize': 'xx-large'}
    #params = {'fontsize': 'xx-large'}

    #data is (N * M), N is chain length, M is parameter number
    data_shape = np.shape(data)
    print(data_shape)
    N = data_shape[0]
    M = data_shape[1]
    fig = plt.figure(figsize = np.array((10, 10))*0.6)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for line in range(M):
        for column in range(M):
            if column > line:
                continue
            ax = fig.add_subplot(M, M, line*M + column+1  )
            if line==column:#plot histogram
                ax.hist(data[:, line], color = 'k', histtype = 'step', linewidth = 0.5)
                ylim = ax.get_ylim()
                ax.set_ylim(ylim)
                sort_data = np.sort(data[:, line])
                quantity = [0.16, 0.5, 0.84]
                for one_quantity in quantity:
                    ax.plot([sort_data[int(one_quantity*N)], sort_data[int(one_quantity*N)]], ylim, 'k--' , linewidth = 0.7 )
            else:#plot scatter
                ax.scatter(data[:, column],data[:, line], color = 'k', s=0.1, alpha = 0.4)
            if line != M-1:
                ax.set_xticks([])
            else:
                ax.set_xticklabels(ax.get_xticks(), rotation=45)#rotate y ticks
                ax.xaxis.set_major_formatter('{x:9<4.3f}')#x-tick significant numbers
            if column != 0 or (column==0 and line==0):
                ax.set_yticks([])
            else:
                ax.set_yticklabels(ax.get_yticks(), rotation=45)#rotate y ticks
                ax.yaxis.set_major_formatter('{x:9<4.3f}')#y-tick significant numbers
                #ax.set_yticklabels(ax.get_yticks(), rotation=45)
            if column == 0 and line!=0:
                ax.set_ylabel(y_labels[line],fontsize=10 )
            if line == M-1:
                ax.set_xlabel(x_labels[column], fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(hspace=.1, wspace=0.1)

    plt.show()
    return fig

def copy_Sebastien_data(one_dir):#only for one ridge curved stretched echelle diagram
    KIC = str(get_star_name(one_dir).replace('KIC', ''))
    print(KIC)
    if len(glob.glob(one_dir+'/*bin_points_for_l_1_backup_m_0.dat'))==0:
        os.system('cp '+glob.glob(one_dir+'/*bin_points_for_l_1_m_0.dat')[0] + ' '+glob.glob(one_dir+'/*bin_points_for_l_1_m_0.dat')[0].replace('_m_0.dat', '_backup_m_0.dat'))
    Sebastien_data_dir = '/Volumes/ligangwork2/solar_like_oscillators/Sebastien_curved_stretched_echelle_diagram/'
    Sebastien_freq_file = glob.glob(Sebastien_data_dir+'/freq_'+KIC+'.txt')[0]
    os.system('cp '+Sebastien_freq_file+' '+one_dir+'/freq_'+KIC+'_from_Sebastien.txt')
    freq_data = np.loadtxt(Sebastien_freq_file)
    np.savetxt(glob.glob(one_dir+'/*bin_points_for_l_1_m_0.dat')[0], freq_data[:,0], header = '# points in the power spectrum, not mode frequencies, in unit of microHz')


if __name__ == '__main__':

    #plot_magnetic_stretched_echelle_diagram('/Volumes/ligangwork2/solar_like_oscillators/Gehan_sample/KIC011515377_MCMC', True)
    #exit()

    dir_list = ['./KIC008684542_MCMC']
    dir_list = sorted(dir_list, key=lambda name: int(name[name.index('KIC')+3:name.index('KIC')+12]))

    if len(glob.glob('./last_KIC_record_for_magnetic_fit.dat'))==0:
        os.system('touch ./last_KIC_record_for_magnetic_fit.dat')

    record = open('./last_KIC_record_for_magnetic_fit.dat').read().replace('\n', '')
    print('record', record)

    start_index = -1
    for one_dir in dir_list:
        start_index += 1
        #print(one_dir)
        if record in one_dir:
            break

    start_index = 0

    if_consider_magnetism = True

    for one_dir in dir_list[start_index:]:

        KIC = str( int( get_star_name(one_dir).replace('KIC', '') ) )

        print('\n\n\n\n\n')
        print(one_dir)

        print('KIC', KIC)
        record_file = open('/Volumes/ligangwork2/solar_like_oscillators/last_KIC_record_for_magnetic_fit.dat', 'w')
        record_file.write(KIC)
        record_file.close()

        fit_magnetic_shift(one_dir, if_consider_magnetism)
        plot_magnetic_stretched_echelle_diagram(one_dir, if_consider_magnetism)
        print('finish one star, sleep 10 seconds')
        tm.sleep(10)