import numpy as np
import matplotlib.pyplot as plt
import os, glob, sys
#from fit_package import fit_a_lorentz_on_one_peak, peak_lorentz
sys.path.insert(0, '/Users/ligang/Library/Mobile Documents/com~apple~CloudDocs/gammaDorcodes')
sys.path.insert(0, '/Users/ligang/Library/Mobile Documents/com~apple~CloudDocs/solarlikecodes/SYDpy-master')
sys.path.insert(0, '/Users/gangli/Library/Mobile Documents/com~apple~CloudDocs/gammaDorcodes')
sys.path.insert(0, '/Users/gangli/Library/Mobile Documents/com~apple~CloudDocs/solarlikecodes/SYDpy-master')
from get_peaks_no_MCMC import *
from power_spectrum_simulation import bin_mode_function
from fit_package import *
from fit_splittings import *
import time as tm
from astropy.io import ascii
from astropy.table import Table


def asymmetric_splitting(freq, l, i, central_frequency, splitting, amplitude, eta, central_freq_shift, background):
    m_list = [1, 0, -1]
    model = np.zeros(len(freq))
    for one_m in m_list:
        this_freq = central_frequency + one_m * splitting
        if one_m == 0:
            this_freq = central_frequency + central_freq_shift
        model = model + visibility(i, l, one_m) * peak_lorentz(freq, amplitude, this_freq, eta, 0)

    model =  model + background
    #plt.figure()
    #plt.plot(freq, model)
    #plt.show()
    #a=input()
    return model

def multi_asymmetric_splittings(freq, theta):  # theta is: central frequency, splitting, amplitude, eta, central_freq_shift, central frequency, splitting, amplitude, eta, central_freq_shift, ..... , inclination, background
    l = 1
    inclination = theta[-2]
    background = theta[-1]
    # background = 1
    model = np.zeros(len(freq))
    for i in range(0, len(theta) - 4, 5):
        a_0 = theta[i + 2]
        centre_freq = theta[i]#central frequency
        centre_freq_shift = theta[i+4]
        splitting = theta[i + 1]
        eta = theta[i + 3]
        model = model + asymmetric_splitting(freq, l, inclination, centre_freq, splitting, a_0, eta, centre_freq_shift, 0)

    model = model + background

    return model


def multi_asymmetric_splittings_likelihood(theta, freq, flat_PSD):
    #print('multi_asymmetric_splittings_likelihood')
    model = multi_asymmetric_splittings(freq, theta)
    ln_p = - np.sum(np.log(model) + flat_PSD / model)
    return ln_p


def multi_asymmetric_splittings_prior(theta, initial_theta, bin_mode_list, m_list):
    counter = -1
    for one_element in theta:
        counter+=1
        if one_element <= 0 and counter % 5 != 4:#except freq shift, everything should be positive
            #print('something negative')
            return -np.inf
    # amplitude larger than 0
    for i in range(0, len(theta) - 4, 5):
        # print(i)
        one_freq = np.array([theta[i] - theta[i + 1], theta[i]+theta[i+4], theta[i] + theta[i + 1]])
        l=1
        max_value = asymmetric_splitting(one_freq,  l, theta[-2], theta[i], theta[i+1], theta[i+2], theta[i+3], theta[i+4], theta[-1])
        if max(max_value) < 6:  # must have some amplitude!
            #print('amplitude too small')
            return -np.inf
        if max(max_value) > 500:
            return -np.inf


    # frequency, splitting, eta cannot be to deviated
    for i in range(0, len(theta) - 2, 5):
        if np.abs(theta[i] - initial_theta[i]) > 0.02:  # frequency
            #print('frequency out of range')
            return -np.inf
        if np.abs(theta[i + 1] - initial_theta[i + 1]) > 0.02:  # splitting
            #print('splitting out of range')
            return -np.inf
        if theta[i + 3] > 0.07:  # eta
            #print('eta out of range')
            return -np.inf
        if np.abs(theta[i+4] - initial_theta[i+4]) > 0.006*10:#too large shift, larger than the splitting, in unit of microHz
            return -np.inf
    # inclination between 0 and 90
    if theta[-2] < 0 or theta[-2] > 90:
        return -np.inf

    if np.abs(theta[-1] - 1) > 1:  # background
        return -np.inf

    distance_prior = 1
    for theta_index in range(0, (len(theta)-2), 5 ):
        #print(theta_index, len(theta))
        m_counter = -1
        for m in m_list:
            m_counter += 1
            model_frequency = theta[theta_index] - m * theta[theta_index+1]
            if np.min(np.abs(model_frequency-bin_mode_list[m_counter]))<0.1:
                HMFW=0.02
                distance_prior = distance_prior * HMFW**2/4/(  np.min(np.abs(model_frequency-bin_mode_list[m_counter]))**2+HMFW**2/4  )

    return np.log(  distance_prior  )

def multi_asymmetric_splittings_prob(theta, freq, flat_PSD, initial_theta, bin_mode_list, m_list):
    prior = multi_asymmetric_splittings_prior(theta, initial_theta, bin_mode_list, m_list)
    if not np.isfinite(prior):
        #print('prior -np.inf')
        return -np.inf
    else:
        return prior + multi_asymmetric_splittings_likelihood(theta, freq, flat_PSD)


# fit multiple splittings automatically
def fit_multi_asymmetric_splittings_for_one_radial_order_automatically(freq_range, freq, flat_PSD, bin_mode_list, m_list, mixed_mode_frequencies, their_zeta, mixed_mode_splittings):
    if 0 in m_list:
        initial_inclination = 30
    else:
        initial_inclination = 80

    # freq_range=[90, 96]
    # freq_range = [99, 105]
    # observed PSD
    index = np.where((freq > min(freq_range)) & (freq < max(freq_range)))[0]
    freq_to_be_fit = freq[index]
    flat_PSD_to_be_fit = flat_PSD[index]
    # calculated frequencies
    # mixed_mode_frequencies=np.array(mixed_mode_frequencies)
    index = np.where((mixed_mode_frequencies > min(freq_range)) & (mixed_mode_frequencies < max(freq_range)))[0]
    mixed_mode_frequencies_used = mixed_mode_frequencies[index]
    their_zeta_used = their_zeta[index]
    mixed_mode_splittings_used = mixed_mode_splittings[index]
    # arange initial guess
    initial_theta = []

    zeta_to_fit = []
    for i in range(len(mixed_mode_frequencies_used)):
        # print('mixed_mode_frequencies_used[i]',mixed_mode_frequencies_used[i] )
        # judge if there are good observed splittings
        m_counter = -1
        observed_freq_list = []
        local_m_component_list = []
        for one_m in m_list:  # match observed and calculated frequencies
            m_counter += 1
            calculated_freq = mixed_mode_frequencies_used[i] - one_m * mixed_mode_splittings_used[i]  # prograde is negative!!!
            close_distance = np.min(np.abs(bin_mode_list[m_counter] - calculated_freq))
            if close_distance > 0.4:  # microHz, there is no good match
                #print('There is no good match, continue')
                continue
            close_index = np.argmin(np.abs(bin_mode_list[m_counter] - calculated_freq))
            observed_freq_list.append(bin_mode_list[m_counter][close_index])
            local_m_component_list.append(one_m)
        if len(observed_freq_list) <= 2:  # cannot find a triple splitting, only one or zero or two frequencies
            #print('smaller than two peaks detected (in the function), continue')
            continue

        # the situation is m components [1, -1, 0]
        m_1_freq = observed_freq_list[np.argmin(np.abs(np.array(local_m_component_list)-1))]
        m_negative_1_freq = observed_freq_list[np.argmin(np.abs(np.array(local_m_component_list)+1))]
        central_freq = (m_1_freq+m_negative_1_freq)/2
        initial_theta.append(  central_freq  )  # central frequency, using the mean value of m=1 and -1 components
        initial_theta.append((max(observed_freq_list) - min(observed_freq_list)) / 2)  # splitting

        zeta_to_fit.append(their_zeta_used[i])
        initial_theta.append(9999)  # power=amplitude square, Mosser 2018, but not used here
        initial_theta.append(0.04 * (1 - their_zeta_used[i]))  # eta, Mosser 2018
        initial_theta[-2] = (max(flat_PSD_to_be_fit) * (1 - their_zeta_used[i]) * (1 - max(their_zeta_used)) * 4 * initial_theta[-1]) ** 0.5  # update the power again
        m_0_index = np.argmin(np.abs(np.array(local_m_component_list)-0))
        #print('m_0_index', m_0_index)
        #print('local_m_component_list', local_m_component_list)
        m_0_freq = observed_freq_list[m_0_index]
        initial_theta.append(m_0_freq - central_freq)#central frequency shift
    if len(initial_theta) <= 5 * 0:  # no pair of splittings
        print('no pair of splittings, continue')
        return [9999], [9999], [9999], [9999]

    initial_theta.append(initial_inclination)
    initial_theta.append(1)  # background is roughly one

    # generate random numbers ranges
    initial_theta_uncertainty = []
    for index in range(0, len(initial_theta) - 2):
        if index % 5 == 0:  # central frequency
            initial_theta_uncertainty.append(0.005)
        if index % 5 == 1:  # splitting
            initial_theta_uncertainty.append(0.005)
        if index % 5 == 2:  # amplitude
            initial_theta_uncertainty.append(initial_theta[index] / 2)
        if index % 5 == 3:  # eta
            initial_theta_uncertainty.append(initial_theta[index] / 20)
        if index % 5 == 4: #central frequency shift
            initial_theta_uncertainty.append(0.002)
    initial_theta_uncertainty.append(1)  # inclination
    initial_theta_uncertainty.append(0.05)  # background
    initial_theta_uncertainty = np.array(initial_theta_uncertainty)

    pos = initial_theta + initial_theta_uncertainty * np.random.randn(2 * len(initial_theta) + 2, len(initial_theta))

    with Pool() as pool:
        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(nwalkers, ndim, multi_asymmetric_splittings_prob, args=(freq_to_be_fit, flat_PSD_to_be_fit, initial_theta, bin_mode_list, m_list), pool=pool)
        sampler.run_mcmc(pos, 4000, progress=True)
    flat_samples = sampler.get_chain(discard=2000, thin=15, flat=True)
    best_theta = np.median(flat_samples, axis=0)
    best_theta_std = np.std(flat_samples, axis=0)

    # plt.figure()
    # plt.plot(freq_to_be_fit, flat_PSD_to_be_fit)
    final_model = multi_asymmetric_splittings(freq_to_be_fit, best_theta)
    initial_model = multi_asymmetric_splittings(freq_to_be_fit, initial_theta)
    # plt.plot(freq_to_be_fit, final_model, label = 'final model')
    # plt.plot(freq_to_be_fit, initial_model, label = 'initial model')
    # plt.legend()
    # plt.title('final fit')

    # plt.figure()
    # zeta_counter = -1
    # for one_index in range(1, len(best_theta)-2, 4):
    #    zeta_counter+=1
    #    plt.plot(zeta_to_fit[zeta_counter], best_theta[one_index], 'o')
    # plt.title('splitting zeta')
    # fig=corner.corner(flat_samples)

    # a=input()
    return best_theta, best_theta_std, zeta_to_fit, flat_samples


def fit_multi_asymmetric_splittings_automatically_shell(freq, flat_PSD, bin_mode_list, m_list, mixed_mode_frequencies, their_zeta, mixed_mode_splittings, nu_max, asymp_para, Delta_Pi, tau, stretched_freq, asymmetric_splitting_work_dir, STAR_TYPE):
    central_radial_order = int(nu_max / asymp_para[0])
    fit_radial_order = np.arange(central_radial_order - 3, central_radial_order + 4)
    l_0_freq_list = []
    for one_n in fit_radial_order:
        l_0_freq_list.append(asymptotic_relation_pure_p_mode(one_n, 0, nu_max, *asymp_para))
    l_0_freq_list = np.array(l_0_freq_list)

    # judge if this radial order will be fitted, if there are serious overlap, then do not fit
    grid_freq_with_different_m = []
    tau_with_different_m_list = []
    for m_index in range(len(m_list)):
        this_m = m_list[m_index]
        grid_freq_with_different_m.append(mixed_mode_frequencies - this_m * mixed_mode_splittings)
        # get tau values of them
        tau_list = []
        for one_freq in grid_freq_with_different_m[-1]:
            closed_index = np.argmin(np.abs(one_freq - stretched_freq))
            tau_list.append(tau[closed_index])
        tau_with_different_m_list.append(tau_list)

    plt.figure()
    x = []  # tau mod Delta_Pi
    y = []  # frequency
    for m_index in range(len(m_list)):
        x.append(tau_with_different_m_list[m_index] % Delta_Pi)
        y.append(grid_freq_with_different_m[m_index])
        plt.plot(x[-1], y[-1], '.')
    #plt.show()
    plt.close()
    # a=input('pause to see the calculated stretched echelle diagram')
    # plt.close()
    # find the closed place
    width_criterion = 0.1  # microHz
    closed_freq_list = []
    for first_index in range(len(m_list)):
        for second_index in range(first_index + 1, len(m_list)):
            # print('first_index', 'second_index', first_index, second_index)
            for one_freq in grid_freq_with_different_m[second_index]:
                # print(one_freq)
                if np.min(np.abs(grid_freq_with_different_m[first_index] - one_freq)) < width_criterion:
                    closed_freq_list.append(one_freq)  # find a freq
    # arange radial order
    closed_freq_n_list = []
    for one_close_freq in closed_freq_list:
        index = np.argmin(np.abs(one_close_freq - l_0_freq_list))
        if one_close_freq < l_0_freq_list[index]:
            closed_freq_n_list.append(fit_radial_order[index - 1])
        else:
            closed_freq_n_list.append(fit_radial_order[index])

    # print('closed_freq_list', closed_freq_list)
    for index in range(len(fit_radial_order) - 1, -1, -1):
        if len(np.where(fit_radial_order[index] == closed_freq_n_list)[0]) >= 3:  # more than 3 peaks overlapping
            fit_radial_order = np.delete(fit_radial_order, index)

    if STAR_TYPE == 'RC':
        fit_radial_order = np.arange(central_radial_order - 3, central_radial_order + 4)

    fit_radial_order = np.arange(central_radial_order - 3, central_radial_order + 4)


    ascii.write([fit_radial_order], asymmetric_splitting_work_dir.replace('asymmetric_solar_like_splittings', '') + '/asymmetric_splitting_fit_radial_order.dat', names=['#the radial orders being fitted'], overwrite=True)
    fit_radial_order = np.loadtxt(asymmetric_splitting_work_dir.replace('asymmetric_solar_like_splittings', '') + '/asymmetric_splitting_fit_radial_order.dat', skiprows=1).astype('int')
    print('these radial orders will be fitted', fit_radial_order)
    # a=input()

    freq_range_list = []
    best_parameter_list = []
    flat_samples_list = []

    splitting_counter = -1
    for one_fit_radial_order in fit_radial_order:
        l_0_freq = asymptotic_relation_pure_p_mode(one_fit_radial_order, 0, nu_max, *asymp_para)
        l_2_freq = asymptotic_relation_pure_p_mode(one_fit_radial_order, 2, nu_max, *asymp_para)
        freq_range = [l_0_freq + 0.02 * asymp_para[0], l_2_freq - 0.05 * asymp_para[0]]

        print('Now the radial order is', one_fit_radial_order)
        print('freq_range', freq_range)

        how_many_peaks_detected_in_this_range = 0
        for one_bin_mode_list in bin_mode_list:  # loop in azimuthal order m
            index = np.where((one_bin_mode_list > min(freq_range)) & (one_bin_mode_list < max(freq_range)))[0]
            how_many_peaks_detected_in_this_range = how_many_peaks_detected_in_this_range + len(index)
        if how_many_peaks_detected_in_this_range <= 1:
            print('smaller than two peaks are detected, continue')
            continue
        best_theta, best_theta_std, zeta_to_fit, flat_samples = fit_multi_asymmetric_splittings_for_one_radial_order_automatically(freq_range, freq, flat_PSD, bin_mode_list, m_list, mixed_mode_frequencies, their_zeta,mixed_mode_splittings)
        if best_theta[0] == 9999:
            print('There is no good solution, continue')
            continue
        freq_range_list.append(freq_range_list)
        best_parameter_list.append(best_theta)
        flat_samples_list.append(flat_samples)
        for index in range(0, len(best_theta) - 2, 5):
            splitting_counter += 1

            # plot corners
            chains = np.zeros((flat_samples.shape[0], 7))
            chains[:, 0] = flat_samples[:, -2]  # inclination
            chains[:, 1] = flat_samples[:, index + 2]  # amplitude
            chains[:, 2] = flat_samples[:, index]  # central freq
            chains[:, 3] = flat_samples[:, index + 1]  # splitting
            chains[:, 4] = flat_samples[:, index + 3]  # eta
            chains[:, 5] = flat_samples[:, index + 4]   #central freq shift
            chains[:, 6] = flat_samples[:, -1]  # background
            fig = corner.corner(chains, labels=['i', 'a_0', 'centre_freq', 'splitting', 'eta', 'shift', 'background'], quantiles=[0.16, 0.5, 0.84], show_titles=True)
            fig.savefig(asymmetric_splitting_work_dir + '/corner_No_' + str(splitting_counter) + '_zeta_' + str(zeta_to_fit[int(index / 5)]) + '.png')
            plt.close(fig)

            # plot fit result for each pair of splitting
            this_central_freq, this_splitting, this_amplitude, this_eta, this_shift = best_theta[index], best_theta[index + 1], best_theta[index + 2], best_theta[index + 3], best_theta[index + 4]
            inclination, background = best_theta[-2], best_theta[-1]
            plot_freq_range = [this_central_freq - 1.5 * this_splitting, this_central_freq + 1.5 * this_splitting]
            freq_index = np.where((freq > min(plot_freq_range)) & (freq < max(plot_freq_range)))[0]
            plt.figure()
            plt.plot(freq[freq_index], flat_PSD[freq_index], 'gray', alpha=0.3)
            one_asymmetric_splitting_model = asymmetric_splitting(freq[freq_index], 1, inclination, this_central_freq, this_splitting, this_amplitude, this_eta, this_shift, background)
            plt.plot(freq[freq_index], one_asymmetric_splitting_model, 'k')
            plt.xlabel('Freq, microHz')
            plt.ylabel('S/N')
            plt.savefig(asymmetric_splitting_work_dir + '/fit_result_No_' + str(splitting_counter) + '_zeta_' + str(zeta_to_fit[int(index / 5)]) + '.png')
            plt.close()

            # save this splitting result
            output_splitting_fid = open(asymmetric_splitting_work_dir + '/fit_result_No_' + str(splitting_counter) + '_zeta_' + str(zeta_to_fit[int(index / 5)]) + '.dat', 'w')
            output_splitting_fid.write('inclination in degrees,amplitude in S/N,centre_freq in microHz,splitting in microHz,eta in microHz, central frequency shift in microHz, background in S/N \n')
            write_down_index = [-2, index + 2, index, index + 1, index + 3, index + 4, -1]
            for one_write_down_index in write_down_index:
                output_splitting_fid.write(str(best_theta[one_write_down_index]) + ' ')
            output_splitting_fid.write('\n')
            for one_write_down_index in write_down_index:
                output_splitting_fid.write(str(best_theta_std[one_write_down_index]) + ' ')
            output_splitting_fid.close()

        # save the result for the whole PSD for each radial order
        plt.figure(figsize=(12, 8))
        plt.plot(freq, flat_PSD, 'gray', alpha=0.5)  # PSD
        plt.plot(freq, multi_asymmetric_splittings(freq, best_theta), 'k', label='Best model')  # best model
        max_ylim = max(flat_PSD[np.where((freq > min(freq_range)) & (freq < max(freq_range)))[0]])

        # plot identified observed frequencies and modes
        # bin_mode_list, m_list
        for m_index in range(len(m_list)):
            current_m = m_list[m_index]
            for one_freq in bin_mode_list[m_index]:
                closed_index = np.argmin(np.abs(freq - one_freq))
                if current_m == 0:
                    color = 'b'
                    marker = 'o'
                if current_m == +1:
                    color = 'g'
                    marker = '+'
                if current_m == -1:
                    color = 'r'
                    marker = '_'
                plt.plot(freq[closed_index], flat_PSD[closed_index], color + marker, markersize=10, markeredgewidth=2)
        # plot calculated frequencies and splittings on that
        # mixed_mode_frequencies, their_zeta, mixed_mode_splittings
        for index in range(len(mixed_mode_frequencies)):
            x1, x2 = min(freq_range), max(freq_range)
            y1, y2 = 0.5 * max_ylim, 0.9 * max_ylim
            y_location = (y2 - y1) / (x2 - x1) * (mixed_mode_frequencies[index] - x1)
            plt.plot([mixed_mode_frequencies[index] - mixed_mode_splittings[index], mixed_mode_frequencies[index], mixed_mode_frequencies[index] + mixed_mode_splittings[index]], [y_location, y_location, y_location], 'r.-')

        plt.ylim([0, max_ylim])
        plt.xlabel('Freq, microHz')
        plt.ylabel('S/N')
        plt.legend()
        plt.xlim(freq_range)
        plt.tight_layout()
        plt.savefig(asymmetric_splitting_work_dir + '/whole_result_for_radial_order_' + str(one_fit_radial_order) + '.png',dpi=500)







def find_asymmetric_splittings(one_dir):
    STAR_NAME = get_star_name(one_dir)
    asymmetry_output_dir = one_dir+'/asymmetry'


    if len(glob.glob(asymmetry_output_dir)) == 0:
        os.system('mkdir '+asymmetry_output_dir)
    #load PSD
    print(one_dir+'/'+STAR_NAME+'_PSD.dat')
    PSD_data = np.loadtxt(glob.glob(one_dir+'/'+STAR_NAME+'_PSD.dat')[0], skiprows=1)
    #plt.figure()
    #plt.plot(PSD_data[:,0], PSD_data[:,1])
    #plt.show()
    #load harvey
    my_harvey = np.loadtxt(glob.glob(one_dir + '/*myself_harvey.dat')[0], skiprows=1)
    #generate flat PSD
    freq, flat_PSD = PSD_data[:,0], PSD_data[:, 1]/my_harvey[:, 1]
    #load p-mode fit
    asymp_para = np.loadtxt(glob.glob(one_dir+'/*asymptotic_relation_fit.dat')[0], skiprows=1)
    asymp_para = asymp_para[:, 0]
    delta_nu = asymp_para[0]
    if len(glob.glob(one_dir + '/' + STAR_NAME + '_nu_max_by_YuJie_because_numax_large.dat')) > 0:
        nu_max = np.loadtxt(one_dir + '/' + STAR_NAME + '_nu_max_by_YuJie_because_numax_large.dat')
    else:
        nu_max = np.loadtxt(one_dir + '/' + STAR_NAME + '_nu_max_by_me.dat')    #load g-mode fit
    optimized_parameters = np.loadtxt(glob.glob(one_dir+'/*grid_optimized_parameters.dat')[0])
    Delta_Pi = optimized_parameters[0, 0]
    #plt.figure()
    #plt.plot(freq, flat_PSD)
    #plt.show()
    #load zeta
    stretched_PSD = np.loadtxt( glob.glob(one_dir+'/*stretched_PSD_updated.dat')[0], skiprows=1 )
    tau = stretched_PSD[:,0]
    stretched_freq = stretched_PSD[:,1]

    #load theoretical results
    theo_result = np.loadtxt(glob.glob(one_dir+'/*best_fit_dipole_freq_from_grid.dat')[0], skiprows=1)
    #load observed peaks
    m_1_list = np.loadtxt(glob.glob(one_dir+'/*bin_points_for_l_1_m_1.dat')[0], skiprows=1)
    m_0_list = np.loadtxt(glob.glob(one_dir+'/*bin_points_for_l_1_m_0.dat')[0], skiprows=1)
    m_negative_1_list = np.loadtxt(glob.glob(one_dir+'/*bin_points_for_l_1_m_-1.dat')[0], skiprows=1)

    m_list = [1, 0, -1]
    selected_mode_list = [m_1_list, m_0_list, m_negative_1_list]
    bin_mode_list = bin_mode_function(selected_mode_list)
    #fit each peaks
    fit_width = 0.2#microHz
    search_width = 0.2 # microHz
    mixed_mode_frequencies, their_zeta, mixed_mode_splittings = theo_result[:, 0], theo_result[:,1], theo_result[:,2]

    fit_multi_asymmetric_splittings_automatically_shell(freq, flat_PSD, bin_mode_list, m_list, mixed_mode_frequencies, their_zeta, mixed_mode_splittings, nu_max, asymp_para, Delta_Pi, tau, stretched_freq, asymmetry_output_dir, 'RGB')
    return 0






    freq_counter = -1
    for one_theo_freq in theo_result[:, 0]:  # negative m is prograde!!!!!!!
        freq_counter += 1
        m_counter = -1

        local_zeta = theo_result[freq_counter, 1]
        print('local_zeta', local_zeta)
        fit_width = theo_result[freq_counter, 2]*1.2 #half of the local splittings

        if_completed_splitting = True
        for one_m in m_list:
            m_counter += 1
            if np.min( np.abs( one_theo_freq - one_m * theo_result[freq_counter, 2] -  bin_mode_list[m_counter]) ) > search_width:#cannot find one component in this splitting
                if_completed_splitting = False
                break
        if if_completed_splitting == False:#there is no three component
            print('cannot find three components')
            continue
        print('find three components, try to fit them')
        #fit one lorentz on it
        m_counter = -1
        plt.figure()
        plt.plot(freq, flat_PSD, 'gray', alpha = 0.4)
        freq_range = [ one_theo_freq - 1.5 * theo_result[freq_counter, 2], one_theo_freq + 1.5 * theo_result[freq_counter, 2] ]
        print('freq_range', freq_range)
        plt.xlim(freq_range)
        plt.ylim([0, np.max(flat_PSD[np.where((freq > min(freq_range)) & (freq < max(freq_range)) )[0]]) ] )
        for one_m in m_list:
            m_counter += 1
            index = np.argmin( np.abs( one_theo_freq -one_m * theo_result[freq_counter, 2]-  bin_mode_list[m_counter]) )
            central_freq = bin_mode_list[m_counter][index]
            index_to_fit = np.where( (freq - central_freq> -0.5*fit_width) & (freq - central_freq < 0.5*fit_width)  )[0]
            input_background = 1
            eta_guess = 0.1*(1-local_zeta)#microhz
            _, _, _, lorentz_parameters, lorentz_paramters_err = fit_a_lorentz_on_one_peak(freq[index_to_fit], flat_PSD[index_to_fit], input_background, eta_guess)
            plt.plot(freq[index_to_fit], peak_lorentz(freq[index_to_fit], * lorentz_parameters), 'k')
        plt.show()

    return 0

def plot_asymmetry_vs_frequency(one_dir):
    STAR_NAME = get_star_name(one_dir)
    asymmetry_dir = one_dir+'/asymmetry'
    if len(glob.glob(asymmetry_dir))==0:
        os.mkdir(asymmetry_dir)
    central_freq, central_freq_err = [], []
    freq_shift, freq_shift_err = [], []
    splitting, splitting_err = [], []

    data_list = glob.glob(asymmetry_dir+'/fit_result*.dat')
    for one_data in data_list:
        #print(one_data)
        fit_data = np.loadtxt(one_data, skiprows=1)
        central_freq.append(fit_data[0, 2])
        central_freq_err.append(fit_data[1, 2])
        splitting.append(fit_data[0, 3])
        splitting_err.append(fit_data[1, 3])
        freq_shift.append(fit_data[0, 5])
        freq_shift_err.append(fit_data[1, 5])

    sort_index = np.argsort(central_freq)

    central_freq = np.array(central_freq)[sort_index]
    central_freq_err = np.array(central_freq_err)[sort_index]
    splitting = np.array(splitting)[sort_index]
    splitting_err = np.array(splitting_err)[sort_index]
    freq_shift = np.array(freq_shift)[sort_index]
    freq_shift_err = np.array(freq_shift_err)[sort_index]
    #print(freq_shift_err)
    eigenfrequency = central_freq + 3 * freq_shift#assuming there is no any shift
    #eigenfrequency = central_freq
    eigenfrequency_err = ( central_freq_err**2+(3*freq_shift_err)**2 )**0.5

    asymmetry_list = - 2 * freq_shift #defined by Eq.2 in Deheuvels 2017
    asymmetry_list_err = 2*freq_shift_err

    relative_asymmetry_list = - freq_shift  /splitting #defined by Eq.2 in Deheuvels 2017
    relative_asymmetry_list_err = ( (freq_shift_err/splitting)**2+(freq_shift/splitting**2*splitting_err)**2 )**0.5

    #load zeta value
    if len(glob.glob(one_dir+'/magnetism/*best_zeta.dat'))>0:
        zeta_data = np.loadtxt(glob.glob(one_dir+'/magnetism/*best_zeta.dat')[0], skiprows=1)
    else:
        zeta_data = np.loadtxt(glob.glob(one_dir + '/*stretched_PSD_updated.dat')[0], skiprows=1)
    zeta_list = []
    zeta_err = []
    eigenfrequency_counter = -1
    for one_eigenfrequency in eigenfrequency:
        eigenfrequency_counter += 1
        close_index = np.argmin(np.abs(zeta_data[:,1]-one_eigenfrequency))
        zeta_list.append(zeta_data[close_index, 3])
        #consider zeta error
        error_index = np.where((zeta_data[:,1] > one_eigenfrequency - eigenfrequency_err[eigenfrequency_counter]) & (zeta_data[:,1] < one_eigenfrequency + eigenfrequency_err[eigenfrequency_counter]))[0]
        zeta_err.append(  0.5*(max(zeta_data[error_index, 3])-min(zeta_data[error_index, 3]))  )

    #fit splitting-zeta relation
    # save the data
    ascii.write([zeta_list, zeta_err, splitting, splitting_err,central_freq, central_freq_err, eigenfrequency, eigenfrequency_err, freq_shift, freq_shift_err, asymmetry_list, asymmetry_list_err], one_dir + '/asymmetry/' + STAR_NAME + '_zeta_splitting_values.dat',names=['#zeta', 'zeta err', 'splitting', 'splitting err in microHz', 'central freq in microHz', 'central freq err', 'eigen freq', 'eigen freq err', 'freq shift', 'freq shift err', 'fractional asymmetry', 'fractional asymmetry err'],overwrite=True)
    #run MCMC
    linear_para = np.polyfit(zeta_list, splitting, 1)
    initial_surface_Omega = linear_para[1]*2*np.pi
    loop_time = 1000
    slope_list = []
    intercept_list = []
    for counter in range(loop_time):
        this_zeta = zeta_list + zeta_err * np.random.normal(0, 1, size=len(zeta_list))
        this_splitting = splitting + splitting_err * np.random.normal(0, 1, size=len(splitting))
        fit_parameter = np.polyfit(this_zeta, this_splitting, 1)
        slope_list.append(fit_parameter[0])
        intercept_list.append(fit_parameter[1])
    slope_list = np.array(slope_list)
    intercept_list = np.array(intercept_list)

    flat_samples = np.zeros((loop_time, 2))
    flat_samples[:, 1] = intercept_list * 2 * np.pi  # surface rotation Omega
    flat_samples[:, 0] = (slope_list + intercept_list) * 2 * 2 * np.pi  # core rotation Omega

#plot splitting-zeta linear relation
    best_rotation_fit = np.median(flat_samples, axis = 0)
    best_rotation_fit_std = np.std(flat_samples, axis = 0)
    fig=corner.corner(flat_samples, labels=['$\Omega_\mathrm{core}$', '$\Omega_\mathrm{surface}$'])
    fig.savefig(one_dir+'/asymmetry/'+STAR_NAME+'_zeta_splitting_corner.png', dpi=150)
    plt.close()
    print('best rotation rates', best_rotation_fit)
    plt.figure(figsize = (5, 4))
    plt.errorbar(zeta_list, splitting, xerr = zeta_err, yerr=splitting_err, fmt='.', color = 'k')
    arti_zeta = np.linspace(min(zeta_list), max(zeta_list), 10)
    random_index = np.random.randint(0, len(flat_samples[:,0]), 100)
    for one_random_index in random_index:
        one_random_index = int(one_random_index)
        plt.plot(  arti_zeta, splitting_zeta_model(arti_zeta, flat_samples[one_random_index, 0], flat_samples[one_random_index, 1]) , '-', c = 'gray', alpha = 0.03 )
    plt.xlabel('$\zeta$')
    plt.ylabel('Splitting ($\mu$Hz)')
    plt.tight_layout()
    plt.savefig(one_dir+'/asymmetry/'+STAR_NAME+'_zeta_splitting.png', dpi=150)
    plt.close()
    np.savetxt(one_dir+'/asymmetry/'+STAR_NAME+'_zeta_splitting_chain.dat', flat_samples, header='Core omega, surface omega, in radians microHz')
    output_fid = open(one_dir+'/asymmetry/'+STAR_NAME+'_zeta_splitting_best_results.dat', 'w')
    output_fid.write('#Core Omega, surface omega, in radians microHz, second line is STD\n')
    output_fid.write(str(best_rotation_fit[0])+' '+str(best_rotation_fit[1])+'\n')
    output_fid.write(str(best_rotation_fit_std[0])+' '+str(best_rotation_fit_std[1])+'\n')
    output_fid.close()



    plt.figure(figsize=(5,4))
    ax = plt.subplot(111)
    ax.errorbar(eigenfrequency, asymmetry_list, xerr=central_freq_err, yerr=asymmetry_list_err, fmt='-', color='gray', alpha=0.5)
    ax.plot(eigenfrequency, asymmetry_list, 'o', c='dimgray')
    x_lim = ax.get_xlim()
    ax.set_xlim(x_lim)
    ax.plot(x_lim, [0, 0], ':', c = 'gray')
    y_lim = ax.get_ylim()
    ax.set_ylim(y_lim)

    #plot zeta
    index = np.where((zeta_data[:,1]>min(x_lim)) & (zeta_data[:,1]<max(x_lim)))[0]
    min_zeta, max_zeta = min( zeta_data[index,3] ), max( zeta_data[index,3] )
    y_number = 200
    x_array = []
    y_array = []
    zeta_colar_array = []

    print('Plotting shaded area for zeta, need some time')
    for one_index in index:

        x_array.append(list(np.zeros(y_number)+zeta_data[one_index,1]))
        y_array.append(    list(   np.linspace(min(y_lim), max(y_lim), y_number)   )   )

        zeta_for_color = -(zeta_data[one_index,3]-max_zeta)/(min_zeta-max_zeta)+1 #max zeta corresponds 1, min zeta corresponds 0
        zeta_for_color = 1-zeta_for_color #reverse
        zeta_colar_array.append(  list(np.zeros(y_number)-zeta_data[one_index,3])   )

    #ax.scatter(x_array, y_array, c=-(-np.array(zeta_colar_array))**0.5, cmap='Greys', alpha = 0.1, zorder = -10)#different color bar scale
    ax.scatter(x_array, y_array, c=zeta_colar_array, cmap='Greys', alpha = 0.1, zorder = -10)
    plt.xlabel('Frequency ($\mu$Hz)')
    plt.ylabel('$\delta_\mathrm{asym}$ ($\mu$Hz)')
    plt.tight_layout()
    plt.savefig(asymmetry_dir+'/'+STAR_NAME+'_asymmetry_vs_freq.png', dpi=200)

    plt.figure(figsize=(5, 4))
    ax = plt.subplot(111)
    ax.errorbar(zeta_list, asymmetry_list, xerr=zeta_err, yerr=asymmetry_list_err, fmt='.', color='gray', alpha=0.5)
    ax.plot(zeta_list, asymmetry_list, 'ko')
    x_lim = ax.get_xlim()
    ax.set_xlim(x_lim)
    ax.plot(x_lim, [0, 0], ':', c = 'gray')
    plt.xlabel('$\zeta$')
    plt.ylabel('$\delta_\mathrm{asym}$ ($\mu$Hz)')
    plt.tight_layout()
    plt.savefig(asymmetry_dir+'/'+STAR_NAME+'_asymmetry_vs_zeta.png', dpi=200)
    #plt.show()
    plt.close()
    print('here here here')


def fit_splitting_vs_zeta_for_magnetic_stars(one_dir):

    STAR_NAME = get_star_name(one_dir)
    if len(glob.glob(one_dir + '/' + STAR_NAME + '_nu_max_by_YuJie_because_numax_large.dat')) > 0:
        nu_max = np.loadtxt(one_dir + '/' + STAR_NAME + '_nu_max_by_YuJie_because_numax_large.dat')
    else:
        nu_max = np.loadtxt(one_dir + '/' + STAR_NAME + '_nu_max_by_me.dat')

    p_mode_parameters = np.loadtxt(glob.glob(one_dir+'/*_asymptotic_relation_fit.dat')[0], skiprows=1)
    g_mode_parameters = np.loadtxt(glob.glob(one_dir+'/magnetism/*_magnetic_parameters_result.dat')[0], skiprows=1)
    asymmetric_splitting_fit_file_list = glob.glob(one_dir+'/asymmetry/fit_result*.dat')
    asymmetric_splitting_fit_result = np.zeros(  (2, len(asymmetric_splitting_fit_file_list), 7 )  )#first dimention: 0 for value, 1 for error. second dimention: number of splittings. third dimention: seven parameters
    counter=-1
    for one_fit_file in asymmetric_splitting_fit_file_list:
        counter+=1
        data = np.loadtxt(one_fit_file, skiprows=1)
        asymmetric_splitting_fit_result[0, counter, :] = data[0, :]
        asymmetric_splitting_fit_result[1, counter, :] = data[1, :]

    loop_time = 400
    zeta_array = np.zeros( (loop_time, len(asymmetric_splitting_fit_file_list)) )

    spliting_list = asymmetric_splitting_fit_result[0, :, 3]
    spliting_uncertainty_list = asymmetric_splitting_fit_result[1, :, 3]

    slope_list = []
    intercept_list = []


    for loop_counter in range(loop_time):
        for splitting_counter in range(len(asymmetric_splitting_fit_file_list)):
            radial_order_max = int(nu_max/p_mode_parameters[0, 0])
            radial_order_list = np.arange(radial_order_max-3, radial_order_max+4)
            perturbed_p_parameters = p_mode_parameters[:, 0] + p_mode_parameters[:, 1] * np.random.randn(len(p_mode_parameters[:, 1]))
            perturbed_g_parameters = g_mode_parameters[0, :] + g_mode_parameters[1, :] * np.random.randn(len(g_mode_parameters[1, :]))

            nu_p = asymptotic_relation_pure_p_mode(radial_order_list, 1, nu_max, *perturbed_p_parameters) + perturbed_g_parameters[3]


            #calculate the m=0 frequency before magnetic perturbations
            #assuming that the shift of m=0 modes is half the shift of m=+-1 modes
            #nu_0'=nu_0+dx
            #nu_1'=nu_0+2dx+dnu
            #solve this equation
            nu_0_before_magnetic_perturbation = asymmetric_splitting_fit_result[0, splitting_counter, 2] + 2*asymmetric_splitting_fit_result[0, splitting_counter, 5]
            uncertainty_nu_0_before_magnetic_perturbation = ( (asymmetric_splitting_fit_result[1, splitting_counter, 2])**2+(2*asymmetric_splitting_fit_result[1, splitting_counter, 5])**2 )**0.5
            perturbed_nu_0_before_magnetic_perturbation = nu_0_before_magnetic_perturbation + uncertainty_nu_0_before_magnetic_perturbation*np.random.randn(1)
            freq_list = np.array([perturbed_nu_0_before_magnetic_perturbation])
            theta_p = theta_p_function(freq_list, nu_p)
            zeta = zeta_function(freq_list, perturbed_g_parameters[1], theta_p, perturbed_p_parameters[0], perturbed_g_parameters[0])
            zeta_array[loop_counter, splitting_counter] = zeta[0]

        #fit a straight line
        linear_fit_parameter = np.polyfit(zeta_array[loop_counter, :], spliting_list, 1)
        slope_list.append( linear_fit_parameter[0] )
        intercept_list.append( linear_fit_parameter[1] )

    slope_list = np.array(slope_list)
    intercept_list = np.array(intercept_list)
    surface_rotation_list = intercept_list*2*np.pi
    core_rotation_list = 2*(slope_list+intercept_list)*2*np.pi

    output_file_name = one_dir+'/asymmetry/'+STAR_NAME+'_zeta_splitting_best_results.dat'
    zeta_splitting_best_results_fid=open(output_file_name, 'w')
    zeta_splitting_best_results_fid.write('#Core Omega, surface omega, in radians microHz, second line is STD\n')
    zeta_splitting_best_results_fid.write(str(np.median(core_rotation_list))+' '+str(np.median(surface_rotation_list))+'\n')
    zeta_splitting_best_results_fid.write(str(np.std(core_rotation_list))+' '+str(np.std(surface_rotation_list)))
    zeta_splitting_best_results_fid.close()


    corner_array = np.zeros((loop_time, 2))
    corner_array[:, 0] = core_rotation_list
    corner_array[:, 1] = surface_rotation_list

    zeta_list = np.mean( zeta_array, axis=0 )
    zeta_uncertainty_list = np.std( zeta_array, axis=0 )

    output_file_name = one_dir+'/asymmetry/'+STAR_NAME+'_zeta_splitting_uncertainties_new.dat'
    ascii.write([zeta_list, zeta_uncertainty_list, spliting_list, spliting_uncertainty_list], output_file_name, names=['zeta', 'zeta uncertainty', 'splitting in microHz', 'splitting uncertainty'], overwrite=True)

    output_file_name = one_dir+'/asymmetry/'+STAR_NAME+'_zeta_splitting_chain.dat'
    ascii.write(corner_array, output_file_name, names=['# Core omega', 'surface omega, in radians microHz'], overwrite=True)


    corner_fig = corner.corner(corner_array, labels=['$\Omega_\mathrm{core}$ ($\mu$Hz)', '$\Omega_\mathrm{env}$ ($\mu$Hz)'], quantiles=[0.16, 0.5, 0.84], figsize=(5, 5), show_titles=True)
    corner_fig.savefig(one_dir+'/asymmetry/'+STAR_NAME+'_zeta_splitting_corner.png')


    plt.figure(figsize=(5, 4))
    plt.errorbar(zeta_list, spliting_list, xerr=zeta_uncertainty_list, yerr=spliting_uncertainty_list, fmt='k.')
    for loop_counter in range(0, loop_time, 2):
        plt.plot(np.sort(zeta_list), np.polyval([slope_list[loop_counter], intercept_list[loop_counter]], np.sort(zeta_list)),  'k-', alpha=0.01)
    plt.xlabel('$\zeta$')
    plt.ylabel(r'$\delta\nu$ ($\mu$Hz)')
    plt.tight_layout()
    plt.savefig(one_dir+'/asymmetry/'+STAR_NAME+'_zeta_splitting.png')
    plt.show()


if __name__ == '__main__':

    #fit_splitting_vs_zeta_for_magnetic_stars('/Volumes/ligangwork2/solar_like_oscillators/Gehan_sample/KIC011515377_MCMC')
    #exit()

    dir_list = ['./KIC008684542_MCMC']


    dir_list = sorted(dir_list, key=lambda name: int(name[name.index('KIC')+3:name.index('KIC')+12]))
    record_file_name = '/Volumes/ligangwork2/solar_like_oscillators/last_KIC_record_for_asymmetry.dat'

    nice_asymmetry_star_list = open('/Volumes/ligangwork2/solar_like_oscillators/undoubted_asymmetry_stars.dat').read()


    wanted_star = '5696081'
    start_index = -1
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
    triplets_counter=0
    for one_dir in dir_list[start_index:]:
        plt.close('all')

        STAR_NAME = get_star_name(one_dir)
        #os.system('open '+one_dir)

        if STAR_NAME.replace('KIC00', '').replace('KIC0', '') not in nice_asymmetry_star_list:
            continue

        print('\n\n\n\n\n\n*****************************')
        print(one_dir)

        record_file = open(record_file_name, 'w')
        record_file.write(STAR_NAME+'\n')
        record_file.close()


        if len(glob.glob(one_dir+'/*bin_points_for_l_1_m*.dat'))<3:
            continue

        else:
            triplets_counter+=1
        #continue




        #if len(glob.glob(one_dir+'/asymmetry/fit_result*dat'))==0:
        #    continue

        print('find triplets, now detect asymmetry')
        #plot_asymmetry_vs_frequency(one_dir)
        #find_asymmetric_splittings(one_dir)
        #plot_asymmetry_vs_frequency(one_dir)

        #continue

        #if len(glob.glob(one_dir+'/asymmetry'))==1:
        #    os.system('rm -rf '+one_dir+'/asymmetry')
        try:
            find_asymmetric_splittings(one_dir)
            #fit_splitting_vs_zeta_for_magnetic_stars(one_dir)
            plot_asymmetry_vs_frequency(one_dir)
        except:
            print('something wrong')

        #tm.sleep(10)

    #print('triplets_counter', triplets_counter)
    #print('how many dir', len(dir_list))