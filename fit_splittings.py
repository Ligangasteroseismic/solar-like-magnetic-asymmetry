import numpy as np
import matplotlib.pyplot as plt
import glob, os
from astropy.io import ascii
from fit_package import *
import emcee
from multiprocessing import Pool

import sys
sys.path.insert(0, '/Users/ligang/Library/Mobile Documents/com~apple~CloudDocs/gammaDorcodes')
sys.path.insert(0, '/Users/ligang/Library/Mobile Documents/com~apple~CloudDocs/solarlikecodes/SYDpy-master')
sys.path.insert(0, '/Users/gangli/Library/Mobile Documents/com~apple~CloudDocs/gammaDorcodes')
sys.path.insert(0, '/Users/gangli/Library/Mobile Documents/com~apple~CloudDocs/solarlikecodes/SYDpy-master')

from get_peaks_no_MCMC import *

#def select_modes_automatically(flat_PSD, bin_mode_list, m_list, n_p, nu_max, asymp_para, g_mode_para):
    #Delta_Pi, q, epsilon_g, one_p_shift, Omega_g = g_mode_para
    #mixed_mode_frequencies, their_zeta = implicit_asymp_relation_for_mixed_modes(n_p, nu_max, asymp_para, Delta_Pi, epsilon_g, q, one_p_shift)
    #mixed_mode_splittings = their_zeta * Omega_g/2/2/np.pi

    #selected_criterion = 1 #microHz
    #for index in range(len(mixed_mode_frequencies)):


def selected_splittings_manually(freq, flat_PSD, echelle_fig, ax_list):
    print('Now click on the echelle diagram to select modes!')

    my_m_list = [1, -1] #m=0 is always in the middle so no need to select

    if_exit_large_loop, if_exit_small_loop = 0, 0

    large_counter = -1
    boundary_array=[] #this array records the location of m=+-1 modes!!!

#click to select splittings
    select_width = 1 #microHz
    while if_exit_large_loop != '9999':
        large_counter+=1
        #determine a pair of splitting
        small_counter=-1
        small_boundary = []
        if_exit_small_loop = 0
        while if_exit_small_loop != '9999':
            small_counter+=1
            if small_counter == 2:
                print('If you click well, you should exit this splitting')
            if_exit_small_loop = input('Now zoom in and click to select the No.'+str(small_counter+1)+' mode in one splitting, note that prograde is negative, 9999 to exit this splitting selection>>>')
            if if_exit_small_loop == '9999':
                break
            coords = plt.ginput(1)
            a=input('Do you select the mode properly? 1 for click again>>>')
            if a != '1':
                small_boundary.append(coords[0][0])
            else:
                small_counter=small_counter-1
            #if_exit_small_loop = input('9999 to exit this splitting and select the next splitting, other to continue to select>>>')
            
        boundary_array.append(sorted(small_boundary))
        for one_ax in ax_list:
            y_lim = one_ax.get_ylim()
            one_ax.fill_between(boundary_array[-1], [min(y_lim), min(y_lim)], [max(y_lim), max(y_lim)], color = 'k', alpha = 0.3)
        if_exit_large_loop=input('9999 to exit the select mode, other to continue>>>')
    return boundary_array


def loop_to_see_if_delete_some(freq, flat_PSD, boundary_array):

    freq_to_be_fit = []
    flat_PSD_to_be_fit = []

#loop to see if the selected area is ok for fitting
    select_width = 0.1 #microHz
    for i in range(len(boundary_array)):#this array records the location of m=+-1 modes!!!
        
        #this_range = np.array(   [min(boundary_array[i])-0.2*(max(boundary_array[i])-min(boundary_array[i])), max(boundary_array[i])+0.2*(max(boundary_array[i])-min(boundary_array[i]))]   )

        this_range = np.array(   [min(boundary_array[i])-0.2, max(boundary_array[i])+0.2]   )


        tmp_fig = plt.figure()
        index = np.where(( freq>min(this_range) ) & ( freq<max(this_range) ))[0]
        this_freq, this_PSD = freq[index], flat_PSD[index]
        plt.plot(this_freq, this_PSD, 'k-')
        plt.plot([min(boundary_array[i]), min(boundary_array[i])], [0, max(this_PSD)], 'b:')
        plt.plot([max(boundary_array[i]), max(boundary_array[i])], [0, max(this_PSD)], 'b:')

        plt.xlabel('Freq, microHz')
        plt.ylabel('S/N')
        if_delete=1
        while if_delete != 9999:
            if_delete = input('Is this ok for fitting a splitting? 9999 to next, other to click on the figure to delete')
            if if_delete == '9999':
                break
            _=input('enter to click the left boundary>>>')
            left=plt.ginput(1)
            plt.plot([left[0][0]], [0], 'ro')
            _ = input('enter to click the right boundary>>>')
            right = plt.ginput(1)

            plt.plot([right[0][0]], [0], 'ro')
            if_record=input('Is it ok? 9999 to delete this boundary, other for not>>>')
            if if_record == '9999':
                left_f = np.min([ left[0][0], right[0][0]  ])
                right_f = np.max([ left[0][0], right[0][0]  ])
                if left_f > right_f:
                    left_f, right_f = min([left_f, right_f]), max([left_f, right_f])
                
                reserve_index = np.where((this_freq<left_f) | (this_freq>right_f) )[0]
                this_freq, this_PSD = this_freq[reserve_index], this_PSD[reserve_index]
                plt.plot(this_freq, this_PSD, 'r-')

        plt.close(tmp_fig)
        freq_to_be_fit.append(this_freq)
        flat_PSD_to_be_fit.append(this_PSD)

    return freq_to_be_fit, flat_PSD_to_be_fit









def fit_splitting_EMCEE_shell(splitting_work_dir):
    boundary = np.loadtxt(splitting_work_dir+'/splitting_boundary.dat')
    print('load boundary')
    splitting_data_file_list = glob.glob(splitting_work_dir+'/splitting_freq_flatPSD_No_*.dat')
    #print(splitting_data_file_list)
    #sort
    one_splitting_file=splitting_data_file_list[0]
    #print(one_splitting_file.index('No_'), one_splitting_file.index('.dat'))
    splitting_data_file_list = sorted(splitting_data_file_list, key = lambda name: int(name[ name.index('No_')+3:name.index('.dat')   ])     )
    #print('sort files')
    counter = -1
    for one_splitting_file in splitting_data_file_list:
        print(one_splitting_file)
        counter+=1
        one_splitting_data = np.loadtxt(one_splitting_file, skiprows=1)
        freq, flat_PSD = one_splitting_data[:,0], one_splitting_data[:,1]
        initial_splitting = 0.5*(  max(boundary[counter, :])-min(boundary[counter, :])  )
        initial_centre_freq = 0.5*(  max(boundary[counter, :])+min(boundary[counter, :])  )
        initial_i = 60
        initial_eta = 0.01
        initial_background = 1
        initial_amplitude = (  16*0.6*np.max(flat_PSD)*initial_eta**2  )**0.5

        initial_guess = np.array([initial_i, initial_amplitude, initial_centre_freq, initial_splitting, initial_eta, initial_background])
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
        pos[:,2] = initial_centre_freq+10**(-2)*np.random.randn(n_walkers)
        pos[:,3] = initial_splitting+10**(-2)*np.random.randn(n_walkers)



        sampler = emcee.EnsembleSampler(   n_walkers, number_of_parameter, one_splitting_log_prob, args=(freq, flat_PSD, initial_guess)   )
        sampler.run_mcmc(pos, 2000, progress=True)

        flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)
        best_parameter_splitting = np.median(flat_samples, axis = 0)
        best_parameter_splitting_std = np.std(flat_samples, axis = 0)

        #save fig
        fig = corner.corner( flat_samples, labels=['i', 'a_0', 'centre_freq', 'splitting', 'eta', 'background' ] )
        fig.savefig(splitting_work_dir+'/corner_No_'+str(counter)+'.png', dpi = 150)
        plt.close()
        plt.figure()
        plt.plot(freq, flat_PSD)
        plt.plot(freq, one_splitting(freq, 1, *best_parameter_splitting))
        #plt.plot(freq, one_splitting(freq, 1, *initial_guess), c = 'r')
        plt.savefig(splitting_work_dir+'/fit_result_No_'+str(counter)+'.png', dpi = 150)
        plt.close()
        output_array=np.zeros([2, len(best_parameter_splitting)])
        output_array[0,:] = best_parameter_splitting
        output_array[1,:]= best_parameter_splitting_std
        #save parameters and chains
        ascii.write(output_array, splitting_work_dir+'/fit_result_No_'+str(counter)+'.dat', names=['inclination in degrees', 'amplitude in S/N', 'centre_freq in microHz', 'splitting in microHz', 'eta', 'background in S/N'], overwrite=True)
        ascii.write(flat_samples, splitting_work_dir+'/chains_No_'+str(counter)+'.dat', overwrite=True)
        #plt.show()
        #a=input('pause to see the corner')


def fit_splitting_vs_zeta(one_dir):
    STAR_NAME = get_star_name(one_dir)
    PSD_data = np.loadtxt(glob.glob(one_dir+'/*PSD.dat')[0], skiprows=1)
    PSD_harvay = np.loadtxt(glob.glob(one_dir+'/*myself_harvey.dat')[0], skiprows=1)
    freq, flat_PSD = PSD_data[:,0], PSD_data[:,1]/PSD_harvay[:,1]
    #plt.figure()
    #plt.plot(freq, PSD_data[:,1])
    #plt.plot(freq, PSD_harvay[:,1])
    #plt.show()
    zeta_data = np.loadtxt(glob.glob(one_dir +'/*stretched_PSD_updated.dat')[0], skiprows=1)
    grid_freq = np.loadtxt(glob.glob(one_dir+'/*best_fit_dipole_freq_from_grid.dat')[0],skiprows=1)
    p_mode_parameter = np.loadtxt(glob.glob(one_dir+'/*_asymptotic_relation_fit.dat')[0], skiprows=1)
    g_mode_parameter = np.loadtxt(glob.glob(one_dir+'/*_grid_optimized_parameters.dat')[0], skiprows=1)
    if len(glob.glob(one_dir + '/' + STAR_NAME + '_nu_max_by_YuJie_because_numax_large.dat')) > 0:
        nu_max = np.loadtxt(one_dir + '/' + STAR_NAME + '_nu_max_by_YuJie_because_numax_large.dat')
    else:
        nu_max = np.loadtxt(one_dir + '/' + STAR_NAME + '_nu_max_by_me.dat')
    envolope = 0.59 * nu_max ** 0.9
    print('nu_max', nu_max)
    ###########
    ###########
    ###########
    Do_you_want_to_use_manual_selection_result_because_the_fit_doesnt_work_well = False
    ###########
    ###########
    ###########
    fit_splitting_list, fit_splitting_err_list = [], []
    corresponding_zeta_list = []
    corresponding_zeta_err_list = []
    obs_zeta_list = []
    obs_freq_list = []
    obs_freq_err_list = []
    grid_freq_list = []
    if Do_you_want_to_use_manual_selection_result_because_the_fit_doesnt_work_well:#not used
        manual_splitting_data = np.loadtxt(glob.glob(one_dir+'/solar_like_splittings/splitting_boundary.dat')[0])
        manual_splitting = 0.5*(manual_splitting_data[:,1]-manual_splitting_data[:,0])
        manual_freq=0.5*(manual_splitting_data[:,1]+manual_splitting_data[:,0])
        for i in range(len(manual_splitting)):
            fit_splitting_list.append(manual_splitting[i])
            fit_splitting_err_list.append(0.01)
            obs_freq_list.append(manual_freq)
            close_index = np.argmin(np.abs(manual_freq[i]-grid_freq[:,0]))
            corresponding_zeta_list.append(grid_freq[close_index,1])
            grid_freq_list.append(grid_freq[close_index,0])
    else:
        splitting_file_list = glob.glob(one_dir+'/solar_like_splittings/fit_result_No_*.dat')
        for one_splitting_file in splitting_file_list:
            tmp_data = np.loadtxt(one_splitting_file, skiprows=1)
            if tmp_data[1, 3] > tmp_data[0, 3]:#error larger than splitting, meaning it is a mistake
                continue
            fit_splitting_list.append(tmp_data[0, 3])
            fit_splitting_err_list.append(tmp_data[1, 3])
            obs_freq_list.append(tmp_data[0, 2])
            obs_freq_err_list.append(tmp_data[1, 2])
            central_freq, central_freq_err =tmp_data[0,2], tmp_data[1,2]
            close_index = np.argmin(np.abs(central_freq-grid_freq[:,0]))
            zeta = grid_freq[close_index,1]
            #print('obs_freq',tmp_data[0, 2], 'grid freq', grid_freq[close_index,0] )
            corresponding_zeta_list.append(zeta)
            #estimate zeta error from frequency error
            close_index_1 = np.argmin(np.abs(central_freq+central_freq_err-grid_freq[:,0]))
            zeta_1 = grid_freq[close_index_1,1]
            close_index_2 = np.argmin(np.abs(central_freq-central_freq_err-grid_freq[:,0]))
            zeta_2 = grid_freq[close_index_2,1]
            zeta_err = max( [abs(zeta_2-zeta), abs(zeta_1-zeta)] )

            corresponding_zeta_err_list.append(zeta_err)

            index = np.argmin( np.abs(obs_freq_list[-1]-zeta_data[:,1]) )
            obs_zeta_list.append(zeta_data[index, 3])
            #print('grid zeta', corresponding_zeta_list[-1], 'obs_zeta', zeta_data[index, 3])

            grid_freq_list.append(grid_freq[close_index,0])

    #print(obs_freq_list, grid_freq_list)
    #a=input()

    corresponding_zeta_list = np.array(corresponding_zeta_list)
    obs_zeta_list = np.array(obs_zeta_list)
    #calculate the zeta uncertainty coming from the uncertainties of the grid parameters
    zeta_uncertainty_loop_time = 50
    zeta_resample_array = np.zeros(  (zeta_uncertainty_loop_time, len(obs_freq_list))  )
    #obs_freq_list
    if_plot_uncertainty = False
    if if_plot_uncertainty:
        plt.figure()
    for zeta_counter in tqdm(range(zeta_uncertainty_loop_time)):
        p_mode_parameter_resample = p_mode_parameter[:, 0] + p_mode_parameter[:, 1] * np.random.randn(4)
        g_mode_parameter_resample = g_mode_parameter[0, :] + g_mode_parameter[1, :] * np.random.randn(6)
        tau_resample, freq_resample, flat_PSD_resample, theta_p_resample, zeta_resample = calculate_stretched_echelle_diagram(freq, flat_PSD, p_mode_parameter_resample, nu_max, envolope, g_mode_parameter_resample[0], g_mode_parameter_resample[1], g_mode_parameter_resample[3], 9999)
        if if_plot_uncertainty:
            plt.plot(freq_resample, zeta_resample, 'k-', alpha = 0.3)
        obs_freq_counter=-1
        for one_obs_freq in obs_freq_list:
            obs_freq_counter+=1
            close_index = np.argmin(np.abs( freq_resample-one_obs_freq ))
            zeta_resample_array[zeta_counter, obs_freq_counter] = zeta_resample[close_index]
    if zeta_uncertainty_loop_time!=0:
        zeta_resample_err = np.std( zeta_resample_array, axis=0 )
    else:
        zeta_resample_err = np.zeros(len(obs_freq_list))
    if if_plot_uncertainty:
        splitting_file_list = glob.glob(one_dir + '/solar_like_splittings/fit_result_No_*.dat')
        obs_freq_for_plot_list = []
        for one_splitting_file in splitting_file_list:
            tmp_data = np.loadtxt(one_splitting_file, skiprows = 1)
            obs_freq_for_plot_list.append(tmp_data[0,2])
            plt.plot(  [ tmp_data[0,2], tmp_data[0,2]], [0, 1],  'k-'  )
            plt.fill_between(  [ tmp_data[0,2]-tmp_data[1, 2], tmp_data[0,2]+tmp_data[1, 2]], [0, 0], [1, 1], facecolor = 'black'  )
        plt.xlabel('freq')
        plt.ylabel('zeta')
        plt.ylim([0.2, 1])
        plt.xlim([min(obs_freq_for_plot_list)-0.1*(max(obs_freq_for_plot_list)-min(obs_freq_for_plot_list)), max(obs_freq_for_plot_list)+0.1*(max(obs_freq_for_plot_list)-min(obs_freq_for_plot_list))])
        plt.show()

    #consider the zeta error from both frequency error and the deviation between the grid and the observed frequency
    corresponding_zeta_err_list = ( np.array( corresponding_zeta_err_list )**2 + (obs_zeta_list-corresponding_zeta_list )**2 + zeta_resample_err**2 )**0.5
    fit_splitting_list = np.array(fit_splitting_list)
    fit_splitting_err_list = np.array(fit_splitting_err_list)
    #save the data
    ascii.write( [corresponding_zeta_list, corresponding_zeta_err_list, fit_splitting_list, fit_splitting_err_list,obs_freq_list], one_dir+'/solar_like_splittings/'+STAR_NAME+'_zeta_splitting_values.dat', names=['#zeta', 'zeta err', 'splitting', 'splitting err in microHz', 'central freq in microHz'], overwrite=True )


    linear_para = np.polyfit(corresponding_zeta_list, fit_splitting_list, 1)
    initial_surface_Omega = linear_para[1]*2*np.pi

    if_run_MCMC = False
    if not if_run_MCMC:#random sampling, not MCMC
        loop_time = 1000
        slope_list = []
        intercept_list = []
        for counter in range(loop_time):
            this_zeta = corresponding_zeta_list + corresponding_zeta_err_list * np.random.normal(0, 1, size = len(corresponding_zeta_list) )
            this_splitting = fit_splitting_list + fit_splitting_err_list * np.random.normal(0, 1, size=len(fit_splitting_err_list) )
            fit_parameter = np.polyfit(this_zeta, this_splitting, 1)
            slope_list.append(fit_parameter[0])
            intercept_list.append(fit_parameter[1])

        slope_list = np.array(slope_list)
        intercept_list = np.array(intercept_list)

        flat_samples = np.zeros((loop_time, 2))
        flat_samples[:,1] = intercept_list*2*np.pi  #surface rotation Omega
        flat_samples[:,0] =( slope_list + intercept_list )*2*2*np.pi #core rotation Omega
    else:#do MCMC
        #if initial_surface_Omega <0:
        #    initial_surface_Omega=0.01
        initial_core_Omega = (linear_para[0]+linear_para[1])*2*2*np.pi
        print('initial guess', initial_core_Omega, initial_surface_Omega)
        nwalkers, ndim = 20, 2
        pos = np.array([initial_core_Omega, initial_surface_Omega]) + 10**(-2)*np.array([initial_core_Omega, initial_surface_Omega])*np.random.randn(nwalkers, ndim)
        #for iii in range(ndim):
        #    print(pos[:, iii])
        #    plt.hist(pos[:, iii], bins = 20)
        #    plt.show()
        sampler = emcee.EnsembleSampler(nwalkers, ndim, splitting_zeta_prob, args=(corresponding_zeta_list, fit_splitting_list, fit_splitting_err_list, initial_core_Omega, initial_surface_Omega))
        sampler.run_mcmc(pos, 1000, progress=True);
        flat_samples = sampler.get_chain(discard=100, thin=15, flat = True)

    best_rotation_fit = np.median(flat_samples, axis = 0)
    best_rotation_fit_std = np.std(flat_samples, axis = 0)
    fig=corner.corner(flat_samples, labels=['$\Omega_\mathrm{core}$', '$\Omega_\mathrm{surface}$'])
    fig.savefig(one_dir+'/solar_like_splittings/'+STAR_NAME+'_zeta_splitting_corner.png', dpi=150)
    plt.close()
    print('best rotation rates', best_rotation_fit)
    plt.figure(figsize = (5, 4))
    plt.errorbar(corresponding_zeta_list, fit_splitting_list, xerr = corresponding_zeta_err_list, yerr=fit_splitting_err_list, fmt='.', color = 'k')
    #plt.plot(obs_zeta_list, fit_splitting_list, 'bh')
    arti_zeta = np.linspace(min(corresponding_zeta_list), max(corresponding_zeta_list), 10)
    #plt.plot(arti_zeta, splitting_zeta_model(arti_zeta, initial_core_Omega, initial_surface_Omega))
    random_index = np.random.randint(0, len(flat_samples[:,0]), 100)
    for one_random_index in random_index:
        one_random_index = int(one_random_index)
        plt.plot(  arti_zeta, splitting_zeta_model(arti_zeta, flat_samples[one_random_index, 0], flat_samples[one_random_index, 1]) , '-', c = 'gray', alpha = 0.03 )
    plt.xlabel('$\zeta$')
    plt.ylabel('Splitting ($\mu$Hz)')
    plt.tight_layout()
    plt.savefig(one_dir+'/solar_like_splittings/'+STAR_NAME+'_zeta_splitting.png', dpi=150)
    plt.close()
    np.savetxt(one_dir+'/solar_like_splittings/'+STAR_NAME+'_zeta_splitting_chain.dat', flat_samples, header='Core omega, surface omega, in radians microHz')
    output_fid = open(one_dir+'/solar_like_splittings/'+STAR_NAME+'_zeta_splitting_best_results.dat', 'w')
    output_fid.write('#Core Omega, surface omega, in radians microHz, second line is STD\n')
    output_fid.write(str(best_rotation_fit[0])+' '+str(best_rotation_fit[1])+'\n')
    output_fid.write(str(best_rotation_fit_std[0])+' '+str(best_rotation_fit_std[1])+'\n')
    output_fid.close()
    #plt.show()


def manually_select_splitting_shell(freq, flat_PSD, echelle_fig, ax_list, one_dir):
    splitting_work_dir = one_dir+'/solar_like_splittings'
    if len(glob.glob(splitting_work_dir))==0:
        os.mkdir(splitting_work_dir)
    #get initial guess of the splittings
    boundary_array = selected_splittings_manually(freq, flat_PSD, echelle_fig, ax_list)
    #save
    np.savetxt(splitting_work_dir+'/splitting_boundary.dat', np.array(boundary_array), header='#splitting begin, end, in microHz')
    #check the data
    freq_to_be_fit, flat_PSD_to_be_fit = loop_to_see_if_delete_some(freq, flat_PSD, boundary_array)
    for i in range(len(freq_to_be_fit)):
        ascii.write(  [freq_to_be_fit[i], flat_PSD_to_be_fit[i]], splitting_work_dir+'/splitting_freq_flatPSD_No_'+str(i)+'.dat', names=['freq in microHz', 'flat PSD'], overwrite = True )

def loop_to_delete_shell(one_dir):
    plt.ion()
    PSD_data = np.loadtxt(glob.glob(one_dir+'/*PSD_flat.dat')[0], skiprows=1)
    freq, flat_PSD = PSD_data[:,0], PSD_data[:,1]
    boundary_data = np.loadtxt(glob.glob(one_dir+'/solar_like_splittings/splitting_boundary.dat')[0])
    boundary_array=[]
    for i in range(  len(boundary_data[:, 0])  ):
        boundary_array.append(boundary_data[i, :])
    #check the data
    freq_to_be_fit, flat_PSD_to_be_fit = loop_to_see_if_delete_some(freq, flat_PSD, boundary_array)
    for i in range(len(freq_to_be_fit)):
        ascii.write(  [freq_to_be_fit[i], flat_PSD_to_be_fit[i]], one_dir+'/solar_like_splittings/splitting_freq_flatPSD_No_'+str(i)+'.dat', names=['freq in microHz', 'flat PSD'], overwrite = True )
    



########################
########################
########################
########################
#fit multiple splittings automatically


def fit_multi_splittings_for_one_radial_order_automatically(freq_range, freq, flat_PSD, bin_mode_list, m_list, mixed_mode_frequencies, their_zeta, mixed_mode_splittings):
    

    if 0 in m_list:
        initial_inclination=30
    else:
        initial_inclination = 80
        
    #freq_range=[90, 96]
    #freq_range = [99, 105]
    #observed PSD
    index = np.where((freq>min(freq_range)) & (freq<max(freq_range)))[0]
    freq_to_be_fit = freq[index]
    flat_PSD_to_be_fit = flat_PSD[index]
    #calculated frequencies
    #mixed_mode_frequencies=np.array(mixed_mode_frequencies)
    index = np.where((mixed_mode_frequencies>min(freq_range)) & (mixed_mode_frequencies<max(freq_range)))[0]
    mixed_mode_frequencies_used = mixed_mode_frequencies[index]
    their_zeta_used = their_zeta[index]
    mixed_mode_splittings_used = mixed_mode_splittings[index]
    #arange initial guess
    initial_theta = []

    zeta_to_fit = []
    for i in range(len(mixed_mode_frequencies_used)):
        #print('mixed_mode_frequencies_used[i]',mixed_mode_frequencies_used[i] )
        #judge if there are good observed splittings
        m_counter = -1
        observed_freq_list = []
        local_m_component_list = []
        for one_m in m_list:#match observed and calculated frequencies
            m_counter+=1
            calculated_freq = mixed_mode_frequencies_used[i]-one_m*mixed_mode_splittings_used[i]#prograde is negative!!!
            close_distance = np.min(np.abs(bin_mode_list[m_counter]-calculated_freq))
            if close_distance > 0.2:#microHz, there is no good match
                continue
            close_index = np.argmin(np.abs(bin_mode_list[m_counter]-calculated_freq))
            observed_freq_list.append(bin_mode_list[m_counter][close_index])
            local_m_component_list.append(one_m)
        if len(observed_freq_list)<=1:#cannot find a splitting, only one or zero frequencies
            continue



        #the situation is m component [1, -1], [1, -1, 0], [1, 0], [-1, 0]
        if 1 in local_m_component_list and -1 in local_m_component_list:#find a pair
            initial_theta.append(   np.mean(observed_freq_list)   )#central frequency, using the mean value of m=1 and -1 components
            initial_theta.append( (max(observed_freq_list)-min(observed_freq_list))/2  )#splitting
        else:
            initial_theta.append(   observed_freq_list[np.argmin(np.abs(local_m_component_list))]   )#central frequency, using the value of m=0 components
            initial_theta.append( max(observed_freq_list)-min(observed_freq_list)  )#splitting, m=1 minus m=0, or m=0 minus m=-1




        zeta_to_fit.append(their_zeta_used[i])
        initial_theta.append(  9999  )#power=amplitude square, Mosser 2018, but not used here
        initial_theta.append(  0.03*(1-their_zeta_used[i])  )#eta, Mosser 2018
        initial_theta[-2]=( max(flat_PSD_to_be_fit)*(1-their_zeta_used[i])*(1-max(their_zeta_used))*4*initial_theta[-1] )**0.5#update the power again


    if len(initial_theta)<=4*0:#no more than 0 pair of splittings
        return [9999], [9999], [9999], [9999]

    initial_theta.append(initial_inclination)
    initial_theta.append(1)#background is roughly one

    #generate random numbers ranges
    initial_theta_uncertainty = []
    for index in range(0, len(initial_theta)-2):
        if index % 4 == 0:#central frequency
            initial_theta_uncertainty.append(0.005)
        if index % 4 == 1:#splitting
            initial_theta_uncertainty.append(0.005)
        if index % 4 == 2:#amplitude
            initial_theta_uncertainty.append(initial_theta[index]/10)
        if index % 4 == 3:#eta
            initial_theta_uncertainty.append(initial_theta[index]/10)
    initial_theta_uncertainty.append(1)#inclination
    initial_theta_uncertainty.append(0.05)#background
    initial_theta_uncertainty = np.array(initial_theta_uncertainty)

    pos = initial_theta + initial_theta_uncertainty*np.random.randn(2*len(initial_theta)+2, len(initial_theta))

    with Pool() as pool:
        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(  nwalkers, ndim, multa_splittings_prob, args=(freq_to_be_fit, flat_PSD_to_be_fit, initial_theta, bin_mode_list, m_list), pool = pool  )
        sampler.run_mcmc(pos, 4000, progress=True)
    flat_samples = sampler.get_chain(discard=2000, thin=15, flat=True)
    best_theta = np.median(flat_samples, axis=0)
    best_theta_std = np.std(flat_samples, axis=0)
    #for inclination, use maximum likelihood, instead of median

    #best_theta_counter = -1
    #likelihood_list_for_inclination = []
    #for best_theta_counter in range(len(flat_samples[:, 0])):
    #    likelihood_list_for_inclination.append( multi_splittings_likelihood(flat_samples[best_theta_counter, :], freq_to_be_fit, flat_PSD_to_be_fit) )
    #largest_likelihood_index = np.argmax(likelihood_list_for_inclination)


    inclination_hist, edge = np.histogram(flat_samples[:,-2], int((max(flat_samples[:,-2])-min(flat_samples[:,-2]))/best_theta_std[-2]*5)  )
    max_inclination_index = np.argmax(inclination_hist)

    print('For inclination, we use maximum likelihood, rather than median')
    print('before using it, the inclination is', best_theta[-2])
    #best_theta[-2] = (edge[max_inclination_index]+edge[max_inclination_index+1])/2
    print('after that. the inclination is', best_theta[-2], 'do not modify the inclination')
    #multi_splittings_likelihood(theta, freq, flat_PSD)




    #plt.figure()
    #plt.plot(freq_to_be_fit, flat_PSD_to_be_fit)
    final_model = multi_splittings(freq_to_be_fit, best_theta)
    initial_model = multi_splittings(freq_to_be_fit, initial_theta)
    #plt.plot(freq_to_be_fit, final_model, label = 'final model')
    #plt.plot(freq_to_be_fit, initial_model, label = 'initial model')
    #plt.legend()
    #plt.title('final fit')

    #plt.figure()
    #zeta_counter = -1
    #for one_index in range(1, len(best_theta)-2, 4):
    #    zeta_counter+=1
    #    plt.plot(zeta_to_fit[zeta_counter], best_theta[one_index], 'o')
    #plt.title('splitting zeta')
    #fig=corner.corner(flat_samples)

    #a=input()
    return best_theta, best_theta_std, zeta_to_fit, flat_samples


def fit_multi_splittings_automatically_shell(freq, flat_PSD, bin_mode_list, m_list, mixed_mode_frequencies, their_zeta, mixed_mode_splittings, nu_max, asymp_para, Delta_Pi, tau, stretched_freq, stretched_flat_PSD, splitting_work_dir, STAR_TYPE):

    central_radial_order = int( nu_max / asymp_para[0] )
    fit_radial_order = np.arange(central_radial_order-3, central_radial_order+4)
    l_0_freq_list = []
    for one_n in fit_radial_order:
        l_0_freq_list.append( asymptotic_relation_pure_p_mode( one_n, 0, nu_max, *asymp_para ) )
    l_0_freq_list = np.array(l_0_freq_list)

    #judge if this radial order will be fitted, if there are serious overlap, then do not fit
    grid_freq_with_different_m = []
    tau_with_different_m_list = []
    for m_index in range(len(m_list)):
        this_m = m_list[m_index]
        grid_freq_with_different_m.append(  mixed_mode_frequencies - this_m * mixed_mode_splittings  )
        #get tau values of them
        tau_list = []
        for one_freq in grid_freq_with_different_m[-1]:
            closed_index = np.argmin(np.abs(one_freq-stretched_freq))
            tau_list.append(tau[closed_index])
        tau_with_different_m_list.append(tau_list)

    #plt.figure()
    #x = []#tau mod Delta_Pi
    #y = []#frequency
    #for m_index in range(len(m_list)):
    #    x.append( tau_with_different_m_list[m_index] % Delta_Pi )
    #    y.append( grid_freq_with_different_m[m_index] )
    #    plt.plot(x[-1], y[-1], '.')
    #plt.show()
    #plt.close()
    #a=input('pause to see the calculated stretched echelle diagram')
    #plt.close()
    #find the closed place
    width_criterion = 0.1 #microHz
    closed_freq_list = []
    for first_index in range(len(m_list)):
        for second_index in range(first_index+1, len(m_list)):
            #print('first_index', 'second_index', first_index, second_index)
            for one_freq in grid_freq_with_different_m[second_index]:
                #print(one_freq)
                if np.min( np.abs( grid_freq_with_different_m[first_index] - one_freq ) ) < width_criterion:
                    closed_freq_list.append(one_freq)#find a freq
    #arange radial order
    closed_freq_n_list = []
    for one_close_freq in closed_freq_list:
        index = np.argmin(np.abs(one_close_freq-l_0_freq_list))
        if one_close_freq < l_0_freq_list[index]:
            closed_freq_n_list.append(fit_radial_order[index-1])
        else:
            closed_freq_n_list.append(fit_radial_order[index])

    #print('closed_freq_list', closed_freq_list)
    for index in range(len(fit_radial_order)-1, -1, -1):
        if len(np.where( fit_radial_order[index] == closed_freq_n_list)[0]) >= 3 : #more than 3 peaks overlapping
            fit_radial_order = np.delete(fit_radial_order, index)

    if STAR_TYPE == 'RC' or STAR_TYPE == 'RGB':
        fit_radial_order = np.arange(central_radial_order-3, central_radial_order+4)

    ascii.write([fit_radial_order], splitting_work_dir.replace('solar_like_splittings', '')+'/splitting_fit_radial_order.dat', names=['#the radial orders being fitted'], overwrite=True)
    fit_radial_order = np.loadtxt(splitting_work_dir.replace('solar_like_splittings', '')+'/splitting_fit_radial_order.dat', skiprows=1).astype('int')

    print('these radial orders will be fitted', fit_radial_order)
    #a=input()

    print('np.shape(fit_radial_order)', np.shape(fit_radial_order))

    freq_range_list = []
    best_parameter_list = []
    flat_samples_list = []

    splitting_counter=-1
    for one_fit_radial_order in fit_radial_order:
        l_0_freq = asymptotic_relation_pure_p_mode( one_fit_radial_order, 0, nu_max, *asymp_para )
        l_2_freq = asymptotic_relation_pure_p_mode( one_fit_radial_order, 2, nu_max, *asymp_para )
        freq_range = [ l_0_freq+0.1*asymp_para[0], l_2_freq-0.1*asymp_para[0] ]

        print('Now the radial order is', one_fit_radial_order)
        print('freq_range', freq_range)

        how_many_peaks_detected_in_this_range=0
        for one_bin_mode_list in bin_mode_list:#loop in azimuthal order m
            index = np.where(  (one_bin_mode_list>min(freq_range)) & (one_bin_mode_list<max(freq_range))  )[0]
            how_many_peaks_detected_in_this_range = how_many_peaks_detected_in_this_range+len(index)
        if how_many_peaks_detected_in_this_range<=2:
            continue
        best_theta, best_theta_std, zeta_to_fit, flat_samples = fit_multi_splittings_for_one_radial_order_automatically(freq_range, freq, flat_PSD, bin_mode_list, m_list, mixed_mode_frequencies, their_zeta, mixed_mode_splittings)
        if best_theta[0] == 9999:
            continue
        freq_range_list.append(freq_range_list)
        best_parameter_list.append(best_theta)
        flat_samples_list.append(flat_samples)
        for index in range(0, len(best_theta)-2, 4):
            splitting_counter+=1

            #plot corners
            chains = np.zeros(( flat_samples.shape[0], 6)  )
            chains[:,0]=flat_samples[:, -2]#inclination
            chains[:,1]=flat_samples[:, index+2]#amplitude
            chains[:,2]=flat_samples[:, index]#central freq
            chains[:,3]=flat_samples[:, index+1]#splitting
            chains[:,4]=flat_samples[:, index+3]#eta
            chains[:,5]=flat_samples[:, -1]#background
            fig=corner.corner(chains, labels=['i', 'a_0', 'centre_freq', 'splitting', 'eta', 'background'], quantiles=[0.16, 0.5, 0.84], show_titles=True)
            fig.savefig(splitting_work_dir+'/corner_No_'+str(splitting_counter)+'_zeta_'+str(zeta_to_fit[int(index/4)])+'.png')
            plt.close(fig)

            #plot fit result for each pair of splitting
            this_central_freq, this_splitting, this_amplitude, this_eta = best_theta[index], best_theta[index+1], best_theta[index+2], best_theta[index+3]
            inclination, background = best_theta[-2], best_theta[-1]
            plot_freq_range = [this_central_freq-2*this_splitting, this_central_freq+2*this_splitting]
            freq_index = np.where(  (freq>min(plot_freq_range)) & (freq<max(plot_freq_range))  )[0]
            plt.figure()
            plt.plot(freq[freq_index], flat_PSD[freq_index], c='gray')
            one_splitting_model = one_splitting(freq[freq_index], 1, inclination, this_amplitude, this_central_freq, this_splitting, this_eta, background)
            plt.plot(freq[freq_index], one_splitting_model, c='k')
            plt.xlabel('Freq, microHz')
            plt.ylabel('S/N')
            plt.savefig(splitting_work_dir+'/fit_result_No_'+str(splitting_counter)+'_zeta_'+str(zeta_to_fit[int(index/4)])+'.png')
            plt.close()

            #save this splitting result
            output_splitting_fid = open(splitting_work_dir+'/fit_result_No_'+str(splitting_counter)+'_zeta_'+str(zeta_to_fit[int(index/4)])+'.dat', 'w')
            output_splitting_fid.write('inclination in degrees,amplitude in S/N,centre_freq in microHz,splitting in microHz,eta in microHz, background in S/N \n')
            write_down_index = [-2, index+2, index, index+1, index+3, -1]
            for one_write_down_index in write_down_index:
                output_splitting_fid.write(str(best_theta[one_write_down_index])+' ')
            output_splitting_fid.write('\n')
            for one_write_down_index in write_down_index:
                output_splitting_fid.write(str(best_theta_std[one_write_down_index])+' ')
            output_splitting_fid.close()

        #save the result for the whole PSD for each radial order
        plt.figure(figsize = (12,8))
        plt.plot(freq, flat_PSD, 'gray', alpha = 0.5)#PSD
        plt.plot(freq, multi_splittings(freq, best_theta), 'k', label = 'Best model')#best model
        max_ylim = max(flat_PSD[np.where((freq > min(freq_range)) & (freq < max(freq_range)) )[0]]  )

        #plot identified observed frequencies and modes
        #bin_mode_list, m_list
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
                plt.plot(freq[closed_index], flat_PSD[closed_index], color+marker, markersize=10, markeredgewidth=2, alpha = 0.6)
        #plot calculated frequencies and splittings on that
        #mixed_mode_frequencies, their_zeta, mixed_mode_splittings
        for index in range(len(mixed_mode_frequencies)):
            x1, x2=min(freq_range), max(freq_range)
            y1, y2 = 0.5*max_ylim, 0.9*max_ylim
            y_location = (y2-y1)/(x2-x1)*(mixed_mode_frequencies[index]-x1)
            plt.plot([mixed_mode_frequencies[index]-mixed_mode_splittings[index], mixed_mode_frequencies[index], mixed_mode_frequencies[index]+mixed_mode_splittings[index]], [y_location,y_location,y_location], 'r.-' )

        plt.ylim([0, max_ylim])
        plt.xlabel('Freq, microHz')
        plt.ylabel('S/N')
        plt.legend()
        plt.xlim(freq_range)
        plt.tight_layout()
        plt.savefig(splitting_work_dir+'/whole_result_for_radial_order_'+str(one_fit_radial_order)+'.png', dpi=500)
########################
########################
########################
########################
#classify different trends in the echelle diagram

def multi_line_likelihood(theta, x, y, ampl, mod_value):
    k, b = theta[0], theta[1]
    model = k * (x-min(x)) + b
    residual = y - model
    index = np.where(residual < mod_value/5)[0]
    if len(index)==0:
        return -np.inf
    ln_likeli = -0.5 * np.sum( ampl[index] * residual[index]**2) 
    return ln_likeli

def multi_line_prior(theta, x, y):
    max_slope = (max(y)-min(y))/(max(x)-min(x))/2
    k, b = theta[0], theta[1]
    if np.abs(k) > max_slope:
        return -np.inf
    return 0

def multi_line_prob(theta, x, y, ampl, mod_value):
    prior = multi_line_prior(theta, x, y)
    if np.isfinite(prior):
        return prior + multi_line_likelihood(theta, x, y, ampl, mod_value)
    else:
        return -np.inf


def classify_trends_in_chelle_diagram(freq_value, ampl, vertical_value, vertical_value_gap, mod_value):
    plt.figure()
    plt.scatter(freq_value % mod_value, vertical_value, s = ampl)
    plt.scatter(freq_value % mod_value+mod_value, vertical_value, s = ampl)

    #rotate the echelle diagram!!!
    x = vertical_value
    y = freq_value % mod_value
    x = np.array( list(x) + list(x) )
    y = np.array( list(y)+list(y+mod_value) )
    y_ampl = np.array( list(ampl) + list(ampl) )
    max_slope = (max(y)-min(y))/(max(x)-min(x))/2
    print('max_slope', max_slope)
    parallel_line = 100
    initial_guess = np.zeros((parallel_line, 2))
    for line in range(parallel_line):
        initial_guess[line, 0] = np.random.uniform(-max_slope*0.5, max_slope*0.5, 1)[0]#slope
        initial_guess[line, 1] = np.random.uniform(0, 2*mod_value,  1)[0]#intercept at minimum x
    initial_guess=np.array(initial_guess)
    print(initial_guess)

    sampler = emcee.EnsembleSampler( parallel_line, 2, multi_line_prob, args=(x, y, y_ampl, mod_value) )
    sampler.run_mcmc(initial_guess, 5000, progress=True)
    flat_samples = sampler.get_chain(discard=2000, thin=15, flat=True)
    plt.figure()
    plt.plot(initial_guess[:,0], initial_guess[:, 1], '.')
    plt.title('initial guess')

    plt.figure()
    plt.plot(flat_samples[:,0], flat_samples[:,1], '.')

    plt.figure()
    plt.scatter(x, y, s=y_ampl)

    for i in np.random.uniform(0, len(flat_samples[:,0]), 100):
        k, b = flat_samples[int(i),0], flat_samples[int(i),1]
        model = k * (x-min(x))+ b
        plt.plot(x, model, 'gray', alpha = 0.2)

    plt.show()    
    a=input()



    slope_list = []
    intercept_list = []#intercept at the minimum x

    for i in tqdm( range(len(x)) ):
        for j in range(len(x)):
            if np.abs(x[i]-x[j])<0.5*vertical_value_gap or np.abs(y[i]-y[j])>0.8*mod_value:#in the same radial order
                continue
            slope = (y[i]-y[j])/(x[i]-x[j])
            intercept = y[j]-slope*x[j]
            slope_list.append(slope)
            intercept_list.append( (min(x)-x[j])/(x[i]-x[j])*(y[i]-y[j])+y[j] )
    plt.figure()
    plt.scatter(x, y, s=y_ampl)
    plt.ylabel('$\\tau$ MOD '+str(mod_value)[0:5]+' in second')
    plt.xlabel('Freq in microHz')
    plt.figure()
    plt.plot(slope_list, intercept_list % mod_value, '.', markersize = 0.5)
    plt.show()
    a=input()

###############################
###############################
###############################
########fir a gaussian on histogram using bayasian approach###########
###############################
###############################
def fit_histo_likelihood(theta, x):
    sigma, centre = theta
    point_number = len(x)
    return point_number * np.log(1/sigma/(2*np.pi)**0.5) + np.sum(  -(x-centre)**2/2/sigma**2  )

def fit_histo_prior(theta):
    sigma, centre = theta
    if sigma <=0 or centre<=0:
        return -np.inf
    else:
        return 0

def fit_histo_prob(theta, x):
    this_prior = fit_histo_prior(theta)
    if np.isfinite(this_prior):
        return this_prior + fit_histo_likelihood(theta, x)
    else:
        return -np.inf

def fit_histo_distribution_for_stretched_echelle_diagram(x, Delta_Pi):
    nwalkers=10
    ndim = 2
    pos = np.zeros([nwalkers, ndim])
    pos[:,0] = np.random.uniform(1, 0.5*Delta_Pi, nwalkers)#sigma
    pos[:,1] = np.random.uniform(1, 0.5*Delta_Pi, nwalkers)#centre

if __name__ == '__main__':

    fit_splitting_vs_zeta('/Volumes/ligangwork2/solar_like_oscillators/YuJie_RGB/KIC004995036_MCMC')
    exit()

    #one_dir = glob.glob('/Volumes/ligangwork/solar_like_oscillators/binaries_sample/long_cadence/significant_splittings/KIC008564976_MCMC')[0]
    #one_dir = glob.glob('/Volumes/ligangwork/solar_like_oscillators/triana_2017_sample/KIC*3744043*MCMC')[0]

    #one_dir = glob.glob('/Volumes/ligangwork/solar_like_oscillators/exoplanet/KIC008219268_MCMC')[0]
    #one_dir = glob.glob('/Volumes/ligangwork/solar_like_oscillators/exoplanet/KIC010864656_MCMC')[0]
    #one_dir = glob.glob('/Volumes/ligangwork/solar_like_oscillators/YuJie_RGB/KIC005731852_MCMC')[0]
    dir_list = glob.glob('/Volumes/ligangwork2/solar_like_oscillators/YuJie_RGB/KIC*_MCMC') + glob.glob('/Volumes/ligangwork2/solar_like_oscillators/Gehan_sample/KIC*_MCMC')
    dir_list=sorted(dir_list, key=lambda name: int(name[name.index('KIC')+3:name.index('KIC')+12]))

    #loop_to_delete_shell(one_dir)
    #exit()
    #fit_splitting_EMCEE_shell(one_dir+'/solar_like_splittings')
    start_index = 0
    wanted = '3965502'
    for one_dir in dir_list:
        if wanted in one_dir:
            break
        start_index += 1
    #start_index = 0
    for one_dir in dir_list[start_index:start_index+1]:
        print('\n\n\n\n')
        print(one_dir)
        if len(glob.glob(one_dir+'/solar_like_splittings')) == 0:
            print('no solar like splitting dir')
            continue
        if len(glob.glob(one_dir+'/solar_like_splittings/not_used_diverge/corner*png'))==0:
            print('no diverge')
            continue
        if len(glob.glob(one_dir+'/solar_like_splittings/corner*png'))<=2:
            print('no enough splittings')
            continue
        fit_splitting_vs_zeta(one_dir)

    exit()
    print('try Emcee')
    fit_splitting_EMCEE_shell('/Volumes/ligangwork/solar_like_oscillators/KIC007509923_MCMC/solar_like_splittings')