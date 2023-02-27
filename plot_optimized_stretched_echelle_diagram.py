import os, glob, sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, '/Users/ligang/Library/Mobile Documents/com~apple~CloudDocs/gammaDorcodes')
sys.path.insert(0, '/Users/ligang/Library/Mobile Documents/com~apple~CloudDocs/solarlikecodes/SYDpy-master')
sys.path.insert(0, '/Users/gang/Library/Mobile Documents/com~apple~CloudDocs/gammaDorcodes')
sys.path.insert(0, '/Users/gang/Library/Mobile Documents/com~apple~CloudDocs/solarlikecodes/SYDpy-master')

from get_peaks_no_MCMC import *
from power_spectrum_simulation import read_frequency_uncertainty


def plot_stretched_echelle_diagram(one_dir):
    print(one_dir)
    #load tau
    stretched_echelle = np.loadtxt( glob.glob(one_dir+'/*_stretched_PSD_updated.dat')[0], skiprows=1 )
    grid = np.loadtxt( glob.glob(one_dir+'/*_grid_optimized_parameters.dat')[0], skiprows=1 )
    Delta_Pi = grid[0, 0]
    mode_index = np.where(stretched_echelle[:,2]>10)[0]
    tau = stretched_echelle[:,0]
    freq = stretched_echelle[:,1]
    S_N = stretched_echelle[:,2]
    zeta = stretched_echelle[:,3]
    best_fit_freq_data = np.loadtxt(glob.glob(one_dir+'/*best_fit_dipole_freq_from_grid.dat')[0], skiprows=1)

    #load modes
    m_list = []
    m_shape = []
    m_color = []
    bin_mode_list = []
    mode_file_name_list = glob.glob(one_dir+'/*bin_points_for_l_1_m_*.dat')

    min_freq = 0
    max_freq = 0

    counter = -1
    for mode_file_name in mode_file_name_list:
        counter += 1
        data = np.loadtxt(mode_file_name, skiprows=1, ndmin=1)
        bin_mode_list.append(data)
        if counter == 0 :
            min_freq = np.min(data)
            max_freq = np.max(data)
        else:
            min_freq = np.min( [min_freq, np.min(data)]  )
            max_freq = np.max( [max_freq, np.max(data)] )
        m = int(  mode_file_name[mode_file_name.index('m_')+2: mode_file_name.index('.dat')]  )
        m_list.append(m)
        if m == -1:
            m_shape.append('_')
            m_color.append('r')
        if m == 0:
            m_shape.append('o')
            m_color.append('b')
        if m == 1:
            m_shape.append('+')
            m_color.append('g')

    #if if_load_lorentzian_fit_frequency:
    bin_mode_list, fit_freq_uncertainty_list = read_frequency_uncertainty(one_dir, bin_mode_list, m_list, False)


    min_freq = min_freq - 1
    max_freq = max_freq + 1

    #plot observed PSD
    plt.figure(figsize=np.array([6,6])*0.7)
    plt.scatter(tau[mode_index]%Delta_Pi, freq[mode_index], s = S_N[mode_index]/4, c='darkgray', zorder = -10)
    plt.scatter(tau[mode_index]%Delta_Pi+Delta_Pi, freq[mode_index], s = S_N[mode_index]/4, c='darkgray', zorder = -10)
    plt.plot([Delta_Pi, Delta_Pi], [min_freq, max_freq], 'k:', zorder = -10)
    plt.ylim([min_freq, max_freq])
    plt.xlim([0, 2*Delta_Pi])

    #plot pick-up points
    for m_index in range(len(m_list)):
        for one_mode in bin_mode_list[m_index]:
            close_index = np.argmin(np.abs(freq-one_mode))
            if m_list[m_index] == 0:
                markersize, markeredgewidth = 3, 3
            else:
                markersize, markeredgewidth = 10, 2

            plt.plot(tau[close_index]%Delta_Pi, freq[close_index], m_shape[m_index]+m_color[m_index], zorder = 0, markersize=markersize, markeredgewidth=markeredgewidth, alpha = 0.6)
            plt.plot(tau[close_index]%Delta_Pi+Delta_Pi, freq[close_index], m_shape[m_index]+m_color[m_index], zorder = 0, markersize=markersize, markeredgewidth=markeredgewidth, alpha = 0.6)

    #plot best-fitting results
    for m_index in range(len(m_list)):
        for i in range(len(best_fit_freq_data[:,0])):
            best_fit_freq = best_fit_freq_data[i,0] - m_list[m_index]*best_fit_freq_data[i,2]
            close_index = np.argmin(np.abs(freq - best_fit_freq))
            best_fit_tau = tau[close_index]
            plt.plot(best_fit_tau%Delta_Pi, best_fit_freq, 'x', color='gray', zorder = -1, alpha = 0.6)
            plt.plot(best_fit_tau%Delta_Pi+Delta_Pi, best_fit_freq, 'x', zorder = -1, color='gray', alpha = 0.6)


    plt.xlabel('$\\tau$ MOD '+str(Delta_Pi)[0:4]+' (s)')
    plt.ylabel('Frequency ($\mu$Hz)')
    plt.tight_layout()
    STAR_NAME = get_star_name(one_dir)
    plt.savefig(one_dir+'/'+STAR_NAME+'_stretched_echelle_diagram_updated.png', dpi=200)
    plt.close()
    #plt.show()



def plot_magnetic_stretched_echelle_diagram(one_dir, if_consider_magnetism):
    print(one_dir)
    if if_consider_magnetism:
        magnetic_dir = one_dir+'/magnetism'
    else:
        magnetic_dir = one_dir+'/non_magnetism'

    #load tau
    stretched_echelle = np.loadtxt( glob.glob(magnetic_dir+'/*_magnetism_best_zeta.dat')[0], skiprows=1 )
    grid = np.loadtxt( glob.glob(magnetic_dir+'/*magnetic_parameters_result.dat')[0], skiprows=1 )
    asymp_para = np.loadtxt(glob.glob(one_dir+'/*asymptotic_relation_fit.dat')[0], skiprows=1)
    Delta_nu = asymp_para[0,0]
    Delta_Pi = grid[0, 0]
    mode_index = np.where(stretched_echelle[:,2]>10)[0]
    tau = stretched_echelle[:,0]
    freq = stretched_echelle[:,1]
    S_N = stretched_echelle[:,2]
    zeta = stretched_echelle[:,3]

    #load modes
    m_list = []
    m_shape = []
    m_color = []
    bin_mode_list = []
    mode_file_name_list = glob.glob(one_dir+'/*bin_points_for_l_1_m_*.dat')

    min_freq = 0
    max_freq = 0

    counter = -1
    for mode_file_name in mode_file_name_list:
        if 'backup' in mode_file_name:
            continue
        counter += 1
        data = np.loadtxt(mode_file_name, skiprows=1, ndmin=1)
        bin_mode_list.append(data)
        if counter == 0 :
            min_freq = np.min(data)
            max_freq = np.max(data)
        else:
            min_freq = np.min( [min_freq, np.min(data)]  )
            max_freq = np.max( [max_freq, np.max(data)] )
        m = int(  mode_file_name[mode_file_name.index('m_')+2: mode_file_name.index('.dat')]  )
        m_list.append(m)
        if m == -1:
            m_shape.append('_')
            m_color.append('r')
        if m == 0:
            m_shape.append('o')
            m_color.append('b')
        if m == 1:
            m_shape.append('+')
            m_color.append('g')

    min_freq = min_freq + 1
    max_freq = max_freq - 1

    bin_mode_list, fit_freq_uncertainty_list = read_frequency_uncertainty(one_dir, bin_mode_list, m_list, if_consider_magnetism)

    plt.figure(figsize=np.array([6,6])*0.7)
    plt.scatter(tau[mode_index]%Delta_Pi, freq[mode_index], s = S_N[mode_index]/4, c='darkgray', zorder = -10, marker = 's')
    plt.scatter(tau[mode_index]%Delta_Pi+Delta_Pi, freq[mode_index], s = S_N[mode_index]/4, c='darkgray', zorder = -10, marker = 's')
    ylim = [min_freq-0.2*Delta_nu, max_freq+0.2*Delta_nu]
    plt.plot([Delta_Pi, Delta_Pi], ylim, 'k:', zorder = -10)
    plt.ylim(ylim)
    plt.xlim([0, 2*Delta_Pi])

    for m_index in range(len(m_list)):
        for one_mode in bin_mode_list[m_index]:
            close_index = np.argmin(np.abs(freq-one_mode))
            if m_list[m_index] == 0:
                markersize, markeredgewidth = 3, 3
            else:
                markersize, markeredgewidth = 10, 2

            plt.plot(tau[close_index]%Delta_Pi, freq[close_index], m_shape[m_index]+m_color[m_index], zorder = 0, markersize=markersize, markeredgewidth=markeredgewidth, alpha = 0.6)
            plt.plot(tau[close_index]%Delta_Pi+Delta_Pi, freq[close_index], m_shape[m_index]+m_color[m_index], zorder = 0, markersize=markersize, markeredgewidth=markeredgewidth, alpha = 0.6)

    #plot best fit frequencies
    best_fit_freq_files = glob.glob(magnetic_dir+'/*best_fitting_magnetic_frequency_m*.dat')
    for one_best_fit_freq_file in best_fit_freq_files:
        current_m = int(  one_best_fit_freq_file[one_best_fit_freq_file.index('_m_')+3:one_best_fit_freq_file.index('.dat')]  )
        if current_m == 0:
            this_marker = '0'
        elif current_m == 1:
            this_marker = '+'
        elif current_m == -1:
            this_marker = '-'
        best_fit_freq = np.loadtxt(one_best_fit_freq_file)
        close_index = []
        for one_best_fit_freq in best_fit_freq:
            i = np.argmin(np.abs(one_best_fit_freq - freq))
            close_index.append(i)
        plt.plot(tau[close_index]%Delta_Pi, freq[close_index], 'x', color = 'gray', zorder = -1, alpha = 0.6)
        plt.plot(tau[close_index]%Delta_Pi+Delta_Pi, freq[close_index], 'x',  color = 'gray', zorder = -1, alpha = 0.6)

    #plt.xlim([0, 2*Delta_Pi])
    #plt.ylim()


    plt.xlabel('$\\tau$ MOD '+str(Delta_Pi)[0:4]+' (s)')
    plt.ylabel('Frequency ($\mu$Hz)')
    plt.tight_layout()
    STAR_NAME = get_star_name(one_dir)
    plt.savefig(magnetic_dir+'/'+STAR_NAME+'_magnetic_stretched_echelle_diagram_updated.png', dpi=200)
    plt.close()
    #plt.show()



if __name__ == '__main__':

    dir_list = glob.glob('/Volumes/ligangwork2/solar_like_oscillators/Gehan_sample/KIC*MCMC')+glob.glob('/Volumes/ligangwork2/solar_like_oscillators/YuJie_RGB/KIC*MCMC')
    #dir_list = glob.glob('/Volumes/ligangwork2/solar_like_oscillators/Gehan_sample/only_delta_pi/*_MCMC') + glob.glob('/Volumes/ligangwork2/solar_like_oscillators/Yujie_RGB/only_delta_pi/*_MCMC')

    #dir_list = glob.glob('/Volumes/ligangwork2/solar_like_oscillators/Gehan_sample/KIC*MCMC')+glob.glob('/Volumes/ligangwork2/solar_like_oscillators/YuJie_RGB/KIC*MCMC') + glob.glob('/Volumes/ligangwork2/solar_like_oscillators/Gehan_sample/only_delta_pi/*_MCMC') + glob.glob('/Volumes/ligangwork2/solar_like_oscillators/Yujie_RGB/only_delta_pi/*_MCMC')

    dir_list = sorted(dir_list, key=lambda name: int(name[name.index('KIC')+3:name.index('KIC')+12]))

    #dir_list=['/Volumes/ligangwork2/solar_like_oscillators/Gehan_sample/only_delta_pi/KIC009474201_MCMC']

    wanted = '8636389'

    start_index = -1
    for one_dir in dir_list:
        start_index+=1
        if wanted in one_dir:
            break
    if_consider_magnetism=False

    #start_index=0

    for one_dir in dir_list[start_index:start_index+1]:


        #os.system('open '+one_dir)
        if 'only_delta_pi' not in one_dir:
            if_load_lorentzian_fit_frequency=True
        else:
            if_load_lorentzian_fit_frequency=False
        #print('if_load_lorentzian_fit_frequency', if_load_lorentzian_fit_frequency)
        plot_stretched_echelle_diagram(one_dir)

        print('\n\n*******************\n\n')

        if 'only_delta_pi' not in one_dir:
            plot_magnetic_stretched_echelle_diagram(one_dir, if_consider_magnetism)