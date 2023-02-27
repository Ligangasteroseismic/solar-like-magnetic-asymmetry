import numpy as np
import matplotlib.pyplot as plt
import os, glob, sys
#sys.path.insert(0, '/Users/ligang/Library/Mobile Documents/com~apple~CloudDocs/gammaDorcodes')
#sys.path.insert(0, '/Users/ligang/Library/Mobile Documents/com~apple~CloudDocs/solarlikecodes/SYDpy-master')
#sys.path.insert(0, '/Users/gang/Library/Mobile Documents/com~apple~CloudDocs/gammaDorcodes')
#sys.path.insert(0, '/Users/gang/Library/Mobile Documents/com~apple~CloudDocs/solarlikecodes/SYDpy-master')
from get_peaks_no_MCMC import *
from solar_like_lightcurves import plot_echelle_diagram
from find_asymmetric_splittings import *
from fit_magnetic_shift import *
from plot_optimized_stretched_echelle_diagram import *

def add_modes_manually(one_dir):
    plt.close('all')
    plt.close()
    STAR_NAME = get_star_name(one_dir)

    #load PSD
    print(one_dir+'/'+STAR_NAME+'_PSD.dat')
    PSD_data = np.loadtxt(glob.glob(one_dir+'/'+STAR_NAME+'_PSD.dat')[0], skiprows=1)
    #load harvey
    my_harvey = np.loadtxt(glob.glob(one_dir + '/*myself_harvey.dat')[0], skiprows=1)
    #generate flat PSD
    freq, flat_PSD = PSD_data[:,0], PSD_data[:, 1]/my_harvey[:, 1]
    #load p-mode fit
    asymp_para = np.loadtxt(glob.glob(one_dir+'/*asymptotic_relation_fit.dat')[0], skiprows=1)
    asymp_para = asymp_para[:, 0]
    delta_nu = asymp_para[0]
    nu_max = (delta_nu/0.28)**(1/0.75)   #deltanu = 0.28*numax**0.75
    envolope = 0.59 * nu_max ** 0.9
    #load g-mode fit
    optimized_parameters = np.loadtxt(glob.glob(one_dir+'/*grid_optimized_parameters.dat')[0])
    Delta_Pi = optimized_parameters[0, 0]
    #load theoretical results
    theo_result = np.loadtxt(glob.glob(one_dir+'/*best_fit_dipole_freq_from_grid.dat')[0], skiprows=1)
    #load zeta
    #stretched_PSD = np.loadtxt( glob.glob(one_dir+'/*stretched_PSD_updated.dat')[0], skiprows=1 )
    #tau = stretched_PSD[:,0]
    #stretched_freq = stretched_PSD[:,1]

    #load observed peaks
    if len(glob.glob(one_dir+'/*bin_points_for_l_1_m_*.dat'))==3:#triplets
        m_1_list = np.loadtxt(glob.glob(one_dir+'/*bin_points_for_l_1_m_1.dat')[0], skiprows=1)
        m_0_list = np.loadtxt(glob.glob(one_dir+'/*bin_points_for_l_1_m_0.dat')[0], skiprows=1)
        m_negative_1_list = np.loadtxt(glob.glob(one_dir+'/*bin_points_for_l_1_m_-1.dat')[0], skiprows=1)
        m_list = [1, 0, -1]
        selected_mode_list = [m_1_list, m_0_list, m_negative_1_list]
    elif len(glob.glob(one_dir+'/*bin_points_for_l_1_m_*.dat'))==2:#only m=+-1
        m_1_list = np.loadtxt(glob.glob(one_dir+'/*bin_points_for_l_1_m_1.dat')[0], skiprows=1)
        m_negative_1_list = np.loadtxt(glob.glob(one_dir+'/*bin_points_for_l_1_m_-1.dat')[0], skiprows=1)
        m_list = [1, -1]
        selected_mode_list = [m_1_list, m_negative_1_list]
    else:
        m_0_list = np.loadtxt(glob.glob(one_dir+'/*bin_points_for_l_1_m_0.dat')[0], skiprows=1)
        m_list = [0]
        selected_mode_list = [m_0_list]

    #plot echelle diagram
    plt.ion()
    echelle_lower_boundary = max([0, nu_max - 0.8 * envolope])
    echelle_upper_boundary = min([max(freq) - 0.05 * delta_nu, nu_max + 0.8 * envolope])
    fig, ax_list, freq_boundary = plot_echelle_diagram(freq, flat_PSD, delta_nu, [echelle_lower_boundary, echelle_upper_boundary])
    #plot identified splittings
    for one_ax in ax_list:
        x_lim = one_ax.get_xlim()
        y_lim = one_ax.get_ylim()
        left_y = 0.1
        right_y = 0.9
        theo_counter = -1
        for one_theo_freq in theo_result[:, 0]:
            theo_counter += 1
            now_y = ( one_theo_freq-min(x_lim) )/ (max(x_lim)-min(x_lim))*(max(y_lim)-min(y_lim))+min(y_lim)
            one_ax.plot([one_theo_freq-theo_result[theo_counter, 2], one_theo_freq, one_theo_freq+theo_result[theo_counter, 2]], [now_y,now_y,now_y], 'r.-')

    #plot observed modes
    lack_m_0 = False
    if len(m_list) == 3:
        marker_list =['+', 'o', '_']
        color_list = ['g', 'b', 'r']
    elif len(m_list) == 2:
        lack_m_0 = True
        marker_list =['+', '_', 'o']
        color_list = ['g', 'r', 'b']
    else:
        marker_list =['o']
        color_list = ['b']
    for m_counter in range(len(m_list)):
        for one_ax in ax_list:
            #print('selected_mode_list', selected_mode_list)
            for one_freq in selected_mode_list[m_counter]:
                index = np.argmin(np.abs(one_freq - freq))
                one_ax.plot(freq[index], flat_PSD[index], color_list[m_counter]+marker_list[m_counter], markersize=10, markeredgewidth=2, alpha = 0.5)


    m_you_want=int(input('input m value you want to add or delete, 9999 to continue>>>'))
    if m_you_want == 9999:
        return
    while m_you_want != 9999:
        if lack_m_0 == True and m_you_want == 0:
            m_list.append(0)
            selected_mode_list.append(np.array([]))
            lack_m_0 = False
            print('add m=0 mode')

        m_index = np.argmin(np.abs(np.array(m_list)-m_you_want))

        judge = input('1 for adding points, 2 for deleting points>>>')
        while judge != '9999':
            if judge == '1' or judge == '':
                judge = '1'
                print('Now click on the echelle diagram to add m='+str(m_you_want)+' modes...')
            if judge == '2':
                print('Now click on the echelle diagram to delete m='+str(m_you_want)+' modes...')
            coords = plt.ginput(1, timeout = 0 )
            this_freq = coords[0][0]
            this_freq_index = np.argmin(np.abs(freq-this_freq))
            this_freq = freq[this_freq_index]
            if judge == '1':#add
                selected_mode_list[m_index] = np.append(selected_mode_list[m_index], this_freq)
                for one_ax in ax_list:#plot on the echelle diagram
                    one_ax.plot(freq[this_freq_index], flat_PSD[this_freq_index], color_list[m_index]+marker_list[m_index], markersize=10, markeredgewidth=2)
            if judge == '2':#delete
                delete_index = np.argmin(np.abs(selected_mode_list[m_index] - this_freq))
                delete_freq = selected_mode_list[m_index][delete_index]
                selected_mode_list[m_index] = np.delete( selected_mode_list[m_index], delete_index )
                index_for_freq = np.argmin(np.abs(freq-delete_freq))
                for one_ax in ax_list:#plot on the echelle diagram
                    one_ax.plot(freq[index_for_freq], flat_PSD[index_for_freq], 'yx', markersize=10, markeredgewidth=2)
            old_judge = judge
            judge = input('enter to continue, or input 1 to add or 2 to delete, 9999 to change m>>>')
            if judge == '':
                judge = old_judge
            print('at the end judge is', judge)

        m_you_want=int(input('input m value you want to add or delete, 9999 to finish>>>'))

    for m_index in range(len(m_list)):
        this_m = m_list[m_index]
        np.savetxt(one_dir + '/'+STAR_NAME+'_bin_points_for_l_1_m_'+str(this_m)+'.dat' ,selected_mode_list[m_index], header='# points in the power spectrum, not mode frequencies, in unit of microHz')

    #check if delta pi list lacks m=0 mode
    delta_pi_list = np.loadtxt(glob.glob(one_dir+'/*best_Delta_P_list.dat')[0], skiprows=1)
    print('delta_pi_list', delta_pi_list)
    if not isinstance(delta_pi_list, list):
        delta_pi_list = [delta_pi_list]
    if len(delta_pi_list) < len(m_list):
        delta_pi_list=np.append(delta_pi_list, np.mean(delta_pi_list))#add m = 0 mode at the end, following the order or m_list
        np.savetxt(one_dir+'/'+STAR_NAME+'_best_Delta_P_list.dat',delta_pi_list, header='# #These numbers give the delta tau in seconds which can be used to calculate the splittings')

    print('new results have been saved!')
    plt.close()
    plt.close('all')


if __name__ == '__main__':

    dir_list = ['./KIC008684542_MCMC']

    dir_list = sorted(dir_list, key=lambda name: int(name[name.index('KIC')+3:name.index('KIC')+12]))

    #wanted_KIC = '5696081'#3216736'
    wanted_KIC = '12070025'
    start_index = -1
    for one_dir in dir_list:
        start_index += 1
        if wanted_KIC in one_dir:
            break


    start_index = 0
    for one_dir in dir_list[start_index:start_index+1]:
        print('\n\n\n\n****************************************************')
        print(one_dir)
        os.system('open '+one_dir)
        add_modes_manually(one_dir)
