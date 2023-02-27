import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt 
#from get_peaks_no_MCMC import get_star_name
file='/Volumes/ligangwork/project1/59_Dra_MCMC/hd-180777.txt'
import glob, os





def bin_data_function(input_time, input_flux, width):#width to bin data

    print('input_time, input_flux',input_time, input_flux)
    #a=input()

    print('input sampling step', width)
    mean_sample_step=np.median(  input_time[1:]-input_time[0:-1]  )
    print('origin data mean sampling step', mean_sample_step)
    bin_step=int( width/mean_sample_step )
    print('bin times', bin_step)


    time, flux=[], []

    for i in np.arange(0, len(input_flux)-bin_step, bin_step):
        #print()
        #print('i+1:i+bin_step', i+1, i+bin_step)
        if np.max(  input_time[i+1:i+bin_step]-input_time[i:i+bin_step-1] )>width*1.5:#there is a gap
            continue

        time.append(np.mean( input_time[i:i+bin_step]) )
        flux.append(np.mean( input_flux[i:i+bin_step]) )


    time=np.array(time)
    flux=np.array(flux)
    #plt.plot(input_time, input_flux, '.')
    #plt.plot(time, flux,  '.')
    #plt.show()

    return time, flux

def bin_data(work_dir):
    input_file=glob.glob(work_dir+'/*light_curves_short_cadence.dat')[0]
    data=np.loadtxt(input_file)
    width = 10/60/24#ten minutes in days
    time, flux=bin_data_function(data[:,0], data[:,1], width)

    #if flux is in mag
    #flux=10**(-flux/2.5)

    ascii.write([time, flux], input_file.replace('_short_cadence',''), names=['#time','flux'], overwrite=True)



if __name__ == '__main__':
    work_dir = glob.glob('/Volumes/ligangwork/project1/TIC308396022_MCMCthreeyears_fundamental_removed')

    for one_dir in work_dir:
        bin_data(one_dir)