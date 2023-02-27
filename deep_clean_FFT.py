
import sys

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import os, glob
from scipy.optimize import curve_fit
import math
import time as module_time
from scipy import interpolate
from astropy.stats import LombScargle
#import scipy.signal as signal
#from read_kepler_data import my_DFT
from time import gmtime, strftime
import math
import scipy.signal


def get_phase(time, flux, target_freq):
    realpart=flux*np.cos(2*math.pi*target_freq*time)
    imagpart=flux*np.sin(2*math.pi*target_freq*time)
    phase=np.arctan2( np.sum(imagpart), np.sum(realpart) )
    if phase<0:
        phase=2*math.pi+phase
    return phase






def tmp_cos_func(t, freq, ampl,phase,zp):
	output=ampl*np.cos(   2*math.pi*freq*t-phase  )+zp
	return output

def get_noise(freq, power, time, flux):
	power_median=np.median(power)
	length=len(freq)
	noise_freq,noise=[],[]
	width=1/(max(time)-min(time))
	width_index=np.where(  (freq>5-0.5*5) & (freq<5+0.5*5)  )[0]

	window=len(width_index)	
	#print('window', window )
	step=int(window/5)



	for i in range(0,length, step):
		#print i
		lower=np.max([int(i-0.5*window), 0])
		upper=np.min([int(i+0.5*window), length-1])
		if lower == 0:
			upper=i+window
		if upper == length-1:
			lower=length-i-window


#get noise
		power_to_be_cal=power[lower:upper]
		for loop in range(3):
			large_index=np.where(np.abs(power_to_be_cal-np.median(power_to_be_cal))>2*np.median(power_to_be_cal))[0]
			power_to_be_cal[large_index]=np.median(power_to_be_cal)
		noise_freq.append(freq[int(0.5*(lower+upper))])
		noise.append( np.median(power_to_be_cal) )
		#print(freq[int(0.5*(lower+upper))], np.median(power_to_be_cal)  )

	noise_freq=np.array(noise_freq)
	noise=np.array(noise)
	index=np.argsort(noise_freq)
	noise_freq, noise= noise_freq[index], noise[index]

	noise=np.interp(freq, noise_freq, noise)

	#index=np.where(freq<60)[0]

	#median=np.median(power[index])

	#noise[:]=median

	return noise













def calculate_amplitude_spectrum(time, flux, oversampling):#the time series is not necessary to be equally spaced
	plt.ion()
	time_sort_index=np.argsort(time)
	time=np.sort(time)-np.min(time)
	flux=flux[time_sort_index]-np.mean(flux)


#interpolate time series
	time_spacing=time[1:]-time[0:-1]
	time_spacing_index=np.where(    np.abs(time_spacing-np.median(time_spacing))<np.std(time_spacing)    )[0]
	time_step=np.median(time_spacing[time_spacing_index])
	#print('time_step', time_step)
	delta_freq=1/(max(time)-min(time))
	arti_time=np.arange(np.min(time), np.max(time), time_step)
#get nyquist frequency
	freq=np.arange(len(arti_time))/(max(arti_time)-min(arti_time))
	length=len(arti_time)
	freq=freq[0:int(0.5*length)]
	nyquist_freq=max(freq)

	ls=LombScargle(time, flux)
	freq,power=ls.autopower(nyquist_factor=1,samples_per_peak=oversampling,  normalization='psd')
	#print(min(freq), max(freq))
	power=2*(power/len(time))**0.5#power is amplitude

	#negative_index=np.where(power<0)[0]
	#power[negative_index]=np.median(power)

	#smaller_nyquist_index=np.where(freq<nyquist_freq)[0]

	#freq, power=freq[smaller_nyquist_index], 2*(power[smaller_nyquist_index]/len(time))**0.5     #amplitude

	return freq, power








def prewhitening(time, flux, output_dir, how_many_peaks, STAR_NAME, if_clean_binary_only, binary_period):#the time series is not necessary to be equally spaced
	plt.ion()
	time_sort_index=np.argsort(time)
	time=np.sort(time)-np.min(time)
	flux=flux[time_sort_index]-np.mean(flux)


#interpolate time series
	time_spacing=time[1:]-time[0:-1]
	time_spacing_index=np.where(    np.abs(time_spacing-np.median(time_spacing))<np.std(time_spacing)    )[0]
	time_step=np.median(time_spacing[time_spacing_index])
	#print('time_step', time_step)
	delta_freq=1/(max(time)-min(time))
	arti_time=np.arange(np.min(time), np.max(time), time_step)
#get nyquist frequency
	freq=np.arange(len(arti_time))/(max(arti_time)-min(arti_time))
	length=len(arti_time)
	freq=freq[0:int(0.5*length)]
	nyquist_freq=max(freq)

	loop_time=how_many_peaks; counter=0
	output_data=np.zeros([loop_time,4])#frequency, amplitude, phase, S/N

	for counter in tqdm(range(loop_time)):

		#freq=np.arange(0.1*delta_freq,nyquist_freq,0.1*delta_freq)
		ls=LombScargle(time, flux)
		freq,power=ls.autopower(nyquist_factor=2,samples_per_peak=10,  normalization='psd')

		negative_index=np.where(power<0)[0]
		power[negative_index]=np.median(power)

		smaller_nyquist_index=np.where(freq<nyquist_freq)[0]
		#print('*******************************')
		#print('*******************************')
		#print('*******************************')
		#print('nyquist_freq',nyquist_freq)
		#print('smaller_nyquist_index', smaller_nyquist_index)
		#for counter in range(len(smaller_nyquist_index)):
		#	print(smaller_nyquist_index[counter], power[smaller_nyquist_index[counter]], 2*(power[smaller_nyquist_index[counter]]/len(time))**0.5)
		freq, power=freq[smaller_nyquist_index], 2*(power[smaller_nyquist_index]/len(time))**0.5     #amplitude

		#plt.figure()
		#plt.plot(freq, power, 'k-')
		#for tmp_i in range(len(freq)):
		#	print(freq[tmp_i], power[tmp_i])
		#a=input()
		#print('counter', counter)

		freq_into_noise=freq*1.0
		power_into_noise=power*1.0
		noise=get_noise(freq_into_noise, power_into_noise, time, flux)#????????
		#noise=np.zeros(len(freq))+np.median(power)

	


		my_lower_limit=0.05 #c/d
		my_upper_limit=nyquist_freq*1.0

		if counter==0: y_max=np.max(power)
		if counter <= 100:
			lower_limit=0.0; upper_limit=nyquist_freq
		else:
			lower_limit=my_lower_limit*1.0; upper_limit=my_upper_limit*1.0


		if if_clean_binary_only:
			binary_harmonic=(  counter+1  )/binary_period
			lower_limit=binary_harmonic-1/(max(time)-min(time))
			upper_limit=binary_harmonic+1/(max(time)-min(time))


		if if_clean_binary_only:
			print(counter, binary_harmonic, lower_limit, upper_limit)

		freq_index=np.where((freq>lower_limit) & (freq<upper_limit))[0]
#before finding the dominent frequency, set the previous frequencies as zero
		#if counter>=1:
		#	for tmp_i in range(counter):
		#		tmp_index=np.where( np.abs(output_data[tmp_i,0]-freq)<2*delta_freq )[0]
				#print tmp_index
				#power[tmp_index]=0.0

		power_after_normalise=power/noise#sorted by S/N
		#power_after_normalise=power*1.0000#sorted by amplitude

#find the dominent frequency
		max_power=np.max(power_after_normalise[freq_index])
		index=np.where(power_after_normalise==max_power)[0]
		#max_power=np.max(power[freq_index])
		#index=np.where(power==max_power)[0]


		peak_freq, peak_ampl=freq[index], power[index]

		#get phase
		#print('Calculating phase')
		peak_phase=get_phase(time, flux, peak_freq)
		#print('calculated phase is', peak_phase)

		#phase=math.pi

		median_power=np.median(power)
		freq_err=0.44*median_power/peak_ampl/(max(time)-min(time))
		ampl_err=0.8*median_power
		phase_err=(2/3.14)**0.5*median_power/peak_ampl
		#judge if the freq step or ampl step is larger than the errors
		#print(index, freq[index], len(freq))
		#print('len(power)', len(power))
		if index+1>=len(freq)-1:
			freq_step=np.abs( freq[index]-freq[index-1] )
			ampl_step=np.abs(power[index]-power[index-1])
			phase_step=np.abs( get_phase(time, flux, freq[index])-get_phase(time, flux, freq[index-1])    )
		else:
			freq_step=np.abs( freq[index+1]-freq[index] )
			ampl_step=np.abs(power[index+1]-power[index])#np.min( [    np.abs(power[index+1]-power[index]), np.abs(power[index]-power[index-1])    ]   )
			phase_step=np.abs( get_phase(time, flux, freq[index+1])-get_phase(time, flux, freq[index])    )
		#print('freq_err, freq_step',freq_err, freq_step)
		#print('ampl_err, ampl_step', ampl_err, ampl_step)
		#print('phase_err, phase_step',phase_err, phase_step)
		
		if freq_step>freq_err:
			#print('freq_step larger than the freq error')
			freq_err=freq_step
		if ampl_step>ampl_err:
			#print('ampl_step larger than ampl_err')
			ampl_err=ampl_step
		if phase_step>phase_err:
			phase_err=phase_step




		#print('freq,freq_err,ampl,ampl_err',peak_freq, freq_err,peak_ampl,ampl_err)
		#print('peak_freq',peak_freq,'peak_ampl',peak_ampl)
		if_exist_directly=False
		if len( peak_freq ) == 0: #if no peak found, exist here
			if_exist_directly=True
			if counter==0:
				output_data = np.zeros( (2, 4) )+9999
			break
		para, pcov=curve_fit(tmp_cos_func, time, flux, p0 = [peak_freq, peak_ampl, peak_phase, 0.0], bounds=((peak_freq-freq_err, peak_ampl-ampl_err, peak_phase-phase_err, -0.05),(peak_freq+freq_err, peak_ampl+ampl_err, peak_phase+phase_err, +0.05)) )
		#print('fitted phase', para[2], 'calculated phase', peak_phase)
		if False:
			plt.plot(time, flux, '.')
			plt.plot(time, tmp_cos_func(time, *para), 'r-')
			plt.show()
			a=input('enter to continue')
			plt.close()
		output_data[counter, 0:3]=para[0:3]; output_data[counter, 3]=para[1]/noise[np.where(peak_freq==freq)[0]]

		small_index=np.where( (freq>my_lower_limit) & (freq < my_upper_limit) )[0]

		S_N_criterion=4

		S_N_index=np.where(power[small_index]/noise[small_index]>S_N_criterion)[0]
		#print S_N_index
		#print 'The number of potential peaks', len(S_N_index)
		#if (np.sort(output_data[0:counter,3])[0:10]<4).all() and counter>=10:		
		if len(S_N_index)<=1 and counter>=10:
			break



		if False:#draw
			plt.figure(figsize=(15,5))
			plt.plot(freq, power)
			plt.plot(peak_freq, peak_ampl,'r.')
			plt.plot(freq, noise,'r')
			plt.plot(freq,noise*S_N_criterion,'g')
			plt.xlim([0,         min([100, nyquist_freq])       ])
			plt.savefig(output_dir+'/'+str(counter)+'.png')
			#plt.show()
			#a=input('enter to the next>>>')
			plt.close()
			#plt.plot(time, flux)
			#plt.plot(time, tmp_cos_func(time, *para))
			#plt.show()
			#a=raw_input('enter to the next>>>')

		#remove this frequency
		flux=flux-tmp_cos_func(time, *para)
		counter+=1
		#module_time.sleep(1)

	if 'KIC' in output_dir:
		KIC_name=output_dir[output_dir.index('KIC'):output_dir.index('KIC')+12]
	else:
		KIC_name=STAR_NAME
	if if_exist_directly==True:#if no peak found, exist here
		np.savetxt(output_dir+'/'+KIC_name+'very_deep_clean.dat',output_data, header='freq, ampl, phase, S/N')
		return output_data
#output light curve and power spectrum in each loop
	output_light_curve=np.zeros((len(flux),2))
	output_light_curve[:,0]=time;output_light_curve[:,1]=flux
	np.savetxt(output_dir+'/light_curve_'+str(counter)+'_freq='+str(peak_freq[0])+'.dat', output_light_curve )
	output_power_spectrum=np.zeros((len(freq), 3))
	output_power_spectrum[:,0]=freq;output_power_spectrum[:,1]=power; output_power_spectrum[:,2]=noise
	np.savetxt(output_dir+'/power_spectrum_'+str(counter)+'_freq='+str(peak_freq[0])+'.dat', output_power_spectrum)
#output frequencies list
	index=np.where(output_data[:,0]!=0)[0]
	output_data=output_data[index,:]
	if 'KIC' in output_dir:
		KIC_name=output_dir[output_dir.index('KIC'):output_dir.index('KIC')+12]
	else:
		KIC_name=STAR_NAME
#calculate S/N based on the last noise
	for i in range(len(output_data[:,0])):
		index=np.argmin(np.abs(output_data[i,0]-freq))
		output_data[i,3]=output_data[i,1]/noise[index]
	np.savetxt(output_dir+'/'+KIC_name+'very_deep_clean.dat',output_data, header='freq, ampl, phase, S/N')
	plt.close()
	return output_data



##############################################################################################################################################################################################################


















def continue_prewhitening(time, flux, output_dir, STAR_NAME):
	print('*******************************')
	print('*******************************')
	print('*******************************')
	plt.ion()
	time_sort_index=np.argsort(time)
	time=np.sort(time)-np.min(time)
	flux=flux[time_sort_index]-np.mean(flux)


#interpolate time series
	time_spacing=time[1:]-time[0:-1]
	time_spacing_index=np.where(    np.abs(time_spacing-np.median(time_spacing))<np.std(time_spacing)    )[0]
	time_step=np.median(time_spacing[time_spacing_index])

	delta_freq=1/(max(time)-min(time))
	arti_time=np.arange(np.min(time), np.max(time), time_step)
#get nyquist frequency
	freq=np.arange(len(arti_time))/(max(arti_time)-min(arti_time))
	length=len(arti_time)
	freq=freq[0:int(0.5*length)]
	nyquist_freq=max(freq)

	print('*******************************')
	print('*******************************')
	print('*******************************')
	print('nyquist_freq',nyquist_freq)


	loop_time=800; counter=0
	output_data=np.zeros([loop_time,4])#frequency, amplitude, phase, S/N

	for counter in range(loop_time):


		freq, power=LombScargle(time, flux).autopower()
		smaller_nyquist_index=np.where(freq<nyquist_freq)[0]

		print('smaller_nyquist_index',smaller_nyquist_index)
		freq, power=freq[smaller_nyquist_index], 2*(power[smaller_nyquist_index]/len(time))**0.5     #amplitude


		print('counter', counter)

		noise=get_noise(freq, power)

		power_after_normalise=power/noise


		if counter==0: y_max=np.max(power)

		lower_limit=0.1; upper_limit=24
		freq_index=np.where((freq>lower_limit) & (freq<upper_limit))[0]
#before finding the dominent frequency, set the previous frequencies as zero
		if counter>=1:
			for tmp_i in range(counter):
				tmp_index=np.where( np.abs(output_data[tmp_i,0]-freq)<2*delta_freq )[0]
				#print tmp_index
				power_after_normalise[tmp_index]=0.0
#find the dominent frequency
		max_power=np.max(power_after_normalise[freq_index])
		index=np.where(power_after_normalise==max_power)[0]

		peak_freq, peak_ampl=freq[index], power[index]

		phase=math.pi

		para, pcov=curve_fit(tmp_cos_func, time, flux, p0 = [peak_freq, peak_ampl, phase, 0.0], bounds=((peak_freq-delta_freq, 0.9*peak_ampl, -np.inf, -0.05),(peak_freq+delta_freq, 1.1*peak_ampl, np.inf, +0.05)) )

		output_data[counter, 0:3]=para[0:3]; output_data[counter, 3]=para[1]/noise[np.where(peak_freq==freq)[0]]



		if (output_data[counter-9:counter+1,3]<1.1).all() and counter>=10:
			break

		if 0==1:#draw
			plt.clf()
			plt.plot(freq, power)
			plt.plot(peak_freq, peak_ampl,'r.')
			plt.plot(freq, noise,'r')
			a=raw_input('enter to the next>>>')


		flux=flux-tmp_cos_func(time, *para)
		counter+=1
		#module_time.sleep(1)



	counter=counter+len(open(output_dir+'/'+KIC_name+'very_deep_clean.dat').readlines())+1
#output light curve and power spectrum in each loop
	output_light_curve=np.zeros((len(flux),2))
	output_light_curve[:,0]=time;output_light_curve[:,1]=flux
	np.savetxt(output_dir+'/light_curve_'+str(counter)+'_freq='+str(peak_freq[0])+'.dat', output_light_curve )
	output_power_spectrum=np.zeros((len(freq), 3))
	output_power_spectrum[:,0]=freq;output_power_spectrum[:,1]=power; output_power_spectrum[:,2]=noise
	np.savetxt(output_dir+'/power_spectrum_'+str(counter)+'_freq='+str(peak_freq[0])+'.dat', output_power_spectrum)
#output frequencies list
	index=np.where(output_data[:,0]!=0)[0]
	output_data=output_data[index,:]

	previous_result=np.loadtxt(output_dir+'/'+KIC_name+'very_deep_clean.dat')

	#combine data
	final_output=np.zeros((len(output_data[:,1])+len(previous_result[:,1]),4))
	final_output[0:len(previous_result[:,1]),:]=previous_result
	final_output[len(previous_result[:,1]):,:]=output_data
	np.savetxt(output_dir+'/'+KIC_name+'very_deep_clean.dat',final_output)
	return output_data



##############################################################################################################################################################################################################

if __name__=='__main__':
	dir_list=glob.glob('/export/pushkin1/gali8292/project1/splitting_stars/slow/KIC*11145123*')
	dir_list=sorted(dir_list, key=lambda name: int(name[name.index('KIC')+3:name.index('KIC')+12]))
	for one_dir in dir_list:
		KIC_name=one_dir[one_dir.index('KIC'):one_dir.index('KIC')+12]
		print(KIC_name)
		output_dir=one_dir+'/'+KIC_name+'_deep_clean'
		if not os.path.isdir(output_dir): 
			os.system('mkdir '+output_dir)
		#else:
			#continue

#FOR NEW STARS
		data=np.loadtxt(glob.glob(one_dir+'/'+KIC_name+'_light_curves.dat')[0])
		how_many_peaks=1500
		results=prewhitening(data[:,0], data[:,1], output_dir, how_many_peaks)

		
#FOR STARS WHICH CONTINUE DIGGING
		#input_lc=glob.glob(output_dir+'/light_curve*.dat')
		#print input_lc

		#if len(input_lc)>=2: continue

		#input_lc=sorted(input_lc, key=lambda name: int(  name[name.index('curve_')+6:name.index('_freq')]   )    )
		#data=np.loadtxt(glob.glob(output_dir+'/light_curve*.dat')[0])
		#try:
	#		results=continue_prewhitening(data[:,0], data[:,1], output_dir)
	#	except:
	#		print 'something wrong, next'



