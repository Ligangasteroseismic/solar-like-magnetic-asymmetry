


#from read_kepler_data import *
import numpy as np
#import PyQt5
import matplotlib.pyplot as plt
#from bin_data import *
#matplotlib.use('Qt5Agg')
from tqdm import tqdm

from pylab import *
import pickle
import subprocess
from time import gmtime, strftime
import os
from shutil import copyfile
import glob
import corner
import gc
from datetime import datetime
from scipy.optimize import curve_fit
#from convert_deep_clean_frequency import *
#from read_kepler_data import *
from deep_clean_FFT import *
from astropy.stats import LombScargle
from convert_deep_clean_frequency import *
from astropy.table import Table, Column, MaskedColumn 
from astropy.io import ascii
from bin_data import bin_data_function

def get_star_name(one_dir):
	location=[]
	counter=-1
	for one_element in one_dir:
		counter+=1
		if one_element == '/':
			location.append(counter)
	if '_MCMC' in one_dir:
		star_name=one_dir[location[-1]+1:one_dir.index('_MCMC')]
	else:
		star_name=one_dir[location[-1]+1:]
	return star_name







def check_combinations(freq, ampl, freq_err, freq_spec, power_spec, work_dir, KIC_number):	
	if 'KIC' in work_dir:
		KIC_number=work_dir[work_dir.index('KIC'):work_dir.index('KIC')+12]
	fid1=open(work_dir+'/'+KIC_number+'fake_frequency_record.dat','w')
	fid1.write('#n freqs, n coefficients, the corresponding freq, the residual, if the combinations(1yes0no)\n')

	print('start to find combinations')
	#sorted by ampl
	index=np.argsort(-ampl)
	freq=freq[index]
	ampl=ampl[index]

	whether=np.zeros(len(index)) #0 for hasn't been checked

	error_magnify=4 #4 years / observation span

	epsilon=0.0002*error_magnify

	coefficient=[2,3]

	combination_freq=[]

	difference=[]


#one parent freqs	
	how_many_freq=len(freq)
	how_many_parents=min( [how_many_freq-4, 20] )
	for index1 in range(how_many_parents):
		f1=freq[index1]
		if whether[index1]>0:continue
		for n1 in coefficient:
			index=np.argmin(np.abs((n1*f1)-freq))#freq[index] is the freq with minimum difference
			residual=np.abs((n1*f1)-freq[index])
					

			fid1.write(str(f1)+' '+str(0.0000)+' '+str(n1)+' '+str(0.0000)+' '+str(freq[index])+' '+str(residual)+' ')

			error=n1*freq_err[index1]+freq_err[index]


			if 0<residual <= 0.002 and ampl[index]<=ampl[index1] and (index not in [index1]) and whether[index1] == 0:
				difference.append(residual)#count a residual
			if residual <= epsilon and ampl[index] <= min([ampl[index1]]) and (index not in [index1]) and whether[index]!=1:
					#find one
					#print(index1, index2,n1, n2)
				fid1.write('1\n')
				whether[index]=1
				combination_freq.append(freq[index])
			else:
				fid1.write('0\n')


#two parent freqs
	coefficient=[1,2,3,-1,-2,-3]
	for index1 in range(how_many_parents):
		for index2 in range(index1+1,how_many_parents):

			f1,f2=freq[index1],freq[index2]
			if whether[index1]+whether[index2] >0:continue
			for n1 in coefficient:
				for n2 in coefficient:
					if np.abs(n1)+np.abs(n2)>2:continue
					index=np.argmin(np.abs((n1*f1+n2*f2)-freq))#freq[index] is the freq with minimum difference
					residual=np.abs((n1*f1+n2*f2)-freq[index])
					
					error=(n1**2*freq_err[index1]**2+n2**2*freq_err[index2]**2)**0.5+freq_err[index]

					fid1.write(str(f1)+' '+str(f2)+' '+str(n1)+' '+str(n2)+' '+str(freq[index])+' '+str(residual)+' ')


					if 0<residual <= 0.002 and ampl[index] <= min([ampl[index1], ampl[index2]]) and (index not in [index1, index2]) and whether[index1]+whether[index2] == 0:
						difference.append(residual)#count a residual
					if residual <= epsilon and ampl[index] <= min([ampl[index1], ampl[index2]]) and (index not in [index1, index2]) and whether[index]!=1:
						#find one
						#print(index1, index2,n1, n2)
						fid1.write('1\n')
						whether[index]=1
						combination_freq.append(freq[index])
					else:
						fid1.write('0\n')


	fid1.close()
#write down the index in the FFT file
	fid1=open(work_dir+'/'+KIC_number+'fake_peaks_locations.dat','w')
	fid1.write('#index of combination freq, from FFT file\n')
	for one_freq in combination_freq:
		index=np.argmin(np.abs(one_freq-freq_spec))
		fid1.write(str(index)+'\n')
	fid1.close()

#write down the index and freq in the FFT file
	fid1=open(work_dir+'/'+KIC_number+'combinations.dat','w')
	fid1.write('#index, freq of combinations\n')
	for one_freq in combination_freq:
		index=np.argmin(np.abs(one_freq-freq_spec))
		fid1.write(str(index)+' '+str(freq_spec[index])+'\n')
	fid1.close()

#write down the 'difference'
	np.savetxt(work_dir+'/'+KIC_number+'fake_peaks_difference_for_histogram.dat', difference)


#plot some figures
	#the histogram
	difference=np.array(difference)
	hist_index=np.where(difference<0.001*error_magnify)[0]
	plt.hist(difference[hist_index], bins=20)
	plt.xlim([0,0.001*error_magnify])
	plt.xlabel('$|f_k-(n_if_i+n_jf_j)|$')
	plt.ylabel('Number')
	plt.savefig(work_dir+'/'+KIC_number+'difference_distribution.png')
	#plt.show()
	plt.close()

	#the spectrum
	plt.figure(figsize=(12,5))
	plt.plot(freq_spec, power_spec)
	for one_freq in freq:
		index=np.argmin(np.abs(freq_spec-one_freq))
		plt.plot(freq_spec[index], power_spec[index],'r.')

	for one_freq in combination_freq:
		index=np.where(np.abs(freq_spec-one_freq)<0.0007/2)[0]
		plt.plot(freq_spec[index], power_spec[index],'r')
	plt.xlim([0,5])
	plt.xlabel('Frequency in c/d')
	plt.ylabel('Amplitude')
	plt.savefig(work_dir+'/'+KIC_number+'fake_frequency.png')
	plt.close()







def read_kepler_data(data_dir):
	#print data_dir
	#print 'printed in this function: ',data_dir
	files_list=glob.glob(data_dir+'/kplr*_slc.dat')
	#print data_dir
	#print files_list

	for filename in tqdm(files_list):
		
		#print filename
		tmp_data=np.loadtxt(filename)

		#fit and correct zero-point shift
		min_time=np.min(tmp_data[:,0])
		max_time=np.max(tmp_data[:,0])	
		tmp_data[:,0]=(tmp_data[:,0]-min_time)/(max_time-min_time)
		
		
		#extract 'inf'
		inf_index=np.where(tmp_data[:,3]==-np.inf)[0]
		notinf_index=np.where(tmp_data[:,3]!=-np.inf)[0]
		tmp_data[inf_index,3]=np.mean(tmp_data[notinf_index,3])
		max_flux=np.max(tmp_data[:,3])
		tmp_data[:,3]=tmp_data[:,3]/max_flux
		#fit
		para=np.polyfit(tmp_data[notinf_index,0],tmp_data[notinf_index,3],2)
		#minus zero point
		tmp_data[notinf_index,3]=tmp_data[notinf_index,3]/np.polyval(para,tmp_data[notinf_index,0])#+1

		tmp_data[:,0]=tmp_data[:,0]*(max_time-min_time)+min_time
		
		

		#print files_list.index(filename)
		#print tmp_data.shape
		if files_list.index(filename)==0:
			data=tmp_data*1.0
			#print data.shape
		else:
			tmp_data=tmp_data[notinf_index,0:5]
			#print data.shape, tmp_data.shape
			data=np.concatenate((data,tmp_data))			
			#print data.shape
	#data[:,0]=np.sort(data[:,0])
	#data[:,1]=data[np.argsort(data[:,0]),1]

	#sort by time
	sort_index=np.argsort(data[:,0])

	data=data[sort_index, :]

	#print data,np.shape(data)
	out_index=np.where( np.abs(data[:,3]-np.mean(data[:,3]))>5*np.std(data[:,3]) )[0]
	data[out_index,3]=np.mean(data[:,3])


	#plt.plot(data[:,0], data[:,3], '.')
	#plt.show()
	#a=input('pause')
	return data;



def remove_slow_trend(time, flux):
	if np.median(flux)<0.1:
		flux=flux+1
	time_difference=np.median( time[1:]-time[0:-1] )

	#bin data for short cadence
	bin_data_mode=False
	#print('^V^V^V^V^V^V^V^V^V')
	#print(time)
	do_you_want_to_bin_data=False
	if do_you_want_to_bin_data:
		bin_data_mode=True
		print('Short cadence data, bin it!')
		original_time, original_flux = time, flux
		time, flux = bin_data_function(time, flux, 30/60/24) #


	#plt.figure()
	#time_difference_array=time[1:]-time[0:-1]
	#index=np.where(time_difference_array<0.2)[0]
	#plt.hist( time_difference_array[index] , bins=500)
	#plt.show()
	flux_median=np.zeros(len(flux))

	boundary=[-1.5]
	 
	width=5 #days
	gap_width = 0.1 #days

	flux_diff_std = np.std( np.diff(flux) )

	for index in tqdm(range(len(time)-1)):

		flux_jump_before=np.abs( flux[index+1]-flux[index] )
		flux_jump_after=np.abs( flux[min([index+2, len(flux)-1])]-flux[index+1]  )
		jump_criterion=3#change this number to be extremely small if you do not want to include this criterion
		if flux_jump_before > jump_criterion*flux_diff_std and flux_jump_after < jump_criterion*flux_diff_std:
			if_there_is_jump=True
		else:
			if_there_is_jump = False
		if_there_is_jump = True



		if (time[index+1]-time[index]>gap_width and if_there_is_jump   ) or index+1==len(time)-1:#there is a gap
		#if (time[index+1]-time[index]>gap_width ) or index+1==len(time)-1:#there is a gap

			boundary.append(index+0.5)
			if index+1==len(time)-1:
				boundary[-1]=index+1+0.5
			local_index=range( int(boundary[-2])+1, int(boundary[-1])+1  )#this section
			#print('local_index',local_index)


			if time[max(local_index)]-time[min(local_index)]<1:
				flux_median[local_index]=1
				continue

			for counter in range(len(local_index)):
				one_local_index=local_index[counter]
				if counter % 4 ==0:
					width_index=np.where((time-time[one_local_index]>-0.5*width) & (time-time[one_local_index]<0.5*width))[0]
					median_index=range(  max(min(local_index), min(width_index)),  min(max(local_index), max(width_index)  )+1                )
					#print(median_index)
					local_median=np.median(flux[median_index])
					flux_median[one_local_index]=local_median
				else:
					flux_median[one_local_index]=flux_median[one_local_index-1]


	#convert boundary from the binned data to the original data

	if bin_data_mode:
		flux_median_interp=np.interp(original_time, time, flux_median)
		original_boundary=[]

		for one_boundary in boundary:
			index=np.argmin(abs( original_time-(time[int(one_boundary)]+1.5*time_difference) ) )
			original_boundary.append(index+0.5)
	
		original_bounary=np.array(original_boundary)
	#print(boundary)
		return original_time, original_flux/flux_median_interp, flux_median_interp, original_bounary
	
	else:
		return time, flux/flux_median, flux_median, boundary
	
	
	
	#return time, flux_median, boundary


def polyfit_remove_slow_trend(time, flux):
	print('Fit a parabola to remove slow trend')
	if np.median(flux)<0.1:
		flux=flux+1
	time_difference=np.median( time[1:]-time[0:-1] )

	#bin data for short cadence
	bin_data_mode=False
	#print('^V^V^V^V^V^V^V^V^V')
	#print(time)
	do_you_want_to_bin_data=False
	if time[2]-time[1]<5/60/24 and do_you_want_to_bin_data:
		bin_data_mode=True
		print('Short cadence data, bin it!')
		original_time, original_flux = time, flux
		time, flux = bin_data_function(time, flux, 60/60/24)


	#plt.figure()
	#time_difference_array=time[1:]-time[0:-1]
	#index=np.where(time_difference_array<0.2)[0]
	#plt.hist( time_difference_array[index] , bins=500)
	#plt.show()
	flux_median=np.zeros(len(flux))

	boundary=[-1.5]
	
	width=np.min([0.1, 30*time_difference]) #days

	height=5 # in unit of sigma

	for index in range(len(time)-1):

		if time[index+1]-time[index]>width or index+1==len(time)-1:#there is a gap
			boundary.append(index+0.5)
			if index+1==len(time)-1:
				boundary[-1]=index+1+0.5
			#print('local_index',local_index)

	flux_median=[]
	for index in range(len(boundary)-1):
		local_index=range( int(boundary[index])+1, int(boundary[index+1])+1  )#this section
		para=np.polyfit(time[local_index], flux[local_index], 2)
		local_median = np.polyval(para, time[local_index])
		for one_local_median in local_median:
			flux_median.append(one_local_median)
############
#			if time[max(local_index)]-time[min(local_index)]<1:
#				flux_median[local_index]=1
#				continue

#			for counter in range(len(local_index)):
#				one_local_index=local_index[counter]
#				if counter % 4 ==0:
#					width_index=np.where((time-time[one_local_index]>-0.5*width) & (time-time[one_local_index]<0.5*width))[0]
#					median_index=range(  max(min(local_index), min(width_index)),  min(max(local_index), max(width_index)  )+1                )
#					#print(median_index)
#					local_median=np.median(flux[median_index])
#					flux_median[one_local_index]=local_median
#				else:
#					flux_median[one_local_index]=flux_median[one_local_index-1]
############

	#convert boundary from the binned data to the original data
	flux_median=np.array(flux_median)
	if bin_data_mode:
		flux_median_interp=np.interp(original_time, time, flux_median)
		original_boundary=[]

		for one_boundary in boundary:
			index=np.argmin(abs( original_time-(time[int(one_boundary)]+1.5*time_difference) ) )
			original_boundary.append(index+0.5)
	
		original_bounary=np.array(original_boundary)
	#print(boundary)
		return original_time, original_flux/flux_median_interp, flux_median_interp, original_bounary
	
	else:
		return time, flux/flux_median, flux_median, boundary
	
	
	
	#return time, flux_median, boundary


def remove_data_at_beginning_or_end_of_each_data_group(time, flux):
	print('************WARNING**************')
	print('************WARNING**************')
	print('************WARNING**************')
	print('************WARNING**************')
	print('************WARNING**************')
	print('************WARNING**************')
	print('This function remove_data_at_beginning_or_end_of_each_data_group may has vital bug, check it clearly!')
	print('Li Gang at 2021.10.23')
	print('************WARNING**************')
	print('************WARNING**************')
	print('************WARNING**************')
	print('************WARNING**************')
	print('************WARNING**************')
	print('************WARNING**************')	
	time_step=np.median( time[1:]-time[0:-1] )

	time_gap_threshold=time_step*30

	day_width=1
	#print(time)
	print('day width, time step')
	print(day_width, time_step)
	how_many_data_to_delete=int(day_width/time_step)#48 points corresponds to 1 day

#if delete
	if True:
		deleted_index=np.arange(0, how_many_data_to_delete)#beginning
		for i in range(len(time)-1):
			if time[i+1]-time[i]<time_gap_threshold or i-how_many_data_to_delete<0:
				continue
			for j in range(i-how_many_data_to_delete, i+1):  #the beginning of gap
				deleted_index=np.append(deleted_index, j)
	
		for i in range(1,len(time)):
			if time[i]-time[i-1]<time_gap_threshold or i+how_many_data_to_delete>len(time):
				continue
			for j in range(i, i+how_many_data_to_delete):  #the end of gap
				deleted_index=np.append(deleted_index, j)
	
	
		for j in range(len(time)-how_many_data_to_delete, len(time)):
			if j < len(time):
				deleted_index=np.append(deleted_index, j)#end
			for one_deleted_index_index in np.argsort(-deleted_index):
				one_deleted_index = deleted_index[one_deleted_index_index]
				if one_deleted_index < len(time):
					time=np.delete(time, one_deleted_index)
					flux=np.delete(flux, one_deleted_index)
	#print('*******')
	#print(time)
	return time, flux

def make_amplitude_to_be_same(time, flux, boundaries):

	overall_std=np.std(flux)
	#print(boundaries)
	for i in np.arange(0, len(boundaries[:,0])  ):
		# print('Now the boundaries are',boundaries[i,0], boundaries[i,1])
		index=np.where((time >= boundaries[i,0]) & (time <= boundaries[i,1]))[0]
		#print('len(index)', len(index))
		if np.std(flux[index]) == 0:
			continue
		#print('np.std(flux[index])', np.std(flux[index]))
		raw_flux_section = flux[index]
		flux[index]=( flux[index]-np.median(flux[index]) )/np.std(flux[index]) * overall_std +1
		flux[index]=flux[index]-np.median(flux[index])+1
		#print('overall_std, np.std(flux[index])', overall_std, np.std(flux[index]))

		#plt.figure()
		#plt.plot(time[index], raw_flux_section, 'k.')
		#plt.plot(time[index], flux[index], 'b.')
		#plt.show()
	return time,flux
	#plt.figure()
	#plt.plot(time, flux, '.')
	#plt.title('corrected flux')
	#plt.show()







def MAIN_KEPLER_PROGRAM_no_emcee(data_dir, STAR_NAME, exit_for_solar_like):
	if 'KIC' in data_dir:	
		STAR_NAME=data_dir[data_dir.index('KIC'):data_dir.index('KIC')+12]
	print('read data...')
	#==============read data==================#

	remove_slow_trend_only=False
	if exit_for_solar_like:
		remove_slow_trend_only = False
	if remove_slow_trend_only:
		input_file_name_extension='_light_curves_raw.dat'
	else:
		input_file_name_extension='_light_curves.dat'



	#input_file_name_extension='_light_curves_raw.dat'
	output_light_curve_name_if_not_remove_slow_trend_only='_light_curves_after_process.dat'
	output_light_curve_name_if_remove_slow_trend_only='_light_curves_with_eclipse.dat'#for eclipsing binaries	

	print(data_dir+'/'+STAR_NAME+input_file_name_extension)
	print("os.path.isfile(data_dir+'/'+STAR_NAME+input_file_name_extension)", os.path.isfile(data_dir+'/'+STAR_NAME+input_file_name_extension))
	if not os.path.isfile(data_dir+'/'+STAR_NAME+input_file_name_extension):
		print('return here')
		return
		data=read_kepler_data(data_dir)
	

		print('save data...')
		time=data[:,0]
		flux=data[:,3]
	
		time_sort_index=np.argsort(time)
		time=time[time_sort_index]
		flux=flux[time_sort_index]

		fid=open(data_dir+'/'+STAR_NAME+input_file_name_extension,'w')
	
		for i in range(0,len(time)):
			fid.write(str(time[i])+'   '+str(flux[i])+'\n')
		fid.close()
	else:
		print('load data now!')
		light_curve=np.loadtxt(data_dir+'/'+STAR_NAME+input_file_name_extension)
		print('load data successfully')
		time=light_curve[:,0]
		flux=light_curve[:,1]

		#bin data only for the third year of TESS data

		#bin_index = np.where(time > 1800)[0]
		#not_bin_index = np.where(time < 1800)[0]
		#tmp_bin_time, tmp_bin_flux = bin_data_function(time[bin_index], flux[bin_index], 30/60/24)
		#time = np.array(list(time[not_bin_index]) + list(tmp_bin_time) )
		#flux = np.array(list(flux[not_bin_index]) + list(tmp_bin_flux) )



	if np.median(flux)>10:
		flux=flux/np.median(flux)



	#print('median',np.median(flux))

	time_sort_index=np.argsort(time)
	time=np.sort(time)#-np.min(time)
	flux=flux[time_sort_index]-np.mean(flux)

	#if np.min(np.diff(time))<1/60/24:#ten minutes
	#	print('Note, bin data')
	#	time, flux=bin_data_function(time, flux, 10/60/24)


	if True:#remove slow trend
		#time, flux=remove_data_at_beginning_or_end_of_each_data_group(time, flux)
		plt.figure(figsize=(18,10))
		ax=plt.subplot(111)
		ax.plot(time, flux+1, 'b.')
		high_pass_filter_mode = True
		polyfit_mode = False
		if high_pass_filter_mode:
			_, _, flux_median, first_boundary=remove_slow_trend(time, flux)
			#ax.plot(time, flux_median, 'k')
			_, _, flux_median, first_boundary=remove_slow_trend(time, flux_median)
			#ax.plot(time, flux_median, 'y')
			_, _, flux_median, boundary=remove_slow_trend(time, flux_median)
			flux=(flux+1)/flux_median
		else:
			_, _, flux_median, boundary = polyfit_remove_slow_trend(time, flux)
			flux=(flux+1)/flux_median

		time_for_flux_median = time 

		#plt.plot(time, flux, '.')
		#plt.show()

		#boundary=np.append(boundary, max(cut_time)+1.5)
		#np.savetxt(data_dir+'/'+STAR_NAME+'_sector_boundaries.dat', time[int(boundary)], header="#Boundaries time begin, end, begin, end, begin, end....")
		boundary_fid = open(data_dir+'/'+STAR_NAME+'_sector_boundaries.dat', 'w')
		boundary_fid.write("#Boundaries time begin, end\n")
		for boundary_counter in range(len(boundary)-1):
			if boundary_counter == 0:
				begin=time[0]
			else:
				begin=time[int(boundary[boundary_counter])+1]

			end = time[int(boundary[boundary_counter+1])]
			boundary_fid.write(  str(begin)+' '+str(end)+'\n'        )
		boundary_fid.close()

		#cut a the beginning and end of data
		if high_pass_filter_mode:
			#cut_time, cut_flux = remove_data_at_beginning_or_end_of_each_data_group(time, flux)
			cut_time, cut_flux = time*1.0, flux*1.0
		else:
			cut_time, cut_flux = time, flux

################
		#make amplitude to be same
		if len(glob.glob(data_dir+'/*sector_boundaries.dat'))!=0:
			#print('qwertyyiyo')
			print(glob.glob(data_dir+'/*sector_boundaries.dat'))
			boundaries=np.loadtxt( glob.glob(data_dir+'/*sector_boundaries.dat')[0] )
			#print('The boundaries are', boundaries)
			#boundaries=np.append(boundaries, max(cut_time)+1.5)
			#print('normalise the amplitudes in different data sections')
			#cut_time, cut_flux=make_amplitude_to_be_same(cut_time, cut_flux, boundaries)
			#cut_time, cut_flux=make_amplitude_to_be_same(cut_time, cut_flux, boundaries)

################


		#plt.plot(time, flux, '.')
		#plt.show()

		#ascii.write([time, flux], data_dir+'/'+STAR_NAME+'_light_curves_with_eclipse.dat', names=['#time', 'flux'], overwrite=True)
		ax.plot(cut_time, cut_flux-0.04, 'r.')
		ax.plot(time_for_flux_median, flux_median, 'g-')
		ylim_now=ax.get_ylim()
		for i in boundary[1:-1]:#plot boundarys
			if int(i)+1<len(time)-1:
				ax.plot([0.5*(time[int(i)+1]+time[int(i)]), 0.5*(time[int(i)+1]+time[int(i)]) ], ylim_now, 'y-')
			else:
				ax.plot([time[int(i)], time[int(i)]], ylim_now, 'y-')

		plt.ylim(ylim_now)
		plt.xlabel('Time, d')
		plt.ylabel('Flux')
		plt.tight_layout()
		if remove_slow_trend_only:
			plt.savefig(data_dir+'/'+STAR_NAME+'_light_curves_with_median_with_eclipse.png', dpi=150)
		else:
			plt.savefig(data_dir+'/'+STAR_NAME+'_light_curves_with_median.png', dpi=150)
		#plt.show()
		plt.close()
		#return

	time, flux = cut_time, cut_flux

	if not remove_slow_trend_only:
		index=np.where(np.abs(flux-np.median(flux))> 5*np.std(flux)   )[0]
		time=np.delete(time, index)
		flux=np.delete(flux, index)
		##############for gaia only!##########
		ascii.write([time, flux], data_dir+'/'+STAR_NAME+output_light_curve_name_if_not_remove_slow_trend_only, names=['#time', 'flux'], overwrite=True)
	else:
		ascii.write([time, flux], data_dir+'/'+STAR_NAME+output_light_curve_name_if_remove_slow_trend_only, names=['#time', 'flux'], overwrite=True)

	plt.figure(figsize=(12, 5))
	plt.plot(time,flux,'.')
	title(STAR_NAME);plt.xlabel('Time in d');plt.ylabel('Normalised flux')
	plt.tight_layout()
	plt.savefig(data_dir+'/'+STAR_NAME+'_light_curves.png', dpi=150)
	#plt.show()
	plt.close()
########################
	if remove_slow_trend_only:
		return	 #return here if you only want to remove slow trend in light curve #======================================================================================================================================================================================================#
########################

######################## for solar-like oscillators
	if exit_for_solar_like:
		return time, flux
########################

	print('clean frequencies...')
	time_spacing=time[1:]-time[0:-1]
	#plt.hist(time_spacing[np.where(( 25/60/24<time_spacing) & (time_spacing<35/60/24))[0]], bins=50)
	#plt.plot(time_spacing)
	#plt.show()
	time_spacing_index=np.where(    np.abs(time_spacing-np.median(time_spacing))<np.std(time_spacing)    )[0]
	time_step= np.median(time_spacing[time_spacing_index]) 
	delta_freq=1/(max(time)-min(time))
	#print(np.min(time), np.max(time), time_step)
	#max_time, min_time=np.min(time), np.max(time)
	arti_time=np.arange(np.min(time), np.max(time), time_step)
	#get nyquist frequency
	freq=np.arange(len(arti_time))/(max(arti_time)-min(arti_time))
	length=len(arti_time)
	freq=freq[0:int(0.5*length)]
	nyquist_freq=max(freq)
	#save amplitude spectrum
	ls=LombScargle(time, flux)
	#freq,power=ls.autopower(nyquist_factor=2,samples_per_peak=10,  normalization='psd')
	#smaller_nyquist_index=np.where(freq<nyquist_freq)[0]
	#freq, power=freq[smaller_nyquist_index], 2*(power[smaller_nyquist_index]/len(time))**0.5     #amplitude


	freq, power = calculate_amplitude_spectrum(time, flux, 10)

	spectrum_output=np.zeros((len(freq),2))
	spectrum_output[:,0], spectrum_output[:,1]=freq, power
	np.savetxt(data_dir+'/'+STAR_NAME+'_FFT_results.dat', spectrum_output, header='#freq, amplitude')
	plt.figure(figsize=(12, 5))
	plt.plot(freq, power)
	plt.title(STAR_NAME);plt.xlabel('Frequency in c/d');plt.ylabel('Fractional amplitude')
	#plt.xlim([0, 100])
	plt.savefig(data_dir+'/'+STAR_NAME+'_FFT.png')
	plt.xlim([0, 5])
	plt.savefig(data_dir+'/'+STAR_NAME+'_FFT_zoom_in.png')
	#a=input()
	plt.close()
	period=1/freq
	index=np.where((period>0.2) & (period<3.0))[0]
	spectrum_output=np.zeros((len(period[index]),2))
	spectrum_output[:,0], spectrum_output[:,1]=period[index], power[index]
	np.savetxt(data_dir+'/'+STAR_NAME+'_g_mode_power_period_data.dat', spectrum_output, header='#freq, amplitude')
	plt.figure(figsize=(12, 5))
	plt.plot(period[index], power[index])
	plt.title(STAR_NAME);plt.xlabel('Period in d');plt.ylabel('Fractional amplitude')
	plt.savefig(data_dir+'/'+STAR_NAME+'_FFT_period.png')
	plt.close()
#====
	# ===========================================
	#return  #return here if you do not want prewhitening
#===============================================
	#high-pass filter
	
	#plt.plot(time, flux, 'r.')
	#plt.show()
	#flux=flux+1  #flux is around 0
	#arti_time=time/max(time)
	##high_pass_filter_parameters=np.polyfit(arti_time, flux, 2)
	#print(high_pass_filter_parameters)
	#flux=flux/(np.polyval(high_pass_filter_parameters, arti_time))-1
	#plt.plot(arti_time*max(time),np.polyval(high_pass_filter_parameters, arti_time)-1, 'b.')
	#plt.show()
	#a=input('enter to continue')

#=============================================
	#prewhitening
	output_dir=data_dir+'/'+STAR_NAME+'_deep_clean'
	if not os.path.exists(output_dir):os.system('mkdir '+output_dir)
	prewhitening(time, flux, output_dir, 1500, STAR_NAME, False, 9999)
#================================================
	#get all peak locations
	convert_deep_clean(one_dir, STAR_NAME)
#================================================


	#test combinations
	combination_control(one_dir, STAR_NAME)
	#data_to_combination=np.loadtxt(output_dir+'/'+STAR_NAME+'very_deep_clean.dat')
	#check_combinations(data_to_combination[:,0], data_to_combination[:,1], freq, power, one_dir)
	return

def combination_control(work_dir, STAR_NAME):#control the function 'check_combinations'
	cwd=os.getcwd()
	os.chdir(work_dir)
	if 'KIC' in STAR_NAME:
		KIC=work_dir[work_dir.index('KIC'):work_dir.index('KIC')+12]
	else:
		KIC=STAR_NAME
	extracted_freq=np.loadtxt(glob.glob('./'+KIC+'*deep_clean/'+KIC+'*very_deep_clean.dat')[0])
	fft=np.loadtxt(glob.glob('./'+KIC+'*FFT_results.dat')[0])

#load light curve
	light_curve=np.loadtxt('./'+KIC+'_light_curves.dat')
	T=np.max(light_curve[:,0])-np.min(light_curve[:,0])
	N=len(light_curve[:,0])

	freq_err=6**0.5/3.14159/extracted_freq[:,3]/T

	check_combinations(extracted_freq[:,0],extracted_freq[:,1], freq_err, fft[:,0], fft[:,1], work_dir, KIC)

	os.chdir(cwd)


if __name__=='__main__':
	import time

	#dir_list = glob.glob('/Volumes/ligangwork/CLUSTER_TESS_LEUVEN/NGC2516/TIC*_MCMC')
	#dir_list = ['/Volumes/ligangwork/CLUSTER_TESS_LEUVEN/NGC2516/gdor_stars/TIC281582674_MCMC']
	dir_list = glob.glob('/Volumes/ligangwork/CLUSTER_TESS_LEUVEN/NGC2516/gdor_stars/TIC*_MCMC')


	#print(dir_list)
	if 'KIC' in dir_list[0]:
		dir_list=sorted(dir_list, key=lambda name: int(name[name.index('KIC')+3:name.index('KIC')+12]))
	if 'TIC' in dir_list[0]:
		dir_list=sorted(dir_list, key=lambda name: int(name[name.index('TIC')+3:name.index('_MCMC')]))
	dir_counter=0
	for one_dir in dir_list:
		if '364398882' in one_dir: #4150611, 4544587,
			break
		dir_counter+=1

	#dir_counter+=1
	dir_counter=0

	#KIC_list=np.loadtxt('/export/pushkin1/gali8292/project1/Nic_Barara_gdor_list/KIC_list.dat')
	#dir_list=[]
	#for one_KIC in KIC_list:
	#	one_KIC=str(int(one_KIC))
	#	dir_list.append('/export/pushkin1/gali8292/project1/Nic_Barara_gdor_list/KIC*'+one_KIC+'*')

	wanted_star_list = open('/Volumes/ligangwork/CLUSTER_TESS_LEUVEN/NGC2516/NGC2516_dsct_list.dat').read() + open('/Volumes/ligangwork/CLUSTER_TESS_LEUVEN/NGC2516/NGC2516_gdor_list.dat').read()
	#print(wanted_star_list)
	#exit()

	#fast rotator with splitting list
	#KIC_list=['3348714','4285040', '4846809', '4952246', '5476473', '5708550', '6765575', '7701947', '7778114', '8352439', '8523871', '9595743', '']

	for one_dir in dir_list[dir_counter:]:
		print('\n\n\n')

		if len(glob.glob(one_dir+'/pixel_file/*png'))==0:
			os.system('mv -f '+one_dir+' /Volumes/ligangwork/CLUSTER_TESS_LEUVEN/NGC2516/bad_data')
			print('bad data', one_dir)
			continue
		#print(one_dir)
		one_dir=glob.glob(one_dir)[0]
		print(dir_counter,one_dir)
		dir_counter+=1
		
		if len(glob.glob(one_dir+'/TIC*light_curves.dat'))==0:
			continue
		#if len(glob.glob(one_dir+'/*difference_distribution.png'))==1:
		#	continue

		#if 'KIC' in one_dir:
		#	STAR_NAME=one_dir[one_dir.index('KIC'):one_dir.index('KIC')+12]
		#else:
		#	STAR_NAME=one_dir[one_dir.index('tic'):one_dir.index('tic')+13]

		STAR_NAME=get_star_name(one_dir)

		print(STAR_NAME)

		#if STAR_NAME.replace('TIC', '') not in wanted_star_list:
		#	continue

		#if STAR_NAME.replace('KIC00','KIC').replace('KIC0', 'KIC').replace('KIC','') not in KIC_list:continue

		#if len(glob.glob(one_dir+'/*deep_clean/*very_deep_clean.dat'))>0:continue

		#get rotation rate

		#if len(glob.glob('/export/pushkin1/gali8292/project1/splitting_stars/slow/KIC*'+STAR_NAME+'*'))>0:continue
		exit_for_solar_like=False
		MAIN_KEPLER_PROGRAM_no_emcee(one_dir, STAR_NAME, exit_for_solar_like)	

		#print('Sleep 10 seconds')
		#time.sleep(10)

#combination_control(one_dir)



























