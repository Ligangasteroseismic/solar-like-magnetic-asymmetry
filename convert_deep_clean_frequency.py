import os, glob, numpy as np


def convert_deep_clean(work_dir, STAR_NAME):
	current_dir=os.getcwd()
	os.chdir(work_dir)
	if 'KIC' in STAR_NAME:
		kic_number=work_dir[work_dir.index('KIC'):work_dir.index('KIC')+12]
	else:
		kic_number=STAR_NAME

	if len(glob.glob('./*deep_clean'))!=1:return

	FFT_data=np.loadtxt(glob.glob('*FFT_results.dat')[0])
	deep_clean_data=np.loadtxt(glob.glob('./'+kic_number+'_deep_clean/'+kic_number+'very_deep_clean.dat')[0])
	
#backup
	if not os.path.isdir('./frequency_from_old_extraction'):
		os.system('mkdir ./frequency_from_old_extraction')
		os.system('cp *peaks_locations.dat ./frequency_from_old_extraction')#if existed, do not backup


#get all_peaks_locations
	out_fid=open('./'+kic_number+'_deep_clean/'+kic_number+'all_peaks_locations.dat','w')
	out_fid.write('#peaks_index      peaks_frequencies\n')
	for freq in deep_clean_data[:,0]:
		index=np.argsort(np.abs(FFT_data[:,0]-freq))
		index=index[0:5]
		index=index[np.argmax(FFT_data[index,1])]
		out_fid.write(str(int(index))+' '+str(FFT_data[index,0])+' '+str(freq)+'\n')

	out_fid.close()

#get peaks_locations (for g mode)
	out_fid=open('./'+kic_number+'_deep_clean/'+kic_number+'peaks_locations.dat','w')
	out_fid.write('#peaks_index      peaks_periods    peaks_accurate_frequency\n')
	for freq in deep_clean_data[:,0]:
		if not 0.2< 1/freq<2: continue
		index=np.argmin(np.abs(FFT_data[:,0]-freq))
		out_fid.write(str(int(index))+' '+str(1/FFT_data[index,0])+' '+str(freq)+'\n')

	out_fid.close()
	

#copy newly-generated files
	os.system('cp ./'+kic_number+'_deep_clean/*peaks_locations.dat ./')


	os.chdir(current_dir)


def combine_old_new_frequencies(work_dir):#combine the frequencies from the old and the new functions
	kic_number=work_dir[work_dir.index('KIC'):work_dir.index('KIC')+12]
	current_dir=os.getcwd()
	os.chdir(work_dir)

#old frequency
	old_data=np.loadtxt(glob.glob('./frequency_from_old_extraction/*all_peaks_locations.dat')[0])

#new frequency
	new_data=np.loadtxt(glob.glob('./KIC*_deep_clean/*all_peaks_locations.dat')[0])

#combine them
	column1, column2, column3=list(old_data[:,0]),list(old_data[:,1]),list(old_data[:,2])
	counter=0
	for counter in range(len(new_data[:,0])):
		if not new_data[counter,0] in column1:
			column1.append(new_data[counter,0])
			column2.append(new_data[counter,1])
			column3.append(new_data[counter,2])
	

#output
	output_data=np.zeros((len(column1),3))
	output_data[:,0],output_data[:,1],output_data[:,2]=column1, column2, column3
	filename=glob.glob('./*all_peaks_locations.dat')[0]
	print(filename)
	np.savetxt(filename, output_data, header='peaks_index      peaks_frequencies')


	output_data[:,0],output_data[:,1],output_data[:,2]=column1, column2, column3
	filename=glob.glob('./'+kic_number+'peaks_locations.dat')[0]
	index=np.where((1/output_data[:,1]>0.2) & (1/output_data[:,1]<2))[0]
	output_data=output_data[index, :]
	output_data[:,1]=1/output_data[:,2]
	print(filename)
	np.savetxt(filename, output_data,header='peaks_index      peaks_periods    peaks_accurate_frequency')



	os.chdir(current_dir)

if __name__ == '__main__':
	dir_list=glob.glob('/export/pushkin1/gali8292/project1/KIC004480321*')
	for one_dir in dir_list:
		print(one_dir)
		try:
			convert_deep_clean(one_dir)
			combine_old_new_frequencies(one_dir)
		except:
			print('something wrong')
