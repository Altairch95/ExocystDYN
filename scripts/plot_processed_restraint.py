import glob
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns


def do_subplots(data_list, key):
	"""
	Plotting the score distribution vs MC steps,
	gathered from the stat.0.out file when processed
	with the process_output.py script.

	Plotting values:
		- 'ConnectivityRestraint'
		- 'DistanceRestraint_Score_restraint'
			- tag_to_protein_restraints
			- PICT restraints
		- 'ExcludedVolumeSphere_None'
	:param data_list: data from dictionary
	:param key: key in dictionary, corresponding to each
	plotting value.
	"""
	num_subplots = len(data_list)  # total number of subplots
	print(f'\tPlotting {key} with a total of {num_subplots} plots...\n')
	if key == 'ConnectivityRestraint' or key == 'tag_to_protein_restraints':
		num_cols = 3  # number of columns per plot
	elif key == 'ExcludedVolumeSphere_None':
		num_cols = 1
	else:
		num_cols = 3
	num_rows = num_subplots // num_cols  # number of rows
	if num_subplots % num_cols != 0:
		num_rows += 1
	position = range(1, num_subplots + 1)  # position index
	# Create main figure
	fig = plt.figure(figsize=(5 * num_rows, 5 * num_cols), dpi=250)
	# Plot the scores for each restraint vs MC steps
	for i in range(num_subplots):
		ax = fig.add_subplot(num_rows, num_cols, position[i])
		value = data_list[i]
		value_name = value.split('/')[-1] \
			.replace(f'{name_input_dir}_', "") \
			.replace('Restraint', '_Restraint') \
			.replace('_', '\n') \
			.replace(".txt", "")
		data = [float(line.split(' ')[-1].strip()) for line in open(value, 'r').readlines()
				if line.startswith('>')]
		print(f'\t\tPlotting {value_name}')
		sns.lineplot(x=range(len(data)), y=data, ax=ax)
		ax.set_xlabel('MC steps')
		ax.set_ylabel(value_name)
		fig.tight_layout(pad=2.0)
	print(f'\tSaving figure ../{output_dir}/{key}.png')
	plt.savefig(f'../{output_dir}/{key}.png')
	plt.clf()

################
#    MAIN
################


if __name__ == '__main__':
	# Create directory to store processed output
	name_input_dir = sys.argv[1]  # 'run_arch_10'
	output_dir = f'{name_input_dir}_processed'
	if not os.path.exists(f'../{output_dir}/'):
		os.mkdir(f'../{output_dir}/')

	# Dump all the possible restraints to process in a file
	restraint_file = f'../{output_dir}/{name_input_dir}_restraints.txt'  # sys.argv[1]

	if not os.path.exists(restraint_file):
		os.system(f'python3 process_output.py -f ../output/{name_input_dir}/stat.0.out -p >> {restraint_file}')

	# Check weight values on the restraints of interest
	# Count if all restraints have been processed (a total of 103)
	num_restraints = int(os.popen(f"less ../{output_dir}/{restraint_file} | "
	 			 "grep 'DistanceRestraint_Score_restraint\|ExcludedVolumeSphere_None\|ConnectivityRestraint' "
	 			 "| wc -l").read().strip())
	print(f'Processing {num_restraints} restraints...\n')
	if len(glob.glob(f'../{output_dir}/*.txt')) != num_restraints + 1:  # + restraint_file.txt
		for line in open(restraint_file, 'r').readlines():
			if 'ConnectivityRestraint' in line\
					or 'DistanceRestraint_Score_restraint' in line\
					or 'ExcludedVolumeSphere_None' in line:
				os.system(f'python3 process_output.py -f ../output/{name_input_dir}/stat.0.out -s {line.strip()} >> '
						  f'../{output_dir}/{name_input_dir}_{line.strip()}.txt')

	print('Plotting Results...\n')
	plots_dict = {
		'ConnectivityRestraint': list(),
		'DistanceRestraint_Score_restraint': list(),
		'ExcludedVolumeSphere_None': list()
	}
	for w_file in glob.glob(f'../{output_dir}/*txt'):
		if w_file != restraint_file:
			if 'ConnectivityRestraint' in w_file:
				plots_dict['ConnectivityRestraint'].append(w_file)
			elif 'DistanceRestraint_Score_restraint' in w_file:
				plots_dict['DistanceRestraint_Score_restraint'].append(w_file)
			elif 'ExcludedVolumeSphere_None' in w_file:
				plots_dict['ExcludedVolumeSphere_None'].append(w_file)

	for k in plots_dict:
		if k == 'DistanceRestraint_Score_restraint':
			sub_plots_dict = dict()
			sub_plots_dict.setdefault('tag_to_protein_restraints', sorted([r for r in plots_dict[k] if 'restraint_1-' in r]))
			sub_plots_dict.setdefault('sec3_FRB_restraints', sorted([r for r in plots_dict[k] if 'Sec3-FRB' in r]))
			sub_plots_dict.setdefault('sec6_FRB_restraints', sorted([r for r in plots_dict[k] if 'Sec6-FRB' in r]))
			sub_plots_dict.setdefault('sec8_FRB_restraints', sorted([r for r in plots_dict[k] if 'Sec8-FRB' in r]))
			sub_plots_dict.setdefault('sec10_FRB_restraints', sorted([r for r in plots_dict[k] if 'Sec10-FRB' in r]))
			sub_plots_dict.setdefault('sec15_FRB_restraints', sorted([r for r in plots_dict[k] if 'Sec15-FRB' in r]))
			sub_plots_dict.setdefault('exo70_FRB_restraints', sorted([r for r in plots_dict[k] if 'Exo70-FRB' in r]))
			sub_plots_dict.setdefault('exo84_FRB_restraints', sorted([r for r in plots_dict[k] if 'Exo84-FRB' in r]))
			for sub_k in sub_plots_dict:
				do_subplots(sub_plots_dict[sub_k], sub_k)
		else:
			do_subplots(plots_dict[k], k)

	print('\n\nDONE\n')
	sys.exit(0)
