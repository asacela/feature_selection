import numpy as np
import math, copy, time


"""

	Todo:
		- Configure for easy of use with given data files
		- Speed up the program
"""


class ai:

	def __init__(self):
		filename = ''

	# read data from file
	def read_and_save_npy(self, filename):

		data = np.genfromtxt(filename + '.txt', delimiter=',', dtype=np.float32)
		fn = filename + '.npy'
		np.save(fn, data)

	# get default rate
	def get_default_rate(self, data):

		ones_cnt = 0

		for i in range(0, data.shape[0]):

			if data[i, 0] == 1:

				ones_cnt += 1


		zeros_cnt = data.shape[0] - ones_cnt

		if ones_cnt > zeros_cnt:

			return ones_cnt / data.shape[0]

		return zeros_cnt / data.shape[0]


	# takes data and current_features with feature to add included as input
	def accuracy(self, data, current_features):

		my_features = copy.deepcopy(current_features)


		# get label array
		my_labels = data[:, 0]

		# delete label array in main array
		data = np.delete(data, 0, 1)

		

		# restructure current_features
		for j in range(0, len(my_features)):
			my_features[j] -= 1

		# delete features that aren't needed
		del_features = list()
		for i in range(0, data.shape[1]):

			if i not in my_features:
				del_features.append(i)
		data = np.delete(data, del_features, 1)

		num_correct = 0
		for x in range(0, data.shape[0]):

			obj_to_classify = data[x, :]
			obj_lbl = my_labels[x] # not sure about this line syntax wise

			nn_dist = float('inf')
			nn_loc = float('inf')

			for y in range(0, data.shape[0]):

				if x != y:

					# get distance of objects
					dist = np.linalg.norm(obj_to_classify - data[y, :])

					if dist < nn_dist:

						nn_dist = dist
						nn_loc = y

			if obj_lbl == my_labels[nn_loc]: # not sure about this line either

				num_correct += 1

		accuracy = num_correct / data.shape[0]
		# print("\tInterim Accuracy: {} %".format(accuracy * 100))
		return accuracy

	# forward selection algorithm
	def forward(self, data):

		my_current_features_list = list()
		best_features = list()
		best_overall_acc = 0

		print('Scanning Search Tree...')
		for x in range(1, data.shape[1]):

			print('On the ' + str(x) + 'th level of the search tree...')

			best_interim_acc = 0
			feature_to_add = []


			for y in range(1, data.shape[1]):

				if y not in my_current_features_list:

					print('\tConsidering adding feature ' + str(y))

					temp_arr = copy.deepcopy(my_current_features_list)
					temp_arr.append(y)
					acc = self.accuracy(data, temp_arr)

					if acc > best_interim_acc:

						best_interim_acc = acc
						feature_to_add = y

			my_current_features_list.append(feature_to_add)


			print('\tBest Interim: ' + str(best_interim_acc * 100) + ' %')
			print('\tBest Overall: ' + str(best_overall_acc * 100) + ' %')
			print('\tCurrent Set of Features: ', my_current_features_list)

			if best_interim_acc > best_overall_acc:
				best_overall_acc = best_interim_acc
				best_features = copy.deepcopy(my_current_features_list)

			# print('Current Accuracy of Set: ' + str(best_interim_acc))
			# print('Current Set of Features: ')
			# print(my_current_features_list)

		print("Top Accuracy: {} %".format(best_overall_acc * 100))
		print('Best Set of Features: {}'.format(best_features))


	# backward elimination algorithm
	def backward(self, data):

		my_current_features_list = list(range(1, data.shape[1]))
		best_features = list()
		best_overall_acc = 0

		print('Scanning Search Tree...')
		for x in range(1, data.shape[1]):


			print('On the ' + str(x) + 'th level of the search tree...')
			best_interim_acc = 0
			feature_to_remove = []

			for y in range(1, data.shape[1]):


				if y in my_current_features_list:

					print('\tConsidering removing feature ' + str(y))

					temp_arr = copy.deepcopy(my_current_features_list)
					temp_arr.remove(y)
					acc = self.accuracy(data, temp_arr)

					if acc > best_interim_acc:

						best_interim_acc = acc
						feature_to_remove = y

			print("\tIndex to pop: {}".format(feature_to_remove))
			my_current_features_list.remove(feature_to_remove)


			print('\tBest Interim: ' + str(best_interim_acc * 100) + ' %')
			print('\tBest Overall: ' + str(best_overall_acc * 100) + ' %')
			print('\tCurrent Set of Features: ', my_current_features_list)


			best_interim_acc = self.accuracy(data, my_current_features_list) 

			if best_interim_acc > best_overall_acc:
				best_overall_acc = best_interim_acc
				best_features = copy.deepcopy(my_current_features_list)

		print("Top Accuracy: {} %".format(best_overall_acc * 100))
		print('Best Set of Features: {}'.format(best_features))



def run_test(filename):

	my_data = np.load(filename, mmap_mode='r')

	my_ai = ai()

	# get default rate
	print("Calculating Default Rate...\n")
	print("\tDefault Rate: {} %\n".format(my_ai.get_default_rate(my_data) * 100))


	# # run forward selection
	print("Starting Forward Selection Search...\n")
	start = time.time()
	my_ai.forward(my_data)
	end = time.time()
	print("Forward Elapsed Time: {} \n".format(str(end - start)))


	# run backward selection 
	print("Starting Backward Selection Search...\n")
	start = time.time()
	my_ai.backward(my_data)
	end = time.time()
	print("Backward Elapsed Time: {} \n".format(str(end - start)))



# main function 
def main():



	"""
		Small Dataset Test
	"""
	print('Small Dataset Testing Starting...')
	run_test('small_dataset71.npy')


	"""
		Large Dataset Test
	"""
	print('Large Dataset Testing Starting...')
	run_test('large_dataset13.npy')


	"""
		Data Generation Helper Fct
	"""


	# generates npy if needed
	# my_ai.read_and_save_npy(filename)

	
	"""
		Extraneous Code for Testing
	"""

	# Accuracy testing
	# for i in range(1, my_data.shape[1]):
	# 	print('Accuracy of ' + str(i))
	# 	accuracy(my_data, [i])

if __name__ == "__main__":
    main()












