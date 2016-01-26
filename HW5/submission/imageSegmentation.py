'''
    Part II (1) Image Segmentation using K-means
    AUTHOR Bahram Banisadr
'''
import sys
import numpy as np
from scipy import misc
from sklearn import preprocessing


def main(argv):

	# Get number of clusters, input file, and output file
	k = int(sys.argv[1])
	img = misc.imread(sys.argv[2])
	output_image = sys.argv[3]

	# Get matrix dimentions
	n,d,l = img.shape

	# Create five dimentional data vector
	five_dimentional = feature_expand(n,d,img)

	# Record mean & std. dev for future use
	mean = np.mean(five_dimentional,axis=0)
	std_dev = np.std(five_dimentional,axis=0)
	
	# Standardize data
	five_dimentional = preprocessing.scale(five_dimentional)

	# Run k-means on feature matrix
	centroids = k_means(k,five_dimentional)

	# Get labels for data
	labels = predict(centroids,five_dimentional)

	# Rescale data
	centroids = np.add(np.multiply(centroids,std_dev),mean)

	# Replace pixel colors with those of nearest centroid
	masked_image = replace_colors(n,d,labels,centroids)

	# Save to output file
	misc.imsave(output_image,masked_image)


def feature_expand(n, d, img):
	'''
	Used to create n*d-by-5 dimetional feature matrix
	'''

	five_dimentional = np.zeros((n*d,5))
	for i in range(n):
		for j in range(d):
			five_dimentional[j + i*d,:] = np.concatenate((img[i,j,:],[i,j]),axis=None)

	return five_dimentional


def k_means(k, five_dimentional):
	'''
	Used to run general K_means and return centroid matrix
	'''

	# Get matrix shape
	n,d = five_dimentional.shape

	# Randomly assign k centroids
	centroids = np.random.rand(k,5)

	# Initialize centroid history
	prev_centroids = np.zeros((k,5))

	while not np.array_equal(centroids,prev_centroids):
		
		# Track history
		prev_centroids = centroids

		# Get labels based on centroids
		labels = predict(centroids,five_dimentional)

		# Get new centroids
		centroids = fit(k,five_dimentional,labels)

	# Upon convergence, return centroids
	return centroids


def predict(centroids, five_dimentional):
	'''
	Takes in centroid matrix and feature matrix

	Returns a vector of labels for the feature matrix
	'''
	# Get matrix shape
	n,d = five_dimentional.shape

	# Get value of k
	k,l = centroids.shape

	# Initialize distance matrix
	distance = np.zeros((n,k))

	# Calculate distances
	for i in range(k):
		distance[:,i] = np.sqrt(np.sum(np.square(five_dimentional - centroids[i,:]),axis=1))

	# Get Labels
	labels = np.argmin(distance,axis=1)

	# Check for missing labels
	missing = np.setdiff1d(range(k),labels)

	# If cluster empty, assign random data point
	for i in range(missing.size):
		rand_index = np.random.randint(0,n)
		labels[rand_index] = missing[i]

	# Return Labels
	return labels


def fit(k, five_dimentional, labels):
	'''
	Takes in featrue matrix and data labels
	Returns k new centroids
	'''

	# Initialize centroid matrix
	centroids = np.zeros((k,5))

	# For each label
	for i in range(k):
		
		# Get points of that label
		labeled_points = labels == i

		# Take mean to calculate centroids
		centroids[i,:] =  np.mean(five_dimentional[labeled_points,:],axis=0)

	return centroids


def replace_colors(n,d,labels,centroids):
	'''
	Takes in dimentions of final matrix, labels, and centroids
	Returns n-by-d matrix of colored pixels
	'''
	
	# Initialize return matrix
	masked = np.zeros((n,d,3))

	# Color pixels
	for i in range(n):
		for j in range(d):
			label = labels[j + i*d]
			masked[i,j,:] = centroids[label,0:3]

	return masked

if __name__ == "__main__":
	main(sys.argv)