from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature
from imutils import build_montages  # Visualization
from imutils import paths  #  Extract the file paths to each images in the dataset
import argparse
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Feature Extraction from each image
def quantify_image(image):
	features = feature.hog(image, orientations = 9,
                        pixels_per_cell = (10, 10), cells_per_block = (2,2),
	                    transform_sqrt = True, block_norm = "L1")
	
	return features

# Load the data
def load_split(path):
	# grab the list of images in the input directory, then initialize
	# the list of data (i.e., images) and class labels
	imagePaths = list(paths.list_images(path))
	data = []
	labels = []

	# loop over the image paths
	for imagePath in imagePaths:
		# extract the class label from the filename
		label = imagePath.split(os.path.sep)[-2]

		# load the input image, convert it to grayscale, and resize
		# it to 200x200 pixels, ignoring aspect ratio
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.resize(image, (200, 200))

		# threshold the image such that the drawing appears as white
		# on a black background
		image = cv2.threshold(image, 0, 255,
			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

		# quantify the image
		features = quantify_image(image)

		# update the data and labels lists, respectively
		data.append(features)
		labels.append(label)

	# return the data and labels
	return (np.array(data), np.array(labels))

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset", required = True, help = "path to input dataset")
ap.add_argument("-t","--trials", type = int, default = 5, help = "# of trials to run")
args = vars(ap.parse_args())

trainingPath = os.path.sep.join([args["dataset"], "training"])
testingPath = os.path.sep.join([args["dataset"], "testing"])

print("[INFO] Loading data...")
(trainX, trainY) = load_split(trainingPath)
(testX, testY) = load_split(testingPath)

# Encode the labels as integers
le = LabelEncoder()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)

# initialize our trials dictionary
trials = {}

# loop over the number of trials to run
for i in range(0, args["trials"]):
	# train the model
	print("[INFO] training model {} of {}...".format(i + 1,
		args["trials"]))
	model = RandomForestClassifier(n_estimators=100)
	model.fit(trainX, trainY)

	# make predictions on the testing data and initialize a dictionary
	# to store our computed metrics
	predictions = model.predict(testX)
	metrics = {}

	# compute the confusion matrix and and use it to derive the raw
	# accuracy, sensitivity, and specificity
	cm = confusion_matrix(testY, predictions).flatten()
	(tn, fp, fn, tp) = cm
	metrics["acc"] = (tp + tn) / float(cm.sum())
	metrics["sensitivity"] = tp / float(tp + fn)
	metrics["specificity"] = tn / float(tn + fp)

	# loop over the metrics
	for (k, v) in metrics.items():
		# update the trials dictionary with the list of values for
		# the current metric
		l = trials.get(k, [])
		l.append(v)
		trials[k] = l
for metric in ("acc", "sensitivity", "specificity"):
    # Grab the list of values for the current metric, then compute the mean and standard deviation
    values = trials[metric]
    mean = np.mean(values)
    std = np.std(values)

    print(metric)
    print("="*len(metric))
    print("u={:.4f}, o={:.4f}".format(mean, std))
    print("")

testingPaths = list(paths.list_images(testingPath))
idxs = np.arange(0, len(testingPath))
idxs = np.random.choice(idxs, size = (25,), replace = False)
images = []


for i in idxs:
    image = cv2.imread(testingPaths[i])
    output = image.copy()
    output = cv2.resize(output, (128, 128))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))
    image = cv2.thresold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THERSH_OTSU)[1]

    features = quantify_images(image)
    preds = mpdel.predict([features])
    label = le.inverse_transform(preds)[0]

    color = (0, 255, 0) if label == "healthy" else (0, 0, 255)
    cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    images.append(output)


montage = build_montages(images, (128, 128), (5,5))[0]

cv2.imshow("output", montage)
cv2.waitKey(0)








