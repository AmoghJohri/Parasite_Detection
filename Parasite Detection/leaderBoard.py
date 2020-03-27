import cv2
import os
import numpy as np
import csv
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy
import random
import sklearn

""" ######################## Helper Functions (BLOB) ######################## """
def adjust_gamma(image, gamma=1.0):
    	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

# loading the dataset
def load_dataset(path) :
    class_images = []
    img_names = os.listdir(path)
    f = open("/home/redhood/Desktop/CollegeWork/Semester_5/Machine_Learning/ProjectMe/out.csv","w")
    f.write("Name\n")
    for names in img_names:
        f.write(names)
        f.write('\n')

    for img_name in img_names :
            try :
                class_images.append(cv2.imread(path + img_name))
            except Exception as e :
                pass
    return np.asarray(class_images)

""" ######################## ############### ########################## """
""" ######################## Hyperparameters (BLOB) ######################## """
# Blob detection (Hyperparameters)  
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.maxArea = 200
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.3
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.0
# Change thresholds
params.minThreshold = -3;
params.maxThreshold = 150;
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.0
detector = cv2.SimpleBlobDetector_create(params)
# other hyperparameters
IMG_SIZE = 90
gamma = 0.95
""" ######################## ############### ########################## """



# takes in the path of the image files and returns a dataframe with the 5 features 
def get_contour_features(DATADIR, dataset):
    dataframe = [[],[],[],[],[]]
    for img in dataset:
        img_ = cv2.GaussianBlur(img, (5,5), 2)
        if not (img_ is None):    
            img_gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(img_gray,127,255,0)
            _,contours,_ = cv2.findContours(thresh,1,2)

            for i in range(5):
                try:
                    area = cv2.contourArea(contours[i])
                    dataframe[i].append(area)
                except:
                    dataframe[i].append(0)  

    return (np.asarray(dataframe).T)

def get_number_of_contours(DATADIR, dataset):
    dataframe = []
    for img in dataset:
        if not (img is None):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edged = cv2.Canny(gray, 30 , 200)
            _,contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            dataframe.append(len(contours))
    return (np.asarray(dataframe).T)

# takes in the path of the image files and returns a dataframe with the number of blobs
def get_blob_features(DATADIR, dataset):
    dataframe = [[],[],[]]
    for img in dataset:
        if not (img is None):
            img_ = cv2.resize(img,(IMG_SIZE, IMG_SIZE))
            img_ = adjust_gamma(img_, gamma)   
            keypoints = detector.detect(img_)
            dataframe[0].append(len(keypoints))
            if(len(keypoints) >= 2):
                dataframe[1].append(keypoints[0].size)
                dataframe[2].append(keypoints[1].size)
            elif(len(keypoints) == 1):
                dataframe[1].append(keypoints[0].size)
                dataframe[2].append(0)
            else:
                dataframe[1].append(0)
                dataframe[2].append(0)
    return np.asarray(dataframe).T


# getting sift features
#Computing the histogram. Can return the indices of the dataset which have null features
def calculate_histogram(images, model):
    count = 0
    feature_vectors=[]
    rmv_index_test = []
    for image in images :
        if not (image is None):
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            #SIFT extraction
            sift = cv2.xfeatures2d.SIFT_create()
            kp, descriptors = sift.detectAndCompute(image,None)
            #classification of all descriptors in the model
            if descriptors is not None :
                predict_kmeans = model.predict(descriptors)
                #calculates the histogram
                hist, bin_edges = np.histogram(predict_kmeans, bins = 10)
                #histogram is the feature vector
                feature_vectors.append(hist)
                temp = hist
            else :
                feature_vectors.append(temp)
                # rmv_index_test.append(count)

            count = count + 1
    feature_vectors=np.asarray(feature_vectors)

    return np.array(feature_vectors)

#Generates the SIFT featuress
def generate_sift_features(train_img):
    sift_keypoints = []
    count = 0
    rmv_index_train = []
    for image in train_img :
         if not (image is None):
            image = cv2.resize(image, (80, 80))
            image =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SIFT_create()
            kp, descriptors = sift.detectAndCompute(image,None)
            if descriptors is not None:
                descriptors = np.array(descriptors)
                sift_keypoints.append(descriptors)
                temp = descriptors

            else :
                 sift_keypoints.append(temp)
                 # rmv_index_train.append(count)

            count = count + 1
    sift_keypoints = np.concatenate(sift_keypoints, axis=0)
    kmeans = KMeans(n_clusters = 10).fit(sift_keypoints)
    print(sift_keypoints.shape)
    print("--------Computed descriptors--------")

    x_Siftfeat_train = calculate_histogram(train_img, kmeans)
    # x_Siftfeat_test = calculate_histogram(test_img, kmeans)
    # test_lbl = np.delete(test_lbl,rmv_index_test,axis = 0)
    print("------Computed Histogram-----")

    # scaler = StandardScaler()
    # scaler.fit(x_Siftfeat_train)
    # x_Siftfeat_train = scaler.transform(x_Siftfeat_train)
    # x_Siftfeat_test = scaler.transform(x_Siftfeat_test)
    # print("------Scaled the features--------")

    return x_Siftfeat_train

def generate_sift_features_(DATADIR, dataset):
    training_data = dataset
    # train_img, test_img, train_lbl, test_lbl = train_test_split(training_data, training_label, test_size=0.1, random_state = 0)

    #generate the sift features
    x_Siftfeat_train = generate_sift_features(training_data)
    x_Siftfeat_train = np.asarray(x_Siftfeat_train)

    #Creating the dataframe
    dataset = pd.DataFrame(data = x_Siftfeat_train)
    return dataset


def get_features():
    DATADIR = "/home/redhood/Desktop/CollegeWork/Semester_5/Machine_Learning/Rough/Group_Project/Parasite/Parasite/test_/test/"
    dataset = load_dataset(DATADIR)

    print("Press 1 to read dataset, 0 otherwise")
    x = int(input())
    if x == 1:
        data1 = get_contour_features(DATADIR, dataset)
        df1 = pd.DataFrame(data = data1)
        
        data2 = get_blob_features(DATADIR, dataset)
        df2 = pd.DataFrame(data = data2)
        
        data3 = generate_sift_features_(DATADIR, dataset)
        df3 = pd.DataFrame(data = data3)
        
        data4 = get_number_of_contours(DATADIR, dataset)
        df4 = pd.DataFrame(data = data4)

        df = pd.concat([df2, df1, df4, df3], axis = 1)
        df.to_csv("test_features.csv")
    

def main():
    get_features()


# sync; echo 3 > /proc/sys/vm/drop_caches - to clear RAM

if __name__ == "__main__" :
    main()


