##### Importing relevant libraries ##############################
import cv2
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
################################################################

############################# Data #############################
# image resizing variables
dimensions = (64, 64)
scaling_factor = 2.0

# local binary pattern @param
radius= 1                                                   
n_points= 8 * radius

# thresholding @param
region_size= 20
threshold_factor = 0.7
normalization_radius= 3

# model training
labels= []
################################################################

########################## Helper functions ####################
# Method to divide LBP image in non-overlapping regions
def divide_into_regions(_lbp, _region_size):
    height, width = _lbp.shape
    regions = []
    for i in range(0, height, _region_size):
        for j in range(0, width, _region_size):
            region= _lbp[i:i+_region_size, j:j+_region_size]
            regions.append(region)
    return regions
#######################################################

# Method to normalize LBP histogram ###################
def normalize_histogram(_histogram):
    return _histogram / np.sum(_histogram)
#######################################################

# Method to threshold computed LBP histogram ##########
def threshold_histogram(_histogram, _threshold_factor):
    _histogram = normalize_histogram(_histogram)
    threshold = _threshold_factor * np.mean(_histogram)
    _histogram[_histogram < threshold] = 0
    _histogram[_histogram >= threshold] = 1
    return _histogram
#######################################################

# Method to normalize LBP feature map/LBP image ##########
def normalize_lbp_image(lbp_image, new_min=0, new_max=255):
    min_val = np.min(lbp_image)
    max_val = np.max(lbp_image)

    # Avoiding zero division
    if min_val == max_val:
        return np.full_like(lbp_image, new_min)

    normalized_image = ((lbp_image - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min
    return normalized_image.astype(np.uint8)

###########################################################################

def extract_features(image_path, threshold_factor):
    img= cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(img, n_points, radius, method='uniform')
    # lbp = normalize_lbp_image(lbp)
    # lbp = np.ravel(lbp)

    ##### overall thresholding (non-regional)
    # variance = cv2.GaussianBlur(lbp, (normalization_radius, normalization_radius), 0)
    # variance = np.var(variance)
    # normalized_lbp = (lbp - np.mean(lbp)) / max(np.sqrt(variance), 1)
    # thresholded_lbp= np.where(normalized_lbp>=(np.mean(normalized_lbp)), 1, 0)
    # lbp_histogram, _= np.histogram(thresholded_lbp, bins=np.arange(0, 3**n_points+1), density=True)


    ##### region-based thresholding
    regions= divide_into_regions(lbp, region_size)
    thresholded_regions= []
    for region in regions:
        histogram, _= np.histogram(region, bins=np.arange(0, 10), density=True)
        histogram = threshold_histogram(histogram, threshold_factor)
        thresholded_regions.append(histogram)
    thresholded_image= np.concatenate(thresholded_regions)

    return thresholded_image.ravel()
    # return normalized_lbp.ravel()
    # return lbp.ravel()
    # return np.ravel(img)
    # return lbp_histogram

############################ Main function FOR OASIS MRI DATABASE ##########################################################
if __name__=="__main__":

    texture_directory = 'C:/Users/HP/Desktop/Python_AI/lbp-descriptor-textureRecog/assets/OASIS_MRI_DB'

    features = []
    labels = []



    for category in range(1, 5):
        category_path = os.path.join(texture_directory, f'OASIS_Cross_{category}_converted')
        for filename in os.listdir(category_path):
            if filename.endswith('.jpg'):  
                image_path = os.path.join(category_path, filename)
                feature = extract_features(image_path, 0.3)
                features.append(feature)
                labels.append(category)

    # features=np.array(features)
    # print("Features list: ", features)
    # labels=np.array(labels)
    # print("Labels list: ", labels)
    # features=features.flatten()
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # classifier = make_pipeline(StandardScaler(), SimpleImputer(strategy='mean'), SVC(kernel='linear', C=1.0))
    # classifier = make_pipeline(StandardScaler(), SimpleImputer(strategy='mean'), RandomForestClassifier(n_estimators=100, random_state=42))
    # classifier = make_pipeline(StandardScaler(), SimpleImputer(strategy='mean'), MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42))
    # classifier = make_pipeline(StandardScaler(), SimpleImputer(strategy='mean'), KNeighborsClassifier(n_neighbors=3))
    classifier= OneVsRestClassifier(svm.SVC())

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy on the test set: {accuracy * 100:.2f}%") ##########################################################
    print("\nPrecision: ", precision_score(y_test, y_pred, average="weighted") * 100, "%")
    print("\nRecall: ", recall_score(y_test, y_pred, average="weighted") * 100, "%")
    print("\nF1 Score: ", f1_score(y_test, y_pred, average="weighted") * 100, "%")
    

#####################################################################################################################################