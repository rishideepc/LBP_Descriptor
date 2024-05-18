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
threshold_factor = 0.3
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

########################## Main function FOR BRODATZ DATASET ############################################################
if __name__=="__main__":

    dataset_path=  "assets/textures/"
    images= []
    for filename in os.listdir(dataset_path):
        img= cv2.imread(os.path.join(dataset_path, filename))
        # img= cv2.resize(img, dimensions)
        images.append(img)

    # cv2.imshow('original image', images[0])
    # displaying original image for reference


    ############ Grayscaling & converting image to NxM graylevel array #########
    gray_images= []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_images.append(gray)

    # cv2.imshow('grayscale image', gray_images[0])
    ###########################################################################

    ######## Filtering grayscale image to further smoothen image ##############
    filtered_images=[]
    for gray in gray_images:
        filtered= cv2.medianBlur(gray, 5)
        filtered_images.append(filtered)

    # cv2.imshow('filtered image', filtered_images[0])
    ###########################################################################


    ## Extracting LBP features from Grayscale image & storing in feature map ##
    lbp_images= []
    for gray in gray_images:
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        lbp = normalize_lbp_image(lbp)
        lbp_images.append(lbp)

    lbp_images_th= []
    for filtered in filtered_images:
        lbp_th = local_binary_pattern(filtered, n_points, radius, method='uniform')
        lbp_th = normalize_lbp_image(lbp_th)
        lbp_images_th.append(lbp_th)

    # cv2.imshow('lbp image', lbp_images[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    ##########################################################################

    ##### Further thresholding based on a threshold factor ###################
    thresholded_images= []
    for lbp in lbp_images:
        regions= divide_into_regions(lbp, region_size)
        thresholded_regions= []
        for region in regions:
            histogram, _= np.histogram(region, bins=np.arange(0, 10), density=True)
            histogram = threshold_histogram(histogram, threshold_factor)
            thresholded_regions.append(histogram)
        thresholded_image= np.concatenate(thresholded_regions)
        thresholded_images.append(thresholded_image)
    thresholded_images= np.array(thresholded_images)

    ########################################################################




    # Labelling & Training & Testing
    data= pd.read_excel('C:/Users/HP/Desktop/Python_AI/lbp-descriptor-textureRecog/Labels.xlsx')

    y= label_binarize(data['Label'], classes=["irregular", "regular"])
    X_train, X_test, y_train, y_test= train_test_split(data['File'], y, test_size=0.15, random_state=35)

    vectorization = TfidfVectorizer()
    xv_train = vectorization.fit_transform(X_train)
    xv_test = vectorization.transform(X_test)

    model_gini = OneVsRestClassifier(svm.SVC())
    model_gini.fit(xv_train, y_train)
    y_pred = model_gini.predict(xv_test)

    print("\nAccuracy Score: ", accuracy_score(y_test, y_pred) * 100, "%")
    print("\nPrecision: ", precision_score(y_test, y_pred, average="weighted") * 100, "%")
    print("\nRecall: ", recall_score(y_test, y_pred, average="weighted") * 100, "%")
    print("\nF1 Score: ", f1_score(y_test, y_pred, average="weighted") * 100, "%")
######################################################################################################################3  