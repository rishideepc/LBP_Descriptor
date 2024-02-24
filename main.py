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
# ADD NOISE TO TESTING PURPOSES
# FIVE FOLD CROSS VALIDATION
# STORE ACCURACY IN ARRAY (lOOP)

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


# incr

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

def extract_features(image_path):
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

    # return thresholded_image.ravel()
    # return normalized_lbp.ravel()
    return lbp.ravel()
    # return np.ravel(img)
    # return lbp_histogram

if __name__=="__main__":

    texture_directory = 'C:/Users/HP/Desktop/Python_AI/lbp-descriptor-textureRecog/assets/OASIS_MRI_DB'

    features = []
    labels = []


    for category in range(1, 5):
        category_path = os.path.join(texture_directory, f'OASIS_Cross_{category}_converted')
        for filename in os.listdir(category_path):
            if filename.endswith('.jpg'):  
                image_path = os.path.join(category_path, filename)
                feature = extract_features(image_path)
                features.append(feature)
                labels.append(category)

    # features=np.array(features)
    # print("Features list: ", features)
    # labels=np.array(labels)
    # print("Labels list: ", labels)
    # features=features.flatten()
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # vectorization = TfidfVectorizer()
    # xv_train= vectorization.fit_transform(X_train)
    # xv_test= vectorization.fit_transform(X_test)

    # classifier = make_pipeline(StandardScaler(), SimpleImputer(strategy='mean'), SVC(kernel='linear', C=1.0))
    # classifier = make_pipeline(StandardScaler(), SimpleImputer(strategy='mean'), RandomForestClassifier(n_estimators=100, random_state=42))
    # classifier = make_pipeline(StandardScaler(), SimpleImputer(strategy='mean'), MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42))
    # classifier = make_pipeline(StandardScaler(), SimpleImputer(strategy='mean'), KNeighborsClassifier(n_neighbors=3))
    classifier= OneVsRestClassifier(svm.SVC())


    # param_grid = {'svc__C': [0.001, 0.01, 0.1, 1, 10, 100]}  #parameter grid for grid search

    # grid_search = GridSearchCV(classifier, param_grid, cv=5, n_jobs=-1)  #grid search with cross validation
    # grid_search.fit(X_train, y_train)

    # print("Best parameters:", grid_search.best_params_)

    # best_classifer = grid_search.best_estimator_

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    print("Predictions for Test set: ", y_pred)
    # y_pred= best_classifer.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

    X5_path = os.path.join(texture_directory, 'OASIS_Cross_gallery_converted')
    X5_features = []
    X5_labels = []

    for filename in os.listdir(X5_path):
        if filename.endswith('.jpg'):
            image_path = os.path.join(X5_path, filename)
            feature = extract_features(image_path)
            X5_features.append(feature)

    sequences = [
    (1, 137),
    (2, 118),
    (3, 91),
    (4, 70)
    ]

    for sequence_number, sequence_length in sequences:
        X5_labels.extend([sequence_number] * sequence_length)

    # print("True values for X5 textures:", X5_labels)
    X5_predictions = classifier.predict(X5_features)
    # print("Predictions for X5 textures:", X5_predictions)

    # accuracy = accuracy_score(X5_labels, X5_predictions)
    # print(f"Accuracy on the X5 set: {accuracy * 100:.2f}%")
    

#####################################################################################


#  Main function #######################################
# if __name__=="__main__":

#     dataset_path=  "assets/textures/"
#     images= []
#     for filename in os.listdir(dataset_path):
#         img= cv2.imread(os.path.join(dataset_path, filename))
#         # img= cv2.resize(img, dimensions)
#         images.append(img)

#     cv2.imshow('original image', images[0])
#     # displaying original image for reference


#     ############ Grayscaling & converting image to NxM graylevel array #########
#     gray_images= []
#     for img in images:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         gray_images.append(gray)

#     cv2.imshow('grayscale image', gray_images[0])
#     ###########################################################################

#     ######## Filtering grayscale image to further smoothen image ##############
#     filtered_images=[]
#     for gray in gray_images:
#         filtered= cv2.medianBlur(gray, 5)
#         filtered_images.append(filtered)

#     cv2.imshow('filtered image', filtered_images[0])
#     ###########################################################################


#     ## Extracting LBP features from Grayscale image & storing in feature map ##
#     lbp_images= []
#     for gray in gray_images:
#         lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
#         lbp = normalize_lbp_image(lbp)
#         lbp_images.append(lbp)

#     lbp_images_th= []
#     for filtered in filtered_images:
#         lbp_th = local_binary_pattern(filtered, n_points, radius, method='uniform')
#         lbp_th = normalize_lbp_image(lbp_th)
#         lbp_images_th.append(lbp_th)

#     cv2.imshow('lbp image', lbp_images[0])
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     ##########################################################################

#     ##### Further thresholding based on a threshold factor ###################
#     thresholded_images= []
#     for lbp in lbp_images:
#         regions= divide_into_regions(lbp, region_size)
#         thresholded_regions= []
#         for region in regions:
#             histogram, _= np.histogram(region, bins=np.arange(0, 10), density=True)
#             histogram = threshold_histogram(histogram, threshold_factor)
#             thresholded_regions.append(histogram)
#         thresholded_image= np.concatenate(thresholded_regions)
#         thresholded_images.append(thresholded_image)
#     thresholded_images= np.array(thresholded_images)

#     cv2.imshow('thresholded lbp image', lbp_images_th[0])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     ########################################################################




#     Labelling & Training & Testing
#     data= pd.read_excel('C:/Users/HP/Desktop/Python_AI/lbp-descriptor-textureRecog/Labels.xlsx')

#     y= label_binarize(data['Label'], classes=["irregular", "regular"])
#     X_train, X_test, y_train, y_test= train_test_split(data['File'], y, test_size=0.15, random_state=35)

#     vectorization = TfidfVectorizer()
#     xv_train = vectorization.fit_transform(X_train)
#     xv_test = vectorization.transform(X_test)

#     model_gini = OneVsRestClassifier(svm.SVC())
#     model_gini.fit(xv_train, y_train)
#     y_pred = model_gini.predict(xv_test)

#     print("\nAccuracy Score: ", accuracy_score(y_test, y_pred) * 100, "%")
    #######################################################################    