##### Importing relevant libraries ##############################
import cv2
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import os
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

##########################################################


# Main function #######################################
if __name__=="__main__":

    dataset_path=  "assets/textures/"
    images= []
    for filename in os.listdir(dataset_path):
        img= cv2.imread(os.path.join(dataset_path, filename))
        # img= cv2.resize(img, dimensions)
        images.append(img)

    cv2.imshow('original image', images[0])
    # displaying original image for reference


    ############ Grayscaling & converting image to NxM graylevel array #########
    gray_images= []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_images.append(gray)

    cv2.imshow('grayscale image', gray_images[0])
    ###########################################################################

    ######## Filtering grayscale image to further smoothen image ##############
    # filtered_images=[]
    # for gray in gray_images:
    #     filtered= cv2.medianBlur(gray, 5)
    #     filtered_images.append(filtered)

    # cv2.imshow('filtered image', filtered_images[0])
    ###########################################################################


    ## Extracting LBP features from Grayscale image & storing in feature map ##
    lbp_images= []
    for gray in gray_images:
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        lbp = normalize_lbp_image(lbp)
        lbp_images.append(lbp)

    cv2.imshow('lbp image', lbp_images[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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
    #########################################################################


    ####### Placeholder code for labeling, model training, and testing ######
    for i in range(len(thresholded_images)):
        labels.append(f'texture{i+1}')
    labels= np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(thresholded_images, labels, test_size=0.15)
    model = LinearSVC()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('\n', y_pred, "\n")
    ########################################################################    