import cv2
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import os

dimensions = (64, 64)
radius= 1                                                   
n_points= 8 * radius
region_size= 20
threshold_factor = 0.3
labels= []
scaling_factor = 2.0


def divide_into_regions(_lbp, _region_size):
    height, width = _lbp.shape
    regions = []
    for i in range(0, height, _region_size):
        for j in range(0, width, _region_size):
            region= _lbp[i:i+_region_size, j:j+_region_size]
            regions.append(region)
    return regions


def normalize_histogram(_histogram):
    """
    Normalize the histogram to have unit L1 norm.

    @param:
        _histogram - the histogram of LBP values for a given region of an image.

    @returns:
        _histogram - the normalized histogram.
    """
    return _histogram / np.sum(_histogram)


def threshold_histogram(_histogram, _threshold_factor):
    _histogram = normalize_histogram(_histogram)
    threshold = _threshold_factor * np.mean(_histogram)
    _histogram[_histogram < threshold] = 0
    _histogram[_histogram >= threshold] = 1
    return _histogram


if __name__=="__main__":

    dataset_path=  "assets/textures/"
    images= []
    for filename in os.listdir(dataset_path):
        img= cv2.imread(os.path.join(dataset_path, filename))
        # img= cv2.resize(img, dimensions)
        images.append(img)

    cv2.imshow('original image', images[0])


    ##############################################################

    gray_images= []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_images.append(gray)

    cv2.imshow('grayscale image', gray_images[0])


    ##############################################################

    filtered_images=[]
    for gray in gray_images:
        filtered= cv2.medianBlur(gray, 5)
        filtered_images.append(filtered)

    cv2.imshow('filtered image', filtered_images[0])


    ##############################################################

    lbp_images= []
    for filtered in filtered_images:
        lbp = local_binary_pattern(filtered, n_points, radius, method='uniform')
        lbp_images.append(lbp)

    cv2.imshow('lbp image', lbp_images[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ##############################################################

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

    cv2.imshow('region', region[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    thresholded_images= np.array(thresholded_images)

    

    ###############################################################
  
    for i in range(len(thresholded_images)):
        labels.append(f'texture{i+1}')
    labels= np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(thresholded_images, labels, test_size=0.15)
    model = LinearSVC()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('\n', y_pred, "\n")


