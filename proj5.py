import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import argparse
from moviepy.editor import VideoFileClip
from skimage.feature import hog
import time
from sklearn import svm, grid_search
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from scipy.ndimage.measurements import label
from sklearn.externals import joblib
import math
from collections import deque

def show_images_side_by_side(img1,title1,img2,title2,is_gray=False):
    fig = plt.figure(figsize=(6, 4))
    subfig1 = fig.add_subplot(1, 2, 1)
    subfig1.imshow(img1)
    subfig1.set_title(title1, fontsize=20)
    subfig2 = fig.add_subplot(1, 2, 2)
    if is_gray:
        subfig2.imshow(img2)
    else:
        subfig2.imshow(img2, cmap='gray')
    subfig2.set_title(title2, fontsize=20)
    fig.tight_layout()
    plt.show()
    return subfig1, subfig2

def show_array_2images_side_by_side(imgs1,titles1,imgs2,titles2,cmap=None):
    assert(len(imgs1)==len(imgs2))
    assert(len(imgs1)==len(titles1))
    assert(len(titles1)==len(titles2))
    n=len(imgs1)
    h_figure=2*n
    fig = plt.figure(figsize=(6, h_figure))
    i=0
    num_cols=2
    for img in imgs1:
        subfig1 = fig.add_subplot(n, num_cols, num_cols*i+1)
        subfig1.imshow(img)
        subfig1.set_title(titles1[i], fontsize=5)

        subfig2 = fig.add_subplot(n, num_cols, num_cols*i+2)
        if cmap==None:
            subfig2.imshow(imgs2[i])
        else:
            subfig2.imshow(imgs2[i],cmap)
        subfig2.set_title(titles2[i], fontsize=5)
        i+=1
    fig.tight_layout()
    plt.show()
    return fig

def visualize_hog(orient, pix_per_cell, cell_per_block):
    car_img_filenames = glob.glob('./vehicles_smallset/**/*.jpeg', recursive=True)
    cars = []
    for car_img_filename in car_img_filenames:
        cars.append(car_img_filename)
    n = 3
    h_figure = 2 * n
    fig = plt.figure(figsize=(6, h_figure))
    i = 0
    num_cols = 4
    for i in range(0,n):
        indx = np.random.randint(0, (len(cars) - 1))
        hog_features_for_one_img = []
        car_image = mpimg.imread(cars[indx])
        for channel in range(car_image.shape[2]):
            hog_feature_throwaway, hog_img = get_hog_features(car_image[:, :, channel],
                                                              orient, pix_per_cell, cell_per_block,
                                                              vis=True, feature_vec=True)
            hog_features_for_one_img.append(hog_img)
        hog_image = np.dstack((hog_features_for_one_img[0], hog_features_for_one_img[1],
                               hog_features_for_one_img[2]))

        subfig1 = fig.add_subplot(n, num_cols, num_cols * i + 1)
        subfig1.imshow(car_image)
        subfig1.set_title("Original Image", fontsize=5)

        for j in range(2,5):
            subfig = fig.add_subplot(n, num_cols, num_cols * i + j)
            subfig.imshow(hog_features_for_one_img[j-2], cmap='gray')
            subfig.set_title("Channel {}".format(j-2), fontsize=5)

        fig.suptitle('HOG Visualization')
        i += 1
    fig.tight_layout()
    plt.show()
    return fig

def read_images_from_directory(directory_name):
    images = glob.glob(directory_name)
    imgs = []
    titles1 = []
    for fname in images:
        img = mpimg.imread(fname)
# if the image has an alpha channel, get rid of it
        if img.shape[2] == 4:
            img=img[:,:,:3]

        imgs.append(img)
        titles1.append((fname))
    return imgs, titles1

def get_arguments():
    ap = argparse.ArgumentParser()

    ap.add_argument("-d", "--diagnostics", required=False, default='False', type=str,
                    help="Show additional diagnostics on video.")
    ap.add_argument("-v", "--visualization", required=False, default='False', type=str,
                    help="Visualize data only and do not create videos.")
    ap.add_argument("-s", "--smallset", required=False, default='False', type=str,
                    help="Use the smaller set of images for training")
    ap.add_argument("-f", "--videofile", required=False, default='project_video.mp4', type=str,
                    help="The video file to use")
    ap.add_argument("-c", "--usesubclip", required=False, default='False', type=str,
                    help="Use subclip of video with specified range")
    ap.add_argument("-r", "--subcliprange", required=False, default='5,7', type=str,
                    help="Range for subclip")
    ap.add_argument("-t", "--forcetrain", required=False, default='False', type=str,
                    help="Train even if saved pickle files are present")

    args = vars(ap.parse_args())
    return args

def get_boolean_arg(args,arg_name):
    if args[arg_name] == "True" or args[arg_name] == "true":
        boolean_arg=True
    elif args[arg_name] == "False" or args[arg_name] == "false":
        boolean_arg=False
    return boolean_arg

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features
# Define a function that takes an image, a color space, and a new image size
# and returns a feature vector. This function was taken from class materials
def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel()
    # Return the feature vector
    return features

def color_hist(img, nbins=32, color_space='RGB',bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    rhist = np.histogram(feature_image[:,:,0], bins=nbins, range=bins_range)
    ghist = np.histogram(feature_image[:,:,1], bins=nbins, range=bins_range)
    bhist = np.histogram(feature_image[:,:,2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, color_space=spatial_bin_colorspace,size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image,color_space=color_hist_colorspace, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                     orient, pix_per_cell, cell_per_block,
                                     vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def find_cars(img, ystart, ystop, scale, svc, X_scaler,color_space,orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,visualize_windows=False):
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255
    rectangles = []
    img_features = []
    img_tosearch = img[ystart:ystop, :, :]
#    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if color_space != 'RGB':
        if color_space == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else:
        ctrans_tosearch = np.copy(img)

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    cnt=0
    for xb in range(nxsteps+1):
        for yb in range(nysteps+1):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            if spatial_feat:
                spatial_features = bin_spatial(subimg,color_space=spatial_bin_colorspace,size=spatial_size)
                img_features.append(spatial_features)

            if hist_feat:
                hist_features = color_hist(subimg, color_space=color_hist_colorspace,nbins=hist_bins)
                img_features.append(hist_features)

            img_features.append(hog_features)
            combined_features = np.concatenate(img_features)

#            if hist_feat or spatial_feat:
#                combined_features=np.concatenate(img_features)
#            else:
#                combined_features=hog_features

#            test_features = X_scaler.transform(
#                np.hstack((spatial_features, hist_features,hog_features)).reshape(1, -1))
            # Scale features and make a prediction
#            test_features = X_scaler.transform(
#                np.hstack((spatial_features, hog_features)).reshape(1, -1))
#            test_features = X_scaler.transform(combined_features)
            test_features=hog_features
            test_prediction = svc.predict(test_features)

            cnt+=1
            if test_prediction == 1 or (visualize_windows and cnt==10):

                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                rectangles.append(((xbox_left, ytop_draw + ystart),(xbox_left + win_draw, ytop_draw + win_draw + ystart)))
    return rectangles

def show_sample_image_details(img):
    image = mpimg.imread(img)
    print("Image shape={}".format(image.shape))

def create_video(input_filename,output_filename):
    clip = VideoFileClip(input_filename)
    if use_subclip:
        clip_to_process=clip.subclip(clip_range[0],clip_range[1])
    else:
        clip_to_process = clip
    processed_clip = clip_to_process.fl_image(process_image) #NOTE: this function expects color images!!
    processed_clip.write_videofile(output_filename, audio=False)

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

class Vehicle_Detection_Buffer():
    def __init__(self):
        # buffer of rectangles for the previous num_rectangles frames
        self.num_rectangles = 15
        self.rectangles_buffer = deque(maxlen=self.num_rectangles)

    def add_rectangles(self, rectangles):
        self.rectangles_buffer.append(rectangles)

def process_image(image,do_visualization=False):
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    draw_image = np.copy(image)

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    image = image.astype(np.float32) / 255
    rectangles = []

#    ystart = 360
#    ystop = 616
#    scale = 1.0
#    hot_windows = find_cars(image, ystart, ystop, scale, svc, X_scaler, color_space, orient, pix_per_cell,
#                            cell_per_block, spatial_size, hist_bins, visualize_windows=False)
#    rectangles.append(hot_windows)

#    window_img1 = draw_boxes(draw_image, hot_windows, color=(0, 255, 0), thick=6)

    ystart = 400
    ystop = 592
    scale = 1.5

    hot_windows = find_cars(image, ystart, ystop, scale, svc, X_scaler, color_space, orient, pix_per_cell,
                            cell_per_block, spatial_size, hist_bins, visualize_windows=False)
    rectangles.append(hot_windows)

#    window_img3 = draw_boxes(draw_image, hot_windows, color=(0, 255, 0), thick=6)

    ystart = 400
    ystop = 656
    scale = 2.0

    hot_windows = find_cars(image, ystart, ystop, scale, svc, X_scaler, color_space, orient, pix_per_cell,
                            cell_per_block, spatial_size, hist_bins, visualize_windows=False)
    rectangles.append(hot_windows)

#    window_img4 = draw_boxes(draw_image, hot_windows, color=(0, 255, 0), thick=6)

    ystart = 500
    ystop = 660
    scale = 2.5

    hot_windows = find_cars(image, ystart, ystop, scale, svc, X_scaler, color_space, orient, pix_per_cell,
                            cell_per_block, spatial_size, hist_bins, visualize_windows=False)
    rectangles.append(hot_windows)

#    window_img5 = draw_boxes(draw_image, hot_windows, color=(0, 255, 0), thick=6)

    rectangles = [item for sublist in rectangles for item in sublist]

    if do_visualization:
        threshold = 1
        heat = add_heat(heat, rectangles)
        heat = apply_threshold(heat, threshold)
    else:
        if len(rectangles) > 0:
            detection_buffer.add_rectangles(rectangles)
        for rectangle_set in detection_buffer.rectangles_buffer:
            heat = add_heat(heat, rectangle_set)
        threshold = 1+math.ceil(len(detection_buffer.rectangles_buffer)/2.0)
        heat = apply_threshold(heat, threshold)

    # Find final boxes from heatmap using label function
    labels = label(heat)
    return_img = draw_labeled_bboxes(np.copy(draw_image), labels)

    if show_diagnostics:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = 'Thresholds: ' + '{}'.format(threshold)
        cv2.putText(return_img, text, (200, 170), font, 1.5, (200, 255, 155), 2, cv2.LINE_AA)

    if do_visualization:
        return return_img, rectangles, heat
    else:
        return return_img

# Begin Main
detection_buffer = Vehicle_Detection_Buffer()
args=get_arguments()
show_diagnostics=get_boolean_arg(args,"diagnostics")
do_visualization=get_boolean_arg(args,"visualization")
use_small_dataset=get_boolean_arg(args,"smallset")
use_subclip=get_boolean_arg(args,"usesubclip")
subclip_range_str=args["subcliprange"]
video_filename=args["videofile"]
force_train=get_boolean_arg(args,"forcetrain")

clip_range=np.empty(2, dtype=int)
parsed_clip_param=subclip_range_str.split(',')
clip_range[0] = np.int(parsed_clip_param[0])
clip_range[1] = np.int(parsed_clip_param[1])

color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16   # Number of histogram bins
spatial_feat = False # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [360, 660] # Min and max in y to search in slide_window()
spatial_bin_colorspace='HSV'
color_hist_colorspace='HSV'

classifier_filename = './car_classifier.joblib.pkl'
scaler_filename = './car_scaler.joblib.pkl'
if os.path.isfile(classifier_filename) and not force_train:
    svc = joblib.load(classifier_filename)
    print("Classifier loaded from file")
    if os.path.isfile(scaler_filename):
        X_scaler = joblib.load(scaler_filename)
    else:
        X_scaler = None
else:
    if use_small_dataset:
        car_img_filenames = glob.glob('./vehicles_smallset/**/*.jpeg', recursive=True)
        non_car_img_filenames = glob.glob('./non-vehicles_smallset/**/*.jpeg', recursive=True)
    else:
        car_img_filenames = glob.glob('./vehicles/**/*.png', recursive=True)
        non_car_img_filenames = glob.glob('./non-vehicles/**/*.png', recursive=True)

    cars = []
    notcars = []
    for car_img_filename in car_img_filenames:
        cars.append(car_img_filename)

    for non_car_img_filename in non_car_img_filenames:
        notcars.append(non_car_img_filename)

    car_features = extract_features(cars, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler

    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    joblib.dump(svc, classifier_filename, compress=9)
    joblib.dump(X_scaler, scaler_filename, compress=9)

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()

if do_visualization:
    test_imgs,test_img_titles=read_images_from_directory('./test_images/test*.jpg')
    window_imgs=[]
    heatmap_imgs=[]
    for image in test_imgs:
        processed_img,rectangles,heatmap=process_image(image,True)
        window_imgs.append(processed_img)
        heatmap_imgs.append(heatmap)

    fig=show_array_2images_side_by_side(test_imgs, test_img_titles, window_imgs, test_img_titles, cmap=None)
    fig.savefig("./visualization/boxes_test_images.jpg")
    fig = show_array_2images_side_by_side(test_imgs, test_img_titles, heatmap_imgs, test_img_titles, cmap='hot')
    fig.savefig("./visualization/heat_test_images.jpg")
    plt.show()

    fig = visualize_hog(orient, pix_per_cell, cell_per_block)
    fig.savefig("./visualization/hog_images.jpg")
else:
    create_video(video_filename, 'vehicle_detection.mp4')

