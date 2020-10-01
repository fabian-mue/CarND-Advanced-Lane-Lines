import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from copy import deepcopy
#%matplotlib qt

# Indicates if output images shall be shown
show_output_images = False

# Indicates if pixel detection has started
pixel_detection_started = False

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
nx = 9
ny = 6
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Camera intrinsic and extrinsic calibration parameters
mtx = 0 # Camera Matrix
dist = 0 # Vector of distortion outputs

# Line finding parameters
nwindows = 9 # Number of sliding windows
margin = 100 # Width of the windows +/- margin
minpix = 50 # Minimum number of pixels found to recenter window

# numbers from description for this project in classroom
y_m_per_pix = 30 / 720  # meters per pixel in y dimension
x_m_per_pix = 3.7 / 700  # meters per pixel in x dimension

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')
#images_test = glob.glob('test_images/*.jpg')
images_test = glob.glob('test_images/test4.jpg')

# Stored line information
last_line_left = None
last_line_right = None

# Indicates if Init-function has been called
initialized = False

# vehicle offset
vehicle_offset = 0

# counts cycles of wrong detections in sequence
wrong_detection_counter = 0

# Ring buffer length
rb_length = 10

'''
Init the stored line information in this module
'''
def init():
    global last_line_left, last_line_right

    last_line_left = Line()
    last_line_right = Line()

# Line class containing important lane information
class Line:
    def __init__(self):
        self.coefficients = np.zeros(3)
        self.coefficients_m = np.zeros(3)
        self.pixels_x = []
        self.pixels_y = []
        self.buffer_x = RingBuffer(rb_length)
        self.buffer_y = RingBuffer(rb_length)
        self.buffer_coeff = RingBuffer(rb_length)
        self.buffer_coeff_m = RingBuffer(rb_length)
        self.curvature = 0
    def init_buffer(self):
        self.buffer_x = RingBuffer(rb_length)
        self.buffer_y = RingBuffer(rb_length)
        self.buffer_coeff = RingBuffer(rb_length)
        self.buffer_coeff_m = RingBuffer(rb_length)

# Circular buffer
class RingBuffer:
    def __init__(self, size):
        self.data = []
        self.size = size

    def append(self, x):
        if len(self.data) == self.size:
            self.data.pop(0)
            self.data.append(x)
        else:
            self.data.append(x)

    def get(self):
        return self.data

''' 
Parts of the code from: Classroom excercise solutions Lesson 6 concept 11, Calibrating your camera

Use chess board images to calibrate camera and get camera matrix as well as distortion parameters
'''
def calibrate_camera(show_corners = False):
    global objpoints, imgpoints, objp, nx, ny, mtx, dist
    # Step through the list and search for chessboard corners
    for fname in images:
        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            # Draw and display the corners
            if show_corners:
                img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)
                cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

''' 
Parts of the code from: Classroom excercise solutions Lesson 6 concept 18, Undistort and transform perspective

Warp/rewarp image using manually determined source and destination points
'''
def warp_image(img, rewarp = False):
    global hood_offset
    img_size = (img.shape[1], img.shape[0])

    s1 = [223, img_size[1]]
    s2 = [572, 468]
    s3 = [716, 468]
    s4 = [1114, img_size[1]]

    d1 = [330, img_size[1]]
    d2 = [330, 1]
    d3 = [950, 1]
    d4 = [950, img_size[1]]

    src = np.float32([s1, s2, s3, s4])
    dst = np.float32([d1, d2, d3, d4])

    if not rewarp:
        M = cv2.getPerspectiveTransform(src, dst)
    else:
        M = cv2.getPerspectiveTransform(dst, src)
    warped_img = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    vertices_s = np.array([s1, s2, s3, s4], np.int32)
    vertices_d = np.array([d1, d2, d3, d4], np.int32)

    return warped_img, vertices_s, vertices_d

''' 
Undistort image using previously determined camera matrix and distortion parameters
'''
def undistort_image(img_in):
    global mtx, dist
    img_out = cv2.undistort(img_in, mtx, dist, None, mtx)
    return img_out

''' 
Parts of the code from: Classroom excercise solutions Lesson 7 concept 12, Color and Gradient

Apply sobel operator and color thresholds to extract relevant line pixels from image 
'''
def filter_image(img, channel_thr = (0, 0), sobelx_thr = (0, 0), dir_thr = (0, 0),  mag_thr = (0, 0)):
    # Convert to HLS color space and separate the channels
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    s_channel = hls[:, :, 2]
    r_channel = img[:,:, 1]

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=15)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)
    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Sobel y
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=15)  # Take the derivative in y
    abs_sobely = np.absolute(sobely)

    # Threshold x gradient
    sobelx_bin = np.zeros_like(scaled_sobelx)
    sobelx_bin[(scaled_sobelx >= sobelx_thr[0]) & (scaled_sobelx <= sobelx_thr[1])] = 1

    # Threshold gradient direction
    abs_sobel_dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    sobel_dir_bin =  np.zeros_like(abs_sobel_dir)
    sobel_dir_bin[(abs_sobel_dir >= dir_thr[0]) & (abs_sobel_dir <= dir_thr[1])] = 1

    # Sobel magnitude
    abs_sobelxy= np.sqrt(abs_sobelx**2 + abs_sobely**2)
    scaled_sobelxy = np.uint8(255*  abs_sobelxy / np.max(abs_sobelxy))
    sobelxy_bin = np.zeros_like(scaled_sobelxy)
    sobelxy_bin[(scaled_sobelxy >= mag_thr[0]) & (scaled_sobelxy <= mag_thr[1])] = 1

    # Threshold color channel s
    s_channel_bin = np.zeros_like(s_channel)
    s_channel_bin[(s_channel >= channel_thr[0]) & (s_channel <= channel_thr[1])] = 1

    # Threshold color channel r
    r_channel_bin = np.zeros_like(r_channel)
    r_channel_bin[(r_channel >= channel_thr[0]) & (r_channel <= channel_thr[1])] = 1

    # Stack each channel
    color_binary = np.dstack((np.zeros_like(sobelx_bin), sobelx_bin, s_channel_bin)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sobelx_bin)
    combined_binary[((sobelx_bin == 1)  & (sobel_dir_bin == 1)  & (sobelxy_bin == 1)) |
                    ((s_channel_bin == 1) & (r_channel_bin == 1))] = 1

    return combined_binary, color_binary

''' 
Code from: Classroom excercise solutions Lesson 8 concept 4, Finding the Lines: Sliding Window

This algorithm is used to detect pixels belonging to the right and left line. 
First a histogram is computed to check where the right and left line begins.
From this starting point sliding windows are calculated to check for defined region if there are pixels belonging to the lines.
These pixels are stored in lists of the left and the right line. 
Finally, the x and y coordinates for the left line pixels as well as for the right line pixels are output.
'''
def start_line_pixel_detection(binary_warped):
    global nwindows, margin, minpix

    # Calculate histogram to find start points of lines
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    window_height = np.int(binary_warped.shape[0] // nwindows)

    # Identify non zero pixels in the binary warped image
    binary_warped_nonzero = binary_warped.nonzero()
    binary_warped_nonzero_x = np.array(binary_warped_nonzero[1])
    binary_warped_nonzero_y = np.array(binary_warped_nonzero[0])

    midpoint = np.int(histogram.shape[0] // 2)
    left_center = np.argmax(histogram[:midpoint])
    right_center = np.argmax(histogram[midpoint:]) + midpoint

    # Lists of left and right line pixel indices
    left_line_candidates = []
    right_line_candidates = []

    # Step through the windows
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = left_center - margin
        win_xleft_high = left_center + margin
        win_xright_low = right_center - margin
        win_xright_high = right_center + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        pixels_in_left_win = ((binary_warped_nonzero_y >= win_y_low) & (binary_warped_nonzero_y < win_y_high) &
                          (binary_warped_nonzero_x >= win_xleft_low) & (binary_warped_nonzero_x < win_xleft_high)).nonzero()[0]
        pixels_in_right_win = ((binary_warped_nonzero_y >= win_y_low) & (binary_warped_nonzero_y < win_y_high) &
                           (binary_warped_nonzero_x >= win_xright_low) & (binary_warped_nonzero_x < win_xright_high)).nonzero()[0]

        # Check and shift center of left and right windows
        if len(pixels_in_left_win) > minpix:
            left_center = np.int(np.mean(binary_warped_nonzero_x[pixels_in_left_win]))
        if len(pixels_in_right_win) > minpix:
            right_center = np.int(np.mean(binary_warped_nonzero_x[pixels_in_right_win]))

        # Add new pixels to the respective lines
        left_line_candidates.append(pixels_in_left_win)
        right_line_candidates.append(pixels_in_right_win)

    # Concatenate the arrays of indices for each line type
    left_line_candidates = np.concatenate(left_line_candidates)
    right_line_candidates = np.concatenate(right_line_candidates)

    # Extract left and right line pixel positions
    left_line_pixels_x = binary_warped_nonzero_x[left_line_candidates]
    left_line_pixels_y = binary_warped_nonzero_y[left_line_candidates]
    right_line_pixels_x = binary_warped_nonzero_x[right_line_candidates]
    right_line_pixels_y = binary_warped_nonzero_y[right_line_candidates]

    return left_line_pixels_x, left_line_pixels_y, right_line_pixels_x, right_line_pixels_y, out_img

''' 
Code from: Classroom excercise solutions Lesson 8 concept 4, Finding the Lines: Sliding Window

This function is used to compute and plot a polynomial along the right and left line by using the pixels belonging to the 
respective line. Furthermore, it updates the stored lane information accordingly, smooths the current line information and performs a sanity check.
'''
def approximate_lines(binary_warped, leftx, lefty, rightx, righty):
    global last_line_left, last_line_right, y_m_per_pix, x_m_per_pix, wrong_detection_counter, pixel_detection_started

    img_out = np.copy(binary_warped)
    last_line_left_copy = deepcopy(last_line_left)
    last_line_right_copy = deepcopy(last_line_right)

    # Find variables to get 2nd order polynomials
    a = np.polyfit(lefty, leftx, 2)
    b = np.polyfit(righty, rightx, 2)
    a_m = np.polyfit(lefty * y_m_per_pix, leftx * x_m_per_pix, 2)
    b_m = np.polyfit(righty * y_m_per_pix, rightx * x_m_per_pix, 2)

    # Append the new coefficients to copies and calculate mean value if possible
    last_line_left_copy.buffer_coeff.append(a)
    if len(last_line_left_copy.buffer_coeff.data) > 1:
        last_line_left_copy.coefficients = np.mean(last_line_left.buffer_coeff.data, 0)
    else:
        last_line_left_copy.coefficients = a
    last_line_right_copy.buffer_coeff.append(b)
    if len(last_line_right_copy.buffer_coeff.data) > 1:
        last_line_right_copy.coefficients = np.mean(last_line_right.buffer_coeff.data, 0)
    else:
        last_line_right_copy.coefficients = b

    # Generate y values for plotting
    y = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    # Generate x values for plotting
    try:
        x_left = last_line_left_copy.coefficients[0] * y ** 2 + last_line_left_copy.coefficients[1] * y + last_line_left_copy.coefficients[2]
        x_right = last_line_right_copy.coefficients[0] * y ** 2 + last_line_right_copy.coefficients[1] * y + last_line_right_copy.coefficients[2]
    except TypeError:
        print('The function failed to fit a line!')
        x_left = y ** 2 + y
        x_right = y ** 2 + y

    # Check if the detected lane is ok
    lane_ok = sanity_check(x_left, x_right)

    if lane_ok:
        last_line_left.buffer_coeff.append(a)
        last_line_right.buffer_coeff.append(b)
        last_line_left.buffer_coeff_m.append(a_m)
        last_line_right.buffer_coeff_m.append(b_m)

        last_line_left.coefficients = np.mean(last_line_left.buffer_coeff.data, 0)
        last_line_right.coefficients = np.mean(last_line_right.buffer_coeff.data, 0)
        last_line_left.coefficients_m = np.mean(last_line_left.buffer_coeff_m.data, 0)
        last_line_right.coefficients_m = np.mean(last_line_right.buffer_coeff_m.data, 0)

        last_line_left.pixels_x = x_left
        last_line_right.pixels_x = x_right
    else:
        # Generate y values for plotting
        y = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        # Generate x values for plotting
        x_left = last_line_left.pixels_x
        x_right = last_line_right.pixels_x

        wrong_detection_counter = wrong_detection_counter + 1
        pixel_detection_started = False if wrong_detection_counter > 1 else pixel_detection_started

    # Colour pixels belonging to the left and right lines
    img_out[lefty, leftx] = [255, 0, 0]
    img_out[righty, rightx] = [0, 0, 255]

    return img_out, x_left, x_right, y

''' 
Checks if the detected lane is ok by checking if the right and left lines are parallel
'''
def sanity_check(x_l, x_r):
    global last_line_right, last_line_left
    last = len(x_l) - 1

    dist_top = abs(x_l[0] - x_r[0])
    dist_center = abs(x_l[int(last / 2)] - x_r[int(last / 2)])
    dist_bottom = abs(x_l[last] - x_r[last])
    tolerance = 100

    if abs(dist_top - dist_center) > tolerance or abs(dist_top - dist_bottom) > tolerance or abs(dist_bottom - dist_center) > tolerance:
        return False
    else:
        return True

''' 
Code from: Classroom excercise solutions Lesson 8 concept 5, Finding the Lines: Search from Prior

This function is used to continue the line pixel detection using information from previously detected lanes.
'''
def continue_line_pixel_detection(binary_warped):
    global margin, last_line_left, last_line_right
    # Identify non zero pixels in the binary warped image
    binary_warped_nonzero = binary_warped.nonzero()
    pixels_x = np.array(binary_warped_nonzero[1])
    pixels_y = np.array(binary_warped_nonzero[0])

    # Use already calculated line polynomials of the last frame to detect new line pixels
    pixels_left_line = ((pixels_x > (last_line_left.coefficients[0] * (pixels_y ** 2) +
                                      last_line_left.coefficients[1] * pixels_y + last_line_left.coefficients[2] - margin)) &
                        (pixels_x < (last_line_left.coefficients[0] * (pixels_y ** 2) +
                                      last_line_left.coefficients[1] * pixels_y + last_line_left.coefficients[2] + margin)))
    pixels_right_line = ((pixels_x > (last_line_right.coefficients[0] * (pixels_y ** 2) +
                                      last_line_right.coefficients[1] * pixels_y + last_line_right.coefficients[2] - margin)) &
                        (pixels_x < (last_line_right.coefficients[0] * (pixels_y ** 2) +
                                      last_line_right.coefficients[1] * pixels_y + last_line_right.coefficients[2] + margin)))

    # Get x and y position of lane pixels
    left_line_pixels_x = pixels_x[pixels_left_line]
    left_line_pixels_y = pixels_y[pixels_left_line]
    right_line_pixels_x = pixels_x[pixels_right_line]
    right_line_pixels_y = pixels_y[pixels_right_line]

    return left_line_pixels_x, left_line_pixels_y, right_line_pixels_x, right_line_pixels_y

'''
Parts of the code from: Classroom excercise solutions Lesson 8 concept 6-7, measure curvatures

This function is used to measure the current curvatures of the left and right line as close to the vehicle as possible. 
Moreover, the vehicle offset is computed.
'''
def measure_lane_information(y_bottom, image_center_x):
    global x_m_per_pix, y_m_per_pix, last_line_left, last_line_right

    # Calculation curvatures of left and right line in meters
    left_curverad = ((1 + (2 * last_line_left.coefficients_m[0] * y_bottom * y_m_per_pix +
                           last_line_left.coefficients_m[1]) ** 2) ** 1.5) / abs(2 * last_line_left.coefficients_m[0])
    right_curverad = ((1 + (2 * last_line_right.coefficients_m[0] * y_bottom * y_m_per_pix +
                            last_line_right.coefficients_m[1]) ** 2) ** 1.5) / abs(2 * last_line_right.coefficients_m[0])

    # Calculate the vehicle offset
    x_left_bottom = last_line_left.coefficients[0] * y_bottom ** 2 + last_line_left.coefficients[1] * y_bottom + last_line_left.coefficients[2]
    x_right_bottom = last_line_right.coefficients[0] * y_bottom ** 2 + last_line_right.coefficients[1] * y_bottom + last_line_right.coefficients[2]
    lane_center_x = x_left_bottom +  (x_right_bottom - x_left_bottom) / 2
    offset = abs(image_center_x - lane_center_x) * x_m_per_pix
    return left_curverad, right_curverad, offset

''' 
Code from: Tips and Tricks for the project

This function is used to draw the identified lane.
'''
def identify_lane(warped_img, undist, x_left, x_right, y):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([x_left, y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([x_right, y])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Rewarp the blank back to original image space
    rewarped_colour, vertices_s, vertices_d = warp_image(color_warp, rewarp = True)
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, rewarped_colour, 0.3, 0)
    return result

''' 
Main function to process input images and optionally display the results
'''
def img_processing(image):
    global pixel_detection_started, vehicle_offset, last_line_left, last_line_right, wrong_detection_counter, show_output_images

    undist = undistort_image(image)
    binary, colored_binary = filter_image(undist, channel_thr = (120, 255), sobelx_thr = (70, 220),
                                          dir_thr = (np.pi/8, np.pi/2), mag_thr = (10, 240))
    warped_NoBinary, vertices_s, vertices_d = warp_image(undist, rewarp=False)
    warped_binary, vertices_s, vertices_d = warp_image(binary, rewarp=False)
    if not pixel_detection_started or show_output_images:
        last_line_left.init_buffer()
        last_line_right.init_buffer()

        left_x, left_y, right_x, right_y, binary_warped_drawn_windows = start_line_pixel_detection(warped_binary)
        binary_warped_drawn, x_left, x_right, y = approximate_lines(binary_warped_drawn_windows, left_x, left_y,
                                                                    right_x, right_y)
        pixel_detection_started = True
        wrong_detection_counter = 0
    else:
        left_x, left_y, right_x, right_y = continue_line_pixel_detection(warped_binary)
        warped_binary_stacked = np.dstack((warped_binary, warped_binary, warped_binary))
        binary_warped_drawn, x_left, x_right, y = approximate_lines(warped_binary_stacked, left_x, left_y, right_x, right_y)

    last_line_left.curvature, last_line_right.curvature, vehicle_offset = measure_lane_information(binary_warped_drawn.shape[0],
                                                                                                   binary_warped_drawn.shape[1] / 2)

    final_img = identify_lane(warped_binary, undist, x_left, x_right, y)

    position1 = (20, 60)
    position2 = (20, 120)

    cv2.putText(final_img, "Lane curvature: " + str((last_line_left.curvature + last_line_right.curvature) / 2) + " m", position1,
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0, 0), 3)
    cv2.putText(final_img, "Vehicle offset: " + str(vehicle_offset) + " m", position2,
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0, 0), 3)


    if show_output_images:
        '''
        f = plt.figure(figsize=(30, 15))
        f.add_subplot(1, 4, 1)
        vertices_s = vertices_s.reshape((-1, 1, 2))
        cv2.polylines(undist, [vertices_s], True, (255, 0, 0), thickness=3)
        plt.imshow(undist, 'gray')
        f.add_subplot(1, 4, 2)
        plt.imshow(binary, 'gray')
        f.add_subplot(1, 4, 3)
        plt.imshow(warped_binary, 'gray')
        f.add_subplot(1, 4, 4)
        plt.imshow(final_img)
        plt.show()
        '''
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))
        vertices_s = vertices_s.reshape((-1, 1, 2))
        cv2.polylines(undist, [vertices_s], True, (255, 0, 0), thickness=3)
        ax1.imshow(undist)
        ax1.set_title('Original image', fontsize=20)
        vertices_d = vertices_d.reshape((-1, 1, 2))
        cv2.polylines(warped_NoBinary, [vertices_d], True, (255, 0, 0), thickness=3)
        ax2.imshow(warped_NoBinary)
        ax2.set_title('Warped colored image', fontsize=20)
        #plt.plot(x_left, y, color='yellow')
        #plt.plot(x_right, y, color='yellow')
        plt.waitforbuttonpress()

    return final_img

if __name__ == "__main__":
    work_with_images = False

    if not initialized:
        init()
        calibrate_camera(show_corners = False)
        initialized = True

    if work_with_images:
        for name in images_test:
            img = mpimg.imread(name)
            img_with_laneInfo = img_processing(img)
    else:
        video_output = 'project_video_out.mp4'

        clip1 = VideoFileClip("project_video.mp4")
        #new_clip = clip1.subclip(0, 2)
        project_clip = clip1.fl_image(img_processing)

        project_clip.write_videofile(video_output, audio=False)

