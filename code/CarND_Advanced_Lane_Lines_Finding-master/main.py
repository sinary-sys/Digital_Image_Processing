import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def get_obj_img_points(images):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    for image in images:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)          
    return objpoints, imgpoints

# Camera calibration & Distortion correction
def cal_undistort(img, objpoints, imgpoints):
#     checkboard_images = glob.glob('./camera_cal/calibration*.jpg')
#     objpoints, imgpoints = get_obj_img_points(checkboard_images)
    
    # mtx: Camera matrix
    # dist: Distortion coefficients
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # undistortion on the original image instead of gray-scaled image
    undist = cv2.undistort(img, mtx, dist, None, mtx) 
    return undist

def gradient_thresh(img, s_thresh=(170,255), sx_thresh=(20,100)):
    img = np.copy(img)
    hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #sobelx
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel>=sx_thresh[0])&(scaled_sobel<=sx_thresh[1])]=1
    
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel>=s_thresh[0])&(s_channel<=s_thresh[1])]=1
    
    gradient_binary = np.zeros_like(sxbinary)
    gradient_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return gradient_binary

def select_yellow(image):
    lower = np.array([15,30,150])
    upper = np.array([40,255,255])
    mask = cv2.inRange(image, lower, upper)
    return mask

def select_white(image):
    lower = np.array([0,200,0])
    upper = np.array([255,255,255])
    mask = cv2.inRange(image, lower, upper)
    return mask

def color_combined_thresh(image):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    yellow = select_yellow(hls)
    white = select_white(hls)
    color_binary = np.zeros_like(yellow)
    color_binary[(yellow >= 1) | (white >= 1)] = 1
    return color_binary

def perspective_tansform(img):
    '''
    src = np.float32([[,],[,],[,],[,]]) as four of the detected corners 
    dst = np.float32([[,],[,],[,],[,]]) as destination points after transformation
    '''
    img_size = (img.shape[1], img.shape[0])
#     src = np.float32([[150,img_size[1]],[580,450],[700,450], [1150, img_size[1]]])
#     offset = 200
    
    src = np.float32([[200,img_size[1]],[580,450],[700,450], [1050, img_size[1]]])
    offset = 200
    
    dst = np.float32([[offset, img_size[1]], [offset, 0], [img_size[0]-offset, 0], [img_size[0]-offset, img_size[1]]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    unwarped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)
    return warped, M, Minv

def find_lane_pixels(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2) # 直方图的中点
    leftx_base = np.argmax(histogram[:midpoint]) # 左半部分最大值坐标
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint #右半部分最大值坐标
    
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 10
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()#np.nonzero函数是numpy中用于得到数组array中非零元素的位置（数组索引）的函数
    # nonzeroy.shape = nonzerox.shape = 1280*760
    nonzeroy = np.array(nonzero[0])# 非0元素的y索引，其实就是坐标
    nonzerox = np.array(nonzero[1])# 非0元素的x索引，其实就是坐标
    
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        
        ### TO-DO: Find the four below boundaries of the window ###
        ### 绘制一个矩形，用直方图最大值向两侧偏离margin
        ### Gonna be updated
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,0,255), 4) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,0,255), 4) 
        
         ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        ### 在对于包围框中的元素，取其x坐标
        ### nonzeroy是一个一维向量,长度为1280*760，存放图中所有像素点的y坐标，x坐标同理
        ### 下面面的操作首先进行逻辑运算筛选符合条件的索引，符合条件的为True，否则为False
        ### 然后用nonzero把所有True的点的索引取出来，也就是说在nonzeroy中满足条件的点的索引
        ### 用该索引取nonzeroy的值，得到的就是该点的y坐标
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        ## 遍历9个窗口，如果窗口内的像素点个数超过了minpix，就把窗口的终点移至这些像素点的终点
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))# 索引取nonzeroy的值，得到的就是该点的x坐标
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    # left_lane_inds 是包含一组列表的列表，concatenate将它们合成一个列表
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
        
    return out_img, leftx, lefty, rightx, righty

def fit_polynomial(binary_warped):
    # Find our lane pixels first
    out_img, leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)
    
    # Fit a second order polynomial to each using `np.polyfit`
    # 拟合车道线的曲线(二项式曲线)
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    ## Visualization ##
    window_img = np.zeros_like(out_img)
    # left: red    right: blue
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    
    # Generate x and y values for plotting
    # y轴每1个像素创建一个坐标点
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    pts_left=np.array([[x,y] for x,y in zip(left_fitx,ploty)],np.int32).reshape((-1,1,2))
    pts_right=np.array([[x,y] for x,y in zip(right_fitx,ploty)],np.int32).reshape((-1,1,2))
    
    # Draw the lane lines
    cv2.polylines(out_img,pts_left,True,(0,255,25),5)
    cv2.polylines(out_img,pts_right,True,(0,255,25),5)

    
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    # Set the width of the windows +/- margin
    margin = 100
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])                        
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])              
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
 
    
    # Draw the lane regions onto the warped blank image   
    # cv2.fillPoly(window_img,np.int_([combined_window]),(0,255, 0))
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    
    # Initialize the out_img without windows
    out_img_ = np.dstack((binary_warped, binary_warped, binary_warped))
    # Color in left and right line pixels
    out_img_[lefty, leftx] = [255, 0, 0]
    out_img_[righty, rightx] = [0, 0, 255]
    # Draw the lane lines
    cv2.polylines(out_img_,pts_left,True,(0,255,25),5)
    cv2.polylines(out_img_,pts_right,True,(0,255,25),5)
    
    # Detect and fit the lane lines without sliding windows
    prev_poly = cv2.addWeighted(out_img_, 1, window_img, 0.3, 0)

    return out_img, prev_poly, left_fitx, right_fitx

def lane_curvature(binary_warped):
    img_shape = binary_warped.shape
    leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)[1:]
    left_fitx, right_fitx = fit_polynomial(binary_warped)[2:]
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension 车道线间距3.7m
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0]) # cover same y-range as image
    y_eval = np.max(ploty)
    
    # Calculate center position
    center_pos = (left_fitx[-1]+right_fitx[-1])/2
    
    # Fit new polynomials to x,y in world space
#     left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
#     right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Calculate the new radius of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + 
                           left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + 
                            right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Now our radius of curvature is in meters
    leftx_int =left_fitx[-1]
    rightx_int = right_fitx[-1]
    center = (center_pos - 1280/2) * xm_per_pix
    
    return left_curverad, right_curverad, center

def drawing(binary_warped, Minv, undist, left_fitx, right_fitx):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    # Mask the region between left_fitx(left lane) and right_fitx(right lane) showing the driving area
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    # Transform to the real world
    newwarp = cv2.warpPerspective(color_warp, Minv, (binary_warped.shape[1], binary_warped.shape[0])) 
    # Combine the result with the original image
    drawed = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # Write text (cte, radius of curvature) on image
    left_curverad, right_curverad, center = lane_curvature(binary_warped)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if center >= 0:
        cv2.putText(drawed, 'Vehicle is {:.2f}m right of center'.format(center), (50,100),
                     font, 1, color=(255,255,255), thickness = 2)
    else:
        cv2.putText(drawed, 'Vehicle is {:.2f}m left of center'.format(abs(center)), (50,100),
                     font, 1, color=(255,255,255), thickness = 2)
    cv2.putText(drawed, 'Radius of curvature is {}m'.format(int((left_curverad + right_curverad)/2)), (50,150),
                     font, 1, color=(255,255,255), thickness = 2)
    
    return drawed

def image_process(img):
    undistort = cal_undistort(img, objpoints, imgpoints)
    # image size
    imshape = img.shape
    # vertices of selected roi
#     roi_vertices = np.array([[(0, imshape[0]),
#                         (imshape[1]*7/15, imshape[0]*3/5),
#                         (imshape[1]*8/15, imshape[0]*3/5),
#                         (imshape[1],imshape[0])]],
#                          dtype=np.int32)
    roi_vertices = np.array([[(200,720),(630,400),(650,400),(1150,720)]],dtype=np.int32)

    # -------------------------------------------------------------
    # Gradient Thresholds
    # -------------------------------------------------------------
    # Choose a Sobel kernel size： a larger odd number to smooth gradient measurements
    gradient_binary= gradient_thresh(undistort)
    # Extract selected regions by roi_vertices
    # gradient_binary = region_of_interest(gradient_combined, roi_vertices)


    # -------------------------------------------------------------
    # Color Thresholds
    # -------------------------------------------------------------
    # selected_image = region_of_interest(undistort, roi_vertices)
    color_binary = color_combined_thresh(undistort)


    # -------------------------------------------------------------
    # Gradient & Color Thresholds
    # -------------------------------------------------------------
    combined_binary = np.zeros_like(color_binary)
    combined_binary[(gradient_binary == 1) | (color_binary ==1)] = 1


    # -------------------------------------------------------------
    # Perspective Transform
    # -------------------------------------------------------------
    warped, _, Minv = perspective_tansform(combined_binary)

    # -------------------------------------------------------------
    # Detect lane lines
    # -------------------------------------------------------------

    left_fitx, right_fitx = fit_polynomial(warped)[2:]
    # Radius of curvature
    left_curverad, right_curverad, center = lane_curvature(warped)


    # -------------------------------------------------------------
    # Drawing
    # -------------------------------------------------------------
    # Draw the lane & Write text onto the warped blank image
    result = drawing(warped, Minv, undistort, left_fitx, right_fitx)

    return result




checkboard_images = glob.glob('./camera_cal/calibration*.jpg')
objpoints, imgpoints = get_obj_img_points(checkboard_images)

test_image = mpimg.imread('./test_images/test4.jpg')
result = image_process(test_image)
plt.figure(figsize=(16,9))
plt.imshow(result)
plt.title("Detected Lane", fontsize=25)

plt.show()








