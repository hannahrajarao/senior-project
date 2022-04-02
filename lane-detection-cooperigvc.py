import sys
import numpy as np
import pyzed.sl as sl
import cv2


def do_canny(frame):
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Applies a 5x5 gaussian blur with deviation of 0 to frame - not mandatory since Canny will do this for us
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Applies Canny edge detector with minVal of 50 and maxVal of 150
    canny = cv2.Canny(blur, 50, 150)
    return canny

def do_segment(frame):
    if perform_segment:
        # Since an image is a multi-directional array containing the relative intensities of each pixel in the image, we can use frame.shape to return a tuple: [number of rows, number of columns, number of channels] of the dimensions of the frame
        # frame.shape[0] give us the number of rows of pixels the frame has. Since height begins from 0 at the top, the y-coordinate of the bottom of the frame is its height
        height = frame.shape[0]
        width = frame.shape[1]
        # Creates a triangular polygon for the mask defined by three (x, y) coordinates
        polygon = np.array([[(0, height), (width, height), (width/2, height/2)]], dtype=np.int32)
        # Creates an image filled with zero intensities with the same dimensions as the frame
        mask = np.zeros_like(frame)
        # Allows the mask to be filled with values of 1 and the other areas to be filled with values of 0
        cv2.fillPoly(mask, polygon, 255)
        # A bitwise and operation between the mask and frame keeps only the triangular area of the frame
        segment = cv2.bitwise_and(frame, mask)
        return segment
    else:
        return frame

def calculate_lines(frame, lines):
    # Empty arrays to store the coordinates of the left and right lines
    left = []
    right = []
    # Loops through every detected line
    if lines is not None:
        for line in lines:
            # Reshapes line from 2D array to 1D array
            x1, y1, x2, y2 = line.reshape(4)
            # Fits a linear polynomial to the x and y coordinates and returns a vector of coefficients which describe the slope and y-intercept
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            y_intercept = parameters[1]
            # If slope is negative, the line is to the left of the lane, and otherwise, the line is to the right of the lane
            if slope < 0:
                left.append((slope, y_intercept))
            else:
                right.append((slope, y_intercept))
    # Averages out all the values for left and right into a single slope and y-intercept value for each line
    left_avg = np.average(left, axis = 0)
    right_avg = np.average(right, axis = 0)
    # Calculates the x1, y1, x2, y2 coordinates for the left and right lines
    left_line = calculate_coordinates(frame, left_avg)
    right_line = calculate_coordinates(frame, right_avg)
    return np.array([left_line, right_line])

def calculate_coordinates(frame, parameters):
    try: 
        slope, intercept = parameters
    except:
        print('there is an error and i don\'t know why')
        print('parameters', parameters)
        slope = 1
        intercept = 0
    # Sets initial y-coordinate as height from top down (bottom of the frame)
    y1 = frame.shape[0]
    # Sets final y-coordinate as 150 above the bottom of the frame
    y2 = int(y1 - 150)
    # Sets initial x-coordinate as (y1 - b) / m since y1 = mx1 + b
    x1 = int((y1 - intercept) / slope)
    # Sets final x-coordinate as (y2 - b) / m since y2 = mx2 + b
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def visualize_lines(frame, lines):
    # Creates an image filled with zero intensities with the same dimensions as the frame
    lines_visualize = np.zeros_like(frame)
    # Checks if any lines are detected
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # Draws lines between two coordinates with green color and 5 thickness
            cv2.line(lines_visualize, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return lines_visualize

def draw_lines(frame, lines):
    if lines is not None:
        for line in lines:
            coords = line[0]
            cv2.line(frame, (coords[0], coords[1]), (coords[2], coords[3]), 200, 3)

def lane_detection(frame):
    canny = do_canny(frame)
    # cv2.imshow("canny", canny)
    segment = do_segment(canny)
    # segment = canny
    hough_lines = cv2.HoughLinesP(segment, 2, np.pi / 180, 100, np.array([]), minLineLength = 100, maxLineGap = 50)
    line_disp = np.zeros_like(frame)
    draw_lines(line_disp, hough_lines)
    cv2.imshow("lines",line_disp)
    lines = calculate_lines(frame, hough_lines) # Averages multiple detected lines from hough into one line for left border of lane and one line for right border of lane
    lines_visualize = visualize_lines(frame, lines) # Visualizes the lines
    cv2.imshow("lines_visualize", lines_visualize)
    output = cv2.addWeighted(frame, 0.9, line_disp, 1, 1) # Overlays lines on frame by taking their weighted sums and adding an arbitrary scalar value of 1 as the gamma argument
    return output
# def solution_lane_detection(frame):
#     canny = do_canny(frame)
#     cv2.imshow("canny", canny)
#     # plt.imshow(frame)
#     # plt.show()
#     segment = do_segment(canny)
#     hough = cv2.HoughLinesP(segment, 2, np.pi / 180, 100, np.array([]), minLineLength = 100, maxLineGap = 50)
#     # Averages multiple detected lines from hough into one line for left border of lane and one line for right border of lane
#     lines = calculate_lines(frame, hough)
#     # Visualizes the lines
#     lines_visualize = visualize_lines(frame, lines)
#     cv2.imshow("hough", lines_visualize)
#     # Overlays lines on frame by taking their weighted sums and adding an arbitrary scalar value of 1 as the gamma argument
#     output = cv2.addWeighted(frame, 0.9, lines_visualize, 1, 1)
#     return output
def main() :

    # Create a ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    input_type = sl.InputType()
    if len(sys.argv) >= 2 :
        input_type.set_from_svo_file(sys.argv[1])
    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.MILLIMETER

    # Open the camera
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed.close()
        exit(1)

    # Set runtime parameters after opening the camera
    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD

    # Prepare new image size to retrieve half-resolution images
    image_size = zed.get_camera_information().camera_resolution
    image_size.width = image_size.width /2
    image_size.height = image_size.height /2

    # Declare your sl.Mat matrices
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)

    global perform_segment
    perform_segment = False
    key = 0
    while key != 113:
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            # Retrieve the left image, depth image in the half-resolution
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            
            image_ocv = image_zed.get_data() #Convert sl.Mat to numpy array
            # cv2.imshow("Image", image_ocv)
            output_image = lane_detection(image_ocv)
            cv2.imshow("Output", output_image)
            key = cv2.waitKey(10)
            if key == 115:
                perform_segment = not perform_segment
                print('perform_segment: ', perform_segment)

    cv2.destroyAllWindows()
    zed.close()

    print("FINISH")

if __name__ == "__main__":
    main()
