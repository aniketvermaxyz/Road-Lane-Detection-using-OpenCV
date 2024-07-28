import numpy as np  # Importing numpy library for numerical operations
import cv2  # Importing OpenCV library for image processing
from moviepy.editor import VideoFileClip  # Importing VideoFileClip for working with videos

def region_selection(image):
    """
    Determine and cut the region of interest in the input image.
    Parameters:
        image: The output from Canny where we have identified edges in the frame.
    """
    mask = np.zeros_like(image)  # Create a mask of zeros with the same dimensions as the input image
    ignore_mask_color = 255  # Set the color for the mask
    rows, cols = image.shape[:2]  # Get the dimensions of the image
    bottom_left = [cols * 0.1, rows * 0.95]  # Define the bottom left point of the region of interest
    top_left = [cols * 0.4, rows * 0.6]  # Define the top left point of the region of interest
    bottom_right = [cols * 0.9, rows * 0.95]  # Define the bottom right point of the region of interest
    top_right = [cols * 0.6, rows * 0.6]  # Define the top right point of the region of interest
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)  # Define the vertices of the region of interest
    cv2.fillPoly(mask, vertices, ignore_mask_color)  # Fill the region of interest in the mask with the specified color
    masked_image = cv2.bitwise_and(image, mask)  # Apply the mask to the input image
    return masked_image

def hough_transform(image):
    """
    Determine and cut the region of interest in the input image.
    Parameter:
        image: Grayscale image which should be an output from the edge detector.
    """
    rho = 1  # Distance resolution of the accumulator in pixels
    theta = np.pi/180  # Angle resolution of the accumulator in radians
    threshold = 20  # Accumulator threshold parameter
    minLineLength = 20  # Minimum line length. Line segments shorter than this are rejected
    maxLineGap = 500  # Maximum allowed gap between line segments to treat them as a single line
    return cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold,
                            minLineLength=minLineLength, maxLineGap=maxLineGap)  # Apply Hough Transform to detect lines
    
def average_slope_intercept(lines):
    """
    Find the slope and intercept of the left and right lanes of each image.
    Parameters:
        lines: Output from Hough Transform.
    """
    left_lines = []  # List to store slope and intercept of left lane lines
    left_weights = []  # List to store weights (lengths) of left lane lines
    right_lines = []  # List to store slope and intercept of right lane lines
    right_weights = []  # List to store weights (lengths) of right lane lines
    
    for line in lines:  # Loop through each line detected by Hough Transform
        for x1, y1, x2, y2 in line:  # Loop through the endpoints of each line
            if x1 == x2:  # If the line is vertical (undefined slope), skip it
                continue
            slope = (y2 - y1) / (x2 - x1)  # Calculate the slope of the line
            intercept = y1 - (slope * x1)  # Calculate the y-intercept of the line
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))  # Calculate the length of the line segment
            if slope < 0:  # If the slope is negative, it belongs to the left lane
                left_lines.append((slope, intercept))  # Append the slope and intercept to the left lane list
                left_weights.append((length))  # Append the length to the left weights list
            else:  # If the slope is positive, it belongs to the right lane
                right_lines.append((slope, intercept))  # Append the slope and intercept to the right lane list
                right_weights.append((length))  # Append the length to the right weights list
                
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None  # Calculate the average slope and intercept of the left lane
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None  # Calculate the average slope and intercept of the right lane
    return left_lane, right_lane

def pixel_points(y1, y2, line):
    """
    Converts the slope and intercept of each line into pixel points.
    Parameters:
        y1: y-value of the line's starting point.
        y2: y-value of the line's end point.
        line: The slope and intercept of the line.
    """
    if line is None:  # If the line is None (not detected), return None
        return None
    slope, intercept = line  # Extract slope and intercept from the line
    x1 = int((y1 - intercept)/slope)  # Calculate x-coordinate of the starting point
    x2 = int((y2 - intercept)/slope)  # Calculate x-coordinate of the end point
    y1 = int(y1)  # Convert y1 to integer
    y2 = int(y2)  # Convert y2 to integer
    return ((x1, y1), (x2, y2))  # Return the pixel points as a tuple

def lane_lines(image, lines):
    """
    Create full-length lines from pixel points.
    Parameters:
        image: The input test image.
        lines: The output lines from Hough Transform.
    """
    left_lane, right_lane = average_slope_intercept(lines)  # Get the average slope and intercept of left and right lanes
    y1 = image.shape[0]  # y-coordinate of the bottom of the image
    y2 = y1 * 0.6  # y-coordinate of the top of the region of interest
    left_line = pixel_points(y1, y2, left_lane)  # Convert slope and intercept to pixel points for left lane
    right_line = pixel_points(y1, y2, right_lane)  # Convert slope and intercept to pixel points for right lane
    return left_line, right_line  # Return pixel points for left and right lanes

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
    """
    Draw lines onto the input image.
    Parameters:
        image: The input test image (video frame in our case).
        lines: The output lines from Hough Transform.
        color (Default = red): Line color.
        thickness (Default = 12): Line thickness. 
    """
    line_image = np.zeros_like(image)  # Create a blank image with the same dimensions as the input image
    for line in lines:  # Loop through the lines
        if line is not None:  # Check if the line is not None
            cv2.line(line_image, *line, color, thickness)  # Draw the line on the blank image
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)  # Add the line image to the original image

def frame_processor(image):
    """
    Process the input frame to detect lane lines.
    Parameters:
        image: Image of a road where one wants to detect lane lines (frames of video).
    """
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    kernel_size = 5  # Define the size of the Gaussian blur kernel
    blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)  # Apply Gaussian blur to reduce noise
    low_t = 50  # Define the lower threshold for Canny edge detection
    high_t = 150  # Define the higher threshold for Canny edge detection
    edges = cv2.Canny(blur, low_t, high_t)  # Detect edges using Canny edge detection
    region = region_selection(edges)  # Select the region of interest
    hough = hough_transform(region)  # Detect lines using Hough Transform
    result = draw_lane_lines(image, lane_lines(image, hough))  # Draw lane lines on the frame
    return result  # Return the processed frame

def process_video(test_video, output_video):
    """
    Read input video stream and produce a video file with detected lane lines.
    Parameters:
        test_video: Location of input video file.
        output_video: Location where output video file is to be saved.
    """
    clip = VideoFileClip(test_video)  # Open the input video file
    processed_clip = clip.fl_image(frame_processor)  # Process each frame of the video
    processed_clip.write_videofile(output_video, audio=False)  # Write the processed frames to a new video file
    print("Video processing completed.")  # Print a message indicating that video processing is complete

# Input and output file paths
input_video_path = r"D:\6TH SEM PROJECT CODES\videos\1\test25.mp4"  # Path to the input video file
output_video_path = r"D:\6TH SEM PROJECT CODES\videos\1\output.mp4"  # Path to save the output video file

# Process the video
process_video(input_video_path, output_video_path)  # Call the function to process the video
