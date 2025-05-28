import cv2
import numpy as np
def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_edges = cv2.Canny(blur, 50, 150)
    return canny_edges
def ROI(img):
    height = img.shape[0]
    triangle = np.array([[(200, height), (1100, height), (550, 250)]], dtype=np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image
def coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    if slope == 0:  # prevent divide by zero
        slope = 0.1
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])
def average_slope(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept = parameters
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    averaged_lines = []
    if left_fit:
        left_avg = np.average(left_fit, axis=0)
        averaged_lines.append(coordinates(image, left_avg))
    if right_fit:
        right_avg = np.average(right_fit, axis=0)
        averaged_lines.append(coordinates(image, right_avg))
    return np.array(averaged_lines)
cap = cv2.VideoCapture("test2.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    canny_image = canny(frame)
    cropped_image = ROI(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    final_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("Lane Detection", final_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
