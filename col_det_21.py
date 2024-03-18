import cv2
import sqlite3
import numpy as np
import colorsys


def detect_color(frame, db_connection):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    print(hsv)
    color_detected = False

    # Retrieve HSV values from the database
    cursor = db_connection.cursor()
    cursor.execute("SELECT color_name, hue, saturation, value FROM color_samples")
    color_samples = cursor.fetchall()
    print(color_samples)

    for color_name, hue, saturation, value in color_samples:
        # Calculate lower and upper bounds for hue, saturation, and value
        hue_lower = hue - 10
        hue_upper = hue + 10
        saturation_lower = max(0, saturation - 50)
        saturation_upper = min(255, saturation + 50)
        value_lower = max(0, value - 50)
        value_upper = min(255, value + 50)

        # Mask pixels within the HSV range
        mask = cv2.inRange(hsv, (hue_lower, saturation_lower, value_lower), (hue_upper, saturation_upper, value_upper))
        print(mask)
        # Count non-zero pixels in the mask
        pixel_count = cv2.countNonZero(mask)

        # If a sufficient number of pixels match the color sample, consider the color detected
        if pixel_count > 0.1 * mask.size:
            print("Detected color:", color_name)
            color_detected = True
            break

    if not color_detected:
        print("No color match found")


def save_color_values(frame, db_connection):
    # Assuming you want to extract color from the center of the frame
    x = frame.shape[1] // 2
    y = frame.shape[0] // 2
    color = extract_color(frame, x, y)
    r = color[2] / 255.0
    g = color[1] / 255.0
    b = color[0] / 255.0

    # Convert to hsv
    (h, s, v) = colorsys.rgb_to_hsv(r, g, b)

    # Expand HSV range
    h = int(h * 179)
    s = int(s * 255)
    v = int(v * 255)

    # Determine color name based on HSV values
    color_name = get_color_name(h, s, v)

    print('HSV : ', h, s, v)
    print('Color Name: ', color_name)

    # Insert the color values into the database
    cursor = db_connection.cursor()
    cursor.execute("INSERT INTO color_samples (color_name, hue, saturation, value) VALUES (?, ?, ?, ?)", (color_name, h, s, v))
    db_connection.commit()

def get_color_name(h, s, v):
    # Example logic to map HSV values to color names
    if h < 10 or h > 170:
        return 'Red'
    elif 10 <= h < 30:
        return 'Orange'
    elif 30 <= h < 70:
        return 'Yellow'
    elif 70 <= h < 100:
        return 'Green'
    elif 100 <= h < 140:
        return 'Blue'
    elif 140 <= h < 160:
        return 'Purple'
    else:
        return 'Unknown'


def extract_color(frame, x, y, window_size=5):
    # Get the region of interest (ROI) around the selected point
    roi = frame[y - window_size:y + window_size, x - window_size:x + window_size]

    # Calculate the average color in the ROI
    avg_color = np.mean(roi, axis=(0, 1))

    return avg_color.astype(int)


# Connect to the database
db_connection = sqlite3.connect('colors.db')

# Create the colors table if it doesn't exist
cursor = db_connection.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS color_samples 
                  (id INTEGER PRIMARY KEY, color_name Char, hue INTEGER, saturation INTEGER, value INTEGER)''')
db_connection.commit()

# Create a VideoCapture object to capture video from the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Call the detect_color function with the frame and the database connection
    detect_color(frame, db_connection)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        save_color_values(frame, db_connection)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Close the database connection
db_connection.close()
