import numpy as np
import cv2

def preprocess(screen, exclude, output):
    # Gray scale the image
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)

    # Crop screen
    screen = screen[exclude[0]:exclude[2], exclude[3]:exclude[1]]

    # Convert to float, and normalized
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255

    # Resize image to 84x84
    screen = cv2.resize(screen, (output, output), interpolation=cv2.INTER_AREA)
    return screen