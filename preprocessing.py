import numpy as np
import cv2

# screen: Contains the frame from the game running in the retro environment
# crop: Tuple in this format (y, y+h, x, x+w)
def preprocess(screen, crop, output):
    # Gray scale the image
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)

    # Crop screen
    screen = screen[crop[0]:crop[1], crop[2]:crop[3]]

    # Convert to float, and normalized
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255

    # Resize image to 84x84
    screen = cv2.resize(screen, (output, output), interpolation=cv2.INTER_AREA)
    return screen