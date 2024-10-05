import numpy as np
import cv2

# screen: Contains the frame from the game running in the retro environment
# crop: Tuple in this format (y, y+h, x, x+w)
# output: Output square image of a given size
def preprocess(screen, crop, output):
    # Gray scale the image
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)

    # Crop screen
    screen = screen[1:-1, 1:-1]

    # Convert to float, and normalized
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255

    # Resize image
    screen = cv2.resize(screen, (output, output), interpolation=cv2.INTER_AREA)
    return screen


def stack_frame(stacked_frames, frame, is_new):
    stacked_frames = np.stack(arrays=[frame])

    return stacked_frames