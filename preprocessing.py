import numpy as np
import cv2

# screen: Contains the frame from the game running in the retro environment
# crop: Tuple in this format (y, y+h, x, x+w)
# output: Output square image of a given size
def preprocess(screen, crop, output):
    # Convert image to strictly black and white
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    screen = cv2.threshold(screen, 1, 255, cv2.THRESH_BINARY)[1]

    # Crop screen [Top:Bot, Left:Right]
    screen = screen[crop[0]:crop[1], crop[2]:crop[3]]

    # Convert to float, and normalized
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255

    # Resize image
    screen = cv2.resize(screen, (output, output), interpolation=cv2.INTER_NEAREST)
    return screen


def stack_frame(stacked_frames, frame, new):
    if new:
        stacked_frames = np.stack(arrays=[frame, frame, frame, frame])
        stacked_frames = stacked_frames
    else:
        stacked_frames[0] = stacked_frames[1]
        stacked_frames[1] = stacked_frames[2]
        stacked_frames[2] = stacked_frames[3]
        stacked_frames[3] = frame

    return stacked_frames