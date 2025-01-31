import cv2
import numpy as np

def stitch_images(images, mode='homology'):
    if len(images)<2:
        return images[0]
    frames = []
    for img in images:
        np_img=np.array(img)
        np_img_bgr=cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        frames.append(np_img_bgr)
    
    if mode == 'panorama':
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    else:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)

    status, pano = stitcher.stitch(frames)

    if status == cv2.Stitcher_OK:
        return pano
    
    else:
        print(status)
        return None