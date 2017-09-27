from __future__ import unicode_literals
import os
import scipy.misc
import numpy as np
import cv2

base = '/ssd2/hmdb'

actions = os.listdir(base)
actions.remove('splits')
actions.remove('gray')
actions.remove('jpegs_256')
actions.remove('jpegs_final')
actions.remove('tvl1_flow')
fourcc = cv2.cv.FOURCC(b'X',b'V',b'I',b'D')

for action in actions:
    videos = os.listdir(os.path.join(base, action))
    for video in videos:
        if not '25fps' in video:
            continue # not a rescaled video
	video_file = os.path.join(base, action, video)

	# extract frames
        if os.path.isdir(base+'/jpegs_final/'+video.split('.')[0]):
            continue
        cap = cv2.VideoCapture(video_file)
        width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))   # float
        height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)) # float
	success, image = cap.read()
        fr = 1
	while success:
            if not os.path.isdir(base+'/jpegs_final/'+video.split('.')[0]):
                os.mkdir(base+'/jpegs_final/'+video.split('.')[0])
            cv2.imwrite(base+'/jpegs_final/'+video.split('.')[0]+'/frame'+str(fr).zfill(6)+'.jpg', image)
	    success, image = cap.read()
            fr += 1
	cap.release()
	
#    os.system('./compute_flow -g 3 -path vid')
    
