from __future__ import unicode_literals
import os
import scipy.misc
import numpy as np
import cv2
import subprocess

base = '/ssd2/hmdb'

actions = os.listdir(base)
actions.remove('splits')
actions.remove('gray')
actions.remove('jpegs_256')
actions.remove('jpegs_final')
actions.remove('tvl1_flow')


resample_command = """ffmpeg -y -i 'FILE' -r 25 -vf scale="'if(gt(iw,ih),-1,256)':'if(gt(ih,iw),-1,256)'" '@&OUT'"""

fourcc = cv2.cv.FOURCC(b'X',b'V',b'I',b'D')

for action in actions:
    videos = os.listdir(os.path.join(base, action))
    for video in videos:
        if '25fps_25fps' in video:
            video_file = os.path.join(base, action, video)
            os.system('rm '+video_file)
        if '25fps' in video:
            continue
        # resample to 25FPS and scale smallest side to 256           
	video_file = os.path.join(base, action, video)
	video_file_out = video_file.split('.avi')[0]+'_25fps.avi'
        if not os.path.exists(video_file_out):
            cmd = resample_command.replace('FILE', video_file).replace('@&OUT', video_file_out)
            print '\n\n'
            print resample_command
            print resample_command.replace('FILE', video_file)
            print cmd
	    subprocess.check_call(cmd, shell=True)

	# convert to greyscale and compute optical flow
        cap = cv2.VideoCapture(video_file_out)
        width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))   # float
        height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)) # float
        out_video = os.path.join(base,'gray',video.split('.avi')[0]+'_gray.avi')
        if os.path.exists(out_video):
            continue
	out = cv2.VideoWriter(out_video, fourcc, 25.0, (width, height), False)
	success, image = cap.read()
	while success:
   	    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	    out.write(gray)
	    success, image = cap.read()
	cap.release()
	out.release()
	print video_file
	
#    os.system('./compute_flow -g 3 -path vid')
    
