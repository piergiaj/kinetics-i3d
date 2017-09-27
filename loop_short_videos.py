from __future__ import division
import os
import numpy as np
import subprocess

base_dir = '/ssd2/hmdb/splits'

splits = ['split1', 'split2', 'split3']

files = os.listdir(base_dir)

splits = ['split1_train.txt', 'split1_test.txt', 'split2_train.txt', 'split2_test.txt', 'split3_train.txt', 'split3_test.txt']

for split_f in splits:
    # read the split file (video, class)
    split_file = os.path.join(base_dir, 'final_'+split_f)
    with open(split_file, 'r') as split_file:
        videos = split_file.read().split('\n')
        videos = [(v.split(' ')[0][:-6]+'.avi', v.split(' ')[2]) for v in videos if len(v) > 0]

    # create dictionary of video to class id (as int)
    for vid in videos:
        num_frames = int(vid[1])
        orig_f = num_frames
        next_frame = num_frames
        #print vid[1], vid[1] < 64
        if num_frames == 1:
            print 'ERROR!', vid
        while num_frames < 66 and num_frames != 1:
            print vid[1], num_frames, vid[0]
            for i in range(1,orig_f):
                subprocess.check_call('cp "'+os.path.join('/ssd2/hmdb', 'jpegs_final', vid[0].split('.')[0]+'_25fps', 'frame'+str(i).zfill(6)+'.jpg')+'" "'+os.path.join('/ssd2/hmdb', 'jpegs_final', vid[0].split('.')[0]+'_25fps', 'frame'+str(next_frame).zfill(6)+'.jpg')+'"', shell=True)
                subprocess.check_call('cp "'+os.path.join('/ssd2/hmdb', 'tvl1_flow/u', vid[0].split('.')[0]+'_gray', 'frame'+str(i).zfill(6)+'.jpg')+'" "'+os.path.join('/ssd2/hmdb', 'tvl1_flow/u/', vid[0].split('.')[0]+'_gray', 'frame'+str(next_frame).zfill(6)+'.jpg')+'"', shell=True)
                subprocess.check_call('cp "'+os.path.join('/ssd2/hmdb', 'tvl1_flow/v', vid[0].split('.')[0]+'_gray', 'frame'+str(i).zfill(6)+'.jpg')+'" "'+os.path.join('/ssd2/hmdb', 'tvl1_flow/v/', vid[0].split('.')[0]+'_gray', 'frame'+str(next_frame).zfill(6)+'.jpg')+'"', shell=True)
                
                next_frame += 1
            num_frames = len(os.listdir(os.path.join('/ssd2/hmdb/', 'jpegs_final', vid[0].split('.')[0]+'_25fps')))
