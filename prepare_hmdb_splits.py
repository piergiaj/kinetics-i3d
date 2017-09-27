from __future__ import division
import os
import numpy as np

base_dir = '/ssd2/hmdb/splits'

splits = ['split1', 'split2', 'split3']

files = os.listdir(base_dir)


for split in splits:
    with open(os.path.join(base_dir, split)+'_train.txt', 'w') as train_out, open(os.path.join(base_dir, split)+'_test.txt', 'w') as test_out:
        for f in files:
            if not split in f:
                continue
            if 'split' in f[:5]:
                continue
            if 'final' in f:
                continue
            action = f.split('_test_')[0]
            videos = [v.split(' ') for v in file(os.path.join(base_dir, f), 'r').read().splitlines()]
            for v in videos:
                if int(v[1]) == 1:
                    train_out.write(v[0]+' '+action+'\n')
                elif int(v[1]) == 2:
                    test_out.write(v[0]+' '+action+'\n')

video_labels = {}
video_frames = {}
class_to_id = {}
next_id = 0

splits = ['split1_train.txt', 'split1_test.txt', 'split2_train.txt', 'split2_test.txt', 'split3_train.txt', 'split3_test.txt']

for split_f in splits:
    # read the split file (video, class)
    split_file = os.path.join(base_dir, split_f)
    with open(split_file, 'r') as split_file:
        videos = split_file.read().split('\n')
        videos = [(v.split(' ')[0], v.split(' ')[1]) for v in videos if len(v) > 0]

    # create dictionary of video to class id (as int)
    for vid in videos:
        if vid[1] not in class_to_id:
            class_to_id[vid[1]] = next_id
            next_id += 1
        video_labels[vid[0]] = class_to_id[vid[1]]
        video_frames[vid[0]] = len(os.listdir(os.path.join('/ssd2/hmdb/', 'jpegs_final', vid[0].split('.')[0]+'_25fps')))


    
# create the final split files of
# RGB-name FLOW-name #Frames Label
for split_f in splits:
    split_file = os.path.join(base_dir, split_f)
        
    with open(split_file, 'r') as split_file:
        videos = split_file.read().split('\n')
        videos = [(v.split(' ')[0], v.split(' ')[1]) for v in videos if len(v) > 0]

    with open(os.path.join(base_dir, 'final_'+split_f), 'w') as out:

        for vid in videos:
            label = str(video_labels[vid[0]])
            num_frames = str(video_frames[vid[0]])
            rgb_name = vid[0].split('.')[0]+'_25fps'
            flow_name = vid[0].split('.')[0]+'_gray'

            if 'IamLegend_run_f_nm_np1_le_med_3' in rgb_name: # theres a problem with this video..
                continue
            
            out.write(rgb_name+' '+flow_name+' '+num_frames+' '+label)
            if vid != videos[-1]:
                out.write('\n')
