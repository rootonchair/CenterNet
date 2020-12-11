import _init_paths
import os, random, time
import torch
from detectors.detector_factory import detector_factory
from tqdm import tqdm
from opts import opts

# Choose to use a config and initialize the detector
config = "/mnt/ai_filestore/home/vincent/mmdetection/configs/mask_rcnn_r50_fpn_1x.py"
checkpoint = "/mnt/ai_filestore/home/vincent/mmdetection/work_dirs/mask_rcnn_r50_fpn_1x/epoch_2.pth"

# initialize the detector
MODEL_PATH = '../models/model_last.pth'
TASK = 'ctdet'
GPU_ID = '-1'
opt = opts().init('{} --load_model {} --gpus {}'.format(TASK, MODEL_PATH, GPU_ID).split(' '))
detector = detector_factory[opt.task](opt)

imgs_dir = "/mnt/ai_filestore/home/vincent/TableBank/Detection/images"
imgs_list = os.listdir(imgs_dir)

k = 500
process_time = []
for i in tqdm(range(k)):
    rand_path = os.path.join(imgs_dir,random.choice(imgs_list))
    t = time.time()
    result = detector.run(rand_path)['results']
    t_e = time.time()-t
    process_time.append(t_e)
avg_time = sum(process_time)/k
print(avg_time)

