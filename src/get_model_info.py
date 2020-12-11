import _init_paths

import torch
from detectors.detector_factory import detector_factory
from opts import opts
from ptflops import get_model_complexity_info

MODEL_PATH = '../models/model_last.pth'
TASK = 'ctdet'
opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
detector = detector_factory[opt.task](opt)

model = detector.model
with torch.cuda.device(0):
    macs, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True,
                                                       print_per_layer_stat=True,
                                                       verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
