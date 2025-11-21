import torch
import torch.nn.functional as F
import torch.optim as optim
# from tensorboard_logger import log_value
import utils.wsad_utils as utils
import numpy as np
from torch.autograd import Variable
from eval.classificationMAP import getClassificationMAP as cmAP
from eval.eval_detection import ANETdetection
import wsad_dataset
from eval.detectionMAP import getDetectionMAP as dmAP
import scipy.io as sio
# from tensorboard_logger import Logger
import multiprocessing as mp
import options
import model
import proposal_methods as PM
import pandas as pd
from collections import defaultdict
import os

torch.set_default_tensor_type('torch.cuda.FloatTensor')
@torch.no_grad()
def test(itr, test_dataset, dataset, args, spotformer, model, logger, device):
    model.eval()
    spotformer.eval()

    proposals = []
    results = defaultdict(dict)

    attn_scores = []
    apex_scores = []
    for batch_idx, (features, labels, mask, vn) in enumerate(dataset):
        seq_len = [features.shape[1]]
        if seq_len == 0:
            continue
        features = features.to(device)
        index = [1, 2, 3, 5, 6, 7, 9, 10, 11, 12]
        features = features[:, :, :, index, :]

        with torch.no_grad():
            features = spotformer(Variable(features))
            outputs = model(features, is_training=False, seq_len=seq_len)
            results[vn] = {'attn': outputs['attn'], 'apex':outputs['apex']}

            attn_scores.append(np.squeeze(outputs['attn'].cpu().numpy()))
            apex_scores.append(np.squeeze(outputs['apex'].cpu().numpy()))
            proposals.append(getattr(PM, args.proposal_method)(vn, outputs, args))

    if not os.path.exists('temp'):
        os.mkdir('temp')
    np.save('temp/{}.npy'.format(args.model_name), results)

    proposals = pd.concat(proposals).reset_index(drop=True)

    iou = [0.5]

    dmap_detect = ANETdetection(test_dataset.path_to_annotations, iou, args=args, subset='validation')
    dmap_detect.prediction = proposals
    dmap = dmap_detect.evaluate()

    return iou, dmap, attn_scores, proposals, None, apex_scores