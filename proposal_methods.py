import numpy as np
import torch
import utils.wsad_utils as utils
from scipy.signal import savgol_filter
import pdb
# from tensorboard_logger import Logger
import pandas as pd
import options


@torch.no_grad()
def multiple_threshold_hamnet(vid_name, data_dict, args):
    element_atn = data_dict['attn']
    element_apex = data_dict['apex']

    cas_supp_atn = element_atn

    pred = np.array([0, 1])

    act_thresh = []
    # NOTE: threshold
    act_thr = np.linspace(0.2, 0.8, 7)
    act_thresh.append(act_thr)
    act_thr = np.linspace(0.2, 0.8, 7)   # 0.1 0.9 9
    act_thresh.append(act_thr)

    num_segments = element_atn[0].shape[0]

    cas_pred_atn = cas_supp_atn[0].cpu().numpy()[:, [0]]
    apex_scores = element_apex[0].cpu().numpy()

    cas_pred_atn = np.reshape(cas_pred_atn, (num_segments, -1, 1))

    proposal_dict = {}

    dataset = args.dataset_name
    if dataset == "CASME2":
        thr1 = 0.5
        thr2 = 0.5
        thr3 = 0.5
    elif dataset == "CASME3":
        thr1 = 0.3
        thr2 = 0.3
        thr3 = 0.5
    else: # SAMM
        thr1 = 0.4 # MaE
        thr2 = 0.3 # 0.3
        thr3 = 0.5 # 0.5

    for c in pred:
        for i in range(len(act_thresh[c])):
            cas_temp_atn = cas_pred_atn.copy()
            temp_apex = apex_scores.copy()
            seg_list = []  # 0 MaE proposals, 1 ME proposals
            pos = np.where(cas_temp_atn[:, 0, 0] > act_thresh[c][i])
            seg_list.append(pos)
            proposals = utils.get_proposal_oic_3(seg_list, vid_name, cas_temp_atn, None, temp_apex, None, c, args.scale, num_segments, args.feature_fps, num_segments, lambda_=0.2, gamma=0.2, action_thr=thr3)
            if len(proposals) > 0:
                try:
                    class_id = proposals[0][0][0]

                    if class_id not in proposal_dict.keys():
                        proposal_dict[class_id] = []

                    proposal_dict[class_id] += proposals[0]
                except IndexError:
                    pass


    final_proposals = []
    for class_id in proposal_dict.keys():
        final_proposals.append(
            utils.soft_nms(proposal_dict[class_id], 0.7, sigma=0.3))

    segment_predict = []
    for i in range(len(final_proposals)):
        for j in range(len(final_proposals[i])):
            [c_pred, c_score, t_start, t_end, t_apex] = final_proposals[i][j]

            if (c_pred == 0.0 and (15 <= (t_end - t_start + 1) <= 240) and c_score >= thr1) or (c_pred == 1.0 and (9 <= (t_end - t_start + 1) <= 15) and c_score >= thr2):
                segment_predict.append([t_start, t_end, c_score, c_pred, t_apex])

    segment_predict = np.array(segment_predict)
    segment_predict = nms(segment_predict, num_segments, args)

    video_lst, t_start_lst, t_end_lst, t_apex_lst = [], [], [], []
    label_lst, score_lst = [], []
    for i in range(np.shape(segment_predict)[0]):
        video_lst.append(vid_name)
        t_start_lst.append(segment_predict[i, 0])
        t_end_lst.append(segment_predict[i, 1])
        score_lst.append(segment_predict[i, 2])
        label_lst.append(segment_predict[i, 3])
        t_apex_lst.append(segment_predict[i, 4])
    prediction = pd.DataFrame(
        {
            "video-id": video_lst,
            "t-start": t_start_lst,
            "t-end": t_end_lst,
            "t-apex": t_apex_lst,
            "label": label_lst,
            "score": score_lst,
        }
    )
    return prediction


def nms(detections, length_v, args):
    """
    Apply soft non-maximum suppression to detections.

    Args:
        detections (list): List of detections, each detection is a list [start, end, score, class].
        sigma (float): Sigma parameter for Gaussian penalty function.
        threshold (float): Threshold for discarding detections.

    Returns:
        List of filtered detections after applying soft NMS.
    """

    dataset = args.dataset_name
    filtered_detections = []

    if dataset == 'CASME2':
        temp = [8, 2]
    elif dataset == 'SAMM':
        temp = [8, 8]
    elif dataset == "CASME3":
        temp = [8, 2]

    while detections.shape[0] > 0:
        if temp[0] == 0 and temp[1] == 0:
            break

        max_index = np.argmax(detections[:, 2])  # Index of detection with maximum score
        max_detection = detections[max_index]

        start_i, end_i, c_score, pred_i, apex_i = max_detection

        if c_score == 0.0:
            break

        if temp[int(pred_i)] == 0:  # 每个类别至多留5个
            detections = np.delete(detections, max_index, axis=0)
        else:
            temp[int(pred_i)] -= 1

            filtered_detections.append(max_detection)

            detections = np.delete(detections, max_index, axis=0)

            for i in range(detections.shape[0]):
                if i >= detections.shape[0]:
                    break
                start_j, end_j, score_j, pred_j, apex_j = detections[i]

                if pred_j != pred_i:
                    continue

                overlap = max(0, min(end_i, end_j) - max(start_i, start_j))
                union = (end_i - start_i) + (end_j - start_j) - overlap
                overlap_ratio = overlap / union

                # Reduce score
                if overlap_ratio > 0.5:
                    detections = np.delete(detections, i, axis=0)
                    i -= 1

    return np.array(filtered_detections)