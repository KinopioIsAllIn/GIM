from __future__ import print_function
import os
import torch
import model
from SpoTFormer import MSSTGT
from wsad_dataset import SampleDataset

import random
from test import test
import options
import numpy as np

import csv

torch.set_default_tensor_type('torch.cuda.FloatTensor')



if __name__ == '__main__':
    args = options.parser.parse_args()
    seed = args.seed
    cuda_n = args.cuda
    print('=============seed: {}, pid: {}============='.format(seed, os.getpid()))
    device = torch.device("cuda:" + str(cuda_n))

    dirname = args.dirname

    overall_tp_MaE, overall_tp_ME, overall_fp_MaE, overall_fp_ME, gt_MaE, gt_ME = 0, 0, 0, 0, 0, 0
    results = {}
    for datasetname in ['SAMM']:
        args.dataset_name = datasetname
        if args.dataset_name == 'SAMM':
            validation_subjects = [
                 "samm_006", "samm_011", "samm_010", "samm_007", "samm_008", "samm_009", "samm_012",
                 "samm_013", "samm_014", "samm_015", "samm_016", "samm_017", "samm_018", "samm_019",
                 "samm_020", "samm_026", "samm_025", "samm_021", "samm_022", "samm_023", "samm_028",  # "samm_024",
                 "samm_032", "samm_033", "samm_034", "samm_035", "samm_036", "samm_037", "samm_030",
                 "samm_031"]
            mae_total = 343
            me_total = 159
        elif args.dataset_name == 'CASME2':
            validation_subjects = [
                "casme_16", "casme_15", "casme_19", "casme_20", "casme_21",
                "casme_24", "casme_22", "casme_23", "casme_25", "casme_26",
                "casme_32", "casme_27", "casme_31", "casme_30", "casme_33",  # "casme_29",
                "casme_37", "casme_38", "casme_36", "casme_40", "casme_34"  # "casme_35",
            ]
            mae_total = 300
            me_total = 57
        elif args.dataset_name == 'CASME3':
            validation_subjects = [
                'casme3_1', 'casme3_10', 'casme3_11', 'casme3_12', 'casme3_13', 'casme3_138', 'casme3_139', 'casme3_14', 'casme3_140', 'casme3_142', 'casme3_143', 'casme3_144',
                    'casme3_145', 'casme3_146', 'casme3_147', 'casme3_148', 'casme3_149', 'casme3_15', 'casme3_150', 'casme3_151', 'casme3_152', 'casme3_153', 'casme3_154', 'casme3_155',
                    'casme3_156', 'casme3_157', 'casme3_158', 'casme3_159', 'casme3_16', 'casme3_160', 'casme3_161', 'casme3_162', 'casme3_163', 'casme3_165', 'casme3_166', 'casme3_167', 'casme3_168',
                    'casme3_169', 'casme3_17', 'casme3_170', 'casme3_171', 'casme3_172', 'casme3_173', 'casme3_174', 'casme3_175', 'casme3_176', 'casme3_177', 'casme3_178', 'casme3_179', 'casme3_180',
                    'casme3_181', 'casme3_182', 'casme3_183', 'casme3_184', 'casme3_185', 'casme3_186', 'casme3_187', 'casme3_188', 'casme3_189', 'casme3_190', 'casme3_192', 'casme3_193', 'casme3_194',
                    'casme3_195', 'casme3_196', 'casme3_197', 'casme3_198', 'casme3_2', 'casme3_200', 'casme3_201', 'casme3_202', 'casme3_203', 'casme3_204', 'casme3_206', 'casme3_207', 'casme3_208',
                    'casme3_209', 'casme3_210', 'casme3_212', 'casme3_213', 'casme3_214', 'casme3_215', 'casme3_216', 'casme3_217', 'casme3_218', 'casme3_3', 'casme3_39',
                    'casme3_4', 'casme3_40', 'casme3_41', 'casme3_42', 'casme3_5', 'casme3_6', 'casme3_7', 'casme3_77', 'casme3_8', 'casme3_9'
                ]
            mae_total = 2231
            me_total = 285

            validation_subjects = [subject[7:] for subject in validation_subjects]

        best_tp_MaE_total, best_tp_ME_total, best_fp_MaE_total, best_fp_ME_total = 0, 0, 0, 0
        best_apex_diff_MaE_total, best_apex_diff_ME_total = 0, 0
        best_tp_MaE_total_Separate, best_tp_ME_total_Separate, best_fp_MaE_total_Separate, best_fp_ME_total_Separate = 0, 0, 0, 0

        for subject in validation_subjects:
            args.validation_subject = subject

            test_dataset = SampleDataset(args, "test")
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                                      generator=torch.Generator(device=device))

            spotformer = MSSTGT(
                seq_len=17,
                num_classes=10,
                dim=256,
                depth=4,
                heads=4,
                mlp_dim=256,
                dropout=0.0,
                emb_dropout=0.1).to(device)

            # model = Model(dataset.feature_size, dataset.num_class).to(device)
            model1 = getattr(model, args.use_model)(256, 2, opt=args).to(device)

            best_f1 = 0.0
            best_epoch = 0
            best_tp_MaE, best_tp_ME, best_fp_MaE, best_fp_ME = 0, 0, 0, 0

            best_tp_MaE_Separate, best_tp_ME_Separate, best_fp_MaE_Separate, best_fp_ME_Separate = 0, 0, 0, 0
            best_apex_diff_MaE, best_apex_diff_ME = 0, 0
            best_f1_MaE, best_f1_ME = 0.0, 0.0

            for i in range(5, 100):
                if os.path.exists(args.pretrained_ckpt + subject +"_%d.pkl" % i):
                    try:
                        model1.load_state_dict(
                            torch.load(args.pretrained_ckpt + subject + "_%d.pkl" % i, map_location='cuda:0')['model1_state_dict'])
                        spotformer.load_state_dict(
                            torch.load(args.pretrained_ckpt + subject + "_%d.pkl" % i, map_location='cuda:0')['spotformer_state_dict'])

                    except Exception as e:
                        continue
                    iou, dmap, attn_scores, proposals, action_scores, apex_scores = test(0, test_dataset, test_loader, args, spotformer, model1, None, device)

                    tp_mae = dmap[0][0][0]
                    tp_me = dmap[0][0][1]
                    fp_mae = dmap[1][0][0]
                    fp_me = dmap[1][0][1]
                    gt_mae = dmap[2][0][0]
                    gt_me = dmap[2][0][1]
                    apex_diff_mae = dmap[3][0][0]
                    apex_diff_me = dmap[3][0][1]
                    f1 = 2 * (tp_mae + tp_me) / (tp_mae + tp_me + fp_mae + fp_me + gt_mae + gt_me)

                    f1_MaE = 2 * (tp_mae) / (tp_mae + fp_mae + gt_mae)
                    f1_ME = 2 * (tp_me) / (tp_me + fp_me + gt_me)

                    if f1 > best_f1 or (f1 == best_f1 and f1_ME > 2*best_tp_ME/(best_tp_ME + best_fp_ME+gt_me)):
                        best_f1 = f1
                        best_epoch = i
                        best_tp_MaE, best_tp_ME, best_fp_MaE, best_fp_ME = tp_mae, tp_me, fp_mae, fp_me
                        best_apex_diff_MaE = apex_diff_mae
                        best_apex_diff_ME = apex_diff_me

                    if f1_MaE > best_f1_MaE:
                        best_f1_MaE = f1_MaE
                        best_tp_MaE_Separate, best_fp_MaE_Separate = tp_mae, fp_mae

                    if f1_ME > best_f1_ME:
                        best_f1_ME = f1_ME
                        best_tp_ME_Separate, best_fp_ME_Separate = tp_me, fp_me

            best_tp_MaE_total += best_tp_MaE
            best_tp_ME_total += best_tp_ME
            best_fp_MaE_total += best_fp_MaE
            best_fp_ME_total += best_fp_ME
            best_apex_diff_MaE_total += best_apex_diff_MaE
            best_apex_diff_ME_total += best_apex_diff_ME

            best_tp_MaE_total_Separate += best_tp_MaE_Separate
            best_tp_ME_total_Separate += best_tp_ME_Separate
            best_fp_MaE_total_Separate += best_fp_MaE_Separate
            best_fp_ME_total_Separate += best_fp_ME_Separate


            print('subject: %s, f1: %f, epoch: %d, tp_mae: %d, tp_me: %d, fp_mae: %d, fp_me: %d' % (subject, best_f1, best_epoch, best_tp_MaE, best_tp_ME, best_fp_MaE, best_fp_ME))
        overall_F1 = 2 * (best_tp_MaE_total + best_tp_ME_total) / (best_tp_MaE_total + best_tp_ME_total + best_fp_MaE_total + best_fp_ME_total + mae_total + me_total)
        MaE_F1 = 2 * (best_tp_MaE_total) / (best_tp_MaE_total + best_fp_MaE_total + mae_total)
        ME_F1 = 2 * (best_tp_ME_total) / (best_tp_ME_total + best_fp_ME_total + me_total)

        MaE_F1_Separate = 2 * (best_tp_MaE_total_Separate) / (best_tp_MaE_total_Separate + best_fp_MaE_total_Separate + mae_total)
        ME_F1_Separate = 2 * (best_tp_ME_total_Separate) / (best_tp_ME_total_Separate + best_fp_ME_total_Separate + me_total)

        MaE_Recall = best_tp_MaE_total / mae_total
        ME_Recall = best_tp_ME_total / me_total
        MaE_Precision = best_tp_MaE_total / (best_tp_MaE_total + best_fp_MaE_total)
        ME_Precision = best_tp_ME_total / (best_tp_ME_total + best_fp_ME_total)
        print('overall: %f, MaE: %f, ME: %f, MaE_Separate: %f, ME_Separate: %f.' % (overall_F1, MaE_F1, ME_F1, MaE_F1_Separate, ME_F1_Separate))
        print('MaE_Recall: %f, MaE_Precision: %f, ME_Recall: %f, ME_Precision: %f.' % (MaE_Recall, MaE_Precision, ME_Recall, ME_Precision))
        print('overall_apex_diff_MaE: %f, overall_apex_diff_ME: %f, overall_apex_diff: %f.' % (best_apex_diff_MaE_total / best_tp_MaE_total, best_apex_diff_ME_total / best_tp_ME_total, (best_apex_diff_MaE_total + best_apex_diff_ME_total) / (best_tp_MaE_total+best_tp_ME_total)))
