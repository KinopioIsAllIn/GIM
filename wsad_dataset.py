from __future__ import print_function
import numpy as np
import utils.wsad_utils as utils
import random
import os
import options
from torch.utils import data
import torch


class SampleDataset(data.Dataset):
    def __init__(self, args, split, mode="both", sampling='random', getproposal=False):
        super().__init__()
        self._split = split
        self.getproposal=getproposal
        self.dataset_name = args.dataset_name
        self.num_class = args.num_class
        self.sampling = sampling
        self.num_segments = args.max_seqlen
        self.feature_size = args.feature_size
        self.path_to_features = os.path.join(args.path_dataset, self.dataset_name + "-SLopticalflow.npy")
        self.path_to_annotations = os.path.join(args.path_dataset, self.dataset_name + "-Annotations/")
        self.validation_subject = args.validation_subject
        self.features = np.load(
            self.path_to_features, encoding="bytes", allow_pickle=True
        )

        self.max_length = np.max(np.array([feature.shape[0] for feature in self.features]))

        self.segments = np.load(
            self.path_to_annotations + "segments.npy", allow_pickle=True
        )
        self.labels = np.load(
            self.path_to_annotations + "labels_all.npy", allow_pickle=True
        )
        self.labeled_frames = np.load(
            self.path_to_annotations + "labels_gau_frame.npy", allow_pickle=True
        )
        self._labels = np.load(
            self.path_to_annotations + "labels.npy", allow_pickle=True
        )
        self.classlist = np.load(
            self.path_to_annotations + "classlist.npy", allow_pickle=True
        )
        self.videonames = np.load(
            self.path_to_annotations + "videoname.npy", allow_pickle=True
        )
        self.batch_size = args.batch_size
        self.trainidx = []
        self.testidx = []
        self.train_test_idx()

        self.classwiseidx = []
        self.currenttestidx = 0
        self.labels_multihot = [
            utils.strlist2multihot(labs, self.classlist)
            for labs in self.labels
        ]

        self.normalize = False
        self.mode = mode

    def __len__(self):
        return len(self.features)

    def train_test_idx(self):

        for i, s in enumerate(self.videonames):
            if not self.getproposal:

                if self.dataset_name == 'CASME3':
                    if s.split('_')[0] == str(self.validation_subject):
                        self.testidx.append(i)
                    else:
                        self.trainidx.append(i)
                else:
                    if s.startswith(self.validation_subject):
                        self.testidx.append(i)
                    else:
                        self.trainidx.append(i)

            else:
                if self.dataset_name == 'CASME3':
                    if s.split('_')[0] == str(self.validation_subject):
                        self.trainidx.append(i)
                    else:
                        self.testidx.append(i)
                else:
                    if s.startswith(self.validation_subject):
                        self.trainidx.append(i)
                    else:
                        self.testidx.append(i)

        if self._split == 'test':
            self.features = self.features[np.array(self.testidx)]
            self.segments = self.segments[np.array(self.testidx)]
            self.labels = self.labels[np.array(self.testidx)]
            self.labeled_frames = self.labeled_frames[np.array(self.testidx)]
            self._labels = self._labels[np.array(self.testidx)]
            self.videonames = self.videonames[np.array(self.testidx)]

        else:
            self.features = self.features[np.array(self.trainidx)]
            self.segments = self.segments[np.array(self.trainidx)]
            self.labels = self.labels[np.array(self.trainidx)]
            self.labeled_frames = self.labeled_frames[np.array(self.trainidx)]
            self._labels = self._labels[np.array(self.trainidx)]
            self.videonames = self.videonames[np.array(self.trainidx)]

    def __getitem__(self, index):
        if self._split == 'train':
            ifeat = self.features[index]
            vn = self.videonames[index]
            length, s1, s2, s3 = ifeat.shape
            ifeat = np.concatenate((ifeat, np.zeros((self.max_length - length, s1, s2, s3))))

            frame_label = np.zeros(self.max_length)

            j = 0
            for frame_index in self.labeled_frames[index]:
                frame_label[frame_index] = 1 if self._labels[index][j] == 'Macro' else 2
                j += 1

            labels = self.labels_multihot[index]

            ifeat = torch.tensor(ifeat).float()
            labels = torch.tensor(labels, dtype=torch.int64)
            frame_label = torch.tensor(frame_label, dtype=torch.int64)

            return ifeat, labels, frame_label, torch.tensor(length, dtype=torch.int64), vn
        else:
            ifeat = self.features[index]
            vn = self.videonames[index]
            length, s1, s2, s3 = ifeat.shape
            frame_label = np.zeros(self.max_length)

            j = 0
            for frame_index in self.labeled_frames[index]:
                frame_label[frame_index] = 1 if self._labels[index][j] == 'Macro' else 2
                j += 1

            ifeat = torch.tensor(ifeat).float()
            frame_label = torch.tensor(frame_label, dtype=torch.int64)

            return ifeat, frame_label, torch.tensor(length, dtype=torch.int64), vn


    def load_data(self, n_similar=0, is_training=True, similar_size=2):
        if is_training:
            labels = []
            idx = []

            # Load similar pairs
            if n_similar != 0:
                rand_classid = np.random.choice(
                    len(self.classwiseidx), size=n_similar
                )
                for rid in rand_classid:
                    rand_sampleid = np.random.choice(
                        len(self.classwiseidx[rid]),
                        size=similar_size,
                        replace=False,
                    )

                    for k in rand_sampleid:
                        idx.append(self.classwiseidx[rid][k])

            # Load rest pairs
            if self.batch_size - similar_size * n_similar < 0:
                self.batch_size = similar_size * n_similar

            rand_sampleid = np.random.choice(
                len(self.trainidx),
                size=self.batch_size - similar_size * n_similar,
            )

            for r in rand_sampleid:
                idx.append(self.trainidx[r])
            feat = []

            print(idx)
            for i in idx:
                ifeat = self.features[i]
                length, s1, s2, s3 = ifeat.shape
                ifeat = np.concatenate((ifeat, np.zeros((self.max_length - length, s1, s2, s3))))
                feat.append(ifeat)

                ##########################################################################
                frame_label = np.zeros(self.max_length)
                # print(self.labeled_frames[i], self._labels[i], self.segments[i])
                for j, index in enumerate(self.labeled_frames[i]):
                    frame_label[index] = 1 if self._labels[i][j] == 'Macro' else 2
                frame_label[length:] = -1  # 超出视频范围的设为-1, 不参与损失函数的计算

            feat = np.array(feat)
            labels = np.array([self.labels_multihot[i] for i in idx])
            return feat, labels, rand_sampleid

        else:
            labs = self.labels_multihot[self.testidx[self.currenttestidx]]
            feat = self.features[self.testidx[self.currenttestidx]]
            vn = self.videonames[self.testidx[self.currenttestidx]]
            if self.currenttestidx == len(self.testidx) - 1:
                done = True
                self.currenttestidx = 0
            else:
                done = False
                self.currenttestidx += 1
            feat = np.array(feat)
            return feat, np.array(labs), vn, done