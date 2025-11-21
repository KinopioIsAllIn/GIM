import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        # torch_init.xavier_uniform_(m.weight)
        # import pdb
        # pdb.set_trace()
        torch_init.kaiming_uniform_(m.weight)
        if type(m.bias) != type(None):
            m.bias.data.fill_(0)


def _focal_loss(output, label, gamma, alpha, lb_smooth):
    output = output.contiguous().view(-1)
    label = label.view(-1)
    mask_class = (label > 0).float() # 正样本
    mask_class1 = (label == 0).float() # 负样本

    # print('mask_class:')
    # print(mask_class)
    # print(mask_class1)

    c_1 = alpha  # 0.99
    c_0 = 1 - c_1  # 0.01
    loss = ((c_1 * torch.abs(label - output) ** gamma * mask_class
             * torch.log(output + 0.00001))
            + (c_0 * torch.abs(label + lb_smooth - output) ** gamma
               * (mask_class1)
               * torch.log(1.0 - output + 0.00001)))
    loss = -torch.mean(loss)
    return loss


def _probability_loss(output, score, gamma, alpha, lb_smooth):
    # output = torch.sigmoid(output)
    loss = _focal_loss(output, score, gamma, alpha, lb_smooth)
    return loss


class SupConLossWithWeight(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.5, contrast_mode='one',
                 base_temperature=1):
        super(SupConLossWithWeight, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, intensity=None, device=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)



        intensity = intensity.contiguous().view(-1, 1)

        intensity_diff = torch.abs(intensity - intensity.T)

        weight_matrix = torch.ones_like(mask).to(device)

        condition_1 = (labels == 0) & ((labels.T == 1) | (labels.T == 2))
        condition_2 = ((labels == 1) | (labels == 2)) & (labels.T == 0)
        weight_matrix[condition_1 | condition_2] = intensity_diff[condition_1 | condition_2]

        condition_3 = ((labels == 1) & (labels.T == 1)) | ((labels == 2) & (labels.T == 2))
        weight_matrix[condition_3] = 1 - intensity_diff[condition_3]

        condition_4 = ((labels == 1) & (labels.T == 2))
        condition_5 = ((labels == 2) & (labels.T == 1))
        weight_matrix[condition_4 | condition_5] = intensity_diff[condition_4 | condition_5]

        # gamma = 2
        # weight_matrix = weight_matrix ** gamma

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        # logits = anchor_dot_contrast
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # 所有的相似度，也就是分母

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)  # mask是所有正样本对的mask
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask  # mask是所有正样本对除了自身。

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask * weight_matrix  #所有样本对的相似度除了自身 分母

        log_prob = torch.log(weight_matrix + 1e-6) + logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)  # 用所有的样本对 - 指数和
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)  # 用所有的样本对 - 指数和

        # 再过滤掉负样本对
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+ 1e-6)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class CO2(torch.nn.Module):
    def __init__(self, n_feature, n_class, **args):
        super().__init__()

        dropout_rate = 0.7

        self.attn_feature_extractor = nn.Sequential(nn.Conv1d(256, 256, 1),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(dropout_rate)
                                       )

        self.attention = nn.Sequential(nn.Conv1d(256, 1, 1),
                                       nn.Dropout(dropout_rate),
                                       nn.Sigmoid()
                                       )

        self.apex_classifier = nn.Sequential(nn.Conv1d(256, 2, 1),
                                        nn.Dropout(dropout_rate),
                                        nn.Sigmoid()
                                        )

        self.apply(weights_init)

    def forward(self, inputs, is_training=True, **args):
        feat = inputs.transpose(-1, -2)

        attn_feat = self.attn_feature_extractor(feat)
        x_atn = self.attention(attn_feat)

        x_apex = self.apex_classifier(feat)

        return {'feat': feat.transpose(-1, -2), 'attn': x_atn.transpose(-1, -2),
                'attn_feat':attn_feat.transpose(-1, -2), 'apex': x_apex.transpose(-1, -2)}#, 'class_feat': class_feat.transpose(-1, -2)}

    def criterion(self, itr, outputs, labels, labels_frame, video_mask, **args):

        feat, element_atn, attn_feat, element_apex = \
            outputs['feat'], outputs['attn'], outputs['attn_feat'], outputs['apex']
        device = args['device']

        if itr < 1:
            pseudo_label, pseudo_apexes = self.soft_pseudo_label_attention_individual_init3(feat, element_atn, labels_frame, video_mask, device)
        elif itr < 5:
            pseudo_label, pseudo_apexes = self.soft_pseudo_label_attention_individual_medium(itr, attn_feat, element_atn, labels_frame, video_mask, device)
        else:
            pseudo_label, pseudo_apexes = self.soft_pseudo_label_attention_individual_adaptive(itr, attn_feat, element_atn, labels_frame, video_mask, device)

        loss_frame = self.soft_frame_loss(itr, feat, element_atn, element_apex, pseudo_label, pseudo_apexes, labels, None, video_mask, False, device, args['opt'].dataset_name)  # 改成了1-atn*neutral


        loss_norm = element_atn.mean()

        if args['opt'].dataset_name == 'SAMM':
            ha = 0.3
        elif args['opt'].dataset_name == 'CASME2':
            ha = 2.5
        else:
            ha = 2.5

        total_loss = loss_frame + ha * loss_norm

        return total_loss

    def soft_pseudo_label_attention_individual_init3(self, feat, attention, labels_frame, video_mask, device):
        """
        · calculate an average feature vector for each BATCH using labeled frames / using highest score;
        · calculate the standard deviation in the general duration;
        · calculate the soft0625 score of each frame in the general duration using mean and variance;

        :param feat: [b, T, C]
        :param attention: [b, T, 1]
        :param labels_frame: [b, T, 1] 0-neutral 1-macro 2-micro
        :param video_mask: [b]
        :return: pseudo label
        """
        pseudo_labels = []

        for i in range(feat.shape[0]):  # i is the index of videos
            feat_video = feat[i, :video_mask[i]]
            label = labels_frame[i, :video_mask[i]]

            pseudo_label = torch.zeros((feat_video.shape[0], 3), dtype=torch.float32).to(device)

            macro_indices = torch.nonzero(label == 1, as_tuple=False)
            micro_indices = torch.nonzero(label == 2, as_tuple=False)
            for ind in macro_indices:
                # find the pseudo apex, centered at apex, assign pseudo label.
                start_index = max(torch.tensor(0).to(device), ind - 5)
                end_index = min(video_mask[i].clone().detach() - 1, ind + 5)
                indices = torch.arange(start_index.item(), end_index.item())
                pseudo_label[indices, 1] = 1.0

            for ind in micro_indices:
                start_index = max(torch.tensor(0).to(device), ind - 3)
                end_index = min(video_mask[i].clone().detach() - 1, ind + 3)
                indices = torch.arange(start_index.item(), end_index.item())
                pseudo_label[indices, 2] = 1.0

            pseudo_labels.append(pseudo_label)
        return pseudo_labels, None

    def soft_pseudo_label_attention_individual_medium(self, itr, feat, attention, labels_frame, video_mask, device):

        pseudo_labels = []

        if itr < 3:
            duration = 1
        elif itr < 5:
            duration = 2

        for i in range(feat.shape[0]):
            feat_video = feat[i, :video_mask[i]]
            label = labels_frame[i, :video_mask[i]]

            pseudo_label = torch.zeros((feat_video.shape[0], 3), dtype=torch.float32).to(device)

            macro_indices = torch.nonzero(label == 1, as_tuple=False)
            micro_indices = torch.nonzero(label == 2, as_tuple=False)
            for ind in macro_indices:
                mean = feat_video[ind]
                start_index = max(torch.tensor(0).to(device), ind - duration * 2)
                end_index = min(video_mask[i].clone().detach() - 1, ind + duration * 2 + 1) # 10
                indices = torch.arange(start_index.item(), end_index.item())

                variance = torch.mean(torch.sum((feat_video[indices] - mean) ** 2, dim=-1))

                pseudo_label[indices, 1] = torch.exp(-torch.sum((feat_video[indices] - mean) ** 2, dim=-1) / (
                                                                 2 * variance + 1e-6))

            for ind in micro_indices:
                mean = feat_video[ind]
                start_index = max(torch.tensor(0).to(device), ind - duration) # 5
                end_index = min(video_mask[i].clone().detach() - 1, ind + duration+1)
                indices = torch.arange(start_index.item(), end_index.item())

                variance = torch.mean(torch.sum((feat_video[indices] - mean) ** 2, dim=-1))

                pseudo_label[indices, 2] = torch.exp(-torch.sum((feat_video[indices] - mean) ** 2, dim=-1) / (
                                                                 2 * variance + 1e-6))

            pseudo_labels.append(pseudo_label)
        return pseudo_labels, None

    def soft_pseudo_label_attention_individual_adaptive(self, itr, feat, attention, labels_frame, video_mask, device):
        pseudo_labels = []
        pseudo_apexes = []

        if itr < 29:
            reliable_thr = 0.8 - (itr - 5) / 80  # 80
        else:
            reliable_thr = 0.5  # 0.5

        for i in range(feat.shape[0]):
            feat_video = feat[i, :video_mask[i]]
            label = labels_frame[i, :video_mask[i]]

            pseudo_label = torch.zeros((feat_video.shape[0], 3), dtype=torch.float32).to(device)
            pseudo_apex = torch.zeros((feat_video.shape[0], 2), dtype=torch.float32).to(device)

            macro_indices = torch.nonzero(label == 1, as_tuple=False)
            micro_indices = torch.nonzero(label == 2, as_tuple=False)

            k_c_mae = 32
            for ind in macro_indices:
                # find the pseudo apex
                start_index = max(torch.tensor(0).to(device), ind - k_c_mae//4)  # 8
                end_index = min(video_mask[i].clone().detach() - 1, ind + k_c_mae//4+1)
                indices = torch.arange(start_index.item(), end_index.item())
                pseudo_apex_ind = torch.argmax(attention[i, indices, 0]) + start_index
                mean = feat_video[pseudo_apex_ind] # save apex feature as mean

                start_index = max(torch.tensor(0).to(device), pseudo_apex_ind - k_c_mae//2)
                end_index = min(video_mask[i].clone().detach() - 1, pseudo_apex_ind + k_c_mae//2 + 1)
                indices = torch.arange(start_index.item(), end_index.item())

                reliable_expression_frames = attention[i, indices, 0] > reliable_thr
                num_reliable_expression_frames = len(torch.nonzero(reliable_expression_frames, as_tuple=False))
                gaussian_duration = min(36, max(k_c_mae//2, round(num_reliable_expression_frames * 1.2))) // 2

                start_index = max(torch.tensor(0).to(device), pseudo_apex_ind - gaussian_duration)
                end_index = min(video_mask[i].clone().detach() - 1, pseudo_apex_ind + gaussian_duration + 1)
                indices = torch.arange(start_index.item(), end_index.item())

                variance = torch.mean(torch.sum((feat_video[indices] - mean) ** 2, dim=-1))

                pseudo_label[indices, 1] = torch.exp(-torch.sum((feat_video[indices] - mean) ** 2, dim=-1) / (2 * variance + 1e-6))

                pseudo_apex[max(pseudo_apex_ind - 2, 0): min(pseudo_apex_ind + 3, feat_video.shape[0]), 0] = 1

            k_c_me = 16
            for ind in micro_indices:
                # find the pseudo apex
                start_index = max(torch.tensor(0).to(device), ind - k_c_me//4)
                end_index = min(video_mask[i].clone().detach() - 1, ind + k_c_me//4 + 1)
                indices = torch.arange(start_index.item(), end_index.item())
                pseudo_apex_ind = torch.argmax(attention[i, indices, 0]) + start_index
                mean = feat_video[pseudo_apex_ind]  # save apex feature as mean

                start_index = max(torch.tensor(0).to(device), pseudo_apex_ind - k_c_me//2)
                end_index = min(video_mask[i].clone().detach() - 1, pseudo_apex_ind + k_c_me//2 + 1)
                indices = torch.arange(start_index.item(), end_index.item())

                reliable_expression_frames = attention[i, indices, 0] > reliable_thr
                num_reliable_expression_frames = len(torch.nonzero(reliable_expression_frames, as_tuple=False))
                gaussian_duration = min(k_c_me, max(k_c_me//2, round(num_reliable_expression_frames * 1.2))) // 2

                start_index = max(torch.tensor(0).to(device), pseudo_apex_ind - gaussian_duration)
                end_index = min(video_mask[i].clone().detach() - 1, pseudo_apex_ind + gaussian_duration + 1)
                indices = torch.arange(start_index.item(), end_index.item())

                variance = torch.mean(torch.sum((feat_video[indices] - mean) ** 2, dim=-1))

                pseudo_label[indices, 2] = torch.exp((-torch.sum((feat_video[indices] - mean) ** 2, dim=-1)) / (2 * variance + 1e-6))

                pseudo_apex[max(pseudo_apex_ind - 1, 0): min(pseudo_apex_ind + 2, feat_video.shape[0]), 1] = 1


            pseudo_labels.append(pseudo_label)
            pseudo_apexes.append(pseudo_apex)
        return pseudo_labels, pseudo_apexes


    def soft_frame_loss(self, itr, feat, attention, apex_scores, pseudo_labels, pseudo_apexes, labels, element_logits, video_mask, withback, device, dataset_name):
        smooth_loss = 0
        milloss = 0

        for i in range(feat.shape[0]):

            smooth_loss += torch.mean(attention[i, :video_mask[i], 0][1:] - attention[i, :video_mask[i], 0][:video_mask[i]-1])

            pseudo_label = pseudo_labels[i].detach()

            # negative sampling
            exp_mask = pseudo_label.sum(dim=-1) == 0
            num_pseudo_expression_frame = len(torch.nonzero(pseudo_label.sum(dim=-1) > 0.0, as_tuple=False))
            num_frame_left = video_mask[i] - num_pseudo_expression_frame
            topk_normal, topk_ind = torch.topk(
                (1 - attention)[i, :video_mask[i], 0][exp_mask],
                k=max(1, min(num_frame_left, num_pseudo_expression_frame)), dim=0)
            negative_indices = torch.nonzero(exp_mask)[topk_ind]

            num_negative = len(negative_indices)

            positive_macro_indices = pseudo_label[:, 1] > 0.0
            num_positive_macro = len(torch.nonzero(positive_macro_indices, as_tuple=False))

            positive_micro_indices = pseudo_label[:, 2] > 0.0
            num_positive_micro = len(torch.nonzero(positive_micro_indices, as_tuple=False))

            attention_loss = 0

            if num_positive_macro > 0:
                attention_loss += F.mse_loss(attention[i, :video_mask[i], 0][positive_macro_indices], pseudo_label[:video_mask[i], 1][positive_macro_indices], reduction='mean')
            if num_positive_micro > 0:
                attention_loss += 1.0 * F.mse_loss(attention[i, :video_mask[i], 0][positive_micro_indices], pseudo_label[:video_mask[i], 2][positive_micro_indices], reduction='mean')
            if num_negative > 0:
                attention_loss += F.mse_loss(attention[i, :video_mask[i], 0][negative_indices], torch.zeros_like(attention[i, :video_mask[i], 0][negative_indices]), reduction='mean')

            positive_macro_indices = pseudo_label[:, 1] > 0.5

            positive_micro_indices = pseudo_label[:, 2] > 0.5

            reward_loss = - 0.4 * attention[i, :video_mask[i], 0][positive_macro_indices].mean() - 0.6 * \
                          attention[i, :video_mask[i], 0][positive_micro_indices].mean()

            milloss += attention_loss + 0.1 * smooth_loss + reward_loss

            if itr > 4:
                pseudo_apex = pseudo_apexes[i].detach()
                apex_loss = 0.0
                window_size = 6
                kernel = torch.ones(1, 1, 2 * window_size + 1).to(device)
                binary_labels = (pseudo_label > 0.5).any(dim=1).float().unsqueeze(0).unsqueeze(0)
                expanded_labels = F.conv1d(binary_labels, kernel, padding=window_size)
                mask = (expanded_labels > 0).squeeze()

                positive_apex_macro_indices = (pseudo_apex[:, 0] > 0)
                apex_macro_label = torch.ones_like(attention[i, :video_mask[i], 0]) * -1
                apex_macro_label[mask] = 0.0
                apex_macro_label[negative_indices] = 0.0
                apex_macro_label[positive_apex_macro_indices] = 1 # pseudo_label[positive_apex_macro_indices, 1] # 1

                positive_apex_micro_indices = (pseudo_apex[:, 1] > 0)
                apex_micro_label = torch.ones_like(attention[i, :video_mask[i], 0]) * -1
                apex_micro_label[mask] = 0.0
                apex_micro_label[negative_indices] = 0.0
                apex_micro_label[positive_apex_micro_indices] = 1 # pseudo_label[positive_apex_micro_indices, 2] # 1

                # if num_apex_positive_macro > 0:
                apex_loss += _probability_loss(apex_scores[i, :video_mask[i], 0], apex_macro_label, 1, 0.90, 0.25)
                # if num_apex_positive_micro > 0:
                apex_loss += 1.8 * _probability_loss(apex_scores[i, :video_mask[i], 1], apex_micro_label, 1, 0.93, 0.25)

                milloss += apex_loss

            contrastive_loss = SupConLossWithWeight(temperature=0.5, base_temperature=0.07)
            action_labels = torch.ones(video_mask[i]).to(device) * -1
            action_labels[negative_indices] = 0
            action_labels[positive_micro_indices] = 2
            action_labels[positive_macro_indices] = 1
            con_indices = action_labels > -1

            pseudo_intensity = torch.ones(video_mask[i]).to(device) * -1
            pseudo_intensity[con_indices] = torch.max(pseudo_label[con_indices], dim=-1)[0]

            if i == 0:
                batch_action_labels = action_labels[con_indices]
                batch_feat = feat[i, :video_mask[i]][con_indices].reshape(-1, 1, feat.shape[-1])
                batch_intensity = pseudo_intensity[con_indices]
            else:
                batch_action_labels = torch.concat((batch_action_labels, action_labels[con_indices]), dim=0)
                batch_feat = torch.concat((batch_feat, feat[i, :video_mask[i]][con_indices].reshape(-1, 1, feat.shape[-1])), dim=0)
                batch_intensity = torch.concat((batch_intensity, pseudo_intensity[con_indices]), dim=0)

            loss_contrastive = contrastive_loss(features=batch_feat, labels=batch_action_labels, intensity=batch_intensity, device=device)


        if dataset_name == 'SAMM':
            coef = 0.000002
        else:
            coef = 0.000014 # 0.00004
        return milloss / feat.shape[0] + coef * loss_contrastive
