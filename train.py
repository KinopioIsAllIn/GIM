import torch

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def train(itr, dataset, args, spotformer, model, optimizer, logger, device):
    model.train()
    spotformer.train()

    total_loss1 = 0.0
    bn = 0

    for batch_idx, (features, labels, labels_frame, video_mask, vn) in enumerate(dataset):
        index = [1, 2, 3, 5, 6, 7, 9, 10, 11, 12]
        features = features[:, :, :, index, :]

        features = features.to(device)
        features = spotformer(features)

        labels = labels.to(device)
        labels_frame = labels_frame.to(device)
        video_mask = video_mask.to(device)

        outputs = model(features, is_training=True, itr=itr, opt=args)
        total_loss = model.criterion(itr, outputs, labels, labels_frame, video_mask, device=device, logger=logger, opt=args, pairs_id=labels_frame, inputs=features)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        total_loss = torch.nan_to_num(total_loss.detach(), nan=0.0)
        total_loss1 += total_loss.data.cpu().numpy()
        bn += 1
    return total_loss1 / bn

