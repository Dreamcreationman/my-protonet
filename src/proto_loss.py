import torch


def ProtoLoss(model_out, gt_label, n_support):
    """
    Compute Loss between model output and ground truth label
    :param model_out: ((num_spt + num_qry)*class_per_epi_tr) × embed_dim
    :param gt_label: ((num_spt + num_qry)*class_per_epi_tr)
    :param n_support: number of samples in support set when training or validating
    :return:
    """
    def supp_idxs(c):
        return gt_label.eq(c).nonzero()[:n_support].squeeze(1)

    classes = torch.unique(gt_label)
    n_classes = len(classes)
    n_query = gt_label.eq(classes[0].item()).sum().item() - n_support

    support_idx = list(map(supp_idxs, classes))
    prototypes = torch.stack([model_out[idx_list].mean(0) for idx_list in support_idx])
    query_idxs = torch.stack(list(map(lambda c: gt_label.eq(c).nonzero()[n_support:], classes))).view(-1)
    dist = euclidean_dis(model_out[query_idxs], prototypes)
    log_p_y = torch.log_softmax(-dist, dim=1).view(n_classes, n_query, -1)
    target_idxs = torch.arange(0, n_classes).view(n_classes, 1, 1).expand(n_classes, n_query, 1).long().cuda()
    loss_val = -log_p_y.gather(2, target_idxs).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_idxs.squeeze()).float().mean()
    return loss_val, acc_val


def euclidean_dis(x, y):
    """
    Distance Metrics of each vector in x over those in y
    :param x: Query Matrix with num_query × embed_dim
    :param y: support Matrix with num_class × embed_dim
    :return: output with num_query × num_class
    """
    nq = x.size(0)
    nc = y.size(0)
    emdim = x.size(1)

    if x.size(1) != y.size(1):
        raise ValueError("The last dim must match between query matrix and support matrix"
                         + " but found {} and {}".format(x.size(1), y.size(1)))

    x = x.unsqueeze(1).expand(nq, nc, emdim)
    y = y.unsqueeze(0).expand(nq, nc, emdim)
    return torch.pow(x - y, 2).sum(2)
