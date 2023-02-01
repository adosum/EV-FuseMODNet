import torch


def compute_event_flow_loss(events, flow_dict):
    # TODO: move device
    if flow_dict.is_cuda:
        events = events.cuda()
    idx = events[:, 0]
    xs = events[:, 1]
    ys = events[:, 2]
    ts = events[:, 3]
    ps = events[:, 4]

    loss_weight_sum = 0.
    total_event_loss = []
    B, C, H, W = flow_dict.size()
    eps = 1e-12
    for batch_idx in range(B):
        x = xs[torch.where(idx == batch_idx)]
        y = ys[torch.where(idx == batch_idx)]
        t = ts[torch.where(idx == batch_idx)]
        p = ps[torch.where(idx == batch_idx)]
        # for flow_idx in range(len(flow_dict)):
        #     flow = flow_dict["flow{}".format(flow_idx)][batch_idx]
        flow = flow_dict[batch_idx, ...]
        neg_mask = p == 0
        pos_mask = p == 1
        t = (t - t[0]) / (t[-1] - t[0] + eps)

        # Resize the event image to match the flow dimension
        # x_ = x / 2 ** (3 - flow_idx)
        # y_ = y / 2 ** (3 - flow_idx)
        x_ = x
        y_ = y
        # Positive events
        xp = x_[pos_mask].type(torch.long)
        yp = y_[pos_mask].type(torch.long)
        tp = t[pos_mask].type(torch.float)

        # Negative events
        xn = x_[neg_mask].type(torch.long)
        yn = y_[neg_mask].type(torch.long)
        tn = t[neg_mask].type(torch.float)

        # # Timestamp for {Forward, Backward} x {Postive, Negative}
        # t_fp = tp[-1] - tp   # t[-1] should be 1
        # t_bp = tp[0]  - tp   # t[0] should be 0
        # t_fn = tn[-1] - tn
        # t_bn = tn[0]  - tn

        # fp_loss = event_loss((xp, yp, t_fp), flow)
        # bp_loss = event_loss((xp, yp, t_bp), flow)
        # fn_loss = event_loss((xn, yn, t_fn), flow)
        # bn_loss = event_loss((xn, yn, t_bn), flow)

        fp_loss = event_loss((xp, yp, tp), flow, forward=True)
        bp_loss = event_loss((xp, yp, tp), flow, forward=False)
        fn_loss = event_loss((xn, yn, tn), flow, forward=True)
        bn_loss = event_loss((xn, yn, tn), flow, forward=False)

        loss_weight_sum += 4
        total_event_loss.append(( fp_loss + bp_loss + fn_loss + bn_loss))
            # total_event_loss += fp_loss + bp_loss

    # total_event_loss /= loss_weight_sum
    return total_event_loss


def event_loss(events, flow, forward=True):
    eps = 1e-12
    H, W = flow.shape[1:]
    # x, y, t = events_average(events, (H, W))
    x, y, t = events

    # Estimate events position after flow
    if forward:
        t_ = t[-1] - t + eps
    else:
        t_ = t[0] - t - eps

    x_ = torch.clamp(x + t_ * flow[0, y, x], min=0, max=W - 1)
    y_ = torch.clamp(y + t_ * flow[1, y, x], min=0, max=H - 1)

    x0 = torch.floor(x_)
    x1 = torch.ceil(x_)
    y0 = torch.floor(y_)
    y1 = torch.ceil(y_)

    # Interpolation ratio
    x0_ratio = 1 - (x_ - x0)
    x1_ratio = 1 - (x1 - x_)
    y0_ratio = 1 - (y_ - y0)
    y1_ratio = 1 - (y1 - y_)

    Ra = x0_ratio * y0_ratio
    Rb = x1_ratio * y0_ratio
    Rc = x0_ratio * y1_ratio
    Rd = x1_ratio * y1_ratio

    # R_sum = Ra + Rb + Rc + Rd
    # Ra /= R_sum
    # Rb /= R_sum
    # Rc /= R_sum
    # Rd /= R_sum

    # Prevent R and T to be zero
    Ra = Ra + eps
    Rb = Rb + eps
    Rc = Rc + eps
    Rd = Rd + eps

    Ta = Ra * t_
    Tb = Rb * t_
    Tc = Rc * t_
    Td = Rd * t_

    # Ta = Ta+eps; Tb = Tb+eps; Tc = Tc+eps; Td = Td+eps

    # Calculate interpolation flatterned index of 4 corners for all events
    Ia = (x0 + y0 * W).type(torch.long)
    Ib = (x1 + y0 * W).type(torch.long)
    Ic = (x0 + y1 * W).type(torch.long)
    Id = (x1 + y1 * W).type(torch.long)

    # Compute the nominator and denominator
    numerator = torch.zeros((W * H), dtype=flow.dtype, device=flow.device)
    denominator = torch.zeros((W * H), dtype=flow.dtype, device=flow.device)

    # denominator.index_add_(0, Ia, Ra)
    # denominator.index_add_(0, Ib, Rb)
    # denominator.index_add_(0, Ic, Rc)
    # denominator.index_add_(0, Id, Rd)

    denominator.index_add_(0, Ia, torch.ones_like(Ra))
    denominator.index_add_(0, Ib, torch.ones_like(Rb))
    denominator.index_add_(0, Ic, torch.ones_like(Rc))
    denominator.index_add_(0, Id, torch.ones_like(Rd))

    numerator.index_add_(0, Ia, Ta)
    numerator.index_add_(0, Ib, Tb)
    numerator.index_add_(0, Ic, Tc)
    numerator.index_add_(0, Id, Td)

    loss = (numerator / (denominator + eps)) ** 2
    return loss.sum()
