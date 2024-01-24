import torch



@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)

    output = torch.cat(tensors_gather, dim=0)
    return output



class FreeMatchThresholdingHook:
    """
    SAT in FreeMatch
    """

    def __init__(self, num_classes, momentum=0.999, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.m = momentum

        self.p_model = torch.ones((self.num_classes)) / self.num_classes
        self.label_hist = torch.ones((self.num_classes)) / self.num_classes
        self.time_p = self.p_model.mean()

    @torch.no_grad()
    def update(self, probs_x_ulb):
        if torch.cuda.device_count() > 1:
            probs_x_ulb = concat_all_gather(probs_x_ulb)
        max_probs, max_idx = torch.max(probs_x_ulb, dim=1, keepdim=True)


        self.time_p = self.time_p * self.m + (1 - self.m) * max_probs.mean()

        # if algorithm.clip_thresh:
        #     self.time_p = torch.clip(self.time_p, 0.0, 0.95)

        self.p_model = self.p_model * self.m + (1 - self.m) * probs_x_ulb.mean(dim=[0,2,3])
        # cc1 = self.p_model * self.m
        # cc2 = (1 - self.m) * probs_x_ulb.mean(dim=0)
        # print("max_probs",max_probs.shape)
        # print("p_model:",self.p_model.shape)
        # print("self.p_model * self.m:",cc1.shape)
        # print("probs_x_ulb:",probs_x_ulb.shape)
        # print("probs_x_ulb.mean(dim=0):",probs_x_ulb.mean(dim=0).shape)
        # print("(1 - self.m) * probs_x_ulb.mean(dim=0):",cc2.shape)
        # hist = torch.bincount(max_idx.reshape(-1), minlength=self.p_model.shape[0]).to(self.p_model.dtype)
        # self.label_hist = self.label_hist * self.m + (1 - self.m) * (hist / hist.sum())

        # algorithm.p_model = self.p_model
        # algorithm.label_hist = self.label_hist
        # algorithm.time_p = self.time_p

    @torch.no_grad()
    def masking(self, logits_x_ulb, softmax_x_ulb=True, *args, **kwargs):

        if not self.p_model.is_cuda:
            self.p_model = self.p_model.to(logits_x_ulb.device)
        if not self.label_hist.is_cuda:
            self.label_hist = self.label_hist.to(logits_x_ulb.device)
        if not self.time_p.is_cuda:
            self.time_p = self.time_p.to(logits_x_ulb.device)

        if softmax_x_ulb:
            probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()

        self.update(probs_x_ulb)

        max_probs, max_idx = probs_x_ulb.max(dim=1)
        mod = self.p_model / torch.max(self.p_model, dim=-1)[0]
        mask = max_probs.ge(self.time_p * mod[max_idx]).to(max_probs.dtype)
        return mask, self.p_model, self.label_hist

