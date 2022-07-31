import torch
import torch.nn.functional as F
from mmdet.core import bbox_overlaps

from teter.core import cal_similarity
from ..builder import TRACKERS


@TRACKERS.register_module()
class TETerBDD(object):
    def __init__(
        self,
        init_score_thr=0.8,
        obj_score_thr=0.5,
        match_score_thr=0.5,
        memo_tracklet_frames=10,
        memo_backdrop_frames=1,
        memo_momentum=0.8,
        nms_conf_thr=0.5,
        nms_backdrop_iou_thr=0.3,
        nms_class_iou_thr=0.7,
        contrastive_thr=0.5,
        match_metric="bisoftmax",
    ):
        assert 0 <= memo_momentum <= 1.0
        assert memo_tracklet_frames >= 0
        assert memo_backdrop_frames >= 0
        self.init_score_thr = init_score_thr
        self.obj_score_thr = obj_score_thr
        self.match_score_thr = match_score_thr
        self.memo_tracklet_frames = memo_tracklet_frames
        self.memo_backdrop_frames = memo_backdrop_frames
        self.memo_momentum = memo_momentum
        self.nms_conf_thr = nms_conf_thr
        self.nms_backdrop_iou_thr = nms_backdrop_iou_thr
        self.nms_class_iou_thr = nms_class_iou_thr
        self.contrastive_thr = contrastive_thr

        assert match_metric in ["bisoftmax", "softmax", "cosine"]
        self.match_metric = match_metric

        self.num_tracklets = 0
        self.tracklets = dict()
        self.backdrops = []

    @property
    def empty(self):
        return False if self.tracklets else True

    def update_memo(self, ids, bboxes, embeds, cls_embeds, labels, frame_id):
        tracklet_inds = ids > -1

        # update memo
        for id, bbox, embed, cls_embed, label in zip(
            ids[tracklet_inds],
            bboxes[tracklet_inds],
            embeds[tracklet_inds],
            cls_embeds[tracklet_inds],
            labels[tracklet_inds],
        ):
            id = int(id)
            if id in self.tracklets.keys():
                velocity = (bbox - self.tracklets[id]["bbox"]) / (
                    frame_id - self.tracklets[id]["last_frame"]
                )
                self.tracklets[id]["bbox"] = bbox
                self.tracklets[id]["embed"] = (1 - self.memo_momentum) * self.tracklets[
                    id
                ]["embed"] + self.memo_momentum * embed
                self.tracklets[id]["cls_embeds"] = cls_embed
                self.tracklets[id]["last_frame"] = frame_id
                self.tracklets[id]["label"] = label
                self.tracklets[id]["velocity"] = (
                    self.tracklets[id]["velocity"] * self.tracklets[id]["acc_frame"]
                    + velocity
                ) / (self.tracklets[id]["acc_frame"] + 1)
                self.tracklets[id]["acc_frame"] += 1
            else:
                self.tracklets[id] = dict(
                    bbox=bbox,
                    embed=embed,
                    cls_embeds=cls_embed,
                    label=label,
                    last_frame=frame_id,
                    velocity=torch.zeros_like(bbox),
                    acc_frame=0,
                )

        backdrop_inds = torch.nonzero(ids == -1, as_tuple=False).squeeze(1)
        ious = bbox_overlaps(bboxes[backdrop_inds, :-1], bboxes[:, :-1])
        for i, ind in enumerate(backdrop_inds):
            if (ious[i, :ind] > self.nms_backdrop_iou_thr).any():
                backdrop_inds[i] = -1
        backdrop_inds = backdrop_inds[backdrop_inds > -1]

        self.backdrops.insert(
            0,
            dict(
                bboxes=bboxes[backdrop_inds],
                embeds=embeds[backdrop_inds],
                cls_embeds=cls_embeds[backdrop_inds],
                labels=labels[backdrop_inds],
            ),
        )

        # pop memo
        invalid_ids = []
        for k, v in self.tracklets.items():
            if frame_id - v["last_frame"] >= self.memo_tracklet_frames:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracklets.pop(invalid_id)

        if len(self.backdrops) > self.memo_backdrop_frames:
            self.backdrops.pop()

    @property
    def memo(self):
        memo_embeds = []
        memo_cls_embeds = []
        memo_ids = []
        memo_bboxes = []
        memo_labels = []
        memo_vs = []
        for k, v in self.tracklets.items():
            memo_bboxes.append(v["bbox"][None, :])
            memo_embeds.append(v["embed"][None, :])
            memo_cls_embeds.append(v["cls_embeds"][None, :])
            memo_ids.append(k)
            memo_labels.append(v["label"].view(1, 1))
            memo_vs.append(v["velocity"][None, :])
        memo_ids = torch.tensor(memo_ids, dtype=torch.long).view(1, -1)

        for backdrop in self.backdrops:
            backdrop_ids = torch.full(
                (1, backdrop["embeds"].size(0)), -1, dtype=torch.long
            )
            backdrop_vs = torch.zeros_like(backdrop["bboxes"])
            memo_bboxes.append(backdrop["bboxes"])
            memo_embeds.append(backdrop["embeds"])
            memo_cls_embeds.append(backdrop["cls_embeds"])
            memo_ids = torch.cat([memo_ids, backdrop_ids], dim=1)
            memo_labels.append(backdrop["labels"][:, None])
            memo_vs.append(backdrop_vs)

        memo_bboxes = torch.cat(memo_bboxes, dim=0)
        memo_embeds = torch.cat(memo_embeds, dim=0)
        memo_cls_embeds = torch.cat(memo_cls_embeds, dim=0)
        memo_labels = torch.cat(memo_labels, dim=0).squeeze(1)
        memo_vs = torch.cat(memo_vs, dim=0)
        return (
            memo_bboxes,
            memo_labels,
            memo_embeds,
            memo_cls_embeds,
            memo_ids.squeeze(0),
            memo_vs,
        )

    def match(
        self,
        bboxes,
        labels,
        embeds,
        cls_embeds,
        frame_id,
        temperature=-1,
        method="teter",
    ):

        _, inds = bboxes[:, -1].sort(descending=True)
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        embeds = embeds[inds, :]
        cls_embeds = cls_embeds[inds, :]

        # duplicate removal for potential backdrops and cross classes
        valids = bboxes.new_ones((bboxes.size(0)))
        ious = bbox_overlaps(bboxes[:, :-1], bboxes[:, :-1])
        for i in range(1, bboxes.size(0)):
            thr = (
                self.nms_backdrop_iou_thr
                if bboxes[i, -1] < self.obj_score_thr
                else self.nms_class_iou_thr
            )
            # remove duplicate
            if (ious[i, :i] > thr).any():
                valids[i] = 0
        valids = valids == 1
        bboxes = bboxes[valids, :]
        labels = labels[valids]
        embeds = embeds[valids, :]
        cls_embeds = cls_embeds[valids, :]

        # init ids container
        ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)

        # match if buffer is not empty
        if bboxes.size(0) > 0 and not self.empty:
            (
                memo_bboxes,
                memo_labels,
                memo_embeds,
                memo_cls_embeds,
                memo_ids,
                memo_vs,
            ) = self.memo
            # iou_sims = bbox_overlaps(bboxes[:,:-1], memo_bboxes[:,:-1])

            if self.match_metric == "bisoftmax":
                feats = torch.mm(embeds, memo_embeds.t())
                d2t_scores = feats.softmax(dim=1)
                t2d_scores = feats.softmax(dim=0)
                scores = (d2t_scores + t2d_scores) / 2
            elif self.match_metric == "softmax":
                feats = torch.mm(embeds, memo_embeds.t())
                scores = feats.softmax(dim=1)
            elif self.match_metric == "cosine":
                scores = torch.mm(
                    F.normalize(embeds, p=2, dim=1),
                    F.normalize(memo_embeds, p=2, dim=1).t(),
                )
            else:
                raise NotImplementedError

            if method == "teter":
                cls_sims = cal_similarity(
                    cls_embeds,
                    memo_cls_embeds,
                    method="cosine",
                    temperature=temperature,
                )
                cat_same = cls_sims > self.contrastive_thr
                scores *= cat_same.float().to(scores.device)

                for i in range(bboxes.size(0)):
                    conf, memo_ind = torch.max(scores[i, :], dim=0)
                    # conf_iou = iou_sims[i, memo_ind]
                    id = memo_ids[memo_ind]
                    if conf > self.match_score_thr:
                        if id > -1:
                            if bboxes[i, -1] > self.obj_score_thr:
                                ids[i] = id
                                scores[:i, memo_ind] = 0
                                scores[i + 1 :, memo_ind] = 0
                            else:
                                if conf > self.nms_conf_thr:
                                    ids[i] = -2
            else:
                raise NotImplementedError

        new_inds = (ids == -1) & (bboxes[:, 4] > self.init_score_thr).cpu()
        num_news = new_inds.sum()
        ids[new_inds] = torch.arange(
            self.num_tracklets, self.num_tracklets + num_news, dtype=torch.long
        )
        self.num_tracklets += num_news

        self.update_memo(ids, bboxes, embeds, cls_embeds, labels, frame_id)

        return bboxes, labels, ids
