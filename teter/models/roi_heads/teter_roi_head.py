import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmdet.core import bbox2roi, build_assigner, build_sampler
from mmdet.models import HEADS, build_head, build_roi_extractor
from mmdet.models.roi_heads import StandardRoIHead


@HEADS.register_module()
class TETerRoIHead(StandardRoIHead):
    def __init__(
        self,
        track_roi_extractor=None,
        track_head=None,
        track_train_cfg=None,
        cem_roi_extractor=None,
        cem_train_cfg=None,
        cem_head=None,
        finetune_cem=False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        if track_head is not None:
            self.init_track_head(track_roi_extractor, track_head)

        if track_train_cfg is not None:
            self.track_train_cfg = track_train_cfg
            self.init_track_assigner_sampler()

        if cem_head is not None:
            self.init_cem_head(cem_roi_extractor, cem_head)
        else:
            self.cem_head = None

        if cem_train_cfg is not None:
            self.cem_train_cfg = cem_train_cfg
            self.init_cem_assigner_sampler()
        self.finetune_cem = finetune_cem

    def init_track_assigner_sampler(self):
        """Initialize assigner and sampler."""
        if self.track_train_cfg.get("assigner", None):
            self.track_roi_assigner = build_assigner(self.track_train_cfg.assigner)
            self.track_share_assigner = False
        else:
            self.track_roi_assigner = self.bbox_assigner
            self.track_share_assigner = True

        if self.track_train_cfg.get("sampler", None):
            self.track_roi_sampler = build_sampler(
                self.track_train_cfg.sampler, context=self
            )
            self.track_share_sampler = False
        else:
            self.track_roi_sampler = self.bbox_sampler
            self.track_share_sampler = True

    def init_cem_assigner_sampler(self):
        """Initialize assigner and sampler."""
        if self.cem_train_cfg.get("assigner", None):
            self.cem_roi_assigner = build_assigner(self.cem_train_cfg.assigner)
            self.cem_share_assigner = False
        else:
            self.track_roi_assigner = self.bbox_assigner
            self.cem_share_assigner = True

        if self.cem_train_cfg.get("sampler", None):
            self.cem_roi_sampler = build_sampler(
                self.cem_train_cfg.sampler, context=self
            )
            self.cem_share_sampler = False
        else:
            self.cem_roi_sampler = self.bbox_sampler
            self.cem_share_sampler = True

    @property
    def with_track(self):
        """bool: whether the RoI head contains a `track_head`"""
        return hasattr(self, "track_head") and self.track_head is not None

    @property
    def with_cem(self):
        """bool: whether the RoI head contains a `track_head`"""
        return hasattr(self, "cem_head") and self.cem_head is not None

    def init_track_head(self, track_roi_extractor, track_head):
        """Initialize ``track_head``"""
        if track_roi_extractor is not None:
            self.track_roi_extractor = build_roi_extractor(track_roi_extractor)
            self.track_share_extractor = False
        else:
            self.track_share_extractor = True
            self.track_roi_extractor = self.bbox_roi_extractor
        self.track_head = build_head(track_head)

    def init_cem_head(self, cem_roi_extractor, cem_head):
        """Initialize ``track_head``"""
        if cem_roi_extractor is not None:
            self.cem_roi_extractor = build_roi_extractor(cem_roi_extractor)
            self.cem_share_extractor = False
        else:
            self.cem_share_extractor = True
            self.cem_roi_extractor = self.bbox_roi_extractor
        self.cem_head = build_head(cem_head)

    def init_weights(self, *args, **kwargs):
        super().init_weights(*args, **kwargs)
        if self.with_track:
            self.track_head.init_weights()
            if not self.track_share_extractor:
                self.track_roi_extractor.init_weights()
        if self.with_cem:
            self.cem_head.init_weights()
            if not self.cem_share_extractor:
                self.cem_roi_extractor.init_weights()

    def forward_train(
        self,
        x,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_match_indices,
        ref_x,
        ref_img_metas,
        ref_proposals,
        ref_gt_bboxes,
        ref_gt_labels,
        gt_bboxes_ignore=None,
        gt_masks=None,
        ref_gt_bboxes_ignore=None,
        *args,
        **kwargs
    ):
        if not self.finetune_cem:
            losses = super().forward_train(
                x,
                img_metas,
                proposal_list,
                gt_bboxes,
                gt_labels,
                gt_bboxes_ignore,
                gt_masks,
                *args,
                **kwargs
            )
        else:
            losses = {}

        num_imgs = len(img_metas)

        if self.with_track or self.with_cem:

            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            if ref_gt_bboxes_ignore is None:
                ref_gt_bboxes_ignore = [None for _ in range(num_imgs)]
            key_sampling_results, ref_sampling_results = [], []
            for i in range(num_imgs):
                assign_result = self.track_roi_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
                )
                sampling_result = self.track_roi_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x],
                )
                key_sampling_results.append(sampling_result)

                ref_assign_result = self.track_roi_assigner.assign(
                    ref_proposals[i],
                    ref_gt_bboxes[i],
                    ref_gt_bboxes_ignore[i],
                    ref_gt_labels[i],
                )
                ref_sampling_result = self.track_roi_sampler.sample(
                    ref_assign_result,
                    ref_proposals[i],
                    ref_gt_bboxes[i],
                    ref_gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in ref_x],
                )
                ref_sampling_results.append(ref_sampling_result)

            key_bboxes = [res.pos_bboxes for res in key_sampling_results]

            if self.with_track and not self.finetune_cem:
                key_feats = self._track_forward(x, key_bboxes)
                ref_bboxes = [res.bboxes for res in ref_sampling_results]
                ref_feats = self._track_forward(ref_x, ref_bboxes)

                match_feats = self.track_head.match(
                    key_feats, ref_feats, key_sampling_results, ref_sampling_results
                )
                asso_targets = self.track_head.get_track_targets(
                    gt_match_indices, key_sampling_results, ref_sampling_results
                )
                loss_track = self.track_head.loss(*match_feats, *asso_targets)

                losses.update(loss_track)

            if self.with_cem:
                key_labels = [res.pos_gt_labels for res in key_sampling_results]
                ref_pos_bboxes = [res.pos_bboxes for res in ref_sampling_results]
                ref_labels = [res.pos_gt_labels for res in ref_sampling_results]
                key_cem_feats = self._cem_forward(x, key_bboxes)
                ref_cem_feats = self._cem_forward(ref_x, ref_pos_bboxes)

                all_feats = torch.cat([key_cem_feats, ref_cem_feats], dim=0)
                label_tensor = torch.cat([torch.cat(key_labels), torch.cat(ref_labels)])
                current_size = len(label_tensor)

                # obtain Tensor size of each rank
                local_size = torch.LongTensor([current_size]).to("cuda")
                size_list = [
                    torch.LongTensor([0]).to("cuda")
                    for _ in range(dist.get_world_size())
                ]
                dist.all_gather(size_list, local_size)
                size_list = [int(size.item()) for size in size_list]
                max_size = max(size_list)
                all_feats = F.pad(all_feats, pad=(0, 0, 0, max_size - current_size))
                label_tensor = F.pad(label_tensor, (0, max_size - current_size))
                combine_label_list = [
                    torch.zeros_like(label_tensor) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(combine_label_list, label_tensor)
                combine_label_tensor = torch.cat(combine_label_list)
                combine_all_feats = torch.cat(GatherLayer.apply(all_feats), dim=0)

                mask = torch.zeros(len(combine_label_tensor), dtype=torch.bool)
                for i in range(len(size_list)):
                    s = size_list[i]
                    mask[(i * max_size) : (i * max_size) + s] = True
                combine_label_tensor = combine_label_tensor[mask]
                combine_all_feats = combine_all_feats[mask]

                normalized_key_feats = F.normalize(combine_all_feats, p=2, dim=1)

                # record the contrastive sample statistics
                loss_cem = self.cem_head.sup_contra_loss(
                    normalized_key_feats[:, None, :], combine_label_tensor
                )
                losses.update(loss_cem)

        return losses

    def _track_forward(self, x, bboxes):
        """Track head forward function used in both training and testing."""
        rois = bbox2roi(bboxes)
        track_feats = self.track_roi_extractor(
            x[: self.track_roi_extractor.num_inputs], rois
        )
        track_feats = self.track_head(track_feats)
        return track_feats

    def _cem_forward(self, x, bboxes):
        """Track head forward function used in both training and testing."""
        rois = bbox2roi(bboxes)
        cem_feats = self.cem_roi_extractor(x[: self.cem_roi_extractor.num_inputs], rois)
        cem_feats = self.cem_head(cem_feats)

        return cem_feats

    def simple_test(self, x, img_metas, proposal_list, rescale):
        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale
        )

       
        det_bboxes = det_bboxes[0]
        det_labels = det_labels[0]

        if det_bboxes.size(0) == 0:
            return det_bboxes, det_labels, None

        track_bboxes = det_bboxes[:, :-1] * torch.tensor(
            img_metas[0]["scale_factor"]
        ).to(det_bboxes.device)
        track_feats = self._track_forward(x, [track_bboxes])
        if self.cem_head is not None:
            cem_feats = self._cem_forward(x, [track_bboxes])
        else:
            cem_feats = None

        return det_bboxes, det_labels, cem_feats, track_feats


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process, supporting backward propagation.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out
