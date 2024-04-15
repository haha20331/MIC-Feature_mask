# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications:
# - Delete tensors after usage to free GPU memory
# - Add HRDA debug visualizations
# - Support ImageNet feature distance for LR and HR predictions of HRDA
# - Add masked image consistency
# - Update debug image system
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.dropout import _DropoutNd
from ..builder import build_loss
from ..losses import accuracy

from mmseg.ops import resize
from mmseg.core import add_prefix
from mmseg.models import UDA, HRDAEncoderDecoder, build_segmentor
from mmseg.models.segmentors.hrda_encoder_decoder import crop
from mmseg.models.uda.masking_consistency_module import \
    MaskingConsistencyModule
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)
from mmseg.models.utils.visualization import prepare_debug_out, subplotimg
from mmseg.utils.utils import downscale_label_ratio


def visualized_gt(img, gt):
    img = img.cpu()
    gt = gt.cpu()
    img_np = img.permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    output_img = np.zeros((512, 512, 3), dtype=np.uint8)

    output_img[gt[0] == 1] = [0, 0, 255] 

    cv2.imwrite('img.png', img_np)
    cv2.imwrite('gt.png', output_img)

    return 0

def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@UDA.register_module()
class DACS(UDADecorator):

    def __init__(self, **cfg):
        super(DACS, self).__init__(**cfg)
        
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.source_only = cfg['source_only']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        #self.enable_fdist = False
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.mask_mode = cfg['mask_mode']
        self.enable_masking = self.mask_mode is not None
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        assert self.mix == 'class'
        #self.fmask_ratio=cfg[fmask_ratio]

        self.mask_feature_ratio = cfg['mask_feature_ratio']
        self.student_consistency_loss_flag = cfg['student_consistency_loss_flag']
        self.mask_img_and_feature_loss_flag = cfg['mask_img_and_feature_loss_flag']

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        if not self.source_only:
            self.ema_model = build_segmentor(ema_cfg)
        self.mic = None
        if self.enable_masking:
            self.mic = MaskingConsistencyModule(require_teacher=False, cfg=cfg)
        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        if self.source_only:
            return
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        if self.source_only:
            return
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        # If the mask is empty, the mean will be NaN. However, as there is
        # no connection in the compute graph to the network weights, the
        # network gradients are zero and no weight update will happen.
        # This can be verified with print_grad_magnitude.
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        # Features from multiple input scales (see HRDAEncoderDecoder)
        if isinstance(self.get_model(), HRDAEncoderDecoder) and \
                self.get_model().feature_scale in \
                self.get_model().feature_scale_all_strs:
            lay = -1
            feat = [f[lay] for f in feat]
            with torch.no_grad():
                self.get_imnet_model().eval()
                feat_imnet = self.get_imnet_model().extract_feat(img)
                feat_imnet = [f[lay].detach() for f in feat_imnet]
            feat_dist = 0
            n_feat_nonzero = 0
            for s in range(len(feat_imnet)):
                if self.fdist_classes is not None:
                    fdclasses = torch.tensor(
                        self.fdist_classes, device=gt.device)
                    gt_rescaled = gt.clone()
                    if s in HRDAEncoderDecoder.last_train_crop_box:
                        gt_rescaled = crop(
                            gt_rescaled,
                            HRDAEncoderDecoder.last_train_crop_box[s])
                    scale_factor = gt_rescaled.shape[-1] // feat[s].shape[-1]
                    gt_rescaled = downscale_label_ratio(
                        gt_rescaled, scale_factor, self.fdist_scale_min_ratio,
                        self.num_classes, 255).long().detach()
                    fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses,
                                           -1)
                    fd_s = self.masked_feat_dist(feat[s], feat_imnet[s],
                                                 fdist_mask)
                    feat_dist += fd_s
                    if fd_s != 0:
                        n_feat_nonzero += 1
                    del fd_s
                    if s == 0:
                        self.debug_fdist_mask = fdist_mask
                        self.debug_gt_rescale = gt_rescaled
                else:
                    raise NotImplementedError
        else:
            with torch.no_grad():
                self.get_imnet_model().eval()
                feat_imnet = self.get_imnet_model().extract_feat(img)
                feat_imnet = [f.detach() for f in feat_imnet]
            lay = -1
            if self.fdist_classes is not None:
                fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
                scale_factor = gt.shape[-1] // feat[lay].shape[-1]
                gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                    self.fdist_scale_min_ratio,
                                                    self.num_classes,
                                                    255).long().detach()
                fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
                feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                                  fdist_mask)
                self.debug_fdist_mask = fdist_mask
                self.debug_gt_rescale = gt_rescaled
            else:
                feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def update_debug_state(self):
        debug = self.local_iter % self.debug_img_interval == 0
        self.get_model().automatic_debug = False
        self.get_model().debug = debug
        if not self.source_only:
            self.get_ema_model().automatic_debug = False
            self.get_ema_model().debug = debug
        if self.mic is not None:
            self.mic.debug = debug
        

    def losses(self, seg_logit, seg_label, seg_weight=None):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=False)
        # if self.sampler is not None:
        #     seg_weight = self.sampler.sample(seg_logit, seg_label)
        seg_label = seg_label.squeeze(1)
        loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0)
        self.loss_decode = build_loss(loss_decode)
        self.loss_decode.debug = False
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=255)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        # if self.debug and hasattr(self.loss_decode, 'debug_output'):
        #     self.debug_output.update(self.loss_decode.debug_output)
        return loss
    
    
    
    def get_pseudo_label_and_weight(self, logits,iter):
        ema_softmax = torch.softmax(logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)

#############################調整threshold大小###################################
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        # ps_large_p = pseudo_prob.ge(self.pseudo_threshold * (1 - iter / 40000)).long() == 1
        # if iter%100 == 0:
        #     print(self.pseudo_threshold * (1 - iter / 40000))
#############################調整threshold大小###################################

        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=logits.device)
        return pseudo_label, pseudo_weight

    def filter_valid_pseudo_region(self, pseudo_weight, valid_pseudo_mask):
        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            assert valid_pseudo_mask is None
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            assert valid_pseudo_mask is None
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        if valid_pseudo_mask is not None:
            pseudo_weight *= valid_pseudo_mask.squeeze(1)
        return pseudo_weight

    def get_student_pseudo_label(self,logits,size,valid_pseudo_mask):
        logits = resize(
                input=logits,
                size=size,
                mode='bilinear',
                align_corners=False)
        label, weight = self.get_pseudo_label_and_weight(
                    logits,iter=0)
        weight = self.filter_valid_pseudo_region(
                    weight, valid_pseudo_mask)
        label = label.unsqueeze(1)
        return label,weight

    def get_student_pseudo_label_ignore_mix(self,logits,size,dev,batch_size,strong_parameters,mix_masks):
        logits = resize(
                        input=logits,
                        size=size,
                        mode='bilinear',
                        align_corners=False)
        label, _ = self.get_pseudo_label_and_weight(
                    logits,iter=0)   
        ignore_mix = torch.full((1024, 1024), 255, device=dev)
        ignore_label =  [None] * batch_size
        for i in range(batch_size):
            strong_parameters['mix'] = 1-mix_masks[i]
            _, ignore_label[i] = strong_transform(
                            strong_parameters,
                            target=torch.stack((label[i], ignore_mix)))
        ignore_label = torch.cat(ignore_label)
        return ignore_label

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      target_img,
                      target_img_metas,
                      rare_class=None,
                      valid_pseudo_mask=None):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training
        if self.mic is not None:
            self.mic.update_weights(self.get_model(), self.local_iter)

        self.update_debug_state()
        seg_debug = {}

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }
        weak_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        # Train on source images
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        seg_debug['Source'] = self.get_model().debug_output
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)


        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                      src_feat)
            log_vars.update(add_prefix(feat_log, 'src'))
############################################    FDLoss    ############################################
            feat_loss = clean_loss + feat_loss
            feat_loss.backward()
############################################    FDLoss    ############################################
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')
        del src_feat, clean_loss
        if self.enable_fdist:
            del feat_loss

        pseudo_label, pseudo_weight = None, None
        if not self.source_only:
            # Generate pseudo-label
            for m in self.get_ema_model().modules():
                if isinstance(m, _DropoutNd):
                    m.training = False
                if isinstance(m, DropPath):
                    m.training = False
            ema_logits = self.get_ema_model().generate_pseudo_label(
                target_img, target_img_metas)
            seg_debug['Target'] = self.get_ema_model().debug_output

            pseudo_label, pseudo_weight = self.get_pseudo_label_and_weight(
                ema_logits,self.local_iter)
            del ema_logits

            pseudo_weight = self.filter_valid_pseudo_region(
                pseudo_weight, valid_pseudo_mask)
            gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

            # Apply mixing
            mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
            strong_img, strong_lbl = [None] * batch_size, [None] * batch_size
            mixed_seg_weight = pseudo_weight.clone()

            mix_masks = get_class_masks(gt_semantic_seg)

            
################## Mix ##################
            for i in range(batch_size):
                strong_parameters['mix'] = mix_masks[i]
                mixed_img[i], mixed_lbl[i] = strong_transform(
                    strong_parameters,
                    data=torch.stack((img[i], target_img[i])),
                    target=torch.stack(
                        (gt_semantic_seg[i][0], pseudo_label[i])))
                _, mixed_seg_weight[i] = strong_transform(
                    strong_parameters,
                    target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
################## wo Mix #########################
            #for i in range(batch_size):
                weak_parameters['mix'] = mix_masks[i]
                strong_img[i], strong_lbl[i] = strong_transform(
                    weak_parameters,
                    data=torch.stack((target_img[i], target_img[i])),
                    target=torch.stack(
                        (pseudo_label[i], pseudo_label[i])))
##############################################################
            del gt_pixel_weight
            mixed_img = torch.cat(mixed_img)
            mixed_lbl = torch.cat(mixed_lbl)
            strong_img = torch.cat(strong_img)
            strong_lbl = torch.cat(strong_lbl)

############## Train on strong augmentation images ##############
            strong_aug_losses = self.get_model().forward_train(
                mixed_img,
                img_metas,
                mixed_lbl,
                seg_weight=pseudo_weight,
                return_feat=False,
                return_logits=self.student_consistency_loss_flag
            )
################# 取得strong_aug predict "strong_aug_logits" ###################
            if self.student_consistency_loss_flag:
                strong_aug_logits = strong_aug_losses['decode.logits'][0]
                strong_aug_label, strong_aug_weight=self.get_student_pseudo_label(strong_aug_logits,mixed_lbl.shape[2:],valid_pseudo_mask)
                del strong_aug_logits,strong_aug_losses['decode.logits']
##########################################################################
            seg_debug['strong_aug'] = self.get_model().debug_output
            strong_aug_losses = add_prefix(strong_aug_losses, 'strong_aug')
            strong_aug_loss, strong_aug_log_vars = self._parse_losses(strong_aug_losses)
            log_vars.update(strong_aug_log_vars)
            strong_aug_loss.backward()
############################################################

############## Train on mixed images ##############
############## get model是拿hrda_encoder_decoder #############
            mix_losses = self.get_model().forward_train(
                mixed_img,
                img_metas,
                mixed_lbl,
                seg_weight=mixed_seg_weight,
                return_feat=False,
                return_logits=self.student_consistency_loss_flag
            )
################# 取得mix predict "mix_logits" ###################
            if self.student_consistency_loss_flag:
                mix_logits = mix_losses['decode.logits'][0]
                del mix_losses['decode.logits']
                if self.local_iter % self.debug_img_interval == 0 and not self.source_only:
                    mix_label, mix_weight=self.get_student_pseudo_label(mix_logits,mixed_lbl.shape[2:],valid_pseudo_mask)
                del mix_logits
##########################################################################
            seg_debug['Mix'] = self.get_model().debug_output
            mix_losses = add_prefix(mix_losses, 'mix')
            mix_loss, mix_log_vars = self._parse_losses(mix_losses)
            log_vars.update(mix_log_vars)
            mix_loss.backward()
############################################################


############## Feature mask功能 ###############
############## 用mask_ratio調整feature mask比例 #############
            if self.mask_feature_ratio['flag']:
                mask_feature_losses = self.get_model().forward_train(
                mixed_img,
                img_metas,
                mixed_lbl,
                seg_weight=mixed_seg_weight,
                return_feat=False,
                mask_feature_ratio = self.mask_feature_ratio,
                return_logits=self.student_consistency_loss_flag
                )
################# 取得mask feature predict "mask_feature_logits" ###################
                if self.student_consistency_loss_flag:
                    mask_feature_logits = mask_feature_losses['decode.logits'][0]
                    del mask_feature_losses['decode.logits']
                    # ignore_mix_mask_feature_label = self.get_student_pseudo_label_ignore_mix(mask_feature_logits,mixed_lbl.shape[2:],dev,batch_size,strong_parameters,mix_masks)
                    
                    #### mask_feature_logits student內部一致性 ####
                    # smf_losses = dict()
                    # student_mask_feature_losses = self.losses(mask_feature_logits, student_pseudo_label, student_pseudo_weight)
                    # del mask_feature_logits
                    # smf_losses.update(add_prefix(student_mask_feature_losses, 'decode'))
                    # student_mask_feature_losses = add_prefix(smf_losses, 'student_masked_feature')
                    # student_mask_feature_loss, student_mask_feature_vars = self._parse_losses(student_mask_feature_losses)
                    # log_vars.update(student_mask_feature_vars)
#############################################################################################
                seg_debug['masked_feature'] = self.get_model().debug_output
                mask_feature_losses = add_prefix(mask_feature_losses, 'masked_feature')
                mask_feature_loss, mask_feature_log_vars = self._parse_losses(mask_feature_losses)
                log_vars.update(mask_feature_log_vars)
                # if self.student_consistency_loss_flag:
                #     mask_feature_loss = (student_mask_feature_loss + mask_feature_loss)
                mask_feature_loss.backward()
                if self.student_consistency_loss_flag:
                    #del student_mask_feature_loss
                    del mask_feature_loss
                    
#############################################################

############## Masked Image Training ##############
        if self.enable_masking and self.mask_mode.startswith('separate'):
            masked_loss = self.mic(self.get_model(), img, img_metas,
                                   gt_semantic_seg, target_img,
                                   target_img_metas, valid_pseudo_mask,
                                   pseudo_label, pseudo_weight,return_logits=self.student_consistency_loss_flag)
################# 取得mask image predict "mask_image_logits" ###################
            if self.student_consistency_loss_flag:
                mask_img_logits = masked_loss['decode.logits'][0]
                del masked_loss['decode.logits']
                mask_img_label, mask_img_weight=self.get_student_pseudo_label(mask_img_logits,mixed_lbl.shape[2:],valid_pseudo_mask)

                #### mask img 在student內的一致性正則化
                smi_losses = dict()
                student_mask_image_losses = self.losses(mask_img_logits, strong_aug_label, strong_aug_weight)
                smi_losses.update(add_prefix(student_mask_image_losses, 'decode'))
                student_mask_image_losses = add_prefix(smi_losses, 'student_masked_image')
                student_mask_image_loss, student_mask_image_vars = self._parse_losses(student_mask_image_losses)
                log_vars.update(student_mask_image_vars)
                del mask_img_logits
##################################################################################
            seg_debug.update(self.mic.debug_output)
            masked_loss = add_prefix(masked_loss, 'masked_image')
            masked_loss, masked_log_vars = self._parse_losses(masked_loss)
            log_vars.update(masked_log_vars)
            if self.student_consistency_loss_flag:
                masked_loss = student_mask_image_loss + masked_loss
            masked_loss.backward()

            if self.student_consistency_loss_flag:
                del masked_loss,student_mask_image_loss
#############################################################

############## Masked Image & Mask Feature Training ##############
        if self.enable_masking and self.mask_mode.startswith('separate') and self.mask_img_and_feature_loss_flag:
            masked_img_and_feature_loss = self.mic(self.get_model(), img, img_metas,
                                   gt_semantic_seg, target_img,
                                   target_img_metas, valid_pseudo_mask,
                                   pseudo_label, pseudo_weight,
                                   mask_feature_ratio = self.mask_feature_ratio,
                                   return_logits=True)
            mask_img_and_feature_logits = masked_img_and_feature_loss['decode.logits'][0]
                #print('mask_img_and_feature_logits',mask_img_and_feature_logits.size())
            del masked_img_and_feature_loss['decode.logits']
            if self.local_iter % self.debug_img_interval == 0 and not self.source_only:
                mask_img_and_feature_debug, _ = self.get_pseudo_label_and_weight(
                mask_img_and_feature_logits,self.local_iter)
                mask_img_and_feature_debug = mask_img_and_feature_debug.unsqueeze(1)

################# 取得 " Masked Image & Mask Feature Logits" ###################
            if self.student_consistency_loss_flag:
                smif_losses = dict()
                student_mask_img_and_feature_losses = self.losses(mask_img_and_feature_logits, ignore_mix_mask_feature, mask_img_weight)
                del mask_img_and_feature_logits
                smif_losses.update(add_prefix(student_mask_img_and_feature_losses, 'decode'))

                student_mask_img_and_feature_losses = add_prefix(smif_losses, 'student_mask_img_and_feature')
                student_mask_img_and_feature_loss, student_mask_img_and_feature_vars = self._parse_losses(student_mask_img_and_feature_losses)
                log_vars.update(student_mask_img_and_feature_vars)
##################################################################################
            masked_img_and_feature_loss = add_prefix(masked_img_and_feature_loss, 'masked_image_and_feature')
            masked_img_and_feature_loss, masked_img_and_feature_log_vars = self._parse_losses(masked_img_and_feature_loss)
            log_vars.update(masked_img_and_feature_log_vars)
            if self.student_consistency_loss_flag:
                masked_img_and_feature_loss = student_mask_img_and_feature_loss + masked_img_and_feature_loss
            masked_img_and_feature_loss.backward()

            if self.student_consistency_loss_flag:
                del masked_img_and_feature_loss
#############################################################
                    
#############################    work_dir debug地方    #############################
        if self.local_iter % self.debug_img_interval == 0 and \
                not self.source_only:
            out_dir = os.path.join(self.train_cfg['work_dir'], 'debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            #for j in range(batch_size):
            for j in range(1):
                rows, cols = 3, 6
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                subplotimg(
                    axs[0][1],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap='cityscapes')
                subplotimg(
                    axs[1][1],
                    pseudo_label[j],
                    'Target Seg (Pseudo) GT',
                    cmap='cityscapes')
                subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed Image')
                subplotimg(
                    axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')
                # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred",
                #            cmap="cityscapes")
                if mixed_lbl is not None:
                    subplotimg(
                        axs[1][3], mixed_lbl[j], 'Seg Targ', cmap='cityscapes')
                if self.student_consistency_loss_flag:    
                    # subplotimg(
                    #     axs[2][0], mix_label[j], 'Student Mix Seg', cmap='cityscapes')
                    # subplotimg(
                    #     axs[2][1], student_mask_feature_debug[j], 'Student Mask Feature Debug',cmap='cityscapes')
                    
                    # subplotimg(
                    #     axs[2][4], ignore_mix_mask_feature[j], 'ignore_mix_mask_feature',cmap='cityscapes')     
                    subplotimg(
                        axs[2][1], strong_aug_label[j], 'Strong Aug Img Debug',cmap='cityscapes')
                    subplotimg(
                        axs[2][0], strong_lbl[j],  'Teacher predict',cmap='cityscapes')
                    subplotimg(
                        axs[2][2], mask_img_label[j], 'Student Mask Img Debug',cmap='cityscapes')
                    subplotimg(
                        axs[2][3], strong_aug_weight[j], 'Student Strong Aug Pseudo W.',vmin=0,vmax=1)
                    subplotimg(
                        axs[2][4], pseudo_weight[j], 'Mask Img Pseudo W.',vmin=0,vmax=1)
                    # subplotimg(
                    #     axs[2][5], student_pseudo_weight[j], 'Student Mix Pseudo W.',vmin=0,vmax=1)
                if self.mask_img_and_feature_loss_flag:
                    subplotimg(
                        axs[2][3], mask_img_and_feature_debug[j], 'Student Mask Img & Feature Debug',cmap='cityscapes')
                    
                subplotimg(
                    axs[0][3],
                    mixed_seg_weight[j],
                    #pseudo_weight[j],
                    'Pseudo W.',
                    vmin=0,
                    vmax=1)
                if self.debug_fdist_mask is not None:
                    subplotimg(
                        axs[0][4],
                        self.debug_fdist_mask[j][0],
                        'FDist Mask',
                        cmap='gray')
                if self.debug_gt_rescale is not None:
                    subplotimg(
                        axs[1][4],
                        self.debug_gt_rescale[j],
                        'Scaled GT',
                        cmap='cityscapes')
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'], 'debug')
            os.makedirs(out_dir, exist_ok=True)
            if seg_debug['Source'] is not None and seg_debug:
                if 'Target' in seg_debug:
                    seg_debug['Target']['Pseudo W.'] = mixed_seg_weight.cpu(
                    ).numpy()
                #for j in range(batch_size):
                for j in range(1):
                    cols = len(seg_debug)
                    rows = max(len(seg_debug[k]) for k in seg_debug.keys())
                    fig, axs = plt.subplots(
                        rows,
                        cols,
                        figsize=(5 * cols, 5 * rows),
                        gridspec_kw={
                            'hspace': 0.1,
                            'wspace': 0,
                            'top': 0.95,
                            'bottom': 0,
                            'right': 1,
                            'left': 0
                        },
                        squeeze=False,
                    )
                    for k1, (n1, outs) in enumerate(seg_debug.items()):
                        for k2, (n2, out) in enumerate(outs.items()):
                            subplotimg(
                                axs[k2][k1],
                                **prepare_debug_out(f'{n1} {n2}', out[j],
                                                    means, stds))
                    for ax in axs.flat:
                        ax.axis('off')
                    plt.savefig(
                        os.path.join(out_dir,
                                     f'{(self.local_iter + 1):06d}_{j}_s.png'))
                    plt.close()
                del seg_debug
        self.local_iter += 1

        return log_vars
