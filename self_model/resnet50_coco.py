import cv2
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from scipy.misc import imresize
import PIL.Image
import math
from types import MethodType

import torchvision.models as models

from self_prm.PRM import pr_conv2d, peak_stimulation

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        model = models.resnet50(pretrained=True)

        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4)

        num_features = model.layer4[1].conv1.in_channels
        self.classifier = nn.Sequential(nn.Conv2d(num_features, 80, kernel_size=1, bias=True))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class Peak_Response_Map(nn.Sequential):
    def __init__(self, *args, **kargs):
        super(Peak_Response_Map, self).__init__(*args)

        self.inferencing = False

        self.peak_filter = self._median_filter

    @staticmethod
    def _median_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold, _ = torch.median(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    @staticmethod
    def _mean_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold = torch.mean(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    @staticmethod
    def _max_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold, _ = torch.max(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    def _patch(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module._original_forward = module.forward
                module.forward = MethodType(pr_conv2d, module)

    def _recover(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) and hasattr(module, '_original_forward'):
                module.forward = module._original_forward

    def instance_nms(self, instance_list, threshold=0.3, merge_peak_response=True):
        selected_instances = []
        while len(instance_list) > 0:
            instance = instance_list.pop(0)
            selected_instances.append(instance)
            src_mask = instance[2].astype(bool)
            src_peak_response = instance[3]

            def iou_filter(x):
                dst_mask = x[2].astype(bool)
                # IoU
                intersection = np.logical_and(src_mask, dst_mask).sum()
                union = np.logical_or(src_mask, dst_mask).sum()
                iou = intersection / (union + 1e-10)
                if iou < threshold:
                    return x
                else:
                    if merge_peak_response:
                        nonlocal src_peak_response
                        src_peak_response += x[3]
                    return None

            instance_list = list(filter(iou_filter, instance_list))
        return selected_instances

    def instance_seg(self, class_response_maps, peak_list, peak_response_maps, retrieval_cfg, img_name = None):
        # cast tensors to numpy array
        # class_response_maps = class_response_maps.squeeze().cpu().numpy()
        peak_list = peak_list.cpu().numpy()
        peak_response_maps = peak_response_maps.cpu().numpy()

        img_height, img_width = peak_response_maps.shape[1], peak_response_maps.shape[2]

        # image size
        img_area = img_height * img_width

        # segment proposals off-the-shelf
        proposals = retrieval_cfg['proposals']

        # proposal contour width
        contour_width = retrieval_cfg.get('contour_width', 5)

        # limit range of proposal size
        proposal_size_limit = retrieval_cfg.get('proposal_size_limit', (0.00002, 0.85))

        # selected number of proposals
        proposal_count = retrieval_cfg['proposal_count']

        # nms threshold
        nms_threshold = retrieval_cfg.get('nms_threshold', 0.3)

        # merge peak response during nms
        merge_peak_response = retrieval_cfg.get('merge_peak_response', True)

        # metric free parameters
        param = retrieval_cfg.get('param', None)

        saliency = retrieval_cfg.get('saliency_map', None)
        saliency = imresize(saliency, (img_height, img_width), interp='bicubic')
        saliency = saliency.astype(bool)

        # process each peak
        instance_list = []
        for i in range(len(peak_response_maps)):
            class_idx = peak_list[i, 1]

            # extract hyper-params
            if isinstance(param, tuple):
                # shared param
                bg_threshold_factor, penalty_factor, balance_factor, saliency_factor = param
            elif isinstance(param, list):
                # independent params between classes
                bg_threshold_factor, penalty_factor, balance_factor, saliency_factor = param[class_idx]
            else:
                raise TypeError('Invalid hyper-params "%s".' % param)

            if isinstance(class_response_maps, list):
                class_response_map = class_response_maps[i]
            else:
                class_response_map = class_response_maps
            class_response = imresize(class_response_map[class_idx], (img_height, img_width), interp='bicubic')
            bg_response = (class_response < bg_threshold_factor * class_response.mean()).astype(np.float32)
            peak_response_map = peak_response_maps[i]

            # select proposal
            max_val = -np.inf
            instance_mask = None

            for j in range(min(proposal_count, len(proposals))):
                raw_mask = imresize(proposals[j].astype(int), peak_response_map.shape, interp='nearest')
                # get contour of the proposal
                contour_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_GRADIENT,
                                                np.ones((contour_width, contour_width), np.uint8)).astype(bool)
                mask = raw_mask.astype(bool)
                # metric
                mask_area = mask.sum()
                if (mask_area >= proposal_size_limit[1] * img_area) or \
                        (mask_area < proposal_size_limit[0] * img_area):
                    continue
                else:
                    val = balance_factor * peak_response_map[mask].sum() + \
                          peak_response_map[contour_mask].sum() - \
                          penalty_factor * bg_response[mask].sum()+\
                          saliency_factor * ((peak_response_map*saliency)[mask]).sum()
                    if val > max_val:
                        max_val = val
                        instance_mask = mask

            if instance_mask is not None:
                instance_list.append((max_val, class_idx, instance_mask, peak_response_map))

                if img_name is not None:
                    none_picture = np.zeros((448, 448)).astype('uint8')
                    none_picture[instance_mask] = 255
                    #test = PIL.Image.fromarray(none_picture)
                    #test.save("/home/zxforchid/Hu_Z/pychram_PRM/COB_test/test_2/%s/%s.png" % (img_name,i))

        instance_list = sorted(instance_list, key=lambda x: x[0], reverse=True)
        if nms_threshold is not None:
            instance_list = self.instance_nms(sorted(instance_list, key=lambda x: x[0], reverse=True), nms_threshold,
                                              merge_peak_response)
        return [dict(category=v[1], mask=v[2], prm=v[3]) for v in instance_list]

    def forward(self, input, class_threshold = 0, peak_threshold = 30, retrieval_cfg = None):
        assert input.dim() == 4, 'PeakResponseMapping layer only supports batch mode.'
        if self.inferencing:
            input.requires_grad_()
        class_response_maps = super(Peak_Response_Map, self).forward(input)
        if self.inferencing is False:
            class_response_maps = F.upsample(class_response_maps, scale_factor=8, mode = 'bilinear', align_corners = True)

        peak_list, aggregation = peak_stimulation(class_response_maps, 3, peak_filter = self.peak_filter)

        if self.inferencing:
            assert class_response_maps.size(0) == 1, 'Currently inference mode (with peak backpropagation) only supports one image at a time.'

            peak_response_maps = []
            valid_peak_list = []
            grad_output = class_response_maps.new_empty(class_response_maps.size())
            for idx in range(peak_list.size(0)):
                if aggregation[peak_list[idx, 0], peak_list[idx, 1]] >= class_threshold:
                    peak_val = class_response_maps[
                        peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2], peak_list[idx, 3]]
                    if peak_val >= peak_threshold:
                        grad_output.zero_()
                        # starting from the peak
                        grad_output[peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2], peak_list[idx, 3]] = 1
                        if input.grad is not None:
                            input.grad.zero_()
                        class_response_maps.backward(grad_output, retain_graph=True)
                        prm = input.grad.detach().sum(1).clone().clamp(min=0)  # peak -> peak rensonse map
                        peak_response_maps.append(prm / prm.sum())
                        valid_peak_list.append(peak_list[idx, :])

            class_response_maps = class_response_maps.detach()
            aggregation = aggregation.detach()

            if len(peak_response_maps)>0:
                valid_peak_list = torch.stack(valid_peak_list)
                peak_response_maps = torch.cat(peak_response_maps, 0)
                if retrieval_cfg is None:
                    return aggregation, class_response_maps, valid_peak_list, peak_response_maps
                else:
                    return self.instance_seg(class_response_maps, valid_peak_list, peak_response_maps, retrieval_cfg)
            else:
                return None
        else:
            return aggregation

    def train(self, mode=True):
        super(Peak_Response_Map, self).train(mode)
        if self.inferencing:
            self._recover()
            self.inferencing = False
        return self

    def inference(self):
        super(Peak_Response_Map, self).train(False)
        self._patch()
        self.inferencing = True
        return self