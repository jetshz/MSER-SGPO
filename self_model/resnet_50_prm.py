import torch
from types import MethodType

import cv2
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.misc import imresize
import PIL.Image
import math

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
        self.classifier = nn.Sequential(nn.Conv2d(num_features, 20, kernel_size=1, bias=True))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ResNet50_AE(nn.Module):
    def __init__(self):
        super(ResNet50_AE, self).__init__()
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

    def forward(self, x):
        x = self.features(x)
        return x


class ResNet50_background(nn.Module):
    def __init__(self):
        super(ResNet50_background, self).__init__()
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
        self.classifier = nn.Sequential(nn.Conv2d(num_features, 21, kernel_size=1, bias=True))

        self.from_scratch_layers = [self.classifier]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups


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

class Peak_Cam_Response_Map(nn.Sequential):
    def __init__(self, *args, **kargs):
        super(Peak_Cam_Response_Map, self).__init__(*args)

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

    def instance_seg(self, class_response_maps, peak_list, peak_response_maps, retrieval_cfg):
        # cast tensors to numpy array
        class_response_maps = class_response_maps.squeeze().cpu().numpy()
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
        proposal_count = retrieval_cfg.get('proposal_count', 500)

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

            class_response = imresize(class_response_maps[class_idx], (img_height, img_width), interp='bicubic')
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
                          penalty_factor * bg_response[mask].sum() + \
                          saliency_factor * saliency[mask].sum()
                    if val > max_val:
                        max_val = val
                        instance_mask = mask

            if instance_mask is not None:
                instance_list.append((max_val, class_idx, instance_mask, peak_response_map))

        instance_list = sorted(instance_list, key=lambda x: x[0], reverse=True)
        if nms_threshold is not None:
            instance_list = self.instance_nms(sorted(instance_list, key=lambda x: x[0], reverse=True), nms_threshold,
                                              merge_peak_response)
        return [dict(category=v[1], mask=v[2], prm=v[3]) for v in instance_list]

    def forward(self, input, cam=None, class_threshold = 0, peak_threshold = 30, retrieval_cfg = None):
        assert input.dim() == 4, 'PeakResponseMapping layer only supports batch mode.'
        if self.inferencing:
            input.requires_grad_()
        class_response_maps = super(Peak_Cam_Response_Map, self).forward(input)
        # class_response_maps = class_response_maps_bg[:,1:21,:,:]
        if self.inferencing is False:
            # class_response_maps_bg = F.upsample(class_response_maps_bg, scale_factor=8, mode = 'bilinear', align_corners = True)
            class_response_maps = F.upsample(class_response_maps, scale_factor=8, mode = 'bilinear', align_corners = True)

        if cam is not None:
            class_response_maps = class_response_maps.mul(cam[:,1:21,:,:])

        peak_list, aggregation = peak_stimulation(class_response_maps, 3, peak_filter = self.peak_filter)
        # aggregation = F.avg_pool2d(cam[:,1:21,:,:], kernel_size=(cam.size(2), cam.size(3)), padding=0)

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

            if len(peak_response_maps) > 0:
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
        super(Peak_Cam_Response_Map, self).train(mode)
        if self.inferencing:
            self._recover()
            self.inferencing = False
        return self

    def inference(self):
        super(Peak_Cam_Response_Map, self).train(False)
        self._patch()
        self.inferencing = True
        return self

class Peak_Response_Map_AE(nn.Sequential):
    # def __init__(self, *args, **kargs):
    def __init__(self, features):
        super(Peak_Response_Map_AE, self).__init__()

        self.inferencing = False

        self.features = features

        self.peak_filter = self._median_filter

        self.cls = nn.Sequential(
            nn.Conv2d(20, 20, kernel_size=3, padding=1,dilation=1),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            nn.Conv2d(20, 20,kernel_size=1,padding=0)
        )

        self.cls_erase = nn.Sequential(
            nn.Conv2d(20, 20, kernel_size=3, padding=1,dilation=1),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            nn.Conv2d(20, 20,kernel_size=1,padding=0)
        )

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

    def instance_seg(self, class_response_maps, peak_list, peak_response_maps, retrieval_cfg):
        # cast tensors to numpy array
        # class_response_maps = class_response_maps.squeeze().cpu().numpy()
        # peak_list = peak_list.cpu().numpy()
        # peak_response_maps = peak_response_maps.cpu().numpy()

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
        proposal_count = retrieval_cfg.get('proposal_count', 100)

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

            class_response = imresize(class_response_maps[class_idx], (img_height, img_width), interp='bicubic')
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
                          penalty_factor * bg_response[mask].sum()
                          # saliency_factor * saliency[mask].sum()
                    if val > max_val:
                        max_val = val
                        instance_mask = mask

            if instance_mask is not None:
                instance_list.append((max_val, class_idx, instance_mask, peak_response_map,))

        instance_list = sorted(instance_list, key=lambda x: x[0], reverse=True)
        if nms_threshold is not None:
            instance_list = self.instance_nms(sorted(instance_list, key=lambda x: x[0], reverse=True), nms_threshold,
                                              merge_peak_response)
        return [dict(category=v[1], mask=v[2], prm=v[3]) for v in instance_list]

    def forward(self, input, class_threshold = 0, peak_threshold = 30, retrieval_cfg = None, img_name = None):
        assert input.dim() == 4, 'PeakResponseMapping layer only supports batch mode.'
        if self.inferencing:
            input.requires_grad_()

        self.map1 = self.features(input)
        class_response_maps = self.cls(self.map1)
        if self.inferencing is False:
            class_response_maps = F.upsample(class_response_maps, scale_factor=8, mode = 'bilinear', align_corners = True)

        peak_list, aggregation = peak_stimulation(class_response_maps, 3, peak_filter = self.peak_filter)

        if self.inferencing:
            assert class_response_maps.size(0) == 1, 'Currently inference mode (with peak backpropagation) only supports one image at a time.'

            pos = torch.le(class_response_maps, 50).float()
            pos_numpy = pos.cpu().numpy()
            erased_map = self.map1.mul(pos)
            class_response_maps_2 = self.cls_erase(erased_map)
            peak_list_2, aggregation_2 = peak_stimulation(class_response_maps_2, 3, peak_filter=self.peak_filter)



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

            peak_response_maps_2 = []
            valid_peak_list_2 = []
            grad_output_2 = class_response_maps_2.new_empty(class_response_maps_2.size())
            for idx in range(peak_list_2.size(0)):
                if aggregation_2[peak_list_2[idx, 0], peak_list_2[idx, 1]] >= class_threshold:
                    peak_val = class_response_maps_2[
                        peak_list_2[idx, 0], peak_list_2[idx, 1], peak_list_2[idx, 2], peak_list_2[idx, 3]]
                    if peak_val >= peak_threshold:
                        grad_output_2.zero_()
                        # starting from the peak
                        grad_output_2[peak_list_2[idx, 0], peak_list_2[idx, 1], peak_list_2[idx, 2], peak_list_2[idx, 3]] = 1
                        if input.grad is not None:
                            input.grad.zero_()
                        class_response_maps_2.backward(grad_output, retain_graph=True)
                        prm_2 = input.grad.detach().sum(1).clone().clamp(min=0)  # peak -> peak rensonse map
                        peak_response_maps_2.append(prm_2 / prm_2.sum())
                        valid_peak_list_2.append(peak_list_2[idx, :])

            class_response_maps = class_response_maps.detach()
            aggregation = aggregation.detach()
            class_response_maps_2 = class_response_maps_2.detach()
            aggregation_2 = aggregation_2.detach()

            print(len(peak_response_maps))
            print(len(peak_response_maps_2))

            if len(peak_response_maps) > 0:
                valid_peak_list = torch.stack(valid_peak_list)
                peak_response_maps = torch.cat(peak_response_maps, 0)
                class_response_maps = class_response_maps.squeeze().cpu().numpy()
                valid_peak_list = valid_peak_list.cpu().numpy()
                peak_response_maps = peak_response_maps.cpu().numpy()
                if retrieval_cfg is None:
                    return aggregation, class_response_maps, valid_peak_list, peak_response_maps
                # else:
                    # return self.instance_seg(class_response_maps, valid_peak_list, peak_response_maps, retrieval_cfg)

            if len(peak_response_maps_2) > 0:
                valid_peak_list_2 = torch.stack(valid_peak_list_2)
                peak_response_maps_2 = torch.cat(peak_response_maps_2, 0)
                class_response_maps_2 = class_response_maps_2.squeeze().cpu().numpy()
                valid_peak_list_2 = valid_peak_list_2.cpu().numpy()
                peak_response_maps_2 = peak_response_maps_2.cpu().numpy()
                if retrieval_cfg is None:
                    return aggregation, class_response_maps, valid_peak_list, peak_response_maps
                else:
                    b = self.instance_seg(class_response_maps_2, valid_peak_list_2, peak_response_maps_2, retrieval_cfg)
                    if len(peak_response_maps) > 0:
                        a = self.instance_seg(class_response_maps, valid_peak_list, peak_response_maps, retrieval_cfg)
                        return a+b
                    else:
                        # a = self.instance_seg(class_response_maps_2, valid_peak_list_2, peak_response_maps_2, retrieval_cfg)
                        if len(b) is 0:
                            return None
                        else:
                            return b
                    # return self.instance_seg(class_response_maps, valid_peak_list, peak_response_maps, retrieval_cfg)
            else:
                if len(peak_response_maps) > 0:
                    return self.instance_seg(class_response_maps, valid_peak_list, peak_response_maps, retrieval_cfg)
                else:
                    return None


        else:
            # mask = torch.ones(erased_map.size()).cuda
            pos = torch.le(class_response_maps, 50).float()
            # mask[pos] = 0.0
            # mask = torch.unsqueeze(mask,dim=1)
            pos = F.upsample(pos, size=self.map1.size()[2:], mode='bilinear', align_corners=True)
            erased_map = self.map1.mul(pos)
            class_response_maps_2 = self.cls_erase(erased_map)
            if self.inferencing is False:
                class_response_maps_2 = F.upsample(class_response_maps_2, scale_factor=8, mode='bilinear',
                                                 align_corners=True)

            peak_list_2, aggregation_2 = peak_stimulation(class_response_maps_2, 3, peak_filter=self.peak_filter)

            return aggregation, aggregation_2

    def train(self, mode=True):
        super(Peak_Response_Map_AE, self).train(mode)
        if self.inferencing:
            self._recover()
            self.inferencing = False
        return self

    def inference(self):
        super(Peak_Response_Map_AE, self).train(False)
        self._patch()
        self.inferencing = True
        return self