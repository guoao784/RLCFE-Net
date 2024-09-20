import argparse
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path
from .revcol import  RevCol

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *

from models.mobilenetv4 import *

from models.fadc import AdaptiveDilatedConv
from models.experimental import *
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)
from utils.tal.anchor_generator import make_anchors, dist2bbox

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    # YOLO Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = max((ch[0] // 4, self.reg_max * 4, 16)), max((ch[0], min((self.nc * 2, 128))))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)


class DDetect(nn.Module):
    # YOLO Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), max((ch[0], min((self.nc * 2, 128))))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3, g=4), nn.Conv2d(c2, 4 * self.reg_max, 1, groups=4)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)

class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class Conv_GN(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.gn = nn.GroupNorm(16, c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.gn(self.conv(x)))

class Detect_LSCD(nn.Module):
    # Lightweight Shared Convolutional Detection Head
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, hidc=256, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.conv = nn.ModuleList(nn.Sequential(Conv_GN(x, hidc, 1)) for x in ch)
        self.share_conv = nn.Sequential(Conv_GN(hidc, hidc, 3), Conv_GN(hidc, hidc, 3))
        self.cv2 = nn.Conv2d(hidc, 4 * self.reg_max, 1)
        self.cv3 = nn.Conv2d(hidc, self.nc, 1)
        self.scale = nn.ModuleList(Scale(1.0) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = self.conv[i](x[i])
            x[i] = self.share_conv(x[i])
            x[i] = torch.cat((self.scale[i](self.cv2(x[i])), self.cv3(x[i])), 1)
        if self.training:  # Training path
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(box)

        if self.export and self.format in ("tflite", "edgetpu"):
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            img_h = shape[2]
            img_w = shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * img_size)
            dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        # for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
        m.cv2.bias.data[:] = 1.0  # box
        m.cv3.bias.data[: m.nc] = math.log(5 / m.nc / (640 / 16) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes):
        """Decode bounding boxes."""
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

try:
    from DCNv4_op.DCNv4.modules.dcnv4 import DCNv4
except ImportError as e:
    pass


class GDDetect(nn.Module):
    # YOLO Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), max((ch[0], min((self.nc * 2, 128))))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3, g=4), nn.Conv2d(c2, 4 * self.reg_max, 1, groups=4)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)

class DyDCNv2(nn.Module):
    """ModulatedDeformConv2d with normalization layer used in DyHead.
    This module cannot be configured with `conv_cfg=dict(type='DCNv2')`
    because DyHead calculates offset and mask from middle-level feature.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int | tuple[int], optional): Stride of the convolution.
            Default: 1.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='GN', num_groups=16, requires_grad=True).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)):
        super().__init__()
        self.with_norm = norm_cfg is not None
        bias = not self.with_norm
        self.conv = ModulatedDeformConv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=bias)
        if self.with_norm:
            self.norm = build_norm_layer(norm_cfg, out_channels)[1]

    def forward(self, x, offset, mask):
        """Forward function."""
        x = self.conv(x.contiguous(), offset, mask)
        if self.with_norm:
            x = self.norm(x)
        return x

class TaskDecomposition(nn.Module):
    def __init__(self, feat_channels, stacked_convs, la_down_rate=8):
        super(TaskDecomposition, self).__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = self.feat_channels * self.stacked_convs
        self.la_conv1 = nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, 1)
        self.relu = nn.ReLU(inplace=True)
        self.la_conv2 = nn.Conv2d(self.in_channels // la_down_rate, self.stacked_convs, 1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.reduction_conv = Conv_GN(self.in_channels, self.feat_channels, 1)
        self.init_weights()

    def init_weights(self):
        # self.la_conv1.weight.normal_(std=0.001)
        # self.la_conv2.weight.normal_(std=0.001)
        # self.la_conv2.bias.data.zero_()
        # self.reduction_conv.conv.weight.normal_(std=0.01)

        torch.nn.init.normal_(self.la_conv1.weight.data, mean=0, std=0.001)
        torch.nn.init.normal_(self.la_conv2.weight.data, mean=0, std=0.001)
        torch.nn.init.zeros_(self.la_conv2.bias.data)
        torch.nn.init.normal_(self.reduction_conv.conv.weight.data, mean=0, std=0.01)

    def forward(self, feat, avg_feat=None):
        b, c, h, w = feat.shape
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        weight = self.relu(self.la_conv1(avg_feat))
        weight = self.sigmoid(self.la_conv2(weight))

        # here we first compute the product between layer attention weight and conv weight,
        # and then compute the convolution between new conv weight and feature map,
        # in order to save memory and FLOPs.
        conv_weight = weight.reshape(b, 1, self.stacked_convs, 1) * \
                      self.reduction_conv.conv.weight.reshape(1, self.feat_channels, self.stacked_convs,
                                                              self.feat_channels)
        conv_weight = conv_weight.reshape(b, self.feat_channels, self.in_channels)
        feat = feat.reshape(b, self.in_channels, h * w)
        feat = torch.bmm(conv_weight, feat).reshape(b, self.feat_channels, h, w)
        feat = self.reduction_conv.gn(feat)
        feat = self.reduction_conv.act(feat)

        return feat


class Detect_TADDH(nn.Module):
    # Task Dynamic Align Detection Head
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, hidc=256, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.share_conv = nn.Sequential(Conv_GN(hidc, hidc // 2, 3), Conv_GN(hidc // 2, hidc // 2, 3))
        self.cls_decomp = TaskDecomposition(hidc // 2, 2, 16)
        self.reg_decomp = TaskDecomposition(hidc // 2, 2, 16)
        self.DyDCNV2 = DyDCNv2(hidc // 2, hidc // 2)
        self.spatial_conv_offset = nn.Conv2d(hidc, 3 * 3 * 3, 3, padding=1)
        self.offset_dim = 2 * 3 * 3
        self.cls_prob_conv1 = nn.Conv2d(hidc, hidc // 4, 1)
        self.cls_prob_conv2 = nn.Conv2d(hidc // 4, 1, 3, padding=1)
        self.cv2 = nn.Conv2d(hidc // 2, 4 * self.reg_max, 1)
        self.cv3 = nn.Conv2d(hidc // 2, self.nc, 1)
        self.scale = nn.ModuleList(Scale(1.0) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            stack_res_list = [self.share_conv[0](x[i])]
            stack_res_list.extend(m(stack_res_list[-1]) for m in self.share_conv[1:])
            feat = torch.cat(stack_res_list, dim=1)

            # task decomposition
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_feat = self.cls_decomp(feat, avg_feat)
            reg_feat = self.reg_decomp(feat, avg_feat)

            # reg alignment
            offset_and_mask = self.spatial_conv_offset(feat)
            offset = offset_and_mask[:, :self.offset_dim, :, :]
            mask = offset_and_mask[:, self.offset_dim:, :, :].sigmoid()
            reg_feat = self.DyDCNV2(reg_feat, offset, mask)

            # cls alignment
            cls_prob = self.cls_prob_conv2(F.relu(self.cls_prob_conv1(feat))).sigmoid()

            x[i] = torch.cat((self.scale[i](self.cv2(reg_feat)), self.cv3(cls_feat * cls_prob)), 1)
        if self.training:  # Training path
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(box)

        if self.export and self.format in ("tflite", "edgetpu"):
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            img_h = shape[2]
            img_w = shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * img_size)
            dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        # for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
        m.cv2.bias.data[:] = 1.0  # box
        m.cv3.bias.data[: m.nc] = math.log(5 / m.nc / (640 / 16) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes):
        """Decode bounding boxes."""
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

class DualDetect(nn.Module):
    # YOLO Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch) // 2  # number of detection layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = max((ch[0] // 4, self.reg_max * 4, 16)), max((ch[0], min((self.nc * 2, 128))))  # channels
        c4, c5 = max((ch[self.nl] // 4, self.reg_max * 4, 16)), max((ch[self.nl], min((self.nc * 2, 128))))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch[:self.nl])
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch[:self.nl])
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, 4 * self.reg_max, 1)) for x in ch[self.nl:])
        self.cv5 = nn.ModuleList(
            nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nc, 1)) for x in ch[self.nl:])
        self.dfl = DFL(self.reg_max)
        self.dfl2 = DFL(self.reg_max)

    def forward(self, x):
        shape = x[0].shape  # BCHW
        d1 = []
        d2 = []
        for i in range(self.nl):
            d1.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
            d2.append(torch.cat((self.cv4[i](x[self.nl+i]), self.cv5[i](x[self.nl+i])), 1))
        if self.training:
            return [d1, d2]
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (d1.transpose(0, 1) for d1 in make_anchors(d1, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([di.view(shape[0], self.no, -1) for di in d1], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        box2, cls2 = torch.cat([di.view(shape[0], self.no, -1) for di in d2], 2).split((self.reg_max * 4, self.nc), 1)
        dbox2 = dist2bbox(self.dfl2(box2), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = [torch.cat((dbox, cls.sigmoid()), 1), torch.cat((dbox2, cls2.sigmoid()), 1)]
        return y if self.export else (y, [d1, d2])

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
        for a, b, s in zip(m.cv4, m.cv5, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)


class DualDDetect(nn.Module):
    # YOLO Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch) // 2  # number of detection layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), max((ch[0], min((self.nc * 2, 128))))  # channels
        c4, c5 = make_divisible(max((ch[self.nl] // 4, self.reg_max * 4, 16)), 4), max((ch[self.nl], min((self.nc * 2, 128))))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3, g=4), nn.Conv2d(c2, 4 * self.reg_max, 1, groups=4)) for x in ch[:self.nl])
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch[:self.nl])
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3, g=4), nn.Conv2d(c4, 4 * self.reg_max, 1, groups=4)) for x in ch[self.nl:])
        self.cv5 = nn.ModuleList(
            nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nc, 1)) for x in ch[self.nl:])
        self.dfl = DFL(self.reg_max)
        self.dfl2 = DFL(self.reg_max)

    def forward(self, x):
        shape = x[0].shape  # BCHW
        d1 = []
        d2 = []
        for i in range(self.nl):
            d1.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
            d2.append(torch.cat((self.cv4[i](x[self.nl+i]), self.cv5[i](x[self.nl+i])), 1))
        if self.training:
            return [d1, d2]
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (d1.transpose(0, 1) for d1 in make_anchors(d1, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([di.view(shape[0], self.no, -1) for di in d1], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        box2, cls2 = torch.cat([di.view(shape[0], self.no, -1) for di in d2], 2).split((self.reg_max * 4, self.nc), 1)
        dbox2 = dist2bbox(self.dfl2(box2), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = [torch.cat((dbox, cls.sigmoid()), 1), torch.cat((dbox2, cls2.sigmoid()), 1)]
        return y if self.export else (y, [d1, d2])
        #y = torch.cat((dbox2, cls2.sigmoid()), 1)
        #return y if self.export else (y, d2)
        #y1 = torch.cat((dbox, cls.sigmoid()), 1)
        #y2 = torch.cat((dbox2, cls2.sigmoid()), 1)
        #return [y1, y2] if self.export else [(y1, d1), (y2, d2)]
        #return [y1, y2] if self.export else [(y1, y2), (d1, d2)]

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
        for a, b, s in zip(m.cv4, m.cv5, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)


class TripleDetect(nn.Module):
    # YOLO Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch) // 3  # number of detection layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = max((ch[0] // 4, self.reg_max * 4, 16)), max((ch[0], min((self.nc * 2, 128))))  # channels
        c4, c5 = max((ch[self.nl] // 4, self.reg_max * 4, 16)), max((ch[self.nl], min((self.nc * 2, 128))))  # channels
        c6, c7 = max((ch[self.nl * 2] // 4, self.reg_max * 4, 16)), max((ch[self.nl * 2], min((self.nc * 2, 128))))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch[:self.nl])
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch[:self.nl])
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, 4 * self.reg_max, 1)) for x in ch[self.nl:self.nl*2])
        self.cv5 = nn.ModuleList(
            nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nc, 1)) for x in ch[self.nl:self.nl*2])
        self.cv6 = nn.ModuleList(
            nn.Sequential(Conv(x, c6, 3), Conv(c6, c6, 3), nn.Conv2d(c6, 4 * self.reg_max, 1)) for x in ch[self.nl*2:self.nl*3])
        self.cv7 = nn.ModuleList(
            nn.Sequential(Conv(x, c7, 3), Conv(c7, c7, 3), nn.Conv2d(c7, self.nc, 1)) for x in ch[self.nl*2:self.nl*3])
        self.dfl = DFL(self.reg_max)
        self.dfl2 = DFL(self.reg_max)
        self.dfl3 = DFL(self.reg_max)

    def forward(self, x):
        shape = x[0].shape  # BCHW
        d1 = []
        d2 = []
        d3 = []
        for i in range(self.nl):
            d1.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
            d2.append(torch.cat((self.cv4[i](x[self.nl+i]), self.cv5[i](x[self.nl+i])), 1))
            d3.append(torch.cat((self.cv6[i](x[self.nl*2+i]), self.cv7[i](x[self.nl*2+i])), 1))
        if self.training:
            return [d1, d2, d3]
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (d1.transpose(0, 1) for d1 in make_anchors(d1, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([di.view(shape[0], self.no, -1) for di in d1], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        box2, cls2 = torch.cat([di.view(shape[0], self.no, -1) for di in d2], 2).split((self.reg_max * 4, self.nc), 1)
        dbox2 = dist2bbox(self.dfl2(box2), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        box3, cls3 = torch.cat([di.view(shape[0], self.no, -1) for di in d3], 2).split((self.reg_max * 4, self.nc), 1)
        dbox3 = dist2bbox(self.dfl3(box3), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = [torch.cat((dbox, cls.sigmoid()), 1), torch.cat((dbox2, cls2.sigmoid()), 1), torch.cat((dbox3, cls3.sigmoid()), 1)]
        return y if self.export else (y, [d1, d2, d3])

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
        for a, b, s in zip(m.cv4, m.cv5, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
        for a, b, s in zip(m.cv6, m.cv7, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)


class TripleDDetect(nn.Module):
    # YOLO Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch) // 3  # number of detection layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), \
                                max((ch[0], min((self.nc * 2, 128))))  # channels
        c4, c5 = make_divisible(max((ch[self.nl] // 4, self.reg_max * 4, 16)), 4), \
                                max((ch[self.nl], min((self.nc * 2, 128))))  # channels
        c6, c7 = make_divisible(max((ch[self.nl * 2] // 4, self.reg_max * 4, 16)), 4), \
                                max((ch[self.nl * 2], min((self.nc * 2, 128))))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3, g=4), 
                          nn.Conv2d(c2, 4 * self.reg_max, 1, groups=4)) for x in ch[:self.nl])
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch[:self.nl])
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3, g=4), 
                          nn.Conv2d(c4, 4 * self.reg_max, 1, groups=4)) for x in ch[self.nl:self.nl*2])
        self.cv5 = nn.ModuleList(
            nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nc, 1)) for x in ch[self.nl:self.nl*2])
        self.cv6 = nn.ModuleList(
            nn.Sequential(Conv(x, c6, 3), Conv(c6, c6, 3, g=4), 
                          nn.Conv2d(c6, 4 * self.reg_max, 1, groups=4)) for x in ch[self.nl*2:self.nl*3])
        self.cv7 = nn.ModuleList(
            nn.Sequential(Conv(x, c7, 3), Conv(c7, c7, 3), nn.Conv2d(c7, self.nc, 1)) for x in ch[self.nl*2:self.nl*3])
        self.dfl = DFL(self.reg_max)
        self.dfl2 = DFL(self.reg_max)
        self.dfl3 = DFL(self.reg_max)

    def forward(self, x):
        shape = x[0].shape  # BCHW
        d1 = []
        d2 = []
        d3 = []
        for i in range(self.nl):
            d1.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
            d2.append(torch.cat((self.cv4[i](x[self.nl+i]), self.cv5[i](x[self.nl+i])), 1))
            d3.append(torch.cat((self.cv6[i](x[self.nl*2+i]), self.cv7[i](x[self.nl*2+i])), 1))
        if self.training:
            return [d1, d2, d3]
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (d1.transpose(0, 1) for d1 in make_anchors(d1, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([di.view(shape[0], self.no, -1) for di in d1], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        box2, cls2 = torch.cat([di.view(shape[0], self.no, -1) for di in d2], 2).split((self.reg_max * 4, self.nc), 1)
        dbox2 = dist2bbox(self.dfl2(box2), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        box3, cls3 = torch.cat([di.view(shape[0], self.no, -1) for di in d3], 2).split((self.reg_max * 4, self.nc), 1)
        dbox3 = dist2bbox(self.dfl3(box3), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        #y = [torch.cat((dbox, cls.sigmoid()), 1), torch.cat((dbox2, cls2.sigmoid()), 1), torch.cat((dbox3, cls3.sigmoid()), 1)]
        #return y if self.export else (y, [d1, d2, d3])
        y = torch.cat((dbox3, cls3.sigmoid()), 1)
        return y if self.export else (y, d3)

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
        for a, b, s in zip(m.cv4, m.cv5, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
        for a, b, s in zip(m.cv6, m.cv7, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)



class Segment(Detect):
    # YOLO Segment head for segmentation models
    def __init__(self, nc=80, nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        p = self.proto(x[0])
        bs = p.shape[0]

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class Panoptic(Detect):
    # YOLO Panoptic head for panoptic segmentation models
    def __init__(self, nc=80, sem_nc=93, nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, ch, inplace)
        self.sem_nc = sem_nc
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.uconv = UConv(ch[0], ch[0]//4, self.sem_nc+self.nc)
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)


    def forward(self, x):
        p = self.proto(x[0])
        s = self.uconv(x[0])
        bs = p.shape[0]

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = self.detect(self, x)
        if self.training:
            return x, mc, p, s
        return (torch.cat([x, mc], 1), p, s) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p, s))
    

class BaseModel(nn.Module):
    # YOLO base model
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        # print(x[0].shape, len(x))
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, DualDetect, TripleDetect, DDetect, Detect_TADDH, Detect_LSCD, GDDetect, DualDDetect, TripleDDetect, Segment)):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
            # m.grid = list(map(fn, m.grid))
        return self


class DetectionModel(BaseModel):
    # YOLO detection model
    def __init__(self, cfg='yolo.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, DDetect, Detect_TADDH, Detect_LSCD, GDDetect, Segment)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment)) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            # check_anchor_order(m)
            # m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            m.bias_init()  # only run once
        if isinstance(m, (DualDetect, TripleDetect, DualDDetect, TripleDDetect)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            #forward = lambda x: self.forward(x)[0][0] if isinstance(m, (DualSegment)) else self.forward(x)[0]
            forward = lambda x: self.forward(x)[0]
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            # check_anchor_order(m)
            # m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            m.bias_init()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        # print(x[0].shape, len(x))
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLO augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y


Model = DetectionModel  # retain YOLO 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    # YOLO segmentation model
    def __init__(self, cfg='yolo-seg.yaml', ch=3, nc=None, anchors=None):
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLO classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  # yaml, model, number of classes, cutoff index
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        # Create a YOLO classification model from a YOLO detection model
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        # Create a YOLO classification model from a *.yaml file
        self.model = None


def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)

    # Args
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]



    # Parse a YOLO model.yaml dictionary
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # m = eval(m) if isinstance(m, str) else m  # eval strings
        try:
            if m == 'node_mode':
                m = d[m]
                if len(args) > 0:
                    if args[0] == 'head_channel':
                        args[0] = int(d[args[0]])
            t = m
            m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]  # get module
        except:
            pass


        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(NameError):
                    try:
                       # args[j] = eval(a) if isinstance(a, str) else a  # eval strings
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
                    except:
                        args[j] = a

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
            Conv, GhostConv, AConv, ConvTranspose, SCConv, AdaptiveDilatedConv, SRFD, DRFD,
            Bottleneck, SPP, SPPF, DWConv, BottleneckCSP, nn.ConvTranspose2d, DWConvTranspose2d, SPPCSPC, ADown, GADown, ELAN1,
            RepNCSPELAN4, DBBNCSPELAN4, GRepNCSPELAN4, OREPANCSPELAN4, SPPELAN, ASPPELAN, GSPPELAN, GASPPELAN, GSConv,
            CSPStage, SPDConv, DGCST, RepNCSPELAN4AKConv1,  HWD_ADown, DRBNCSPELAN4, RepNCSPELAN4_CAA, RGCSPELAN, C3, C2f, C2f_ContextGuided}:

            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, SPPCSPC}:
                args.insert(2, n)  # number of repeats
                n = 1



        elif m is nn.BatchNorm2d:
            args = [ch[f]]

        ###attention #####
        elif m in {EMA_attention, NAMAttention,
                       DoubleAttention}:
            c2 = ch[f]
            args = [c2, *args]
        ###attention #####

        elif m is CSMHSA:
            c1 = [ch[x] for x in f]
            c2 = ch[f[-1]]
            args = [c1, c2]

        elif m is CFC_CRB:
            c1 = ch[f]
            c2 = c1 // 2
            args = [c1, *args]
        elif m is SFC_G2:
            c1 = [ch[x] for x in f]
            c2 = c1[0]
            args = [c1]

        elif m is Concat:
            c2 = sum(ch[x] for x in f)

        elif m is SDI:
            args = [[ch[x] for x in f]]

        elif m is space_to_depth:
            c2 = 4 * ch[f]

        elif m is FocusFeature:
            c1 = [ch[x] for x in f]
            c2 = int(c1[1] * 0.5 * 3)
            args = [c1, *args]

        elif m is Shortcut:
            c2 = ch[f[0]]
        elif m is ReOrg:
            c2 = ch[f] * 4
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        # TODO: channel, gw, gd
        elif m in {Detect, DualDetect, TripleDetect, DDetect, Detect_TADDH, Detect_LSCD, GDDetect, DualDDetect, TripleDDetect, Segment}:
            args.append([ch[x] for x in f])
            # if isinstance(args[1], int):  # number of anchors
            #     args[1] = [list(range(args[1] * 2))] * len(f)
            if m in {Segment}:
                args[2] = make_divisible(args[2] * gw, 8)

        elif m in {RevCol,
                   MobileNetV4ConvSmall, MobileNetV4ConvMedium, MobileNetV4ConvLarge, MobileNetV4HybridMedium,
                   MobileNetV4HybridLarge
                   }:
            if m is RevCol:
                args[1] = [make_divisible(min(k, max_channels) * width, 8) for k in args[1]]
                args[2] = [max(round(k * depth), 1) for k in args[2]]
            m = m(*args)
            c2 = m.channel

        elif m in {CPCA, LocalWindowAttention, CARAFE, DySample, TransNeXt_AggregatedAttention, ELA_HSFPN}:
            c2 = ch[f]
            args = [c2, *args]

        elif m in {CGAFusion, CAFMFusion}:
            c2 = ch[f[1]]
            args = [c2, *args]

        elif m is Contract:
            c2 = ch[f] * args[0] ** 2

        elif m in {WaveletUnPool}:
            c2 = ch[f] // 4

        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

    #     if isinstance(c2, list):
    #         is_backbone = True
    #         m_ = m
    #         m_.backbone = True
    #     else:
    #
    #        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
    #        t = str(m)[8:-2].replace('__main__.', '')  # module type
    #     m.np = sum(x.numel() for x in m_.parameters())  # number params
    #     # m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
    #     # LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
    #     # save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
    #     # layers.append(m_)
    #     m_.i, m_.f, m_.type = i + 4 if is_backbone else i, f, t  # attach index, 'from' index, type
    #     if verbose:
    #         LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}')  # print
    #     save.extend(x % (i + 4 if is_backbone else i) for x in ([f] if isinstance(f, int) else f) if
    #                 x != -1)  # append to savelist
    #     layers.append(m_)
    #
    #     if i == 0:
    #         ch = []
    #     # ch.append(c2)
    #     if isinstance(c2, list):
    #         ch.extend(c2)
    #         for _ in range(5 - len(ch)):
    #             ch.insert(0, 0)
    #     else:
    #         ch.append(c2)
    # return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolo.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)
    model.eval()

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  # report fused model summary
        model.fuse()
