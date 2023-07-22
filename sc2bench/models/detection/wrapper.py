import torch
from torchdistill.common.main_util import load_ckpt

from .registry import load_detection_model
from .transform import RCNNTransformWithCompression
from ..registry import get_compression_model
from ..wrapper import register_wrapper_class, WRAPPER_CLASS_DICT
from ...analysis import AnalyzableModule, check_if_analyzable


@register_wrapper_class
class InputCompressionDetectionModel(AnalyzableModule):
    """
    Wrapper module for input compression model followed by detection model.

    :param detection_model: object detection model
    :type detection_model: nn.Module
    :param device: torch device
    :type device: torch.device or str
    :param codec_params: transform sequence configuration for codec
    :type codec_params: dict or None
    :param compression_model: compression model
    :type compression_model: nn.Module or None
    :param uses_cpu4compression_model: whether to use CPU instead of GPU for `comoression_model`
    :type uses_cpu4compression_model: bool
    :param pre_transform_params: pre-transform parameters
    :type pre_transform_params: dict or None
    :param post_transform_params: post-transform parameters
    :type post_transform_params: dict or None
    :param adaptive_pad_kwargs: keyword arguments for AdaptivePad
    :type adaptive_pad_kwargs: dict or None
    """
    def __init__(self, detection_model, device, codec_params=None, compression_model=None,
                 uses_cpu4compression_model=False, pre_transform_params=None, post_transform_params=None,
                 analysis_config=None, adaptive_pad_kwargs=None, **kwargs):
        if analysis_config is None:
            analysis_config = dict()

        super().__init__()
        detection_model.transform = \
            RCNNTransformWithCompression(
                detection_model.transform, device, codec_params, analysis_config.get('analyzer_configs', list()),
                analyzes_after_compress=analysis_config.get('analyzes_after_compress', False),
                compression_model=compression_model, uses_cpu4compression_model=uses_cpu4compression_model,
                pre_transform_params=pre_transform_params, post_transform_params=post_transform_params,
                adaptive_pad_kwargs=adaptive_pad_kwargs
            )
        self.device = device
        self.uses_cpu4compression_model = uses_cpu4compression_model
        self.detection_model = detection_model

    def use_cpu4compression(self):
        """
        Changes the device of the compression model to CPU.
        """
        if self.uses_cpu4compression_model and self.detection_model.transform.compression_model is not None:
            self.detection_model.transform.compression_model = self.detection_model.transform.compression_model.cpu()

    def forward(self, x):
        """
        Args:
            x (ImageList): input sample.

        Returns:
            Tensor: output tensor from self.detection_model.
        """
        return self.detection_model(x)

    def activate_analysis(self):
        self.activated_analysis = True
        if check_if_analyzable(self.detection_model.transform):
            self.detection_model.transform.activate_analysis()

    def deactivate_analysis(self):
        self.activated_analysis = False
        self.detection_model.transform.deactivate_analysis()
        if check_if_analyzable(self.detection_model.transform):
            self.detection_model.transform.deactivate_analysis()

    def analyze(self, compressed_obj):
        if not self.activated_analysis:
            return

        for analyzer in self.analyzers:
            analyzer.analyze(compressed_obj)
        if check_if_analyzable(self.detection_model.transform):
            self.detection_model.transform.analyze(compressed_obj)

    def summarize(self):
        for analyzer in self.analyzers:
            analyzer.summarize()
        if check_if_analyzable(self.detection_model.transform):
            self.detection_model.transform.summarize()

    def clear_analysis(self):
        for analyzer in self.analyzers:
            analyzer.clear()
        if check_if_analyzable(self.detection_model.transform):
            self.detection_model.transform.clear_analysis()


def get_wrapped_detection_model(wrapper_model_config, device):
    """
    Get a wrapped object detection model

    :param wrapper_model_config: wrapper model configuration
    :type wrapper_model_config: dict
    :param device: torch device
    :type device: torch.device
    :return: model: wrapped object detection model
    :rtype: model: nn.Module
    """
    wrapper_model_name = wrapper_model_config['name']
    if wrapper_model_name not in WRAPPER_CLASS_DICT:
        raise ValueError('wrapper_model_name `{}` is not expected'.format(wrapper_model_name))

    compression_model_config = wrapper_model_config.get('compression_model', None)
    compression_model = get_compression_model(compression_model_config, device)
    detection_model_config = wrapper_model_config['detection_model']
    model = load_detection_model(detection_model_config, device)
    wrapped_model = WRAPPER_CLASS_DICT[wrapper_model_name](model, compression_model=compression_model, device=device,
                                                           **wrapper_model_config['params'])
    ckpt_file_path = wrapper_model_config.get('ckpt', None)
    if ckpt_file_path is not None:
        load_ckpt(ckpt_file_path, model=wrapped_model, strict=False)
    return wrapped_model
