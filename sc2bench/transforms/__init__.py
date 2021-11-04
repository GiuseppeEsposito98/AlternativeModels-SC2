from .codec import CODEC_TRANSFORM_MODULE_DICT
from .misc import MISC_TRANSFORM_MODULE_DICT

TRANSFORM_MODULE_DICT = dict()
TRANSFORM_MODULE_DICT.update(CODEC_TRANSFORM_MODULE_DICT)
TRANSFORM_MODULE_DICT.update(MISC_TRANSFORM_MODULE_DICT)
