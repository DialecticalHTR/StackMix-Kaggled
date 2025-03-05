# -*- coding: utf-8 -*-
from typing import Dict

from .base import BaseConfig

from .bentham import BenthamConfig
from .peter import PeterConfig
from .iam import IAMConfig
from .iam_tbluche import TblucheIAMConfig
from .hkr import HKRConfig
from .saintgall import SaintGallConfig
from .washington import WashingtonConfig
from .schwerin import SchwerinConfig
from .konzil import KonzilConfig
from .patzig import PatzigConfig
from .ricordi import RicordiConfig
from .schiller import SchillerConfig

from .cyrillic import CyrillicConfig
from .dialectic import DialecticConfig

CONFIGS: Dict[str, BaseConfig] = {
    'bentham': BenthamConfig,
    'peter': PeterConfig,
    'iam': IAMConfig,
    'iam_tbluche': TblucheIAMConfig,
    'hkr': HKRConfig,
    'saintgall': SaintGallConfig,
    'washington': WashingtonConfig,
    'schwerin': SchwerinConfig,
    'konzil': KonzilConfig,
    'patzig': PatzigConfig,
    'ricordi': RicordiConfig,
    'schiller': SchillerConfig,
    'cyrillic': CyrillicConfig,
    'dialectic': DialecticConfig,
}

__all__ = [
    'CONFIGS',
]
