from .inference import AudioTagging, SoundEventDetection
from .config import SAMPLE_RATE, CLASSES_NUM, labels, classes_num

__version__ = "0.1.1"

__all__ = [
    'SAMPLE_RATE',
    'CLASSES_NUM',
    'labels',
    'classes_num',
    'AudioTagging',
    'SoundEventDetection'
]