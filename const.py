## Constants

from enum import Enum

class AnnotationSource(str, Enum):
    HUMAN = "human"
    MODEL = "model"


class DemoSelect(str, Enum):
    RANDOM = "random"
    SIMILAR = "similar"
    DIVERSE = "diverse" # not implemented yet