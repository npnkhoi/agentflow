## Constants

from enum import Enum

class AnnotationSource(str, Enum):
    HUMAN = "human"
    MODEL = "model"


class DemoSelect(str, Enum):
    """Built-in demo selection strategies.

    Not a closed set: `DemoConfig.select` also accepts any name passed to
    DemoPool.register_strategy(). See agentflow/demo.py.
    """

    RANDOM = "random"
    SIMILAR = "similar"
    DIVERSE = "diverse"