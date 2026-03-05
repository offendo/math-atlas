from .herald import HeraldFormatter
from .goedel import GoedelFormatter
from .atlas import ATLASFormatter
from .kimina import KiminaFormatter
from .few_shot import FewShotFormatter
from .reform import ReformFormatter
from enum import Enum, auto


class Models(Enum):
    HERALD = auto()
    GOEDEL = auto()
    ATLAS = auto()
    KIMINA = auto()
    GPTOSS = auto()
    REFORM = auto()

    @classmethod
    def from_str(cls, model_name):
        if "herald" in model_name:
            return cls.HERALD
        elif "goedel" in model_name:
            return cls.GOEDEL
        elif "atlas" in model_name:
            return cls.ATLAS
        elif "kimina" in model_name:
            return cls.KIMINA
        elif "gpt" in model_name:
            return cls.GPTOSS
        elif "qwen" in model_name:
            return cls.GPTOSS
        elif "llama" in model_name or "l\"lama" in model_name:
            # treat various llama-style names the same way
            return cls.GPTOSS
        elif "reform" in model_name:
            return cls.REFORM
        else:
            raise ValueError(f"unsupported model: {model_name}")


def get_formatter(model_name):
    model = Models.from_str(model_name)
    formatter_dict = {
        Models.HERALD: HeraldFormatter,
        Models.GOEDEL: GoedelFormatter,
        Models.ATLAS: ATLASFormatter,
        Models.KIMINA: KiminaFormatter,
        Models.GPTOSS: FewShotFormatter,
        Models.REFORM: ReformFormatter,
    }
    if model in formatter_dict:
        return formatter_dict[model]()
    raise NotImplementedError(f"no support for {model_name}")
