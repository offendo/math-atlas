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
    FEW_SHOT = auto()
    REFORM = auto()

    @classmethod
    def from_str(cls, model_name):
        model_name = model_name.lower()
        if "herald" in model_name:
            return cls.HERALD
        elif "goedel" in model_name:
            return cls.GOEDEL
        elif "atlas" in model_name:
            return cls.ATLAS
        elif "kimina" in model_name:
            return cls.KIMINA
        elif "reform" in model_name:
            return cls.REFORM
        elif "gpt" in model_name:
            return cls.FEW_SHOT
        elif "qwen" in model_name:
            return cls.FEW_SHOT
        elif "llama" in model_name or "l\"lama" in model_name:
            # treat various llama-style names the same way
            return cls.FEW_SHOT
        else:
            raise ValueError(f"unsupported model: {model_name}")


def get_formatter(model_name, **kwargs):
    model = Models.from_str(model_name)
    formatter_dict = {
        Models.HERALD: HeraldFormatter,
        Models.GOEDEL: GoedelFormatter,
        Models.ATLAS: ATLASFormatter,
        Models.KIMINA: KiminaFormatter,
        Models.FEW_SHOT: FewShotFormatter,
        Models.REFORM: ReformFormatter,
    }
    if model in formatter_dict:
        # FewShotFormatter accepts additional prompt file parameters
        if model == Models.FEW_SHOT:
            return formatter_dict[model](model=model_name, **kwargs)
        else:
            return formatter_dict[model](model=model_name)
    raise NotImplementedError(f"no support for {model_name}")
