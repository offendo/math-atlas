from .few_shot import FewShotFormatter

# simple subclass for clarity; the generic few-shot behavior is provided
# by `FewShotFormatter`, so this file exists purely for backwards
# compatibility if something explicitly imports `Qwen35Formatter`.

class Qwen35Formatter(FewShotFormatter):
    pass
