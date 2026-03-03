from .herald import format_example as herald_format_example, parse_output as herald_parse_output
from .goedel import format_example as goedel_format_example, parse_output as goedel_parse_output
from .atlas import format_example as atlas_format_example, parse_output as atlas_parse_output
from .kimina import format_example as kimina_format_example, parse_output as kimina_parse_output
from .gptoss import format_example as gptoss_format_example, parse_output as gptoss_parse_output


def get_formatter(model_name):
    model_name, og = model_name.lower(), model_name
    if "herald" in model_name:
        return herald_format_example
    elif "goedel" in model_name:
        return goedel_format_example
    elif "atlas" in model_name:
        return atlas_format_example
    elif "kimina" in model_name:
        return kimina_format_example
    elif "gpt" in model_name:
        return gptoss_format_example

    raise NotImplementedError(f"no support for {og}")


def get_output_parser(model_name):
    model_name, og = model_name.lower(), model_name
    if "herald" in model_name:
        return herald_parse_output
    elif "goedel" in model_name:
        return goedel_parse_output
    elif "atlas" in model_name:
        return atlas_parse_output
    elif "kimina" in model_name:
        return kimina_parse_output
    elif "gpt" in model_name:
        return gptoss_parse_output

    raise NotImplementedError(f"no support for {og}")
