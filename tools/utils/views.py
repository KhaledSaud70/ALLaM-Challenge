from colorama import Fore, Style
from enum import Enum


class ToolColor(Enum):
    MeterClassifier = Fore.LIGHTGREEN_EX
    RhymeClassifier = Fore.CYAN
    ProsodyWriter = Fore.LIGHTBLUE_EX
    Diacritizer = Fore.LIGHTRED_EX
    PoemRetrieval = Fore.LIGHTWHITE_EX


def print_tool_output(output: str, tool: str):
    print(f"{ToolColor[tool].value}{tool}: {output}{Style.RESET_ALL}")
