from colorama import Fore, Style
from enum import Enum


class OperationColor(Enum):
    PoemGenerator = Fore.LIGHTYELLOW_EX
    VerseReviewer = Fore.LIGHTMAGENTA_EX
    VerseReviser = Fore.LIGHTGREEN_EX
    QueryTransform = Fore.GREEN
    PoemEvaluator = Fore.BLUE


def print_operation_output(output: str, operation: str):
    print(f"{OperationColor[operation].value}{operation}: \n{output}{Style.RESET_ALL}")
