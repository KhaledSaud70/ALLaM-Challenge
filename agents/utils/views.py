from colorama import Fore, Style
from enum import Enum


class OperationColor(Enum):
    PoetryGeneratorAgent = Fore.GREEN
    VerseValidation = Fore.LIGHTRED_EX


def print_output(output: str, operation: str):
    print(f"{OperationColor[operation].value}{operation}: {output}{Style.RESET_ALL}")
