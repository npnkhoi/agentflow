import re
import signal
import threading


# allows lockout of termination while writing to results file
class ShutdownFlag:
    def __init__(self):
        self.lock = threading.Lock()

        def raise_flag(_signum, _frame):
            with self.lock:
                raise InterruptedError()

        signal.signal(signal.SIGTERM, raise_flag)
        signal.signal(signal.SIGINT, raise_flag)


def camel_to_snake(camel_case_string: str) -> str:
    """
    Converts a CamelCase string to snake_case.

    Args:
        camel_case_string: The input string in CamelCase.

    Returns:
        The converted string in snake_case.

    # Example usage
    camel_string_1 = "myVariableName"
    snake_string_1 = camel_to_snake(camel_string_1)
    print(f"'{camel_string_1}' converted to snake_case: '{snake_string_1}'")

    camel_string_2 = "AnotherExampleString"
    snake_string_2 = camel_to_snake(camel_string_2)
    print(f"'{camel_string_2}' converted to snake_case: '{snake_string_2}'")

    camel_string_3 = "HTTPStatus"
    snake_string_3 = camel_to_snake(camel_string_3)
    print(f"'{camel_string_3}' converted to snake_case: '{snake_string_3}'")
    """
    # Use a regex to find uppercase letters that are not at the beginning
    # and insert an underscore before them.
    # Then, convert the entire string to lowercase.
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", camel_case_string)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    snake_case_string = s.lower()
    return snake_case_string
