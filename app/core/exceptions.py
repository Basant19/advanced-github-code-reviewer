#E:\advanced-github-code-reviewer\app\core\exceptions.py
import sys
from typing import Any

def error_message_detail(error: Any, error_detail: sys):
    exc_type, exc_value, exc_tb = error_detail.exc_info()
    
    if exc_tb:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "Unknown"
        line_number = "Unknown"

    return (
        f"Error occurred in script [{file_name}] "
        f"line [{line_number}] "
        f"message [{str(error)}]"
    )

class CustomException(Exception):
    def __init__(self, error_message: Any, error_details: sys = sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail=error_details
        )

    def __str__(self):
        return self.error_message