import sys
import logging

def error_message_detail(error, error_detail: sys):
    _, _, exc_tab = error_detail.exc_info()
    file_name = exc_tab.tb_frame.f_code.co_filename
    line_num = exc_tab.tb_lineno
    error_message = f"Error occurred in Python script [{file_name}] line number [{line_num}] error message [{error}]"
    return error_message

class custom_exception(Exception):
    def __init__(self, error, error_detail: sys):
        super().__init__(error)
        self.error_message = error_message_detail(error, error_detail=error_detail)

    def __str__(self):
        return self.error_message




