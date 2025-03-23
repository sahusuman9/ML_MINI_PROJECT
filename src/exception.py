import sys  # Importing sys module to handle system-specific parameters and functions
from src.logger import logging  # Importing custom logging module for logging errors


def error_message_detail(error, error_detail: sys):
    """
    Extracts detailed error message including file name and line number where the error occurred.
    
    Args:
        error (Exception): The error that occurred.
        error_detail (sys): System details from sys module to extract traceback information.
    
    Returns:
        str: Formatted error message with script name, line number, and error details.
    """
    _, _, exc_tb = error_detail.exc_info()  # Extract traceback details
    file_name = exc_tb.tb_frame.f_code.co_filename  # Get the filename where error occurred
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)  # Formatting error message with details
    )
    return error_message  # Returning formatted error message


class CustomException(Exception):
    """
    Custom Exception class to handle and format error messages.
    """
    def __init__(self, error_message, error_details: sys):
        """
        Initializes the CustomException class with a detailed error message.
        
        Args:
            error_message (str): The error message to be displayed.
            error_details (sys): System details to extract traceback information.
        """
        super().__init__(error_message)  # Calling parent Exception class constructor
        self.error_message = error_message_detail(error_message, error_detail=error_details)  # Generating detailed error message
        
    def __str__(self):
        """
        Returns the formatted error message when exception is printed.
        
        Returns:
            str: Formatted error message.
        """
        return self.error_message
