#%% --- --- --- --- --- --- --- --- ---
# Imports


#%% --- --- --- --- --- --- --- --- ---
# Simple Progress Bar
def simple_progress_bar(progress: float,
                        data: dict = None,
                        bar_length: int = 10,
                        bar_open_char: str = "|",
                        bar_fill_char: str = "█",
                        void_char: str = " ",
                        bar_close_char: str = "|",
                        print_percentage: bool = True) -> str:
    """
    Return a simple progress bar with the given data.

    Args:
        batch_index (int): The current batch index.
        batches_amount (int): The total number of batches.
        data (dict, optional): A dictionary of extra data to display. Defaults to None (no extra data printed).
        bar_length (int, optional): The length of the progress bar. Defaults to 10.
        bar_open_char (str, optional): The character to use for the opening of the progress bar. Defaults to "|".
        bar_fill_char (str, optional): The character to use for filling the progress bar. Defaults to "█".
        void_char (str, optional): The character to use for empty space in the progress bar. Defaults to " ".
        bar_close_char (str, optional): The character to use for the closing of the progress bar. Defaults to "|".

    Returns:
        str: A string representing the progress bar and metrics.
    """
    # Create a string to represent the losses/metrics
    info_str_vals = []
    for key, value in data.items():
        # In case of a list
        if isinstance(value, list):
            value_str = [f"{v:.3f}" for v in value]
            info_str_vals.append(f"{key}: [{' '.join(value_str)}]")
        # In case of a single value
        else:
            info_str_vals.append(f"{key}: {value:.4f}")
    info_str = ', '.join(info_str_vals)
    
    # Create a string to represent the loading bar
    fill_amount = int(progress * bar_length)
    void_amount = bar_length - fill_amount
    bar = bar_open_char + bar_fill_char * fill_amount + void_char * void_amount + bar_close_char

    # Add progress percentage if solicited
    if print_percentage:
        bar = f"{bar} {progress * 100 : 6.2f}%"
    
    # Return the loading bar and info string
    return f"{bar} | {info_str}"

def print_simple_progress_bar(progress: float,
                              data: dict = ModuleNotFoundError,
                              bar_length: int = 10,
                              bar_open_char: str = "|",
                              bar_fill_char: str = "█",
                              void_char: str = " ",
                              bar_close_char: str = "|",
                              print_percentage: bool = True) -> None:
    """
    Display a simple progress bar with the given data.

    Args:
        batch_index (int): The current batch index.
        batches_amount (int): The total number of batches.
        data (dict, optional): A dictionary of extra data to display. Defaults to None (no extra data printed).
        bar_length (int, optional): The length of the progress bar. Defaults to 10.
        bar_open_char (str, optional): The character to use for the opening of the progress bar. Defaults to "|".
        bar_fill_char (str, optional): The character to use for filling the progress bar. Defaults to "█".
        void_char (str, optional): The character to use for empty space in the progress bar. Defaults to " ".
        bar_close_char (str, optional): The character to use for the closing of the progress bar. Defaults to "|".

    Returns:
        str: A string representing the progress bar and metrics.
    """
    if progress >= 1.0:
        print_end = '\n'
    else:
        print_end = '\r'
    bar = simple_progress_bar(
        progress=progress,
        data=data,
        bar_length=bar_length,
        bar_open_char=bar_open_char,
        bar_fill_char=bar_fill_char,
        void_char=void_char,
        bar_close_char=bar_close_char,
        print_percentage=print_percentage,
    )
    # Print the loading bar and info string to the console
    print(bar, end=print_end)
    
# %%
