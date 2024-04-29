#%% --- --- --- --- --- --- --- --- ---
# Imports
import json
import random
from datetime import datetime

#%% --- --- --- --- --- --- --- --- ---
# Functions
def generate_random_name(delimiter='-') -> str:
    """
    Get a random adjective and noun from the `names_options` JSON file
    and return them joined by a delimiter.

    Args:
        delimiter (str, optional): The delimiter to use when joining the adjective and noun. Default is '-'.

    Returns:
        str: A string containing a random adjective and noun joined by the delimiter.
    """
    with open('data/names_options.json') as f:
        data = json.load(f)
    adjectives = data['adjectives']
    nouns = data['nouns']
    random_adj = random.choice(adjectives)
    random_noun = random.choice(nouns)
    return delimiter.join([random_adj, random_noun])

def generate_random_name_with_current_time(time_format='%y%m%d-%H%M%S', delimiter='-'):
    """
    Get a random adjective and noun from the `names_options` JSON file, join them with a delimiter,
    and prefix the string with the current time formatted according to the time_format.

    Args:
        time_format (str, optional): The format for the current time. Default is '%y%m%d-%H%M%S'.
        delimiter (str, optional): The delimiter to use when joining the adjective and noun. Default is '-'.

    Returns:
        str: A string containing the current time prefix, a random adjective, and noun joined by the delimiter.
    """
    current_time = datetime.datetime.now().strftime(time_format)
    random_name = generate_random_name(delimiter)
    return delimiter.join([current_time, random_name])