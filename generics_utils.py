#%% --- --- --- --- --- --- --- --- ---
# Iterables
def print_iterable(iterable:iter, item_prefix=None, item_sufix=None):
    """
    Print each item in an iterable with optional prefixes and suffixes.

    Args:
        iterable (iter): The iterable to print items from.
        item_prefix (str, optional): Prefix to add before each item. Defaults to None.
        item_sufix (str, optional): Suffix to add after each item. Defaults to None.
    """
    for item in iterable:
        print(f"{item_prefix}{item}{item_sufix}")

#%% --- --- --- --- --- --- --- --- ---
# Dictionaries
def print_dictionary(dictionary:dict, level_sep:str="\t", key_sufix:str="- ", current_level:int=0):
    """
    Print a dictionary with nested dictionaries and iterable values in a structured format.

    Args:
        dictionary (dict): The dictionary to print.
        level_sep (str, optional): The separator to use for indentation. Defaults to "\t".
        key_sufix (str, optional): The suffix to add after each key. Defaults to "- ".
        current_level (int, optional): The current level of indentation. Defaults to 0.
    """
    def _print_dict(subdictionary, level):
        print_dictionary(subdictionary, level_sep=level_sep, key_sufix=key_sufix, current_level=level)
    
    def _print_iter(subiter, level):
        for i,item in enumerate(subiter):
            print(f"{level * level_sep}{key_sufix}{i}:", end="")
            if isinstance(item, dict):
                print("")
                _print_dict(item, level+1)
            elif isinstance(item, (list, tuple, type(iter))):
                print("")
                _print_iter(item, level+1)
            else:
                print(f" {item}")
    
    for k,v in dictionary.items():
        print(f"{current_level * level_sep}{key_sufix}{k}:", end="")
        if isinstance(v, dict):
            print("")
            _print_dict(v, current_level+1)
        elif isinstance(v, (list, tuple, type(iter))):
            print("")
            _print_iter(v, current_level+1)
        else:
            print(f" {v}")

def merge_dictionaries_with_subfixed_keys(dictionary:dict, subdictionary:dict, prefix_key:str):
    """
    Merge a dictionary with another dictionary, where the keys of the second dictionary are prefixed with a given string.

    Args:
        dictionary (dict): The dictionary to merge into.
        subdictionary (dict): The dictionary to merge.
        prefix_key (str): The prefix to add to the keys of the second dictionary.

    Returns:
        dict: The merged dictionary.
    """
    for k in subdictionary:
        dictionary[f"{prefix_key}_{k}"] = subdictionary[k]
    return dictionary

def add_value_or_merge_dictionary_with_subfixed_keys(dictionary:dict, key:str, val) -> None:
    """
    Add a value to a dictionary, or merge a dictionary with a subfixed key.

    Args:
        dictionary (dict): The dictionary to add the value to.
        key (str): The key to add the value to.
        val (any): The value to add.

    Returns:
        None: The updated dictionary.
    """
    # add each key-value pair from the dictionary with the given key as a prefix
    if isinstance(val, dict):
        dictionary = merge_dictionaries_with_subfixed_keys(dictionary, val, key)
    # add the simple value as a key-value pair with the given key
    else:
        dictionary[key] = val
    return dictionary
        
def flatten_dictionary_with_joined_keys(dictionary:dict, key_join_char:str="_", previous_key:str=None):
    """
    Flatten a nested dictionary by joining keys with a given character.

    Args:
        dictionary (dict): The dictionary to flatten.
        key_join_char (str): The character to join keys with.
        previous_key (str, optional): The previous key in the flattened dictionary. Defaults to None.

    Returns:
        dict: The flattened dictionary.
    """
    current_key = ""
    new_dictionary = {}
    if not previous_key:
        previous_key = ""
    else:
        previous_key = previous_key + key_join_char
    for k,v in dictionary.items():
        current_key = previous_key + k
        if isinstance(v, dict):
            v = flatten_dictionary_with_joined_keys(v, previous_key=current_key, key_join_char=key_join_char)
            for sk,sv in v.items():
                new_dictionary[sk] = sv
        else:
            new_dictionary[current_key] = v
    return new_dictionary
