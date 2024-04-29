#%% --- --- --- --- --- --- --- --- ---
# Imports
from sklearn.model_selection import GroupShuffleSplit

#%% --- --- --- --- --- --- --- --- ---
# Subset splits
def train_test_group_split(dataframe,
                           X_column_name:str,
                           y_column_name:str,
                           group_column_name:str,
                           train_size:float=.8,
                           print_percentage:bool=False,
                           print_comparison:bool=False):
    """
    Split a dataframe into training and testing sets based on a grouping column.

    Args:
        dataframe (pd.DataFrame): The input dataframe.
        X_column_name (str): The name of the column containing the features.
        y_column_name (str): The name of the column containing the target variable.
        group_column_name (str): The name of the column containing the grouping variable.
        train_size (float, optional): The proportion of the data to include in the training set. Defaults to 0.8.
        print_percentage (bool, optional): Whether to print the training and testing amounts. Defaults to False.
        print_comparison (bool, optional): Whether to print a comparison of the training and testing IDs. Defaults to False.

    Returns:
        tuple: The training and testing dataframes.
    """
    
    X, y, groups = dataframe[X_column_name], dataframe[y_column_name], dataframe[group_column_name]
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size)
    train_index, test_index = next(gss.split(X=X, y=y, groups=groups))

    if print_percentage:
        print("Train and test amounts:")
        print(f"  Train: {len(train_index)} ({100*len(train_index)/(len(train_index)+len(test_index)):.2f}%)")
        print(f"  Test: {len(test_index)} ({100*len(test_index)/(len(train_index)+len(test_index)):.2f}%)")
        print("")

    if print_comparison:
        print("IDs in train and test:")
    train_ids = list(dataframe.loc[train_index,:][group_column_name].unique())
    test_ids = list(dataframe.loc[test_index,:][group_column_name].unique())
    for id in train_ids:
        if id in test_ids:
            raise ValueError(f"Something went wrong. Training ID '{id}' is also in testing.")
    if print_comparison:
        print("  No training ID in testing. All good.")
        print("")
    
    return dataframe.iloc[train_index], dataframe.iloc[test_index]