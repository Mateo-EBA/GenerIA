#%% --- --- --- --- --- --- --- --- ---
# Imports
import os
import random
import traceback
import time
import json
from datetime import datetime
import numpy as np

import pandas as pd
import wandb

from name_generation import generate_random_name_with_current_time

#%% --- --- --- --- --- --- --- --- ---
# Debug Mockups
class WandbRunMockup():
    def __init__(self, name, dir, config):
        self.name = name
        self.dir = dir
        self.config = config
        
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        json_path = os.path.join(self.dir, "setup.json")
        with open(json_path, 'w') as file:
            json.dump(self.config, file, indent=4)
            
    def log(self, data:dict):
        df = pd.DataFrame(data, index=[0])
        path = os.path.join(self.dir, "logs.csv")
        if os.path.exists(path):
            df.to_csv(path, mode='a', index=False, header=False)
        else:
            df.to_csv(path, index=False)

class WandbSweepMockup():
    def __init__(self, sweep_config):
        self.name = ''.join(random.choices("abcdefghijklmnopqrstuvwxyz", k=6))
        self.raw_config = sweep_config.copy()
        self.pruned_config = self._prune_parameters(self.raw_config["parameters"])

    def _prune_parameters(self, params:dict):
        config = {}
        for k,v in params.items():
            if "parameters" in v:
                v = v["parameters"]
            if isinstance(v, dict):
                config[k] = self._prune_parameters(v)
        return config
    
    def _get_random_config(self, src_config):
        config = {}
        try:
            for k,v in src_config.items():
                if "value" in v:
                    config[k] = v["value"]
                elif "values" in v:
                    i = np.random.choice(range(len(v["values"])))
                    config[k] = v["values"][i]
                elif "min" in v:
                    min_v = v["min"]
                    max_v = v["max"]
                    config[k] = min_v + np.random.random() * (max_v - min_v)
                elif "parameters" in v:
                    config[k] = self._get_random_config(v["parameters"])
        except:
            print(f"Error on '{k}: {v}'")
            raise ValueError("BOOM")
        return config
    
    def init_run(self, name, dir):
        config = self._get_random_config(self.pruned_config)
        return WandbRunMockup(name, dir, config)

#%% --- --- --- --- --- --- --- --- ---
#%% Functions
def wandb_simple_run(run_config:dict, train_func, run_name:str=None, entity:str=None, project:str=None,
                     login_key:str=None, storage_dir:str=None, run_mode:str="online", use_wandb:bool=True):
    """
    Runs a machine learning experiment with Weights and Biases.
    
    This function logs the run configuration, initializes a WandB
    run with the specified entity and project, calls the given training function
    with the initialized run, and finishes the run once training is complete.
    
    Args:
        run_config (dict): A dictionary of hyperparameters and other settings for this run.
        train_func (callable): The function that trains the machine learning model and logs results. Recieves the current wandb Run.
        run_name (str, optional): Name of the run. Default is None, in which case a random name with the current date is used.
        entity (str, optional): Name of the company or team to which this project belongs. Default is None.
        project (str, optional): Name of the project to which this run belongs. Default is None.
        login_key (str, optional): Your Weights and Biases API key. Default is None.
        storage_dir (str, optional): The directory in which to store run information locally. Default is None.
        run_mode (str, optional): "online", "offline" or "disabled". Default is "online".
    
    Returns:
        None
    
    Raises:
        Any errors that occur during the training process may be raised.
    """
    
    # If a storage directory is provided, set the required environment variables to point to it
    if storage_dir is not None:
        os.environ['WANDB_DIR'] = storage_dir
        os.environ['WANDB_CACHE_DIR'] = storage_dir
        os.environ['WANDB_CONFIG_DIR'] = storage_dir
    
    # If a login key is provided, log in to Weights and Biases automatically
    if not use_wandb and login_key is not None:
        wandb.login(key=login_key, relogin=True)
    
    # Initialize name if none given
    if run_name is None:
        run_name = generate_random_name_with_current_time(time_format='%y%m%d-%H%M%S', delimiter='-')
    
    # Initialize a new run with the provided configuration options
    # and the entity and project (if provided)
    run = wandb.init(
        name=run_name,
        config=run_config,
        entity=entity,
        project=project,
        dir=storage_dir,
        mode=run_mode,
    )
    
    # Call the provided training function, passing in the initialized run object
    train_func(run)
    
    # Complete the run and close the Weights and Biases session
    wandb.finish()
    
def wandb_start_sweep(sweep_config:dict, train_func, sweep_name:str=None, sweep_runs_count:int=None,
                      entity:str=None, project:str=None, login_key:str=None, storage_dir:str=None, run_mode:str="online",
                      use_wandb:bool=True, random_seed:int=None):
    """
    Performs a sweep training with Weights and Biases.
    
    This function sets the sweep configuration, initializes it on WandB
    with the specified entity and project, calls the given training function
    on each run, and shows the traceback if an exception happens.
    
    Args:
        sweep_config (dict): A dictionary of hyperparameters and other settings for this sweep.
        sweep_name (str, optional): A name for the sweep, if left as None a random string will be given.
        train_func (callable): The function that trains the machine learning model and logs results. Recieves the current wandb Run.
        sweep_runs_count (int, optional): Number of maximum runs to perform on this sweep.
        entity (str, optional): Name of the company or team to which this project belongs. Default is None.
        project (str, optional): Name of the project to which this run belongs. Default is None.
        login_key (str, optional): Your Weights and Biases API key. Default is None.
        storage_dir (str, optional): The directory in which to store run information locally. Default is None.
        run_mode (str, optional): "online", "offline" or "disabled". Default is "online".
        random_seed (int, optional): the seed for the random package. Default is current time.
    
    Returns:
        None
    
    Raises:
        Any errors that occur during the training process may be raised.
    """
    
    # If mode is provided, set the required environment variable
    if run_mode is not None:
        os.environ['WANDB_MODE'] = run_mode
    
    # Add sweep name if given
    if sweep_name:
        sweep_config["name"] = sweep_name
    
    # Create sweep ID
    if use_wandb:
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            entity=entity,
            project=project,
        )
    else:
        sweep_id = WandbSweepMockup(sweep_config)
    
    wandb_run_sweep(
        sweep_id=sweep_id,
        train_func=train_func,
        sweep_runs_count=sweep_runs_count,
        entity=entity,
        project=project,
        login_key=login_key,
        storage_dir=storage_dir,
        run_mode=run_mode,
        use_wandb=use_wandb,
        random_seed=random_seed,
    )
    
def wandb_run_sweep(sweep_id:str, train_func, sweep_runs_count:int=None, entity:str=None, project:str=None,
                    login_key:str=None, storage_dir:str=None, run_mode:str="online", use_wandb:bool=True,
                    random_seed:int=None):
    """
    Performs a sweep training with Weights and Biases.
    
    This function sets the sweep configuration, initializes it on WandB
    with the specified entity and project, calls the given training function
    on each run, and shows the traceback if an exception happens.
    
    Args:
        sweep_id (str): Sweep ID generated by wandb.sweep with a config of hyperparameters and other settings.
        train_func (callable): The function that trains the machine learning model and logs results. Recieves the current wandb Run.
        sweep_runs_count (int, optional): Number of maximum runs to perform on this sweep.
        entity (str, optional): Name of the company or team to which this project belongs. Default is None.
        project (str, optional): Name of the project to which this run belongs. Default is None.
        login_key (str, optional): Your Weights and Biases API key. Default is None.
        storage_dir (str, optional): The directory in which to store run information locally. Default is the project's dir.
        run_mode (str, optional): "online", "offline" or "disabled". Default is "online".
        random_seed (int, optional): the seed for the random package. Default is current time.

    Returns:
        None
    
    Raises:
        Any errors that occur during the training process may be raised.
    """
    
    # If mode is provided, set the required environment variable
    if run_mode is not None:
        os.environ['WANDB_MODE'] = run_mode
        
    # If a storage directory is provided, set the required environment variables to point to it
    if storage_dir is not None:
        os.environ['WANDB_DIR'] = storage_dir
        os.environ['WANDB_CACHE_DIR'] = storage_dir
        os.environ['WANDB_CONFIG_DIR'] = storage_dir
                
    # Login to wandb
    if use_wandb and login_key is not None:
        wandb.login(key=login_key, relogin=True)
    else:
        sweep_mockup = sweep_id
    
    # The method called on each run
    def sweep_run_catch():
        try:
            # Generate run name
            random.seed(datetime.now()) # Make sure its different than previous one
            run_name = generate_random_name_with_current_time(time_format='%y%m%d-%H%M%S', delimiter='-')
            if random_seed: # Change seed
                random.seed(random_seed)
            else:
                random.seed(datetime.now())
            
            # Initialize
            if use_wandb:
                run = wandb.init(name=run_name, dir=storage_dir, mode=run_mode)
            else:
                run = sweep_mockup.init_run(name=run_name, dir=storage_dir)
            
            # Train
            train_func(run, use_wandb)
            
            # Finish
            if use_wandb:
                wandb.finish()
            
            # Sleep 10 seconds to make sure run finishes properly
            time.sleep(10)
            
        # Handle any exceptions
        except Exception:
            print("Error during sweep:")
            traceback.print_exc() # Print the traceback to know where the exception comes from
            print()
            if use_wandb:
                wandb.finish()
            exit(1) # Exit the program with a non-zero status code to let WandB know that the run crashed

    # Making sure no run is active
    if use_wandb:
        wandb.finish()
    
    # Start sweep
    if use_wandb:
        wandb.agent(
            sweep_id,
            function=sweep_run_catch,
            entity=entity,
            project=project,
            count=sweep_runs_count,
        )
    else:
        for _ in range(sweep_runs_count):
            sweep_run_catch()