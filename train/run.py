from importlib import import_module
import inspect
from sys import argv
from typing import Optional, TypeVar

from TSA.train.trainer import Trainer

def find_trainer_class(module_name) -> Optional['Trainer']:
    module = import_module(module_name)
    target = None

    # Return the last class that is the subclass of `Trainer`.
    for _, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, Trainer) and obj.__module__ == module_name:
            target = obj

    return target

if __name__ == "__main__":
    # Take the `model_name` by finding the last part after splitting python script module name by `.`.
    script_name = argv[1]
    argv.remove(script_name)
    model_name = script_name.split(".")[-1]
    
    print(f"*** {model_name} is removed from the argument vector. ***", flush=True)
    module_name = f"TSA.train.{script_name}"
    trainer_cls = find_trainer_class(module_name)
    trainer: 'Trainer' = trainer_cls()
    trainer.run()