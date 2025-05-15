from functools import wraps
import networkx as nx
import numpy as np
from typing import Any, Callable

def counter(call_init: int=0):
    def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
        """ decorator to record number of calls of a function """
        @wraps(function)
        def wrapper(*args, **kwargs):
            out = function(*args, **kwargs)
            wrapper.call += 1
            return out
        
        wrapper.call = call_init
        return wrapper
    return decorator
