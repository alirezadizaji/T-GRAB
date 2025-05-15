from typing_extensions import Any, Callable, List, Union

def verify_kwd_existence(kwd_name: Union[str, List[str]]):
    """ This decorator supports verifying the existence of a keyword or a list of keywords among the output of a function. 
    The output of function should be an instance of dictionary."""
    def decorator(func: Callable[[Any], Any]):
        def wrapper(*args, **kwargs):
            nonlocal kwd_name
            output = func(*args, **kwargs)
            assert isinstance(output, dict), "This decorator only supports functions that returns a dictionary type."
            if isinstance(kwd_name, str):
                kwd_name = [kwd_name]
            for k in kwd_name:
                assert k in output.keys(), f"{k} is missing among the output of the {func.__name__}."
            
            return output
        return wrapper
    return decorator
