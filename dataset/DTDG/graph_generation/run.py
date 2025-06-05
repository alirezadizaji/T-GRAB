from importlib import import_module
import inspect
import sys
from sys import argv
from typing import Optional

from .graph_generator import GraphGenerator



def find_graph_generation_class(module_name) -> GraphGenerator:
    module = import_module(module_name)
    target = None

    # Return the last class in the module file that is the subclass of `GraphGenerator`.
    for _, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, GraphGenerator) and obj.__module__ == module_name:
            target = obj

    assert target is not None
    return target

if __name__ == "__main__":
    script_name = argv[1]
    argv.remove(script_name)
    module_name = f"T-GRAB.dataset.DTDG.graph_generation.{script_name}"
    cls_graph_generator = find_graph_generation_class(module_name)
    parser = cls_graph_generator.get_parser()
    try:
        args = parser.parse_args()
        args_dict = vars(args)
    except:
        parser.print_help()
        sys.exit(0)

    graph_generator: 'GraphGenerator' = cls_graph_generator(args)
    graph_generator.create_data(args_dict)