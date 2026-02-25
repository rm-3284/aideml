import copy
import json
from pathlib import Path
from typing import Type, TypeVar

import dataclasses_json
from ..journal import Journal


def dumps_json(obj: dataclasses_json.DataClassJsonMixin):
    """Serialize AIDE dataclasses (such as Journals) to JSON."""
    if isinstance(obj, Journal):
        obj = copy.deepcopy(obj)
        node2parent = {n.id: n.parent.id for n in obj.nodes if n.parent is not None}
        for n in obj.nodes:
            n.parent = None
            n.children = set()

    obj_dict = obj.to_dict()

    if isinstance(obj, Journal):
        obj_dict["node2parent"] = node2parent  # type: ignore
        obj_dict["__version"] = "2"

    return json.dumps(obj_dict, separators=(",", ":"))


def dump_json(obj: dataclasses_json.DataClassJsonMixin, path: Path):
    with open(path, "w") as f:
        f.write(dumps_json(obj))


G = TypeVar("G", bound=dataclasses_json.DataClassJsonMixin)


def loads_json(s: str, cls: Type[G]) -> G:
    """Deserialize JSON to AIDE dataclasses."""

    import logging
    obj_dict = json.loads(s)
    obj = cls.from_dict(obj_dict)

    # Validate node.step for Journal nodes
    if isinstance(obj, Journal):
        valid_nodes = []
        for n in obj.nodes:
            if not (isinstance(n.step, int) and n.step >= 0):
                logging.warning(f"Invalid node.step detected during deserialization: node.id={n.id}, step={n.step}. Setting to None.")
                n.step = None
            valid_nodes.append(n)
        obj.nodes = valid_nodes

        id2nodes = {n.id: n for n in obj.nodes}
        for child_id, parent_id in obj_dict["node2parent"].items():
            if child_id in id2nodes and parent_id in id2nodes:
                id2nodes[child_id].parent = id2nodes[parent_id]
                id2nodes[child_id].__post_init__()
            else:
                logging.warning(f"Invalid parent/child reference in node2parent: child_id={child_id}, parent_id={parent_id}")
    return obj


def load_json(path: Path, cls: Type[G]) -> G:
    with open(path, "r") as f:
        return loads_json(f.read(), cls)
