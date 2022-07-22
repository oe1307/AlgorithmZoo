import yaml
from attrdict import AttrDict

from utils import setup_logger

logger = setup_logger(__name__)


def read_yaml(path):
    obj = dict()
    logger.debug(f"\n [ READ ] {path}")
    with open(path, mode="r") as f:
        obj = yaml.safe_load(f)

    msg = ""
    for key, value in obj.items():
        msg += f"\n {key} ← {value}"
    msg += "\n"

    logger.debug(msg)

    return AttrDict(obj)


def make_config(path, args):
    logger.debug(f"\n [ READ ] {path}\n")
    with open(path, mode="r") as f:
        obj = yaml.safe_load(f)
    args = vars(args)
    for k in args.keys():
        if args[k] is None and k in obj.keys():
            args[k] = obj[k]
    obj.update(args)

    msg = ""
    for key, value in obj.items():
        msg += f"\n {key} ← {value}"
    msg += "\n"
    logger.debug(msg)

    return AttrDict(obj)
