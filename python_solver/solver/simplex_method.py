from attrdict import AttrDict


class SimplexMethod:
    """単体法

    LP(線形計画問題)のみ有効

    Attributes:
    属性の名前 (属性の型): 属性の説明

    Args:
    引数の名前 (引数の型): 引数の説明

    """

    def __init__(self, timelimit=60):
        self.timelimit = timelimit

    def solve(problem):
        result = dict()
        result["objective"] = None
        return AttrDict(result)
