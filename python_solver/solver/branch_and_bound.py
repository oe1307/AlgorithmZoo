from attrdict import AttrDict


class BranchAndBound:
    """分枝限定法

    MIP(混合整数計画問題)を解くことができる

    Attributes:
    属性の名前 (属性の型): 属性の説明

    Args:
    引数の名前 (引数の型): 引数の説明

    """

    def __init__(self, timelimit=60, stype="depth"):
        self.timelimit = timelimit
        self.stype = stype

    def solve(problem):
        result = dict()
        result["objective"] = None
        return AttrDict(result)
