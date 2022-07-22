_VARIABLE_NO = -1


class Variable:
    """変数を管理するためのクラス

    Attributes:
    属性の名前 (属性の型): 属性の説明

    Args:
    引数の名前 (引数の型): 引数の説明

    """

    def __init__(self, name=None, vtype="continuous"):
        global _VARIABLE_NO
        _VARIABLE_NO += 1

        self.value = None
        self.vtype = vtype
        self.name = f"x[{_VARIABLE_NO}]" if name is None else name
#         参考： https://github.com/flab-coder/flopt/blob/681d2f98824a52bb95de73676823d7ae59c6a013/flopt/variable.py#L366-L521
