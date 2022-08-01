from base import Variable


class Problem:
    """問題を管理するためのクラス

    Attributes:
    属性の名前 (属性の型): 属性の説明

    Args:
    引数の名前 (引数の型): 引数の説明

    """

    def __init__(self, name="problem1", case="minimize"):
        self.name = name
        self.case = case
        self.variables = list()
        self.objectibe_function = 0
        self.constrains = None

    def __add__(self, other):
        print(other)
