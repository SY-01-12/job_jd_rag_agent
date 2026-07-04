import ast
import operator as op
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool


class CalculatorInput(BaseModel):
    expression: str = Field(..., description="数学表达式，例如：3 * (5 + 2)")


_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _safe_eval(node):
    """
    安全计算 AST 节点，只允许数字和基础数学运算。
    """
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("只支持数字常量")

    if isinstance(node, ast.Num):  # 兼容旧版本 Python
        return node.n

    if isinstance(node, ast.BinOp):
        operator_type = type(node.op)
        if operator_type not in _ALLOWED_OPERATORS:
            raise ValueError("不支持的运算符")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _ALLOWED_OPERATORS[operator_type](left, right)

    if isinstance(node, ast.UnaryOp):
        operator_type = type(node.op)
        if operator_type not in _ALLOWED_OPERATORS:
            raise ValueError("不支持的一元运算符")
        operand = _safe_eval(node.operand)
        return _ALLOWED_OPERATORS[operator_type](operand)

    raise ValueError("表达式中包含不允许的内容")


def calculator_func(expression: str) -> str:
    """
    安全计算数学表达式。
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
        return f"计算结果：{result}"
    except ZeroDivisionError:
        return "计算失败：除数不能为 0"
    except Exception as e:
        return f"计算失败：{str(e)}"


calculator_tool = StructuredTool.from_function(
    func=calculator_func,
    name="calculator",
    description="用于计算基础数学表达式，例如加减乘除、括号、幂运算等。",
    args_schema=CalculatorInput,
)