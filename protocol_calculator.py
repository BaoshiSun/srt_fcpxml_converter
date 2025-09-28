"""Command-line calculator with safe expression evaluation.

This module evaluates arithmetic expressions using Python's AST so
that only a restricted set of operations is allowed. It can be used as a
stand-alone program or imported as a helper function.
"""

from __future__ import annotations

import argparse
import ast
import operator
from typing import Any, Callable, Dict

# Map supported binary operations to the underlying Python operator.
_BINARY_OPERATORS: Dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Map supported unary operations.
_UNARY_OPERATORS: Dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# The collection of nodes that are allowed in the parsed AST.
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
)

_ALLOWED_OPERATOR_NODES = tuple(_BINARY_OPERATORS) + tuple(_UNARY_OPERATORS)


class CalculatorError(ValueError):
    """Raised when an invalid expression is encountered."""


def _ensure_allowed(node: ast.AST) -> None:
    """Ensure every node in the AST is allowed.

    Parameters
    ----------
    node:
        The AST node to validate.
    """

    if isinstance(node, _ALLOWED_OPERATOR_NODES):
        return

    if not isinstance(node, _ALLOWED_NODES):
        raise CalculatorError(f"Unsupported expression component: {type(node).__name__}")

    for child in ast.iter_child_nodes(node):
        _ensure_allowed(child)


def _eval_ast(node: ast.AST) -> Any:
    """Recursively evaluate a validated AST node."""

    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise CalculatorError(f"Unsupported constant type: {type(node.value).__name__}")

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPERATORS:
            raise CalculatorError(f"Unsupported unary operator: {op_type.__name__}")
        return _UNARY_OPERATORS[op_type](_eval_ast(node.operand))

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINARY_OPERATORS:
            raise CalculatorError(f"Unsupported binary operator: {op_type.__name__}")
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        return _BINARY_OPERATORS[op_type](left, right)

    raise CalculatorError(f"Unsupported expression component: {type(node).__name__}")


def evaluate(expression: str) -> Any:
    """Evaluate a mathematical expression safely.

    Parameters
    ----------
    expression:
        A string containing the expression to evaluate.

    Returns
    -------
    Any
        The numeric result of the expression.

    Raises
    ------
    CalculatorError
        If the expression contains unsupported syntax or operations.
    """

    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:  # pragma: no cover - simple error passthrough
        raise CalculatorError("Invalid expression") from exc

    _ensure_allowed(parsed)
    return _eval_ast(parsed)


def main() -> None:
    """Entry-point for the command-line interface."""

    parser = argparse.ArgumentParser(description="Evaluate arithmetic expressions safely.")
    parser.add_argument("expression", help="Arithmetic expression to evaluate, e.g. '2*(3+4)'.")
    args = parser.parse_args()

    result = evaluate(args.expression)
    print(result)


if __name__ == "__main__":
    main()
