"""
Calculator tool - Safe mathematical expression evaluation.
Supports basic arithmetic and common mathematical functions.
"""

import ast
import math
import operator

from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    """Input schema for calculator."""

    expression: str = Field(
        description=(
            "Mathematical expression to evaluate " "(e.g., '2 + 2', 'sqrt(16)', 'sin(3.14159/2)')"
        )
    )


# Supported operators
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Supported functions
FUNCTIONS = {
    # Basic math
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    # Powers and logarithms
    "sqrt": math.sqrt,
    "pow": pow,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    # Trigonometric
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    # Hyperbolic
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    # Angular conversion
    "degrees": math.degrees,
    "radians": math.radians,
    # Rounding
    "ceil": math.ceil,
    "floor": math.floor,
    "trunc": math.trunc,
    # Other
    "factorial": math.factorial,
    "gcd": math.gcd,
}

# Supported constants
CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}


class SafeEvaluator(ast.NodeVisitor):
    """Safe AST-based expression evaluator."""

    def visit_Expression(self, node):
        """Handle Expression node."""
        return self.visit(node.body)

    def visit_Constant(self, node):
        """Handle numeric constants."""
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    def visit_Num(self, node):
        """Handle numeric literals (Python < 3.8 compatibility)."""
        return node.n

    def visit_Name(self, node):
        """Handle named constants like pi, e."""
        name = node.id.lower()
        if name in CONSTANTS:
            return CONSTANTS[name]
        raise ValueError(f"Unknown constant: {node.id}")

    def visit_BinOp(self, node):
        """Handle binary operations."""
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)

        if op_type not in OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")

        return OPERATORS[op_type](left, right)

    def visit_UnaryOp(self, node):
        """Handle unary operations."""
        operand = self.visit(node.operand)
        op_type = type(node.op)

        if op_type not in OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")

        return OPERATORS[op_type](operand)

    def visit_Call(self, node):
        """Handle function calls."""
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are supported")

        func_name = node.func.id.lower()
        if func_name not in FUNCTIONS:
            raise ValueError(f"Unknown function: {node.func.id}")

        args = [self.visit(arg) for arg in node.args]
        return FUNCTIONS[func_name](*args)

    def visit_Tuple(self, node):
        """Handle tuples (for functions like min, max with multiple args)."""
        return tuple(self.visit(el) for el in node.elts)

    def visit_List(self, node):
        """Handle lists."""
        return [self.visit(el) for el in node.elts]

    def generic_visit(self, node):
        """Reject any other node types."""
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.

    Supports:
    - Basic arithmetic: +, -, *, /, //, %, **
    - Functions: sqrt, sin, cos, tan, log, exp, abs, round, min, max, etc.
    - Constants: pi, e, tau

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        Result of the calculation as a string

    Examples:
        calculate("2 + 2") -> "4"
        calculate("sqrt(16)") -> "4.0"
        calculate("sin(pi/2)") -> "1.0"
        calculate("2**10") -> "1024"
    """
    try:
        # Clean up the expression
        expr = expression.strip()

        # Handle empty expression
        if not expr:
            return "Error: Empty expression"

        # Parse the expression into an AST
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            return f"Error: Invalid syntax - {e}"

        # Evaluate safely
        evaluator = SafeEvaluator()
        result = evaluator.visit(tree)

        # Format the result
        if isinstance(result, float):
            # Clean up floating point representation
            if result.is_integer() and abs(result) < 1e15:
                return str(int(result))
            elif abs(result) < 1e-10:
                return "0"
            else:
                # Round to reasonable precision
                formatted = f"{result:.10g}"
                return formatted
        elif isinstance(result, complex):
            return str(result)
        else:
            return str(result)

    except ZeroDivisionError:
        return "Error: Division by zero"
    except ValueError as e:
        return f"Error: {e}"
    except OverflowError:
        return "Error: Result too large"
    except Exception as e:
        return f"Error evaluating expression: {e}"


def get_supported_functions() -> str:
    """Return a list of supported mathematical functions and constants."""
    funcs = ", ".join(sorted(FUNCTIONS.keys()))
    consts = ", ".join(sorted(CONSTANTS.keys()))
    return f"Supported functions: {funcs}\nSupported constants: {consts}"


# Tool metadata for registry
TOOL_CONFIG = {
    "name": "calculate",
    "description": (
        "Evaluate a mathematical expression. Supports basic arithmetic (+, -, *, /, **, %), "
        "functions (sqrt, sin, cos, tan, log, exp, abs, round, min, max, factorial, etc.), "
        "and constants (pi, e, tau). Examples: '2 + 2', 'sqrt(16)', 'sin(pi/2)', '2**10'"
    ),
    "input_schema": CalculatorInput,
    "requires_confirmation": False,
}

__all__ = [
    "calculate",
    "get_supported_functions",
    "CalculatorInput",
    "TOOL_CONFIG",
]
