"""
Python code execution tool - Execute Python code snippets.

Features:
- Persistent variables between executions (per session)
- REPL-style automatic result display
- True timeout enforcement via multiprocessing
- Special commands: %vars, %clear, %reset, %history
- Execution history tracking
- Optional NumPy/Pandas support (if installed)
- Restricted execution environment for safety

Requires user confirmation for safety.
"""

import ast
import io
import multiprocessing as mp
import threading
import traceback
from collections import OrderedDict
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from queue import Empty
from typing import Any

from pydantic import BaseModel, Field

# Configuration constants
MAX_OUTPUT_SIZE = 10000  # Maximum output size in characters
MAX_RESULT_SIZE = 2000  # Maximum result repr size
MAX_HISTORY_SIZE = 50  # Maximum history entries per session

# Security limits
MAX_LOOP_ITERATIONS = 100000  # Maximum iterations for any loop
MAX_RECURSION_DEPTH = 100  # Maximum recursion depth
MAX_COLLECTION_SIZE = 10000  # Maximum size for lists/dicts/sets
MAX_STRING_LENGTH = 100000  # Maximum string length


class PythonExecInput(BaseModel):
    """Input schema for Python code execution."""

    code: str = Field(description="Python code to execute")
    timeout: int = Field(
        default=30,
        description="Execution timeout in seconds (default: 30, max: 60)",
    )
    persistent: bool = Field(
        default=True,
        description="Persist variables between calls (default: True)",
    )
    session_id: str = Field(
        default="default",
        description="Session ID for isolating persistent state between users/chats",
    )


@dataclass
class ExecutionRecord:
    """Record of a code execution."""

    code: str
    success: bool
    output: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


@dataclass
class SessionState:
    """State for a session including variables and history."""

    variables: dict[str, Any] = field(default_factory=dict)
    history: list[ExecutionRecord] = field(default_factory=list)


_MAX_SESSIONS = 1000

# Session execution contexts (persistent variables and history); LRU-evicted at _MAX_SESSIONS
_session_states: OrderedDict[str, SessionState] = OrderedDict()
_session_states_lock: threading.RLock = threading.RLock()

# Current session ID for interactive (single-threaded) mode.
# WARNING: not thread-safe — do not rely on this in concurrent (assistant) contexts.
# In assistant mode, pass session_id explicitly via the tool input schema instead.
_current_session: str = "default"


def set_session(session_id: str) -> None:
    """Set the current session for persistent state.

    For backward compatibility in interactive (single-threaded) mode only.
    Not thread-safe — do not use in concurrent contexts such as assistant mode.
    """
    global _current_session
    _current_session = session_id


def _get_session_state(session_id: str | None = None) -> SessionState:
    """Get or create session state for the given session_id.

    Falls back to "default" (never to the global _current_session) so that
    concurrent callers without an explicit session_id cannot interfere with each other.

    Applies LRU eviction when the number of sessions reaches _MAX_SESSIONS.
    """
    sid = session_id if session_id is not None else "default"
    with _session_states_lock:
        if sid in _session_states:
            _session_states.move_to_end(sid)
            return _session_states[sid]
        if len(_session_states) >= _MAX_SESSIONS:
            _session_states.popitem(last=False)
        _session_states[sid] = SessionState()
        return _session_states[sid]


def get_context(session_id: str | None = None) -> dict[str, Any]:
    """Get or create execution context for a session."""
    return _get_session_state(session_id).variables


def get_history(session_id: str | None = None) -> list[ExecutionRecord]:
    """Get execution history for a session."""
    return _get_session_state(session_id).history


def add_to_history(code: str, success: bool, output: str, session_id: str | None = None) -> None:
    """Add an execution record to history."""
    with _session_states_lock:
        state = _get_session_state(session_id)
        state.history.append(ExecutionRecord(code=code, success=success, output=output))
        if len(state.history) > MAX_HISTORY_SIZE:
            state.history = state.history[-MAX_HISTORY_SIZE:]


def clear_context(session_id: str | None = None) -> None:
    """Clear execution context for a session (keeps history)."""
    with _session_states_lock:
        state = _get_session_state(session_id)
        state.variables.clear()


def clear_history(session_id: str | None = None) -> None:
    """Clear execution history for a session."""
    with _session_states_lock:
        state = _get_session_state(session_id)
        state.history.clear()


# Module names to block via AST Import/ImportFrom node inspection.
# These are checked structurally so that the name appearing in a string
# literal, comment, or variable name does NOT trigger a false positive.
_BLOCKED_MODULES: frozenset[str] = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "pathlib",
        "importlib",
        "pickle",
        "marshal",
        "shelve",
        "socket",
        "urllib",
        "requests",
        "http",
        "ftplib",
        "smtplib",
        "telnetlib",
        "ctypes",
        "multiprocessing",
        "threading",
        "asyncio",
        "concurrent",
        "signal",
        "pty",
        "tty",
        "fcntl",
        "termios",
        "resource",
        "sysconfig",
        "platform",
        "tempfile",
        "glob",
        "fnmatch",
    }
)

# Built-in call names checked via AST to avoid substring false positives
# (e.g. "profile(" containing "file(", "hexec(" containing "exec(").
_DANGEROUS_CALL_NAMES: frozenset[str] = frozenset({"eval", "exec", "compile", "open", "file"})

# Dunder/special-name patterns that remain as substring checks — these are
# distinctive enough that false positives are not a concern.
_DANGEROUS_DUNDER_PATTERNS: list[str] = [
    "__import__",
    "__builtins__",
    "__class__",
    "__bases__",
    "__subclasses__",
    "__mro__",
    "__globals__",
    "__code__",
    "__reduce__",
    "__getstate__",
    "__setstate__",
]

# Alias kept for code that iterates _DANGEROUS_CALL_PATTERNS directly.
_DANGEROUS_CALL_PATTERNS: list[str] = _DANGEROUS_DUNDER_PATTERNS

# Keep DANGEROUS_PATTERNS as a combined list for backward compatibility
DANGEROUS_PATTERNS: list[str] = (
    [f"{name}(" for name in sorted(_DANGEROUS_CALL_NAMES)]
    + _DANGEROUS_DUNDER_PATTERNS
    + sorted(_BLOCKED_MODULES)
)


# Dangerous attribute names that could be used for sandbox escape
DANGEROUS_ATTRS = {
    "__class__",
    "__bases__",
    "__subclasses__",
    "__mro__",
    "__globals__",
    "__code__",
    "__builtins__",
    "__import__",
    "__loader__",
    "__spec__",
    "__reduce__",
    "__reduce_ex__",
    "__getstate__",
    "__setstate__",
    "__init_subclass__",
    "__set_name__",
    "__class_getitem__",
    "__dict__",
    "gi_frame",
    "gi_code",
    "f_globals",
    "f_locals",
    "f_builtins",
    "co_code",
    "func_globals",
    "func_code",
    "tb_frame",
    "tb_next",
    "tb_lasti",
    "tb_lineno",
    "__traceback__",
}


def _safe_getattr(obj: Any, name: str, *args: Any) -> Any:
    """getattr wrapper that blocks access to dangerous attributes at runtime."""
    if not isinstance(name, str):
        raise TypeError(f"attribute name must be string, not '{type(name).__name__}'")
    if name in DANGEROUS_ATTRS:
        raise AttributeError(f"Access to attribute '{name}' is blocked in sandbox")
    return getattr(obj, name, *args)


def _safe_hasattr(obj: Any, name: str) -> bool:
    """Safe hasattr that respects attribute restrictions."""
    try:
        _safe_getattr(obj, name)
        return True
    except (AttributeError, RuntimeError):
        return False


def _safe_setattr(obj: Any, name: str, value: Any) -> None:
    """setattr wrapper that blocks assignment to dangerous attributes at runtime."""
    if not isinstance(name, str):
        raise TypeError(f"attribute name must be string, not '{type(name).__name__}'")
    if name in DANGEROUS_ATTRS:
        raise AttributeError(f"Access to attribute '{name}' is blocked in sandbox")
    setattr(obj, name, value)


def _safe_type(*args: Any) -> Any:
    """Restricted type() — single-arg form only; 3-arg metaclass form is blocked."""
    if len(args) == 1:
        return type(args[0])
    raise TypeError("type() with 3 arguments is not allowed in sandbox")


# Safe built-in functions (restricted set)
SAFE_BUILTINS = {
    # Types
    "bool": bool,
    "int": int,
    "float": float,
    "str": str,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "frozenset": frozenset,
    "bytes": bytes,
    "bytearray": bytearray,
    "complex": complex,
    "object": object,
    "type": _safe_type,
    "slice": slice,
    # Functions
    "abs": abs,
    "all": all,
    "any": any,
    "ascii": ascii,
    "bin": bin,
    "callable": callable,
    "chr": chr,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "format": format,
    "getattr": _safe_getattr,
    "hasattr": _safe_hasattr,
    "hash": hash,
    "hex": hex,
    "id": id,
    "input": None,  # Disabled
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "setattr": _safe_setattr,
    "sorted": sorted,
    "sum": sum,
    "zip": zip,
    "vars": None,  # Disabled for security
    "dir": dir,
    "property": property,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    "super": super,
    # Exceptions (for handling)
    "BaseException": BaseException,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "ZeroDivisionError": ZeroDivisionError,
    "StopIteration": StopIteration,
    "RuntimeError": RuntimeError,
    "NotImplementedError": NotImplementedError,
    "OverflowError": OverflowError,
    "RecursionError": RecursionError,
    "MemoryError": MemoryError,
    "AssertionError": AssertionError,
    "ImportError": ImportError,
    "ModuleNotFoundError": ModuleNotFoundError,
    # Constants
    "True": True,
    "False": False,
    "None": None,
    "Ellipsis": Ellipsis,
    "NotImplemented": NotImplemented,
}

# Safe modules that can be imported
SAFE_MODULES = {
    "math",
    "random",
    "string",
    "re",
    "json",
    "datetime",
    "collections",
    "itertools",
    "functools",
    "operator",
    "statistics",
    "decimal",
    "fractions",
    "textwrap",
    "unicodedata",
    "hashlib",
    "hmac",
    # "copy" removed: deepcopy calls __reduce_ex__ via C code, bypassing sandbox
    "pprint",
    "typing",
    # Additional safe modules
    "csv",
    "dataclasses",
    "enum",
    "uuid",
    "struct",
    "contextlib",
    "abc",
    "numbers",
    "bisect",
    "heapq",
    "array",
    "cmath",
    "time",  # Limited functionality (sleep, time)
}

# Optional data science modules (added if available)
OPTIONAL_MODULES = {"numpy", "pandas", "scipy"}

# Check for optional modules and add them if available
_AVAILABLE_OPTIONAL: dict[str, bool] = {}
for _mod_name in OPTIONAL_MODULES:
    try:
        __import__(_mod_name)
        SAFE_MODULES.add(_mod_name)
        _AVAILABLE_OPTIONAL[_mod_name] = True
    except ImportError:
        _AVAILABLE_OPTIONAL[_mod_name] = False


def get_available_modules() -> dict[str, bool]:
    """Return dict of optional modules and their availability."""
    return _AVAILABLE_OPTIONAL.copy()


def _check_ast_security(tree: ast.AST) -> tuple[bool, str]:
    """
    Perform deep AST analysis for security issues.

    Checks for:
    - Dangerous attribute access
    - Dynamic attribute access with suspicious strings
    - Chained attribute access that could escape sandbox

    Returns:
        Tuple of (is_safe, error_message)
    """
    for node in ast.walk(tree):
        # Check direct attribute access
        if isinstance(node, ast.Attribute):
            attr_name = node.attr
            if attr_name in DANGEROUS_ATTRS:
                return (
                    False,
                    f"Blocked: Access to '{attr_name}' is not allowed (security)",
                )

            # Check for double-underscore attributes (dunder)
            if attr_name.startswith("__") and attr_name.endswith("__"):
                # Allow some safe dunders
                safe_dunders = {
                    "__len__",
                    "__iter__",
                    "__next__",
                    "__getitem__",
                    "__setitem__",
                    "__delitem__",
                    "__contains__",
                    "__str__",
                    "__repr__",
                    "__add__",
                    "__sub__",
                    "__mul__",
                    "__truediv__",
                    "__floordiv__",
                    "__mod__",
                    "__pow__",
                    "__eq__",
                    "__ne__",
                    "__lt__",
                    "__le__",
                    "__gt__",
                    "__ge__",
                    "__bool__",
                    "__hash__",
                    "__call__",
                    "__enter__",
                    "__exit__",
                    "__init__",
                    "__new__",
                    "__del__",
                    "__name__",
                    "__doc__",
                    "__slots__",
                }
                if attr_name not in safe_dunders:
                    return (
                        False,
                        f"Blocked: Access to '{attr_name}' is not allowed",
                    )

        # Check for getattr/setattr with string literals containing dangerous names
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in ("getattr", "setattr"):
                # Check if second argument is a dangerous string
                if len(node.args) >= 2:
                    arg = node.args[1]
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        if arg.value in DANGEROUS_ATTRS:
                            return (
                                False,
                                f"Blocked: getattr/setattr with '{arg.value}' " "is not allowed",
                            )

            # AST-based detection of dangerous built-in calls (avoids substring
            # false positives such as "profile(" matching "file(").
            call_name: str | None = None
            if isinstance(func, ast.Name):
                call_name = func.id
            elif isinstance(func, ast.Attribute):
                call_name = func.attr
            if call_name in _DANGEROUS_CALL_NAMES:
                return (
                    False,
                    f"Blocked: Call to '{call_name}' is not allowed (security)",
                )

    return True, ""


def _check_ast_imports(tree: ast.Module) -> tuple[bool, str]:
    """Walk the AST for Import/ImportFrom nodes and reject blocked module names.

    Returns:
        Tuple of (is_safe, error_message)
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _BLOCKED_MODULES:
                    return False, f"Blocked: Import of '{top}' is not allowed"
                if top not in SAFE_MODULES:
                    return (
                        False,
                        f"Blocked: Import of '{top}' is not allowed. "
                        f"Allowed: {', '.join(sorted(SAFE_MODULES))}",
                    )
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top in _BLOCKED_MODULES:
                return False, f"Blocked: Import from '{top}' is not allowed"
            if top not in SAFE_MODULES:
                return (
                    False,
                    f"Blocked: Import from '{top}' is not allowed. "
                    f"Allowed: {', '.join(sorted(SAFE_MODULES))}",
                )
    return True, ""


def _check_code_safety(code: str) -> tuple[bool, str]:
    """
    Check if code contains dangerous patterns.

    Performs both string-based and AST-based security checks.

    Returns:
        Tuple of (is_safe, error_message)
    """
    code_lower = code.lower()

    # Check non-module patterns via substring matching
    for pattern in _DANGEROUS_CALL_PATTERNS:
        if pattern.lower() in code_lower:
            return (
                False,
                f"Blocked: Code contains restricted pattern '{pattern}'",
            )

    # Parse AST to check for dangerous constructs
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    # Check import statements via AST (avoids false positives from string literals)
    is_safe, error = _check_ast_imports(tree)
    if not is_safe:
        return False, error

    # Deep AST security analysis
    is_safe, error = _check_ast_security(tree)
    if not is_safe:
        return False, error

    return True, ""


class _LoopLimiter(ast.NodeTransformer):
    """
    AST transformer that adds iteration limits to loops.

    Transforms:
        for x in iterable:
            body

    Into:
        _loop_counter_N = 0
        for x in iterable:
            _loop_counter_N += 1
            if _loop_counter_N > MAX_LOOP_ITERATIONS:
                raise RuntimeError("Loop iteration limit exceeded")
            body

    Similarly for while loops.
    """

    def __init__(self):
        self.counter_id = 0
        super().__init__()

    def _get_counter_name(self) -> str:
        self.counter_id += 1
        return f"_loop_counter_{self.counter_id}"

    def _create_limit_check(self, counter_name: str, lineno: int) -> ast.If:
        """Create an if statement that raises error if limit exceeded."""
        return ast.If(
            test=ast.Compare(
                left=ast.Name(id=counter_name, ctx=ast.Load()),
                ops=[ast.Gt()],
                comparators=[ast.Constant(value=MAX_LOOP_ITERATIONS)],
            ),
            body=[
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id="RuntimeError", ctx=ast.Load()),
                        args=[
                            ast.Constant(
                                value=f"Loop iteration limit exceeded "
                                f"({MAX_LOOP_ITERATIONS} iterations)"
                            )
                        ],
                        keywords=[],
                    ),
                    cause=None,
                )
            ],
            orelse=[],
            lineno=lineno,
            col_offset=0,
        )

    def _create_counter_increment(self, counter_name: str, lineno: int) -> ast.AugAssign:
        """Create counter += 1 statement."""
        return ast.AugAssign(
            target=ast.Name(id=counter_name, ctx=ast.Store()),
            op=ast.Add(),
            value=ast.Constant(value=1),
            lineno=lineno,
            col_offset=0,
        )

    def _create_counter_init(self, counter_name: str, lineno: int) -> ast.Assign:
        """Create counter = 0 statement."""
        return ast.Assign(
            targets=[ast.Name(id=counter_name, ctx=ast.Store())],
            value=ast.Constant(value=0),
            lineno=lineno,
            col_offset=0,
        )

    def visit_For(self, node: ast.For) -> list:
        """Transform for loops to add iteration limiting."""
        self.generic_visit(node)  # Transform nested loops first

        counter_name = self._get_counter_name()
        lineno = node.lineno

        # Add counter increment and check at the start of the loop body
        increment = self._create_counter_increment(counter_name, lineno)
        check = self._create_limit_check(counter_name, lineno)
        node.body = [increment, check] + node.body

        # Return init statement followed by the loop
        init = self._create_counter_init(counter_name, lineno)
        return [init, node]

    def visit_While(self, node: ast.While) -> list:
        """Transform while loops to add iteration limiting."""
        self.generic_visit(node)  # Transform nested loops first

        counter_name = self._get_counter_name()
        lineno = node.lineno

        # Add counter increment and check at the start of the loop body
        increment = self._create_counter_increment(counter_name, lineno)
        check = self._create_limit_check(counter_name, lineno)
        node.body = [increment, check] + node.body

        # Return init statement followed by the loop
        init = self._create_counter_init(counter_name, lineno)
        return [init, node]


def _add_loop_limits(code: str) -> str:
    """
    Transform code to add iteration limits to all loops.

    Args:
        code: Python source code

    Returns:
        Transformed code with loop limits
    """
    try:
        tree = ast.parse(code)
        transformer = _LoopLimiter()
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        return ast.unparse(new_tree)
    except SyntaxError:
        return code  # Return original if can't parse


def _transform_for_last_expr(code: str) -> tuple[str, bool]:
    """
    Transform code to capture the last expression's value.

    Returns:
        Tuple of (transformed_code, has_last_expr)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, False

    if not tree.body:
        return code, False

    last_stmt = tree.body[-1]

    # If last statement is an expression (not assignment, not print, etc.)
    if isinstance(last_stmt, ast.Expr):
        # Check if it's not a None, print call, or assignment expression
        if isinstance(last_stmt.value, ast.Constant) and last_stmt.value.value is None:
            return code, False

        # Transform: expr -> _last_result_ = expr
        new_assign = ast.Assign(
            targets=[ast.Name(id="_last_result_", ctx=ast.Store())],
            value=last_stmt.value,
            lineno=last_stmt.lineno,
            col_offset=last_stmt.col_offset,
        )
        tree.body[-1] = new_assign
        ast.fix_missing_locations(tree)

        return ast.unparse(tree), True

    return code, False


def _restricted_import(name: str, *args: Any, **kwargs: Any) -> Any:
    """Restricted import function that only allows safe modules."""
    module_name = name.split(".")[0]
    if module_name not in SAFE_MODULES:
        raise ImportError(
            f"Import of '{name}' is not allowed in restricted mode. "
            f"Allowed modules: {', '.join(sorted(SAFE_MODULES))}"
        )
    import builtins

    return builtins.__import__(name, *args, **kwargs)


def _format_error_with_context(error: Exception, code: str, error_type: str) -> str:
    """
    Format an error message with line context.

    Args:
        error: The exception that occurred
        code: The original code that was executed
        error_type: Type of error (e.g., "Syntax Error", "Name Error")

    Returns:
        Formatted error message with context
    """
    error_str = str(error)
    lines = code.split("\n")

    # Try to extract line number from error
    lineno = None
    if hasattr(error, "lineno") and error.lineno:
        lineno = error.lineno
    elif "line " in error_str.lower():
        # Try to parse line number from error message
        import re

        match = re.search(r"line (\d+)", error_str, re.IGNORECASE)
        if match:
            lineno = int(match.group(1))

    result_parts = [f"{error_type}: {error_str}"]

    # Add line context if we have a line number
    if lineno and 1 <= lineno <= len(lines):
        result_parts.append("")
        # Show context: line before, error line, line after
        start = max(0, lineno - 2)
        end = min(len(lines), lineno + 1)

        for i in range(start, end):
            line_num = i + 1
            prefix = "→ " if line_num == lineno else "  "
            result_parts.append(f"{prefix}{line_num:3d} | {lines[i]}")

    return "\n".join(result_parts)


def _handle_special_command(
    code: str, context: dict[str, Any], session_id: str | None = None
) -> str | None:
    """
    Handle special % commands.

    Returns:
        Response string if command was handled, None otherwise.
    """
    code_stripped = code.strip()
    parts = code_stripped.split(maxsplit=1)
    cmd = parts[0] if parts else ""
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "%vars":
        if not context:
            return "No variables in current context."
        with _session_states_lock:
            vars_snapshot = dict(context)
        var_list = []
        for name, value in sorted(vars_snapshot.items()):
            if name.startswith("_"):
                continue
            val_repr = repr(value)
            val_type = type(value).__name__
            if len(val_repr) > 60:
                val_repr = val_repr[:57] + "..."
            var_list.append(f"  {name}: {val_type} = {val_repr}")
        if not var_list:
            return "No user variables in current context."
        return "Variables:\n" + "\n".join(var_list)

    if cmd in ("%clear", "%reset"):
        with _session_states_lock:
            context.clear()
        return "Context cleared. All variables removed."

    if cmd == "%history":
        history = get_history(session_id)
        if not history:
            return "No execution history."
        # Show last N entries (default 10, or specified)
        try:
            n = int(arg) if arg else 10
        except ValueError:
            n = 10
        n = min(n, len(history))
        entries = history[-n:]
        lines = [f"Last {n} execution(s):"]
        for i, rec in enumerate(entries, 1):
            status = "✓" if rec.success else "✗"
            code_preview = rec.code.replace("\n", " ")[:40]
            if len(rec.code) > 40:
                code_preview += "..."
            lines.append(f"  {i}. [{rec.timestamp}] {status} {code_preview}")
        return "\n".join(lines)

    if cmd == "%modules":
        available = []
        unavailable = []
        for mod, is_avail in sorted(_AVAILABLE_OPTIONAL.items()):
            if is_avail:
                available.append(mod)
            else:
                unavailable.append(mod)
        lines = ["Optional modules:"]
        if available:
            lines.append(f"  Available: {', '.join(available)}")
        if unavailable:
            lines.append(f"  Not installed: {', '.join(unavailable)}")
        lines.append(f"\nCore modules ({len(SAFE_MODULES) - len(available)}):")
        core = sorted(SAFE_MODULES - OPTIONAL_MODULES)
        # Show in columns
        lines.append(f"  {', '.join(core)}")
        return "\n".join(lines)

    if cmd == "%help":
        return (
            "Special commands:\n"
            "  %vars       - List all variables with types\n"
            "  %clear      - Clear all variables\n"
            "  %reset      - Same as %clear\n"
            "  %history N  - Show last N executions (default: 10)\n"
            "  %modules    - Show available modules\n"
            "  %help       - Show this help"
        )

    return None


def _preload_safe_modules() -> dict[str, Any]:
    """Pre-import safe modules for use in restricted environment."""
    preloaded = {}
    for module_name in SAFE_MODULES:
        try:
            preloaded[module_name] = __import__(module_name)
        except ImportError:
            pass  # Module not available
    return preloaded


# Pre-load modules at module initialization
_PRELOADED_MODULES = _preload_safe_modules()


def _safe_range(*args):
    """Safe range that limits iteration count."""
    r = range(*args)
    if len(r) > MAX_LOOP_ITERATIONS:
        raise RuntimeError(
            f"Range too large: {len(r)} > {MAX_LOOP_ITERATIONS}. "
            "Use smaller ranges or generators."
        )
    return r


def _safe_list(iterable=None):
    """Safe list constructor that limits size."""
    if iterable is None:
        return []
    result = list(iterable)
    if len(result) > MAX_COLLECTION_SIZE:
        raise RuntimeError(
            f"List too large: {len(result)} > {MAX_COLLECTION_SIZE}. "
            "Use generators or process in chunks."
        )
    return result


def _safe_set(iterable=None):
    """Safe set constructor that limits size."""
    if iterable is None:
        return set()
    result = set(iterable)
    if len(result) > MAX_COLLECTION_SIZE:
        raise RuntimeError(
            f"Set too large: {len(result)} > {MAX_COLLECTION_SIZE}. " "Use smaller data sets."
        )
    return result


def _safe_dict(*args, **kwargs):
    """Safe dict constructor that limits size."""
    result = dict(*args, **kwargs)
    if len(result) > MAX_COLLECTION_SIZE:
        raise RuntimeError(
            f"Dict too large: {len(result)} > {MAX_COLLECTION_SIZE}. " "Use smaller data sets."
        )
    return result


def _safe_str(obj=""):
    """Safe str constructor that limits length."""
    result = str(obj)
    if len(result) > MAX_STRING_LENGTH:
        raise RuntimeError(
            f"String too long: {len(result)} > {MAX_STRING_LENGTH}. "
            "Use shorter strings or truncate."
        )
    return result


# Safe builtins with size-limited versions
SAFE_BUILTINS_EXTENDED = {
    **SAFE_BUILTINS,
    "range": _safe_range,
    # Note: We keep original list/dict/set for compatibility but the loop limiter
    # prevents creation of huge collections via iteration
}


def _execute_code_internal(
    code: str,
    context: dict[str, Any],
    max_output: int = 10000,
) -> dict[str, Any]:
    """
    Execute code and return results dict.

    This function is called in a subprocess for timeout enforcement.
    """
    # Check for special commands
    special_result = _handle_special_command(code, context)
    if special_result is not None:
        return {
            "success": True,
            "output": special_result,
            "result": None,
            "variables": list(context.keys()),
        }

    # Check code safety
    is_safe, error = _check_code_safety(code)
    if not is_safe:
        return {"success": False, "error": error}

    # Transform for last expression capture
    transformed_code, has_last_expr = _transform_for_last_expr(code)

    # Apply loop iteration limits
    try:
        transformed_code = _add_loop_limits(transformed_code)
    except Exception:  # noqa: BLE001  # nosec B110
        pass  # If AST transformation fails, continue with original code

    # Create restricted globals with context
    restricted_globals: dict[str, Any] = {
        "__builtins__": {**SAFE_BUILTINS_EXTENDED, "__import__": _restricted_import},
        "__name__": "__main__",
        "__doc__": None,
    }

    # Make preloaded modules available for import
    # This allows 'import math' to work by providing the module
    for mod_name, mod in _PRELOADED_MODULES.items():
        if mod_name not in restricted_globals:
            restricted_globals[mod_name] = mod

    # Track user variables before execution
    user_vars_before = set(context.keys())

    # Load existing context (may override preloaded modules with user vars)
    restricted_globals.update(context)

    # Capture output
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    last_result = None

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(transformed_code, restricted_globals)  # noqa: S102  # nosec B102

        # Get last expression result
        if has_last_expr and "_last_result_" in restricted_globals:
            last_result = restricted_globals.pop("_last_result_")

        # Update context with new user-defined variables only
        # Exclude: internal names, builtins, preloaded modules (unless user reassigned)
        internal_names = {"__builtins__", "__name__", "__doc__", "_last_result_"}
        module_names = set(_PRELOADED_MODULES.keys())

        for k, v in restricted_globals.items():
            # Skip internal names
            if k in internal_names or k.startswith("_"):
                continue

            # Skip preloaded modules unless user explicitly imported/assigned
            if k in module_names and k not in user_vars_before:
                # Check if this is still the same preloaded module
                if k in _PRELOADED_MODULES and v is _PRELOADED_MODULES[k]:
                    continue

            # Keep existing user variables and new assignments
            try:
                repr(v)  # Test if value can be represented
                context[k] = v
            except Exception:  # noqa: BLE001  # nosec B110
                pass

        # Get output
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()

        # Truncate if needed
        if len(stdout_output) > max_output:
            stdout_output = (
                stdout_output[:max_output] + f"\n... (truncated, {len(stdout_output)} chars total)"
            )

        return {
            "success": True,
            "stdout": stdout_output,
            "stderr": stderr_output,
            "result": last_result,
            "variables": [k for k in context.keys() if not k.startswith("_")],
        }

    except SyntaxError as e:
        return {
            "success": False,
            "error": _format_error_with_context(e, code, "Syntax Error"),
        }
    except NameError as e:
        return {
            "success": False,
            "error": _format_error_with_context(e, code, "Name Error"),
        }
    except TypeError as e:
        return {
            "success": False,
            "error": _format_error_with_context(e, code, "Type Error"),
        }
    except ValueError as e:
        return {
            "success": False,
            "error": _format_error_with_context(e, code, "Value Error"),
        }
    except ZeroDivisionError as e:
        return {
            "success": False,
            "error": _format_error_with_context(e, code, "Division Error"),
        }
    except ImportError as e:
        return {"success": False, "error": f"Import Error: {e}"}
    except RecursionError:
        return {
            "success": False,
            "error": "Recursion Error: Maximum recursion depth exceeded. "
            "Check for infinite recursion in your code.",
        }
    except MemoryError:
        return {
            "success": False,
            "error": "Memory Error: Out of memory. "
            "Try processing smaller data or using generators.",
        }
    except Exception as e:
        tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
        filtered = [line for line in tb_lines if "python_exec.py" not in line]
        return {"success": False, "error": f"Error:\n{''.join(filtered)}"}
    finally:
        stdout_capture.close()
        stderr_capture.close()


def _worker_process(
    code: str,
    context_data: dict[str, Any],
    result_queue: mp.Queue,  # type: ignore[type-arg]
) -> None:
    """Worker process for code execution with timeout."""
    try:
        # Set recursion limit for this process
        import sys

        sys.setrecursionlimit(MAX_RECURSION_DEPTH + 50)  # Small buffer

        # Recreate context (some objects may not serialize)
        context: dict[str, Any] = {}
        for k, v in context_data.items():
            try:
                context[k] = v
            except Exception:  # noqa: BLE001  # nosec B110
                pass

        result = _execute_code_internal(code, context)

        # Filter context to only serializable values
        serializable_context: dict[str, Any] = {}
        dropped_vars: list[str] = []
        for k, v in context.items():
            try:
                import pickle  # nosec B403

                pickle.dumps(v)
                serializable_context[k] = v
            except Exception:  # noqa: BLE001  # nosec B110
                dropped_vars.append(k)

        # Include updated context in result
        result["context"] = serializable_context
        if dropped_vars:
            result["dropped_vars"] = dropped_vars

        # Ensure result itself is serializable
        if result.get("result") is not None:
            try:
                import pickle  # nosec B403

                pickle.dumps(result["result"])
            except Exception:  # noqa: BLE001 — subprocess; fall back to repr()
                result["result"] = repr(result["result"])

        result_queue.put(result)
    except Exception as e:
        tb = traceback.format_exc()
        result_queue.put({"success": False, "error": f"Worker error: {e}\n{tb}"})


def execute_python(
    code: str, timeout: int = 30, persistent: bool = True, session_id: str = "default"
) -> str:
    """
    Execute Python code in a restricted environment with persistent state.

    Features:
    - Variables persist between calls (use %clear to reset)
    - Last expression value is automatically displayed
    - True timeout enforcement via subprocess

    Special commands:
    - %vars  - List all stored variables
    - %clear - Clear all variables
    - %help  - Show available commands

    Restrictions:
    - Limited built-in functions (no open, eval, exec, etc.)
    - Only safe modules can be imported (math, json, datetime, etc.)
    - No file system access
    - No network access
    - No system commands

    Args:
        code: Python code to execute
        timeout: Execution timeout in seconds (max 60)
        persistent: Whether to persist variables between calls
        session_id: Session ID for isolating state between users/chats

    Returns:
        Output from the code execution or error message
    """
    # Validate timeout
    timeout = min(max(1, timeout), 60)

    # Get context scoped to the explicit session_id (never falls back to global)
    context = get_context(session_id) if persistent else {}

    # Handle special commands directly (no subprocess needed)
    if code.strip().startswith("%"):
        special_result = _handle_special_command(code, context, session_id=session_id)
        if special_result is not None:
            return special_result

    # Prepare serializable context data
    context_data: dict[str, Any] = {}
    if persistent:
        with _session_states_lock:
            snapshot = dict(context)
    else:
        snapshot = context
    for k, v in snapshot.items():
        try:
            # Test pickling
            import pickle  # nosec B403

            pickle.dumps(v)
            context_data[k] = v
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    # Execute in subprocess with timeout
    result_queue: mp.Queue = mp.Queue()  # type: ignore[type-arg]

    process = mp.Process(
        target=_worker_process,
        args=(code, context_data, result_queue),
    )

    try:
        process.start()
        process.join(timeout=timeout)

        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
            return f"Error: Execution timed out after {timeout} seconds"

        # Get result (wait a bit for the result to arrive)
        try:
            result = result_queue.get(timeout=2)
        except Empty:
            return "Error: No result from execution (queue empty)"

        # Update context from result
        dropped_vars: list[str] = result.get("dropped_vars", [])
        if persistent and "context" in result:
            with _session_states_lock:
                fresh_state = _get_session_state(session_id)
                fresh_state.variables.clear()
                fresh_state.variables.update(result.get("context", {}))

        # Format output
        if not result.get("success"):
            error_output = result.get("error", "Unknown error")
            # Record failed execution in history
            if persistent:
                add_to_history(
                    code, success=False, output=error_output[:100], session_id=session_id
                )
            return error_output

        output_parts = []

        # Standard output
        stdout = result.get("stdout", "")
        if stdout:
            # Truncate if too long
            if len(stdout) > MAX_OUTPUT_SIZE:
                stdout = stdout[:MAX_OUTPUT_SIZE] + f"\n... (truncated, {len(stdout)} chars total)"
            output_parts.append(stdout.rstrip())

        # Standard error
        stderr = result.get("stderr", "")
        if stderr:
            if output_parts:
                output_parts.append("")
            output_parts.append(f"Stderr: {stderr.rstrip()}")

        # Last expression result
        last_result = result.get("result")
        if last_result is not None:
            result_repr = repr(last_result)
            if len(result_repr) > MAX_RESULT_SIZE:
                result_repr = result_repr[: MAX_RESULT_SIZE - 3] + "..."
            if output_parts:
                output_parts.append("")
            output_parts.append(f"Result: {result_repr}")

        # Variables info (if any were created/modified)
        variables = result.get("variables", [])
        if variables and persistent:
            if output_parts:
                output_parts.append("")
            output_parts.append(f"[Variables: {', '.join(sorted(variables))}]")

        # Warn about non-picklable variables lost during IPC
        if dropped_vars and persistent:
            if output_parts:
                output_parts.append("")
            output_parts.append(
                f"Warning: {len(dropped_vars)} variable(s) could not be persisted "
                f"(not picklable): {', '.join(sorted(dropped_vars))}"
            )

        if not output_parts:
            final_output = "Code executed successfully (no output)"
        else:
            final_output = "\n".join(output_parts)

        # Record successful execution in history
        if persistent:
            output_preview = final_output[:100]
            if len(final_output) > 100:
                output_preview += "..."
            add_to_history(code, success=True, output=output_preview, session_id=session_id)

        return final_output

    except Exception as e:
        return f"Error: Failed to execute code: {e}"
    finally:
        # Ensure process is cleaned up
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()


# Build dynamic description based on available modules
_optional_available = [m for m, avail in _AVAILABLE_OPTIONAL.items() if avail]
_optional_str = f" Optional: {', '.join(_optional_available)}." if _optional_available else ""

# Tool metadata for registry
TOOL_CONFIG = {
    "name": "execute_python",
    "description": (
        "Execute Python code in a restricted environment. "
        "Variables persist between calls - build up computations step by step. "
        "Commands: %vars (list variables), %clear (reset), %history (show history), "
        "%modules (list available). "
        "Last expression value is automatically returned. "
        "Core modules: math, random, json, datetime, re, collections, itertools, "
        "functools, statistics, csv, dataclasses, enum, uuid, time."
        f"{_optional_str} "
        "No file/network access."
    ),
    "input_schema": PythonExecInput,
    "requires_confirmation": True,
}

__all__ = [
    "execute_python",
    "PythonExecInput",
    "TOOL_CONFIG",
    "set_session",
    "get_context",
    "get_history",
    "clear_context",
    "clear_history",
    "get_available_modules",
    "ExecutionRecord",
    "SessionState",
]
