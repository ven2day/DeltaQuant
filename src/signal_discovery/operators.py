"""Allowlisted formula compiler for cross-sectional OHLCV panels.

This is a local safety-focused adaptation of the operator/formula idea in
NVIDIA's Apache-2.0 quantitative-signal-discovery-agent. LLM output is never
executed as Python source.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

OHLCV_FIELDS = ("Open", "High", "Low", "Close", "Volume")
Panel = dict[str, pd.DataFrame]
Value = pd.DataFrame | float | int


def build_ohlcv_panel(histories: Mapping[str, pd.DataFrame]) -> Panel:
    """Convert per-symbol OHLCV frames into date x symbol field matrices."""

    series_by_field: dict[str, list[pd.Series]] = {field: [] for field in OHLCV_FIELDS}
    for symbol, raw in histories.items():
        if raw is None or raw.empty:
            continue
        frame = raw.copy()
        frame.columns = [str(column).lower() for column in frame.columns]
        if not {field.lower() for field in OHLCV_FIELDS}.issubset(frame.columns):
            continue
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        for field in OHLCV_FIELDS:
            values = pd.to_numeric(frame[field.lower()], errors="coerce")
            values.name = symbol
            series_by_field[field].append(values)

    panel: Panel = {}
    for field, values in series_by_field.items():
        if values:
            panel[field] = pd.concat(values, axis=1).sort_index()
        else:
            panel[field] = pd.DataFrame()
    return panel


def _window(value: Value) -> int:
    if isinstance(value, pd.DataFrame):
        raise ValueError("rolling window must be a numeric constant")
    number = int(value)
    if number < 2 or number > 2500 or float(value) != number:
        raise ValueError("rolling window must be an integer between 2 and 2500")
    return number


def _clean(value: Value) -> Value:
    if isinstance(value, pd.DataFrame):
        return value.replace([np.inf, -np.inf], np.nan)
    return value


def _rank(value: Value) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise ValueError("Rank requires a DataFrame")
    return value.rank(axis=1, pct=True)


def _ts_rank(value: Value, days: Value) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise ValueError("TS_Rank requires a DataFrame")
    window = _window(days)
    return value.rolling(window).rank(pct=True)


def _decay_linear(value: Value, days: Value) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise ValueError("Decay_Linear requires a DataFrame")
    window = _window(days)
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()
    return value.rolling(window).apply(lambda row: float(np.dot(row, weights)), raw=True)


def _binary(operation: Callable[[Value, Value], Value]) -> Callable[[Value, Value], Value]:
    return lambda left, right: _clean(operation(left, right))


def _unary(operation: Callable[[Value], Value]) -> Callable[[Value], Value]:
    return lambda value: _clean(operation(value))


def _rolling_unary(method: str) -> Callable[[Value, Value], pd.DataFrame]:
    def apply(value: Value, days: Value) -> pd.DataFrame:
        if not isinstance(value, pd.DataFrame):
            raise ValueError(f"TS_{method.title()} requires a DataFrame")
        return getattr(value.rolling(_window(days)), method)()

    return apply


OPERATORS: dict[str, tuple[Callable[..., Value], int]] = {
    "Add": (_binary(lambda x, y: x + y), 2),
    "Sub": (_binary(lambda x, y: x - y), 2),
    "Mul": (_binary(lambda x, y: x * y), 2),
    "Div": (_binary(lambda x, y: x / y), 2),
    "Power": (_binary(lambda x, y: x**y), 2),
    "Max": (_binary(lambda x, y: np.maximum(x, y)), 2),
    "Min": (_binary(lambda x, y: np.minimum(x, y)), 2),
    "Abs": (_unary(lambda x: abs(x)), 1),
    "Sign": (_unary(lambda x: np.sign(x)), 1),
    "Log": (_unary(lambda x: np.log(x)), 1),
    "Sqrt": (_unary(lambda x: np.sqrt(x)), 1),
    "Inv": (_unary(lambda x: 1 / x), 1),
    "Rank": (_rank, 1),
    "TS_Return": (lambda x, d: x.pct_change(_window(d)), 2),
    "TS_Delta": (lambda x, d: x - x.shift(_window(d)), 2),
    "Delay": (lambda x, d: x.shift(_window(d)), 2),
    "TS_Mean": (_rolling_unary("mean"), 2),
    "TS_Std": (_rolling_unary("std"), 2),
    "TS_Min": (_rolling_unary("min"), 2),
    "TS_Max": (_rolling_unary("max"), 2),
    "TS_Sum": (_rolling_unary("sum"), 2),
    "TS_Median": (_rolling_unary("median"), 2),
    "TS_Var": (_rolling_unary("var"), 2),
    "TS_Skew": (_rolling_unary("skew"), 2),
    "TS_Kurt": (_rolling_unary("kurt"), 2),
    "TS_Rank": (_ts_rank, 2),
    "Decay_Linear": (_decay_linear, 2),
    "EMA": (lambda x, d: x.ewm(span=_window(d), adjust=False).mean(), 2),
    "TS_Corr": (lambda x, y, d: x.rolling(_window(d)).corr(y), 3),
    "TS_Cov": (lambda x, y, d: x.rolling(_window(d)).cov(y), 3),
}

OPERATOR_DESCRIPTIONS: dict[str, str] = {
    "Add": "element-wise addition: Add(x, y)",
    "Sub": "element-wise subtraction: Sub(x, y)",
    "Mul": "element-wise multiplication: Mul(x, y)",
    "Div": "element-wise division: Div(x, y)",
    "Rank": "cross-sectional percentile rank: Rank(x)",
    "TS_Return": "return over d settled bars: TS_Return(x, d)",
    "TS_Delta": "difference from d settled bars ago: TS_Delta(x, d)",
    "Delay": "value d settled bars ago: Delay(x, d)",
    "TS_Mean": "rolling mean: TS_Mean(x, d)",
    "TS_Std": "rolling standard deviation: TS_Std(x, d)",
    "TS_Min": "rolling minimum: TS_Min(x, d)",
    "TS_Max": "rolling maximum: TS_Max(x, d)",
    "TS_Sum": "rolling sum: TS_Sum(x, d)",
    "TS_Median": "rolling median: TS_Median(x, d)",
    "TS_Var": "rolling variance: TS_Var(x, d)",
    "TS_Skew": "rolling skew: TS_Skew(x, d)",
    "TS_Kurt": "rolling kurtosis: TS_Kurt(x, d)",
    "TS_Rank": "time-series percentile rank: TS_Rank(x, d)",
    "Decay_Linear": "linear weighted rolling mean: Decay_Linear(x, d)",
    "EMA": "exponential moving average: EMA(x, d)",
    "TS_Corr": "rolling correlation: TS_Corr(x, y, d)",
    "TS_Cov": "rolling covariance: TS_Cov(x, y, d)",
    "Power": "element-wise power: Power(x, y)",
    "Max": "element-wise maximum: Max(x, y)",
    "Min": "element-wise minimum: Min(x, y)",
    "Abs": "absolute value: Abs(x)",
    "Sign": "sign of value: Sign(x)",
    "Log": "natural logarithm: Log(x)",
    "Sqrt": "square root: Sqrt(x)",
    "Inv": "reciprocal: Inv(x)",
}


@dataclass(frozen=True)
class CompiledFormula:
    formula: str
    tree: ast.Expression
    operators_used: tuple[str, ...]
    fields_used: tuple[str, ...]

    def evaluate(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        value = _evaluate_node(self.tree.body, panel)
        if not isinstance(value, pd.DataFrame):
            raise ValueError("formula must produce a cross-sectional DataFrame")
        return value.replace([np.inf, -np.inf], np.nan)


class FormulaCompiler:
    """Validate formulas and compile them into a safe AST representation."""

    def compile(self, formula: str) -> CompiledFormula:
        if not formula or len(formula) > 2000:
            raise ValueError("formula is empty or too long")
        try:
            tree = ast.parse(formula, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"invalid formula syntax: {exc.msg}") from exc
        nodes = list(ast.walk(tree))
        if len(nodes) > 150:
            raise ValueError("formula is too complex")

        operators: set[str] = set()
        fields: set[str] = set()
        for node in nodes:
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in OPERATORS:
                    raise ValueError("formula contains an unknown or unsafe function")
                if node.keywords:
                    raise ValueError("keyword arguments are not allowed")
                expected = OPERATORS[node.func.id][1]
                if len(node.args) != expected:
                    raise ValueError(
                        f"{node.func.id} expects {expected} arguments, got {len(node.args)}"
                    )
                operators.add(node.func.id)
            elif isinstance(node, ast.Name):
                if node.id in OHLCV_FIELDS:
                    fields.add(node.id)
                elif node.id not in OPERATORS:
                    raise ValueError(f"unknown name: {node.id}")
            elif isinstance(node, ast.Constant):
                if not isinstance(node.value, (int, float)) or abs(float(node.value)) > 1_000_000:
                    raise ValueError("only bounded numeric constants are allowed")
            elif isinstance(node, (ast.Expression, ast.Load, ast.UnaryOp, ast.USub, ast.UAdd)):
                continue
            elif not isinstance(node, ast.Name):
                raise ValueError(f"unsafe syntax: {type(node).__name__}")
        if not fields:
            raise ValueError("formula must reference at least one OHLCV field")
        return CompiledFormula(formula, tree, tuple(sorted(operators)), tuple(sorted(fields)))

    def operator_prompt(self) -> str:
        return "\n".join(
            f"- {name}: {OPERATOR_DESCRIPTIONS[name]}" for name in sorted(OPERATOR_DESCRIPTIONS)
        )


def _evaluate_node(node: ast.AST, panel: Mapping[str, pd.DataFrame]) -> Value:
    if isinstance(node, ast.Name):
        if node.id not in OHLCV_FIELDS or node.id not in panel:
            raise ValueError(f"panel field unavailable: {node.id}")
        return panel[node.id]
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp):
        value = _evaluate_node(node.operand, panel)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        raise ValueError("unsupported unary operator")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function, _ = OPERATORS[node.func.id]
        arguments = [_evaluate_node(argument, panel) for argument in node.args]
        return _clean(function(*arguments))
    raise ValueError(f"unsupported formula node: {type(node).__name__}")
