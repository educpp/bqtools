from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import math
import re
import warnings
from mpmath import iv
import pint

Number = Union[int, float]
StrOrNum = Union[str, Number]
TupleInput = Tuple[StrOrNum, StrOrNum, str]

_SYMBOL_TO_OBJ: Dict[str, int] = {}
_OBJ_TO_SYMBOL: Dict[int, str] = {}
_AUTO_STORES: Dict[str, "BQStore"] = {}
_INJECTED_KEY_TARGETS: Dict[str, Tuple[str, str]] = {}


def _inject_global(name: str, value: Any) -> None:
    try:
        try:
            from IPython import get_ipython

            ip = get_ipython()
        except Exception:
            ip = None

        if ip is not None and hasattr(ip, "user_ns"):
            g = ip.user_ns
        else:
            import inspect

            frame = inspect.currentframe()
            caller = frame.f_back.f_back if frame and frame.f_back else None
            if caller is None:
                return
            g = caller.f_globals
        if name in g and g[name] is not value:
            existing = g.get(name)

            def _is_bq_obj(v: Any) -> bool:
                for cls_name in ("BQStore", "BQKeyView", "BQListView", "BoundedQuantity"):
                    cls = globals().get(cls_name)
                    if cls is not None and isinstance(v, cls):
                        return True
                return False

            if _is_bq_obj(existing) and _is_bq_obj(value):
                g[name] = value
                return

            warnings.warn(
                f"Global '{name}' already exists. Use del {name} (or restart the kernel) and re-run.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        g[name] = value
    finally:
        try:
            del frame
        except Exception:
            pass


def _register_injected_key(store: "BQStore", key: str) -> None:
    _INJECTED_KEY_TARGETS[key] = (store._name, key)


def get_injected_target(name: str) -> Optional[Tuple[str, str]]:
    return _INJECTED_KEY_TARGETS.get(name)


def reset_symbol_registry() -> None:
    """Clear the symbol registry. If you use DATA/CALC/CONST, prefer reset_all()."""
    _SYMBOL_TO_OBJ.clear()
    _OBJ_TO_SYMBOL.clear()


def _unregister_symbol(symbol: str) -> None:
    """Remove a symbol->object mapping to allow overwrite in DATA/CALC/CONST."""
    sym = (symbol or "").strip()
    if not sym:
        return
    oid = _SYMBOL_TO_OBJ.pop(sym, None)
    if oid is not None:
        _OBJ_TO_SYMBOL.pop(oid, None)


def _register_symbol(obj: Any, symbol: str) -> None:
    """
    Enforce:
      - one object cannot have two different symbols
      - one symbol cannot refer to two different objects
    """
    sym = (symbol or "").strip()
    if not sym:
        return
    oid = id(obj)

    if oid in _OBJ_TO_SYMBOL and _OBJ_TO_SYMBOL[oid] != sym:
        raise ValueError(
            f"Object already has symbol '{_OBJ_TO_SYMBOL[oid]}'. "
            f"Refusing to assign a second symbol '{sym}'."
        )
    if sym in _SYMBOL_TO_OBJ and _SYMBOL_TO_OBJ[sym] != oid:
        raise ValueError(f"Symbol '{sym}' is already assigned to another object.")
    _OBJ_TO_SYMBOL[oid] = sym
    _SYMBOL_TO_OBJ[sym] = oid


@dataclass(frozen=True)
class BoundsConfig:
    require_symbol_on_print: bool = True
    strict_decimal_match_on_input: bool = True
    show_interval_in_print: bool = True
    show_sigfig_warning: bool = True
    enforce_unique_symbols: bool = True


CONFIG = BoundsConfig()


def set_config(
    *,
    require_symbol_on_print: Optional[bool] = None,
    strict_decimal_match_on_input: Optional[bool] = None,
    show_interval_in_print: Optional[bool] = None,
    show_sigfig_warning: Optional[bool] = None,
    enforce_unique_symbols: Optional[bool] = None,
) -> None:
    """Update global CONFIG flags."""
    global CONFIG
    CONFIG = BoundsConfig(
        require_symbol_on_print=CONFIG.require_symbol_on_print
        if require_symbol_on_print is None
        else require_symbol_on_print,
        strict_decimal_match_on_input=CONFIG.strict_decimal_match_on_input
        if strict_decimal_match_on_input is None
        else strict_decimal_match_on_input,
        show_interval_in_print=CONFIG.show_interval_in_print
        if show_interval_in_print is None
        else show_interval_in_print,
        show_sigfig_warning=CONFIG.show_sigfig_warning
        if show_sigfig_warning is None
        else show_sigfig_warning,
        enforce_unique_symbols=CONFIG.enforce_unique_symbols
        if enforce_unique_symbols is None
        else enforce_unique_symbols,
    )


U_SIG_DIGITS = 1


def set_u_sig_digits(n: int = 2) -> None:
    """
    Configure how many significant digits to keep in the FINAL uncertainty u.
    u is always rounded UP (ceiling) to this many sig digits.
    """
    global U_SIG_DIGITS
    if n < 1:
        raise ValueError("Only positive integers are supported.")
    U_SIG_DIGITS = n


ureg = pint.UnitRegistry()
Q_ = ureg.Quantity


def parse_unit(unit_str: str) -> pint.Unit:
    unit_str = (unit_str or "").strip()
    return ureg.parse_units(unit_str) if unit_str else ureg.dimensionless


def unit_str(unit: pint.Unit) -> str:
    """Pretty unit string (e.g., cm^3 instead of cm ** 3)."""
    if unit == ureg.dimensionless:
        return ""
    s = f"{Q_(1, unit):~}".replace("1 ", "").strip()
    return s.replace(" ** ", "^")


def conversion_factor(from_unit: pint.Unit, to_unit: pint.Unit) -> float:
    if from_unit == to_unit:
        return 1.0
    return Q_(1, from_unit).to(to_unit).magnitude


def sigfigs_from_str(s: str) -> int:
    """Classroom convention: decimals count trailing zeros; integers don't."""
    s = s.strip().lower().replace("_", "").lstrip("+")
    mant = s.split("e", 1)[0] if "e" in s else s
    mant = mant.strip().lstrip("+").lstrip("-")
    if mant in ("", "."):
        return 1
    if set(mant.replace(".", "")) <= {"0"}:
        digits = mant.replace(".", "")
        return max(1, len(digits))
    if "." in mant:
        digits = mant.replace(".", "").lstrip("0")
        return max(1, len(digits))
    digits = mant.lstrip("0").rstrip("0")
    return max(1, len(digits))


def order10(x: float) -> int:
    if x == 0:
        return 0
    return int(math.floor(math.log10(abs(x))))


def place_from_val_sig(x: float, sig: int) -> float:
    """Return the decimal place (power of 10) corresponding to sig figs."""
    if sig < 1:
        raise ValueError("sig must be >= 1")
    if x == 0:
        return 10 ** (-(sig - 1))
    return 10 ** (order10(x) - sig + 1)


def round_to_place(x: float, place: float) -> float:
    if place == 0:
        return x
    dp = -order10(place)
    return round(x, dp)


def fmt_place(x: float, place: float) -> str:
    """Format rounded to place, preserving trailing zeros where applicable."""
    if place == 0:
        return f"{x:g}"
    dp = -order10(place)
    xr = round_to_place(x, place)
    if dp >= 0:
        return f"{xr:.{dp}f}"
    return str(int(round(xr, 0)))


def decimal_places(s: str) -> int:
    return 0 if "." not in s else len(s.split(".", 1)[1])


def place_from_str(s: str) -> float:
    """Infer last-place from numeric string via its sig figs."""
    x = float(s)
    sig = sigfigs_from_str(s)
    return place_from_val_sig(x, sig)


def ceil_to_place(x: float, place: float) -> float:
    """
    Ceiling to a place, but robust to tiny floating errors.
    Example: x=0.30000000000000004 at place=0.1 should stay 0.3, not jump to 0.4.
    """
    if place == 0:
        return x
    k = x / place
    k_round = round(k)
    if abs(k - k_round) < 1e-12:
        return k_round * place
    return math.ceil(k - 1e-12) * place


def iv_mid_half(I: Any) -> Tuple[float, float, float, float]:
    """Return (mid, u, lo, hi) for interval I."""
    lo = float(I.a)
    hi = float(I.b)
    return 0.5 * (lo + hi), 0.5 * (hi - lo), lo, hi


def iv_convert(I: Any, from_unit: pint.Unit, to_unit: pint.Unit) -> Any:
    """Convert an interval magnitude between units by scaling endpoints."""
    if from_unit == to_unit:
        return I
    return I * conversion_factor(from_unit, to_unit)


def _overlap_fraction(true_lo: float, true_hi: float, rep_lo: float, rep_hi: float) -> float:
    """Fraction of the TRUE interval covered by the REPORTED interval (0..1)."""
    w = true_hi - true_lo
    if w <= 0:
        return 1.0
    overlap = max(0.0, min(true_hi, rep_hi) - max(true_lo, rep_lo))
    return overlap / w


@dataclass
class BoundedQuantity:
    """
    Bounds-only quantity backed by exact interval arithmetic (mpmath.iv).

    This class also tracks a lightweight, classroom-style precision metadata
    (place for +/-, sig for */) that is used only for warnings and trace output.
    """

    I: Any
    unit: pint.Unit
    symbol: str = ""
    expr: str = ""
    op: Optional[str] = None
    lhs: Optional["BoundedQuantity"] = None
    rhs: Optional["BoundedQuantity"] = None
    node_kind: str = "node"
    rule_mode: str = "place"
    rule_place: float = 0.0
    rule_sig: int = 1

    def __post_init__(self) -> None:
        if CONFIG.enforce_unique_symbols and self.symbol.strip():
            _register_symbol(self, self.symbol)

    def set_symbol(self, symbol: str) -> "BoundedQuantity":
        sym = (symbol or "").strip()
        if CONFIG.enforce_unique_symbols:
            _register_symbol(self, sym)
        self.symbol = sym
        return self

    def _expr_for_print(self) -> str:
        return self.expr.strip() if self.expr.strip() else self.symbol.strip()

    @staticmethod
    def from_value_bound_unit(
        value: StrOrNum,
        bound: StrOrNum = 0,
        unit: str = "",
        *,
        symbol: str,
        strict_decimal_match: Optional[bool] = None,
    ) -> "BoundedQuantity":
        """
        Create x = value ± bound (bounds-only).

        If strict_decimal_match is enabled, value and bound must imply the same last-place,
        except when bound == 0 (exact).
        """
        strict = (
            CONFIG.strict_decimal_match_on_input
            if strict_decimal_match is None
            else strict_decimal_match
        )
        sym = (symbol or "").strip()
        if not sym:
            raise ValueError("symbol is required for inputs.")

        v = float(value)
        b = float(bound)
        if b < 0:
            raise ValueError("bound must be >= 0")

        if strict and isinstance(value, str) and isinstance(bound, str):
            if float(bound) != 0.0:
                pv = place_from_str(value)
                pb = place_from_str(bound)
                if pv != pb:
                    raise ValueError(
                        f"Decimal-place mismatch: value '{value}' implies place {pv}, "
                        f"but bound '{bound}' implies place {pb}. "
                        "Fix by matching decimals (e.g., '10.00 ± 0.10' or '10.000 ± 0.100')."
                    )

        if isinstance(bound, str) and float(bound) != 0.0:
            rule_place = place_from_str(bound)
        elif isinstance(value, str):
            rule_place = place_from_str(value)
        else:
            rule_place = place_from_val_sig(v if v != 0 else 1.0, 3)

        I = iv.mpf([v - b, v + b])
        return BoundedQuantity(
            I=I,
            unit=parse_unit(unit),
            symbol=sym,
            expr=sym,
            node_kind="input",
            rule_mode="place",
            rule_place=rule_place,
            rule_sig=1,
        )

    @staticmethod
    def from_tuple(
        t: TupleInput, *, symbol: str, strict_decimal_match: Optional[bool] = None
    ) -> "BoundedQuantity":
        if len(t) != 3:
            raise ValueError("Expected (value, bound, unit).")
        return BoundedQuantity.from_value_bound_unit(
            t[0], t[1], t[2], symbol=symbol, strict_decimal_match=strict_decimal_match
        )

    @staticmethod
    def const(x: Number) -> "BoundedQuantity":
        """Exact dimensionless constant (for mixing scalars)."""
        I = iv.mpf([float(x), float(x)])
        return BoundedQuantity(
            I=I,
            unit=ureg.dimensionless,
            symbol="",
            expr=f"{float(x):g}",
            node_kind="const",
            rule_mode="sig",
            rule_place=0.0,
            rule_sig=999,
        )

    def _coerce(self, other: Any) -> "BoundedQuantity":
        return other if isinstance(other, BoundedQuantity) else BoundedQuantity.const(float(other))

    def _rule_place_for_warning(self) -> float:
        mid = iv_mid_half(self.I)[0]
        if self.rule_mode == "place":
            return self.rule_place
        return place_from_val_sig(mid if mid != 0 else 1.0, self.rule_sig)

    def __str__(self) -> str:
        mid, u, _lo, _hi = iv_mid_half(self.I)
        place = self.rule_place if self.rule_mode == "place" else place_from_val_sig(mid if mid != 0 else 1.0, self.rule_sig)
        mid_s = fmt_place(mid, place)
        u_s = fmt_place(u, place)
        unit_s = unit_str(self.unit)
        return f"{mid_s}, {u_s}, {unit_s}" if unit_s else f"{mid_s}, {u_s}"

    def __repr__(self) -> str:
        return self.__str__()

    def __add__(self, other: Any) -> "BoundedQuantity":
        if "BQListView" in globals() and isinstance(other, (BQListView, BQKeyView)):
            return NotImplemented
        o = self._coerce(other)
        oI = iv_convert(o.I, o.unit, self.unit)
        I = self.I + oI
        place = max(self._rule_place_for_warning(), o._rule_place_for_warning())
        expr = self._op_expr("+", o)
        return BoundedQuantity(
            I=I,
            unit=self.unit,
            expr=expr,
            op="+",
            lhs=self,
            rhs=o,
            node_kind="op",
            rule_mode="place",
            rule_place=place,
            rule_sig=1,
        )

    def __sub__(self, other: Any) -> "BoundedQuantity":
        if "BQListView" in globals() and isinstance(other, (BQListView, BQKeyView)):
            return NotImplemented
        o = self._coerce(other)
        oI = iv_convert(o.I, o.unit, self.unit)
        I = self.I - oI
        place = max(self._rule_place_for_warning(), o._rule_place_for_warning())
        expr = self._op_expr("-", o)
        return BoundedQuantity(
            I=I,
            unit=self.unit,
            expr=expr,
            op="-",
            lhs=self,
            rhs=o,
            node_kind="op",
            rule_mode="place",
            rule_place=place,
            rule_sig=1,
        )

    def __mul__(self, other: Any) -> "BoundedQuantity":
        if "BQListView" in globals() and isinstance(other, (BQListView, BQKeyView)):
            return NotImplemented
        o = self._coerce(other)
        I = self.I * o.I
        unit = self.unit * o.unit

        def est_sig(q: "BoundedQuantity") -> int:
            if q.rule_mode == "sig":
                return q.rule_sig
            mid = iv_mid_half(q.I)[0]
            if q.rule_place == 0:
                return 999
            return max(1, order10(mid if mid != 0 else 1.0) - order10(q.rule_place) + 1)

        sig = min(est_sig(self), est_sig(o))
        expr = self._op_expr("*", o)
        return BoundedQuantity(
            I=I,
            unit=unit,
            expr=expr,
            op="*",
            lhs=self,
            rhs=o,
            node_kind="op",
            rule_mode="sig",
            rule_place=0.0,
            rule_sig=sig,
        )

    def __truediv__(self, other: Any) -> "BoundedQuantity":
        if "BQListView" in globals() and isinstance(other, (BQListView, BQKeyView)):
            return NotImplemented
        o = self._coerce(other)
        I = self.I / o.I
        unit = self.unit / o.unit

        def est_sig(q: "BoundedQuantity") -> int:
            if q.rule_mode == "sig":
                return q.rule_sig
            mid = iv_mid_half(q.I)[0]
            if q.rule_place == 0:
                return 999
            return max(1, order10(mid if mid != 0 else 1.0) - order10(q.rule_place) + 1)

        sig = min(est_sig(self), est_sig(o))
        expr = self._op_expr("/", o)
        return BoundedQuantity(
            I=I,
            unit=unit,
            expr=expr,
            op="/",
            lhs=self,
            rhs=o,
            node_kind="op",
            rule_mode="sig",
            rule_place=0.0,
            rule_sig=sig,
        )

    def __radd__(self, other: Any) -> "BoundedQuantity":
        return self.__add__(other)

    def __rmul__(self, other: Any) -> "BoundedQuantity":
        return self.__mul__(other)

    def _u_round_up_for_report(self, mid: float, u: float) -> Tuple[float, float, float]:
        """
        Returns (mid_print, u_print, place) such that:
          - u_print has U_SIG_DIGITS sig digits and is rounded UP
          - mid_print rounded to the same place

        This does not inflate u_print to guarantee enclosure of the exact iv interval.
        If rounding reduces coverage, format() emits a warning.
        """
        if u <= 0:
            return mid, 0.0, self._rule_place_for_warning()

        place = place_from_val_sig(u, U_SIG_DIGITS)
        u_p = ceil_to_place(u, place)
        place = place_from_val_sig(u_p, U_SIG_DIGITS) if u_p > 0 else place
        mid_p = round_to_place(mid, place)
        return mid_p, u_p, place

    def _warning_sigfig(self, place_used: float, mid: float) -> Optional[str]:
        if not CONFIG.show_sigfig_warning:
            return None
        rule_place = self._rule_place_for_warning()
        if (place_used != 0.0) and (rule_place != 0.0) and (place_used < rule_place):
            example_mid = fmt_place(round_to_place(mid, rule_place), rule_place)
            return (
                f"WARNING: inputs suggest rounding to place {rule_place:g} "
                f"(e.g., {example_mid}), but bounds-based uncertainty implies place {place_used:g}. "
                "Check inputs (sig figs/decimals) or cancellation."
            )
        return None

    def format(self) -> str:
        if CONFIG.require_symbol_on_print and not self.symbol.strip():
            raise ValueError("This quantity has no symbol. Use .set_symbol('name').")

        mid, u, lo_true, hi_true = iv_mid_half(self.I)
        mid_p, u_p, place = self._u_round_up_for_report(mid, u)

        mid_s = fmt_place(mid_p, place)
        u_s = fmt_place(u_p, place)

        if place != 0 and decimal_places(mid_s) != decimal_places(u_s):
            raise ValueError(f"Decimal mismatch in output: value '{mid_s}' and uncertainty '{u_s}'.")

        ut = unit_str(self.unit)
        usp = f" {ut}" if ut else ""

        expr = self._expr_for_print()
        head = (
            f"{self.symbol} = "
            if not (expr and expr != self.symbol)
            else f"{self.symbol} = {expr} = "
        )

        rep_lo = mid_p - u_p
        rep_hi = mid_p + u_p

        if CONFIG.show_interval_in_print:
            line1 = (
                f"{head}{mid_s} ± {u_s}{usp}   "
                f"(interval [{fmt_place(rep_lo, place)}, {fmt_place(rep_hi, place)}]{usp})"
            )
        else:
            line1 = f"{head}{mid_s} ± {u_s}{usp}"

        cov = _overlap_fraction(lo_true, hi_true, rep_lo, rep_hi)
        warn_cov = None
        if cov < 0.999999:
            warn_cov = (
                f"WARNING: reported interval overlaps {100.0*cov:.1f}% of the exact iv interval "
                f"[{lo_true:.6g}, {hi_true:.6g}]{usp}."
            )

        warn_sig = self._warning_sigfig(place, mid)

        if warn_cov and warn_sig:
            return line1 + "\n" + warn_cov + "\n" + warn_sig
        if warn_cov:
            return line1 + "\n" + warn_cov
        if warn_sig:
            return line1 + "\n" + warn_sig
        return line1

    def __str__(self) -> str:
        return self.format()


def _bq_est_sig(q: BoundedQuantity) -> int:
    """Estimate sig figs from Option-3 metadata (warning-only)."""
    if q.rule_mode == "sig":
        return q.rule_sig
    mid = iv_mid_half(q.I)[0]
    if q.rule_place == 0:
        return 999
    return max(1, order10(mid if mid != 0 else 1.0) - order10(q.rule_place) + 1)


_OP_PREC = {"+": 1, "-": 1, "*": 2, "/": 2, "**": 3}


def _wrap_parens(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    if s.startswith("(") and s.endswith(")"):
        return s
    return f"({s})"


def _needs_parens(child_op: Optional[str], parent_op: str, side: str) -> bool:
    if not child_op or not parent_op:
        return False

    cp = _OP_PREC.get(child_op, 99)
    pp = _OP_PREC.get(parent_op, 99)

    if cp < pp:
        return True
    if cp > pp:
        return False

    if parent_op == "-" and side == "rhs":
        return True
    if parent_op == "/" and side == "rhs":
        return True
    if parent_op == "**" and side == "lhs":
        return True

    return False


def _bq_op_expr(self: "BoundedQuantity", op: str, other: "BoundedQuantity") -> str:
    a = self._expr_for_print()
    b = other._expr_for_print()

    if _needs_parens(getattr(self, "op", None), op, "lhs"):
        a = _wrap_parens(a)
    if _needs_parens(getattr(other, "op", None), op, "rhs"):
        b = _wrap_parens(b)

    return f"{a} {op} {b}" if (a and b) else ""


BoundedQuantity._op_expr = _bq_op_expr


def _bq_pow(self: "BoundedQuantity", p: Any) -> "BoundedQuantity":
    if isinstance(p, BoundedQuantity):
        if p.unit != ureg.dimensionless:
            raise TypeError("Exponent must be dimensionless.")
        if float(p.I.a) != float(p.I.b):
            raise TypeError("Exponent must be an exact constant (no uncertainty).")
        p = float(p.I.a)

    if isinstance(p, bool):
        p = int(p)
    if isinstance(p, float):
        if not p.is_integer():
            raise TypeError("Only integer powers are supported (e.g., **2, **-1).")
        p = int(p)
    if not isinstance(p, int):
        raise TypeError("Only integer powers are supported (e.g., **2, **-1).")

    I = self.I ** p
    unit = (self.unit ** p) if p != 0 else ureg.dimensionless

    exp_node = BoundedQuantity.const(p)

    base = self._expr_for_print()
    if getattr(self, "op", None):
        base = _wrap_parens(base)
    expr = f"{base}**{p}"

    return BoundedQuantity(
        I=I,
        unit=unit,
        symbol="",
        expr=expr,
        op="**",
        lhs=self,
        rhs=exp_node,
        node_kind="op",
        rule_mode="sig",
        rule_place=0.0,
        rule_sig=_bq_est_sig(self),
    )


BoundedQuantity.__pow__ = _bq_pow


def _bq_convert_to(self: "BoundedQuantity", target_unit: str) -> "BoundedQuantity":
    """
    Convert this BoundedQuantity to target_unit (in place) and return self.

    Example:
        r.to("mm/s^2")
        print(r)
    """
    tgt = parse_unit(target_unit)
    old_unit = self.unit

    lo = Q_(float(self.I.a), old_unit).to(tgt).magnitude
    hi = Q_(float(self.I.b), old_unit).to(tgt).magnitude
    self.I = iv.mpf([min(lo, hi), max(lo, hi)])
    self.unit = tgt

    if self.rule_mode == "place" and self.rule_place:
        f = abs(Q_(1, old_unit).to(tgt).magnitude)
        self.rule_place = abs(self.rule_place * f)

    return self


BoundedQuantity.convert_to = _bq_convert_to
BoundedQuantity.to = _bq_convert_to
BoundedQuantity.in_units = _bq_convert_to


def Qb(t: TupleInput, *, symbol: str, strict_decimal_match: Optional[bool] = None) -> BoundedQuantity:
    """Create a bounds quantity from (value, bound, unit)."""
    sym = (symbol or "").strip()
    if not sym:
        raise ValueError("Qb(..., symbol=...) requires a non-empty symbol.")
    return BoundedQuantity.from_tuple(t, symbol=sym, strict_decimal_match=strict_decimal_match)


def trace_str(x: "BoundedQuantity") -> str:
    """Return the multiline trace as a string (no printing)."""
    if not isinstance(x, BoundedQuantity):
        raise TypeError("trace_str(x) requires a BoundedQuantity")

    lines: List[str] = []
    seen: Set[int] = set()

    def lab(q: BoundedQuantity) -> str:
        if q.symbol.strip():
            return q.symbol.strip()
        e = q._expr_for_print()
        return e if e else "<unnamed>"

    def g(v: float) -> str:
        return f"{v:.6g}"

    def _decimals_in(s: str) -> int:
        mant = s.split("e", 1)[0].split("E", 1)[0]
        if "." not in mant:
            return 0
        return len(mant.split(".", 1)[1])

    def _fmt_like_iv(uval: float, lo_s: str, hi_s: str) -> str:
        if ("e" in lo_s.lower()) or ("e" in hi_s.lower()):
            return g(uval)
        d = max(_decimals_in(lo_s), _decimals_in(hi_s))
        return f"{uval:.{d}f}"

    def tracked(q: BoundedQuantity) -> str:
        return f"tracked place={q.rule_place:g}" if q.rule_mode == "place" else f"tracked sig={q.rule_sig}"

    def visit(q: BoundedQuantity, depth: int) -> None:
        indent = "  " * depth
        oid = id(q)
        if oid in seen:
            lines.append(f"{indent}{lab(q)}  (reused object)")
            return
        seen.add(oid)

        if q.node_kind == "const":
            return

        if q.op and q.lhs is not None and q.rhs is not None:
            visit(q.lhs, depth + 1)
            visit(q.rhs, depth + 1)

        mid, u, lo_true, hi_true = iv_mid_half(q.I)
        mid_p, u_p, place = q._u_round_up_for_report(mid, u)

        ut = unit_str(q.unit)
        usp = f" {ut}" if ut else ""

        head = lab(q)
        if q.op and q.expr and head != q.expr:
            head += f" = {q.expr}"

        rep_lo = mid_p - u_p
        rep_hi = mid_p + u_p
        cov = _overlap_fraction(lo_true, hi_true, rep_lo, rep_hi)

        lines.append(f"{indent}{head}")

        lo_s = g(lo_true)
        hi_s = g(hi_true)
        lines.append(f"{indent}  iv       [{lo_s}, {hi_s}]{usp}")

        u_iv_s = _fmt_like_iv(u, lo_s, hi_s)
        lines.append(f"{indent}  u(iv)    {u_iv_s}{usp}")

        lines.append(
            f"{indent}  report   {fmt_place(mid_p, place)} ± {fmt_place(u_p, place)}{usp}  (LSD={place:g})"
        )
        if cov < 0.999999:
            lines.append(f"{indent}  WARNING: coverage {100.0*cov:.1f}% of iv interval")

        lines.append(f"{indent}  {tracked(q)}")

        warn_sig = q._warning_sigfig(place, mid)
        if warn_sig:
            lines.append(f"{indent}  {warn_sig}")

    lines.append(f"TRACE: {lab(x)}")
    visit(x, 0)
    return "\n".join(lines)


def trace(x: "BoundedQuantity") -> None:
    """Print the trace."""
    print(trace_str(x))


class BQColumn:
    """
    Column-style access into a BQStore.

    DATA.y[0] maps to DATA["y_0"].
    CALC.l[1] maps to CALC["l_1"].
    """

    def __init__(self, store: "BQStore", prefix: str):
        self._store = store
        self._prefix = (prefix or "").strip()
        if not self._prefix:
            raise ValueError("Column prefix cannot be empty.")
        if not self._prefix.isidentifier():
            raise ValueError(f"Column prefix {self._prefix!r} must be a valid identifier.")

    def _key(self, i: int) -> str:
        if not isinstance(i, int) or isinstance(i, bool):
            raise TypeError("Column index must be an integer.")
        if i < 0:
            raise ValueError("Column index must be >= 0.")
        return f"{self._prefix}_{i}"

    def __getitem__(self, idx: Union[int, slice]) -> Any:
        if isinstance(idx, slice):
            start = 0 if idx.start is None else idx.start
            stop = idx.stop
            step = 1 if idx.step is None else idx.step
            if stop is None:
                raise ValueError("Slice stop is required, e.g. DATA.y[0:5].")
            if not all(isinstance(v, int) or v is None for v in (idx.start, idx.stop, idx.step)):
                raise TypeError("Slice indices must be integers.")
            if step == 0:
                raise ValueError("Slice step cannot be 0.")
            out: List[BoundedQuantity] = []
            for i in range(start, stop, step):
                out.append(self._store[self._key(i)])
            return out
        return self._store[self._key(idx)]

    def __setitem__(self, i: int, value: Any) -> None:
        self._store[self._key(i)] = value


class BQListView:
    """Read-only view for a list of BoundedQuantity items with table-like printing."""

    def __init__(self, items: List[BoundedQuantity], name: str = "y"):
        self._items = items
        self._col_widths = (3, 7, 7, 6)
        self._name = (name or "y").strip() or "y"

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, idx: Union[int, slice]) -> Any:
        return self._items[idx]

    def _binary_op(self, other: Any, op) -> List[BoundedQuantity]:
        if isinstance(other, BQListView):
            other_items = other._items
        elif "BQKeyView" in globals() and isinstance(other, BQKeyView):
            other_items = other._items()
        elif isinstance(other, list):
            other_items = other
        else:
            other_items = None

        if other_items is None:
            return [op(a, other) for a in self._items]

        if len(other_items) != len(self._items):
            raise ValueError("List lengths must match for element-wise operations.")
        return [op(a, b) for a, b in zip(self._items, other_items)]

    def __add__(self, other: Any) -> List[BoundedQuantity]:
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other: Any) -> List[BoundedQuantity]:
        return self._binary_op(other, lambda a, b: b + a)

    def __sub__(self, other: Any) -> List[BoundedQuantity]:
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other: Any) -> List[BoundedQuantity]:
        return self._binary_op(other, lambda a, b: b - a)

    def __mul__(self, other: Any) -> List[BoundedQuantity]:
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other: Any) -> List[BoundedQuantity]:
        return self._binary_op(other, lambda a, b: b * a)

    def __truediv__(self, other: Any) -> List[BoundedQuantity]:
        return self._binary_op(other, lambda a, b: a / b)

    def __rtruediv__(self, other: Any) -> List[BoundedQuantity]:
        return self._binary_op(other, lambda a, b: b / a)

    def __pow__(self, other: Any) -> List[BoundedQuantity]:
        return self._binary_op(other, lambda a, b: a ** b)

    def __rpow__(self, other: Any) -> List[BoundedQuantity]:
        return self._binary_op(other, lambda a, b: b ** a)

    def _format_row(self, q: BoundedQuantity, idx: int) -> str:
        mid, u, _lo, _hi = iv_mid_half(q.I)
        place = q.rule_place if q.rule_mode == "place" else place_from_val_sig(mid if mid != 0 else 1.0, q.rule_sig)
        mid_s = fmt_place(mid, place)
        u_s = fmt_place(u, place)
        unit_s = unit_str(q.unit)
        w_idx, w0, w1, w2 = self._col_widths
        if not unit_s:
            unit_s = ""
        return f"{idx:>{w_idx}} | {mid_s:>{w0}} | {u_s:>{w1}} | {unit_s:<{w2}}"

    def __str__(self) -> str:
        w_idx, w0, w1, w2 = self._col_widths
        header = f"{'#':>{w_idx}} | {self._name:>{w0}} | {'±':>{w1}} | {'unit':<{w2}}"
        rows = "\n".join(self._format_row(q, i) for i, q in enumerate(self._items))
        return f"{header}\n{rows}\n" if rows else f"{header}\n"

    def __repr__(self) -> str:
        return self.__str__()


class BQKeyView:
    """View for a single key that prints as a table and supports list-like ops."""

    def __init__(self, store: "BQStore", key: str):
        self._store = store
        self._key = (key or "").strip()

    def _items(self) -> List[BoundedQuantity]:
        v = self._store._store.get(self._key)
        if isinstance(v, list):
            return v
        if isinstance(v, BoundedQuantity):
            return [v]
        return []

    def __len__(self) -> int:
        return len(self._items())

    def __iter__(self):
        return iter(self._items())

    def __getitem__(self, idx: Union[int, slice]) -> Any:
        items = self._items()
        return items[idx]

    def __str__(self) -> str:
        return str(BQListView(self._items(), name=self._key))

    def __repr__(self) -> str:
        return self.__str__()

    def __add__(self, other: Any) -> List[BoundedQuantity]:
        return BQListView(self._items(), name=self._key).__add__(other)

    def __radd__(self, other: Any) -> List[BoundedQuantity]:
        return BQListView(self._items(), name=self._key).__radd__(other)

    def __sub__(self, other: Any) -> List[BoundedQuantity]:
        return BQListView(self._items(), name=self._key).__sub__(other)

    def __rsub__(self, other: Any) -> List[BoundedQuantity]:
        return BQListView(self._items(), name=self._key).__rsub__(other)

    def __mul__(self, other: Any) -> List[BoundedQuantity]:
        return BQListView(self._items(), name=self._key).__mul__(other)

    def __rmul__(self, other: Any) -> List[BoundedQuantity]:
        return BQListView(self._items(), name=self._key).__rmul__(other)

    def __truediv__(self, other: Any) -> List[BoundedQuantity]:
        return BQListView(self._items(), name=self._key).__truediv__(other)

    def __rtruediv__(self, other: Any) -> List[BoundedQuantity]:
        return BQListView(self._items(), name=self._key).__rtruediv__(other)

    def __pow__(self, other: Any) -> List[BoundedQuantity]:
        return BQListView(self._items(), name=self._key).__pow__(other)

    def __rpow__(self, other: Any) -> List[BoundedQuantity]:
        return BQListView(self._items(), name=self._key).__rpow__(other)


class Cols:
    """
    Storage for plot-ready lists prepared from DATA/CALC/CONST.

    After DATA.prepare("y"), you can use:
      DATA.cols.y
      DATA.cols.uy
      DATA.cols.y_unit
      DATA.cols.y_i
    """

    def __init__(self, owner_name: str):
        object.__setattr__(self, "_owner", str(owner_name))
        object.__setattr__(self, "_d", {})

    def clear(self) -> None:
        """Remove all prepared lists."""
        self._d.clear()

    def keys(self) -> List[str]:
        """Return stored prepared names."""
        return list(self._d.keys())

    def __getattr__(self, name: str) -> Any:
        if name in self._d:
            return self._d[name]
        raise AttributeError(f"{self._owner}.cols has no item {name!r}. Run {self._owner}.prepare(...) first.")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        self._d[name] = value


class BQStore:
    """
    Student-friendly container for lab variables.

    Scalar access:
      DATA["y_0"] = ("12.4","0.3","cm")
      DATA.y_0 is the same object.

            Reassigning the same key turns it into an indexed list:
            DATA["y"] = ("12.4","0.3","cm")
            DATA["y"] = ("12.6","0.3","cm")
            DATA.y  -> [y_0, y_1]

    Column access:
      DATA.y[0] = ("12.4","0.3","cm")
      DATA.y[0] is the same as DATA["y_0"].

    Derived quantities:
      CALC["l_1"] = DATA.y[1] - DATA.y[0]
      CALC.l[1] is the same as CALC["l_1"].

    Prepared plot lists:
      DATA.prepare("y")
      DATA.cols.y, DATA.cols.uy, DATA.cols.y_unit, DATA.cols.y_i
    """

    def __init__(self, name: str):
        object.__setattr__(self, "_name", str(name))
        object.__setattr__(self, "_store", {})
        object.__setattr__(self, "cols", Cols(str(name)))

    def clear(self) -> None:
        """Remove all stored quantities and unregister their symbols."""
        for k in list(self._store.keys()):
            v = self._store.get(k)
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, BoundedQuantity) and item.symbol.strip():
                        _unregister_symbol(item.symbol)
            else:
                _unregister_symbol(k)
        self._store.clear()
        self.cols.clear()

    def _store_list(self, key: str, items: List[BoundedQuantity]) -> None:
        if key in self._store:
            old = self._store.get(key)
            if isinstance(old, list):
                for item in old:
                    if isinstance(item, BoundedQuantity) and item.symbol.strip():
                        _unregister_symbol(item.symbol)
            else:
                _unregister_symbol(key)

        for i, q in enumerate(items):
            q.set_symbol(f"{key}_{i}")
        self._store[key] = items
        if len(items) == 1:
            _inject_global(key, items[0])
            _inject_global(f"{key}_0", items[0])
        else:
            _inject_global(key, BQKeyView(self, key))
            for i, q in enumerate(items):
                _inject_global(f"{key}_{i}", q)
        _register_injected_key(self, key)

    def keys(self) -> List[str]:
        """Return stored keys in insertion order."""
        return list(self._store.keys())

    def __contains__(self, key: str) -> bool:
        if key in self._store:
            return True
        base, idx = self._split_indexed_key(key)
        if base and idx is not None:
            v = self._store.get(base)
            return isinstance(v, list) and 0 <= idx < len(v)
        return False

    def __getitem__(self, key: str) -> Any:
        if key in self._store:
            v = self._store[key]
            return BQListView(v, name=key) if isinstance(v, list) else v
        base, idx = self._split_indexed_key(key)
        if base and idx is not None:
            v = self._store.get(base)
            if isinstance(v, list) and 0 <= idx < len(v):
                return v[idx]
        raise KeyError(key)

    def _coerce_to_bq(self, k: str, value: Any, *, symbol: str) -> BoundedQuantity:
        if isinstance(value, tuple):
            if len(value) != 3:
                raise ValueError(f"{self._name}[{k!r}] tuple must be (value, bound, unit).")
            return Qb(value, symbol=symbol)

        if isinstance(value, BoundedQuantity):
            return value.set_symbol(symbol) if value.symbol.strip() != symbol else value

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return BoundedQuantity.const(float(value)).set_symbol(symbol)

        raise TypeError(
            f"{self._name}[{k!r}] expects a (value, bound, unit) tuple or a BoundedQuantity expression."
        )

    def _coerce_to_bq_with_prev(
        self, k: str, value: Any, *, symbol: str, prev: BoundedQuantity
    ) -> BoundedQuantity:
        if isinstance(value, tuple):
            if len(value) == 1:
                v = value[0]
                b = None
            elif len(value) == 3:
                return self._coerce_to_bq(k, value, symbol=symbol)
            elif len(value) == 2:
                v, b = value
            else:
                raise ValueError(
                    f"{self._name}[{k!r}] tuple must be (value, bound, unit) or (value,) for copy-forward."
                )
        elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
            v = value
            b = None
        else:
            return self._coerce_to_bq(k, value, symbol=symbol)

        mid_prev, u_prev, _lo, _hi = iv_mid_half(prev.I)
        place = prev.rule_place if prev.rule_mode == "place" else place_from_val_sig(
            mid_prev if mid_prev != 0 else 1.0, prev.rule_sig
        )
        u_str = fmt_place(u_prev, place) if b is None else str(b)
        unit_s = unit_str(prev.unit)
        return Qb((v, u_str, unit_s), symbol=symbol)

    def _split_indexed_key(self, key: str) -> Tuple[Optional[str], Optional[int]]:
        k = (key or "").strip()
        m = re.match(r"^(?P<base>.+)_(?P<idx>\d+)$", k)
        if not m:
            return None, None
        base = m.group("base")
        if not base.isidentifier():
            return None, None
        return base, int(m.group("idx"))

    def __setitem__(self, key: str, value: Any) -> None:
        k = (key or "").strip()
        if not k:
            raise ValueError(f"{self._name}[key] requires a non-empty key.")
        if not k.isidentifier():
            raise ValueError(f"{self._name}[{k!r}]: key must be a valid identifier (example: 'y_0', 'l_1').")

        if isinstance(value, BQListView):
            self._store_list(k, list(value))
            return
        if isinstance(value, list) and all(isinstance(v, BoundedQuantity) for v in value):
            self._store_list(k, value)
            return

        base, idx = self._split_indexed_key(k)
        if base and idx is not None:
            existing_base = self._store.get(base)
            if isinstance(existing_base, list):
                if idx < len(existing_base):
                    _unregister_symbol(existing_base[idx].symbol)
                    existing_base[idx] = self._coerce_to_bq(k, value, symbol=f"{base}_{idx}")
                    return
                if idx == len(existing_base):
                    existing_base.append(self._coerce_to_bq(k, value, symbol=f"{base}_{idx}"))
                    return
                raise IndexError(f"{self._name}[{k!r}] index {idx} is out of range.")

        if k not in self._store:
            self._store[k] = self._coerce_to_bq(k, value, symbol=k)
            _inject_global(k, self._store[k])
            _inject_global(f"{k}_0", self._store[k])
            _register_injected_key(self, k)
            return

        existing = self._store.get(k)
        if isinstance(existing, list):
            idx = len(existing)
            q = self._coerce_to_bq_with_prev(k, value, symbol=f"{k}_{idx}", prev=existing[0])
            existing.append(q)
            _inject_global(f"{k}_{idx}", q)
            return

        if isinstance(existing, BoundedQuantity):
            _unregister_symbol(existing.symbol)
            existing.set_symbol(f"{k}_0")
            q = self._coerce_to_bq_with_prev(k, value, symbol=f"{k}_1", prev=existing)
            self._store[k] = [existing, q]
            _inject_global(k, BQKeyView(self, k))
            _inject_global(f"{k}_0", existing)
            _inject_global(f"{k}_1", q)
            _register_injected_key(self, k)
            return

        raise TypeError(f"{self._name}[{k!r}] expects a (value, bound, unit) tuple or a BoundedQuantity expression.")

    def __getattr__(self, name: str) -> Any:
        if name in self._store:
            v = self._store[name]
            return BQListView(v, name=name) if isinstance(v, list) else v
        base, idx = self._split_indexed_key(name)
        if base and idx is not None:
            v = self._store.get(base)
            if isinstance(v, list) and 0 <= idx < len(v):
                return v[idx]
        if name.startswith("_"):
            raise AttributeError(name)
        return BQColumn(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        self[name] = value

    def _scan_indexed_prefix(self, prefix: str) -> List[Tuple[int, str]]:
        """
        Return [(i, key), ...] for keys matching f"{prefix}_<int>" sorted by i.
        """
        p = (prefix or "").strip()
        if not p:
            raise ValueError("prefix cannot be empty.")
        if not p.isidentifier():
            raise ValueError(f"prefix {p!r} must be a valid identifier.")
        out: List[Tuple[int, str]] = []
        pat = re.compile(rf"^{re.escape(p)}_(\d+)$")
        for k in self._store.keys():
            m = pat.match(k)
            if m:
                out.append((int(m.group(1)), k))
        if p in self._store and isinstance(self._store[p], list):
            for i, _q in enumerate(self._store[p]):
                out.append((i, f"{p}_{i}"))
        out.sort(key=lambda t: t[0])
        return out

    def prepare(self, name: str, to: Optional[str] = None) -> None:
        """
        Prepare plot-ready lists for ONE column and store them under <STORE>.cols.

        Reads existing keys like:
          y_0, y_1, y_2, ...

        Stores:
          cols.y       = [mid0, mid1, ...]
          cols.uy      = [u0, u1, ...]
          cols.y_unit  = "cm"  (or target unit if converted)
          cols.y_i     = [0, 1, 2, ...]

        If 'to' is provided, each entry is converted to this unit for the prepared lists only.
        The stored BoundedQuantity objects are not modified.
        """
        col = (name or "").strip()
        if not col:
            raise ValueError("prepare(name, to=None): name cannot be empty.")
        if not col.isidentifier():
            raise ValueError(f"prepare({col!r}): name must be a valid identifier (example: 'y', 't').")

        items = self._scan_indexed_prefix(col)
        if not items:
            raise ValueError(f"{self._name}.prepare({col!r}): no items found like '{col}_0', '{col}_1', ...")

        tgt_unit: Optional[pint.Unit] = parse_unit(to) if to is not None else None

        idx: List[int] = []
        mids: List[float] = []
        us: List[float] = []
        out_unit: Optional[pint.Unit] = None

        for i, k in items:
            if k in self._store:
                q = self._store[k]
            else:
                base, idx = self._split_indexed_key(k)
                if base is None or idx is None:
                    raise KeyError(k)
                v = self._store.get(base)
                if not isinstance(v, list) or idx >= len(v):
                    raise KeyError(k)
                q = v[idx]
            mid, u, _lo, _hi = iv_mid_half(q.I)

            if tgt_unit is None:
                if out_unit is None:
                    out_unit = q.unit
                elif q.unit != out_unit:
                    raise ValueError(f"{self._name}.prepare({col!r}): mixed units found (example: '{k}').")
            else:
                f = Q_(1, q.unit).to(tgt_unit).magnitude
                mid = mid * float(f)
                u = abs(float(f)) * u
                out_unit = tgt_unit

            idx.append(i)
            mids.append(mid)
            us.append(u)

        setattr(self.cols, col, mids)
        setattr(self.cols, "u" + col, us)
        setattr(self.cols, col + "_unit", unit_str(out_unit if out_unit is not None else ureg.dimensionless))
        setattr(self.cols, col + "_i", idx)


def reset_all() -> None:
    """Clear DATA, CALC, CONST, and the symbol registry."""
    DATA.clear()
    CALC.clear()
    CONST.clear()
    for k, store in list(_AUTO_STORES.items()):
        store.clear()
        _AUTO_STORES.pop(k, None)
    try:
        from IPython import get_ipython

        ip = get_ipython()
    except Exception:
        ip = None
    if ip is not None and hasattr(ip, "user_ns"):
        ns = ip.user_ns
        for name in list(ns.keys()):
            if name in ("DATA", "CALC", "CONST"):
                continue
            v = ns.get(name)
            if isinstance(v, BQStore) or isinstance(v, BQKeyView) or isinstance(v, BoundedQuantity):
                ns.pop(name, None)
    _INJECTED_KEY_TARGETS.clear()
    reset_symbol_registry()


def make_store(name: str) -> BQStore:
    """Create a new BQStore instance (e.g., ELONGATION = make_store("ELONGATION"))."""
    return BQStore(name)


def get_or_create_store(name: str) -> BQStore:
    """Get or create a named BQStore for auto-created globals (uppercase names only)."""
    n = (name or "").strip()
    if not n or not n.isidentifier() or not n.isupper():
        raise AttributeError(name)
    if n in ("DATA", "CALC", "CONST"):
        return globals()[n]
    if n not in _AUTO_STORES:
        _AUTO_STORES[n] = BQStore(n)
    return _AUTO_STORES[n]




DATA = BQStore("DATA")
CALC = BQStore("CALC")
CONST = BQStore("CONST")
