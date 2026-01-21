# ubq (in bqtools)

**ubq** is a student-friendly, bounds-based uncertainty helper for lab notebooks running in **Microsoft Teams / Jupyter** (Pyodide-style environments). It lets students enter measurements as `(value, bound, unit)`, compute derived quantities with **exact interval arithmetic**, and print results consistently with a simple “uncertainty rounding” rule and clear warnings.

The goal is: **students never edit the helper code**. They only edit lab input cells. You distribute ubq as a **pure-Python wheel** and install it with `micropip`.

---

## What this solves

Typical pain points in student lab reports:

- Students break the framework code while trying to “just fill in numbers”
- Mixed units and conversions are error-prone
- Uncertainty reporting and rounding rules drift between students
- Hard to audit how a result was computed

ubq addresses this with:

- A **DATA / CALC / CONST** container workflow:
  - `DATA` for measured values
  - `CONST` for constants (e.g., `g`)
  - `CALC` for computed results
- **Interval arithmetic** via `mpmath.iv` for bounds propagation (no stepwise rounding in computation)
- Output formatting: `x = ... = value ± u (interval [..., ...])`
- Warnings if rounding reduces coverage of the exact interval
- A student-readable computation tree via `trace(x)`

---

## Install (Microsoft Teams / Pyodide)

Students install ubq from a **wheel URL** (recommended: GitHub tag + jsDelivr).

Example install cell:

```python
import micropip

await micropip.install(
    "https://cdn.jsdelivr.net/gh/<YOUR_GITHUB_USER>/bqtools@v0.1.0/wheels/ubq-0.1.0-py3-none-any.whl"
)

from ubq import DATA, CALC, CONST, trace, set_u_sig_digits, reset_all
```

> Why a wheel URL? `micropip` installs **wheels**, not raw source from GitHub.

---

## Quick start

### 1) Reset all state (recommended at top of each notebook)
```python
reset_all()
set_u_sig_digits(1)   # keep 1 significant digit in final uncertainty (rounded up)
```

### 2) Enter measured data
Scalar style:
```python
DATA["x"] = ("12.4", "0.3", "cm")
DATA["t"] = ("2.88", "0.07", "s")

print(DATA.x)
print(DATA.t)
```

Column style (indexed trials):
```python
DATA.l[0] = ("0.105", "0.001", "m")
DATA.l[1] = ("0.115", "0.001", "m")
DATA.l[2] = ("0.125", "0.001", "m")
```

### 3) Compute derived quantities
```python
CALC["v"] = DATA.x / DATA.t
print(CALC.v)

# Convert units (in-place on that object)
CALC.v.to("m/s")
print(CALC.v)
```

### 4) Inspect the calculation tree
```python
trace(CALC.v)
```

---

## Example: spring constant from 10 trials

We measure mass `m` (grams) and length/extension `l` (meters). Compute force:

\[
F = (m \times 10^{-3})\, g
\]

Then fit a line \(F = k\,l + b\). The slope is the spring constant \(k\).

```python
reset_all()
set_u_sig_digits(1)

# Constant
CONST["g"] = ("9.80", "0", "m/s^2")

# 10 trials: masses in grams, extensions in meters
DATA.m[0] = ("50.0", "0.1", "g")
DATA.m[1] = ("100.0","0.1", "g")
DATA.m[2] = ("150.0","0.1", "g")
DATA.m[3] = ("200.0","0.1", "g")
DATA.m[4] = ("250.0","0.1", "g")
DATA.m[5] = ("300.0","0.1", "g")
DATA.m[6] = ("350.0","0.1", "g")
DATA.m[7] = ("400.0","0.1", "g")
DATA.m[8] = ("450.0","0.1", "g")
DATA.m[9] = ("500.0","0.1", "g")

DATA.l[0] = ("0.105", "0.001", "m")
DATA.l[1] = ("0.115", "0.001", "m")
DATA.l[2] = ("0.125", "0.001", "m")
DATA.l[3] = ("0.135", "0.001", "m")
DATA.l[4] = ("0.145", "0.001", "m")
DATA.l[5] = ("0.155", "0.001", "m")
DATA.l[6] = ("0.165", "0.001", "m")
DATA.l[7] = ("0.175", "0.001", "m")
DATA.l[8] = ("0.185", "0.001", "m")
DATA.l[9] = ("0.195", "0.001", "m")

# Compute F_i = (m_i in kg) * g
for i in range(10):
    CALC.F[i] = (DATA.m[i].to("kg") * CONST.g).to("N")

print(CALC.F[0])
print(CALC.F[9])

# Prepare plot/fit columns (numeric lists)
DATA.prepare("l", to="m")
CALC.prepare("F", to="N")

x = DATA.cols.l
y = CALC.cols.F
uy = CALC.cols.uF   # optional weights for fitting
```

Now you can do a linear fit using whatever your notebook environment provides (e.g., NumPy’s `polyfit` if available).

---

## Public API reference (docstring-style)

### `reset_all() -> None`
Clear `DATA`, `CALC`, `CONST`, and internal symbol registry. Use at the top of each notebook run.

**Example**
```python
reset_all()
```

---

### `set_u_sig_digits(n: int) -> None`
Configure how many significant digits to keep in the **final reported uncertainty**.

- `n >= 1`
- Uncertainty is rounded **up** to the selected precision (ceiling-style rule)
- Computation remains exact interval math; this only affects final display

**Example**
```python
set_u_sig_digits(1)   # classroom-friendly
set_u_sig_digits(2)   # more precise reporting
```

---

### `DATA`, `CALC`, `CONST` (instances of `BQStore`)
Student-facing containers.

#### Scalar assignment
```python
DATA["x"] = ("12.4", "0.3", "cm")
print(DATA.x)
```

#### Column assignment (indexed)
```python
DATA.y[0] = ("1.23", "0.05", "m")
DATA.y[1] = ("1.25", "0.05", "m")

print(DATA.y[0])
```

#### Computed assignment
```python
CALC["v"] = DATA.x / DATA.t
print(CALC.v)
```

#### Constants (same interface)
```python
CONST["g"] = ("9.80", "0", "m/s^2")
```

---

### `BQStore.prepare(name: str, to: Optional[str] = None) -> None`
Prepare **one column** as plain numeric lists for plotting/fitting.

Reads keys like:
- `name_0, name_1, name_2, ...`

Stores on `STORE.cols`:
- `cols.<name>`     list of midpoints
- `cols.u<name>`    list of half-widths (uncertainties from interval)
- `cols.<name>_unit` string
- `cols.<name>_i`    list of indices

If `to` is given, values are converted into that unit **for the prepared lists only** (objects are not modified).

**Example**
```python
DATA.prepare("l", to="m")
x = DATA.cols.l
ux = DATA.cols.ul
```

---

### `trace(x: BoundedQuantity) -> None`
Print a readable, multi-line breakdown of how `x` was built, including:

- Exact interval `[lo, hi]`
- Half-width `u(iv)`
- Reported rounded `value ± u`
- Coverage warning if rounding reduces overlap with exact interval
- Lightweight “tracked place/sig” metadata used for warnings

**Example**
```python
trace(CALC.v)
```

---

### `BoundedQuantity`
Internal numeric type representing a bounds-based quantity with units.

Students normally do not construct this directly; they use `DATA[...] = (...)` and `CALC[...] = expression`.

Common methods students use:
- `.to("m/s")`  convert units in-place (Pint)
- `print(q)`    formatted report line

---

## Design notes (instructor-facing)

- ubq uses **exact interval arithmetic** (`mpmath.iv`) for propagation of measurement bounds.
- “Sig-fig rules” are used only as *metadata warnings* (not for core math).
- Final reporting rounds the uncertainty to `U_SIG_DIGITS` significant digits (rounded up) and rounds the value to the same least significant digit.
- If rounding reduces overlap with the exact interval, ubq emits a coverage warning instead of inflating uncertainty.

---

## Building and publishing (maintainer)

### Build wheel
```bash
python -m pip install --upgrade pip
python -m pip install build
python -m build
```

Wheel appears in `dist/`, e.g.:
- `dist/ubq-0.1.0-py3-none-any.whl`

### Publish for students (recommended: jsDelivr)
Commit wheel into `wheels/`, tag, and push.

```bash
mkdir -p wheels
cp dist/ubq-0.1.0-py3-none-any.whl wheels/
git add wheels/ubq-0.1.0-py3-none-any.whl
git commit -m "Add wheel ubq 0.1.0"
git tag v0.1.0
git push --tags
git push
```

Students then install:
```python
await micropip.install("https://cdn.jsdelivr.net/gh/<YOUR_GITHUB_USER>/bqtools@v0.1.0/wheels/ubq-0.1.0-py3-none-any.whl")
```

---

## License
GNU GENERAL PUBLIC LICENSE v3
