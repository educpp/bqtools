# ubq (in bqtools)

**ubq** is a student‑friendly uncertainty tool for lab notebooks. Students only type simple values in notebook cells; the library handles bounds, units, and propagation.

---

## Install (Microsoft Teams / Jupyter)

Example install cell:

```python
import micropip

await micropip.install(
  "https://cdn.jsdelivr.net/gh/<YOUR_GITHUB_USER>/bqtools@v0.1.0/wheels/ubq-0.1.0-py3-none-any.whl"
)

from ubq import DATA, CONST, set_u_sig_digits, reset_all
```

---

## Quick start

### 1) Reset (top of notebook)
```python
reset_all()
set_u_sig_digits(1)
```

### 2) Enter measurements (simple, repeated lines)
```python
DATA.M = ("0.000", "0.001", "kg")
DATA.y = ("165.0", "0.5", "cm")

print(M)
print(y)
```

### 3) Add more trials using short input
After the first entry, you can omit uncertainty and/or unit.

```python
M = ("0.100")
y = ("153.0")

print(M)
print(y)
```

### 4) Compute derived quantities (column math)
```python
ELONGATION["x"] = y - y_0
print(x)
```

---

## Student‑friendly rules

### Auto tables for repeated assignments
```python
DATA.M = ("0.000", "0.001", "kg")
M = ("0.100")
print(M)
```

### Short‑hand input signatures
```python
DATA.y = ("165.0", "0.5", "cm")
y = ("153.0")
y = ("148.0", "0.2")
```

### Column math (element‑wise)
```python
ELONGATION["x"] = y - y_0
SUMS["s"] = y + z
```

### Capitalized names are stores
```python
FORCE["F"] = M * CONST.g
print(F)
```

### Notebook‑only rewrite
In notebooks, `M = ("0.100")` is automatically treated as an append to `DATA.M`.

---

## Example (short)
```python
reset_all()
set_u_sig_digits(1)

DATA.M = ("0.000", "0.001", "kg")
DATA.y = ("165.0", "0.5", "cm")

M = ("0.100")
y = ("153.0")

ELONGATION["x"] = y - y_0
print(x)
```

---

## License
GNU GENERAL PUBLIC LICENSE v3
