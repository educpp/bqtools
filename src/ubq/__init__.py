from .core import (
    DATA,
    CALC,
    CONST,
    trace,
    trace_str,
    set_u_sig_digits,
    set_config,
    reset_all,
    reset_symbol_registry,
    BoundedQuantity,
    BQStore,
    make_store,
    get_or_create_store,
    add_quantity,
)


def __getattr__(name: str):
    if name in globals():
        return globals()[name]
    store = get_or_create_store(name)
    globals()[name] = store
    return store


def add_quantity(name: str):
    store = get_or_create_store(name)
    globals()[name] = store
    try:
        import inspect

        frame = inspect.currentframe()
        if frame is not None and frame.f_back is not None:
            frame.f_back.f_globals[name] = store
            ann = frame.f_back.f_globals.get("__annotations__")
            if ann is None:
                ann = {}
                frame.f_back.f_globals["__annotations__"] = ann
            ann[name] = BQStore
    finally:
        try:
            del frame
        except Exception:
            pass
    return store
