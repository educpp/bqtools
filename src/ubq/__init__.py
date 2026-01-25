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
    get_injected_target,
)


def __getattr__(name: str):
    if name in globals():
        return globals()[name]
    store = get_or_create_store(name)
    globals()[name] = store
    return store


def enable_assignment_rewrite() -> bool:
    try:
        from IPython import get_ipython
        import ast
    except Exception:
        return False

    ip = get_ipython()
    if ip is None or not hasattr(ip, "ast_transformers"):
        return False

    class _UbqAssignTransformer(ast.NodeTransformer):
        def _ubq_attr(self, name: str) -> ast.Attribute:
            return ast.Attribute(
                value=ast.Name(id="ubq", ctx=ast.Load()),
                attr=name,
                ctx=ast.Load(),
            )

        def visit_Assign(self, node: ast.Assign):
            self.generic_visit(node)
            if len(node.targets) != 1:
                return node
            target = node.targets[0]
            if isinstance(target, ast.Name):
                info = get_injected_target(target.id)
                if info:
                    store_name, key = info
                    node.targets[0] = ast.Attribute(
                        value=ast.Name(id=store_name, ctx=ast.Load()),
                        attr=key,
                        ctx=ast.Store(),
                    )
                    return node
                if target.id.isupper():
                    node.value = self._ubq_attr(target.id)
                    return node
            return node

        def visit_AnnAssign(self, node: ast.AnnAssign):
            self.generic_visit(node)
            target = node.target
            if isinstance(target, ast.Name):
                info = get_injected_target(target.id)
                if info:
                    store_name, key = info
                    node.target = ast.Attribute(
                        value=ast.Name(id=store_name, ctx=ast.Load()),
                        attr=key,
                        ctx=ast.Store(),
                    )
                    return node
                if target.id.isupper():
                    node.value = self._ubq_attr(target.id)
                    return node
            return node

        def visit_Subscript(self, node: ast.Subscript):
            self.generic_visit(node)
            if isinstance(node.value, ast.Name) and node.value.id.isupper():
                node.value = self._ubq_attr(node.value.id)
            return node

        def visit_Attribute(self, node: ast.Attribute):
            self.generic_visit(node)
            if isinstance(node.value, ast.Name) and node.value.id.isupper():
                node.value = self._ubq_attr(node.value.id)
            return node

    for t in ip.ast_transformers:
        if t.__class__.__name__ == "_UbqAssignTransformer":
            return True

    ip.ast_transformers.append(_UbqAssignTransformer())
    return True


enable_assignment_rewrite()
