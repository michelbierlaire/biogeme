def pretty_model(model):
    """Print a human-readable snapshot of a PyMC model (version-robust).

    This avoids `model.describe()` (not present in all versions) and instead
    inspects the standard attributes exposed by `pm.Model`.
    """

    # Helper: summarize an RV
    def _rv_row(rv, kind):
        op = getattr(getattr(rv, "owner", None), "op", None)
        dist = getattr(op, "name", None) or type(op).__name__
        # Robust shape representation: avoid truth-testing tensors and int-casting symbolic dims
        shp_attr = getattr(rv, "shape", ())
        try:
            shp_iter = tuple(shp_attr)  # may contain symbolic dims
        except (TypeError, ValueError, AttributeError):
            shp_iter = ()
        shp = tuple(s if isinstance(s, int) else str(s) for s in shp_iter)
        return {
            "name": rv.name,
            "kind": kind,
            "dist": dist,
            "shape": shp,
            "dtype": str(getattr(rv, "dtype", "?")),
        }

    rows = []
    for rv in getattr(model, "free_RVs", []):
        rows.append(_rv_row(rv, "free"))
    for rv in getattr(model, "observed_RVs", []):
        rows.append(_rv_row(rv, "observed"))
    for d in getattr(model, "deterministics", []):
        rows.append(
            {
                "name": d.name,
                "kind": "deterministic",
                "dist": "Deterministic",
                # Avoid boolean evaluation on tensor shapes; stringify symbolic dims
                "shape": (
                    lambda _a: (
                        (
                            lambda _it: tuple(
                                _s if isinstance(_s, int) else str(_s) for _s in _it
                            )
                        )(tuple(_a) if not isinstance(_a, (int,)) else (_a,))
                    )
                )(getattr(d, "shape", ())),
                "dtype": str(getattr(d, "dtype", "?")),
            }
        )

    # Pretty print table without pandas dependency
    print("\n=== Model variables ===")
    if not rows:
        print("(no variables found)")
    else:
        headers = ("name", "kind", "dist", "shape", "dtype")

        # compute column widths
        def _fmt(val):
            if isinstance(val, tuple):
                return str(val)
            return str(val)

        widths = {h: max(len(h), max(len(_fmt(r[h])) for r in rows)) for h in headers}
        # header
        line = "  ".join(h.ljust(widths[h]) for h in headers)
        print(line)
        print("  ".join("-" * widths[h] for h in headers))
        # rows
        for r in rows:
            print("  ".join(_fmt(r[h]).ljust(widths[h]) for h in headers))

    # Lists by category
    print("\nFREE RVs :", [rv.name for rv in getattr(model, "free_RVs", [])])
    print("DETERMIN.:", [d.name for d in getattr(model, "deterministics", [])])
    print("OBSERVED :", [rv.name for rv in getattr(model, "observed_RVs", [])])

    # Try evaluating logp at the initial point (value space, not transformed)
    try:
        # Prefer explicit value variables to avoid transform name mismatches (e.g., *_interval__)
        rvs_to_values = getattr(model, "rvs_to_values", {})
        if rvs_to_values:
            val_vars = list(rvs_to_values.values())
        else:
            val_vars = list(getattr(model, "value_vars", []))

        if not val_vars:
            raise AttributeError("No value variables available to compile logp.")

        # Compile logp expecting the untransformed value variables as inputs
        logp_fn = model.compile_logp(inputs=val_vars)

        # Build an input point in value space using each variable's test_value
        ip_val = {}
        for v in val_vars:
            tv = getattr(getattr(v, "tag", object()), "test_value", None)
            if tv is None:
                raise ValueError(
                    f"No test_value found for value var '{getattr(v, 'name', '?')}'."
                )
            ip_val[v.name] = tv

        lp = float(logp_fn(ip_val))
        print("\nlogp(initial point):", lp)
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        print("\n(logp at initial point not available)", e)
