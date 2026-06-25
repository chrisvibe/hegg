def compute_reservoir_metrics(p) -> dict:
    from project.boolean_reservoir.code.reservoir import BooleanReservoir
    from project.boolean_reservoir.code.lut import calc_lut_p
    params = p.to_pydantic()
    model = BooleanReservoir.build_graph_and_lut(params)
    return {
        # "spectral_radius": float(calc_spectral_radius(model.graph)),
        "lut_p": calc_lut_p(model.lut),
    }

def reservoir_key(p):
    return (p.M.R.seed, p.M.R.n_nodes)

def get_reservoir_metrics(p) -> dict:
    """Return stored reservoir metrics if available, else compute them.

    Uses stored metrics if lut_p is present. spectral_radius is intentionally
    null in all current parquets and is excluded from the availability check.
    """
    rm = getattr(p.L, 'reservoir_metrics', None)
    if rm is not None:
        stored = rm.model_dump() if hasattr(rm, 'model_dump') else dict(rm)
        if stored.get('lut_p') is not None:
            return stored
    return compute_reservoir_metrics(p)
