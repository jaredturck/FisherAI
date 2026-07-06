"""Build long-format benchmark rows without touching production hot paths."""

CSV_FIELDS = (
    "category",
    "scope",
    "component",
    "metric",
    "value",
    "unit",
    "count",
    "mean",
    "p50",
    "p95",
    "p99",
    "max",
    "share_percent",
    "notes",
)


def percentile(values, percent):
    """Return one linearly interpolated percentile."""
    if not values:
        return ""
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(percent) / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def metric_row(
    category,
    scope,
    component,
    metric,
    value,
    unit="",
    count="",
    mean="",
    p50="",
    p95="",
    p99="",
    maximum="",
    share_percent="",
    notes="",
):
    """Return one normalized long-format benchmark row."""
    return {
        "category": category,
        "scope": scope,
        "component": component,
        "metric": metric,
        "value": value,
        "unit": unit,
        "count": count,
        "mean": mean,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "max": maximum,
        "share_percent": share_percent,
        "notes": notes,
    }


def distribution_row(
    category,
    scope,
    component,
    metric,
    values,
    unit,
    share_base=None,
    notes="",
):
    """Summarize repeated observations in one CSV row."""
    values = [float(value) for value in values]
    total = sum(values)
    share_percent = ""
    if share_base:
        share_percent = total / share_base * 100.0
    return metric_row(
        category,
        scope,
        component,
        metric,
        total,
        unit=unit,
        count=len(values),
        mean=total / len(values) if values else "",
        p50=percentile(values, 50),
        p95=percentile(values, 95),
        p99=percentile(values, 99),
        maximum=max(values) if values else "",
        share_percent=share_percent,
        notes=notes,
    )


def find_row(rows, scope, component, metric):
    """Find a benchmark row by its stable identity."""
    for row in rows:
        if (
            row["scope"] == scope
            and row["component"] == component
            and row["metric"] == metric
        ):
            return row
    return None
