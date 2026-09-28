"""Read literal preset metadata without importing or executing the frozen solver."""
import ast
from pathlib import Path


def preset_volume_fractions(path):
    """Read ProblemSpec.volfrac from the frozen ALL_PRESETS dictionary.

    Process dictionary entries and unpackings in source order, preserving Python's
    last-key-wins behavior. Only literal volume fractions are supported; a changed
    preset representation must be reviewed rather than evaluated as Python code.
    """
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    tables = {}
    for statement in tree.body:
        if isinstance(statement, ast.AnnAssign):
            target, value = statement.target, statement.value
        elif isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target, value = statement.targets[0], statement.value
        else:
            continue
        if not isinstance(target, ast.Name) or not (
            target.id.startswith("PRESETS_") or target.id == "ALL_PRESETS"
        ):
            continue
        if not isinstance(value, ast.Dict):
            raise ValueError(f"Expected a literal preset dictionary: {target.id}")
        table = {}
        for key, spec in zip(value.keys, value.values):
            if key is None:
                if not isinstance(spec, ast.Name) or spec.id not in tables:
                    raise ValueError("Expected an earlier preset dictionary in unpacking")
                table.update(tables[spec.id])
                continue
            name = ast.literal_eval(key)
            if not isinstance(name, str) or not isinstance(spec, ast.Call) or not (
                isinstance(spec.func, ast.Name) and spec.func.id == "ProblemSpec"
            ) or spec.args or any(k.arg is None for k in spec.keywords):
                raise ValueError(f"Expected an explicit ProblemSpec for {name!r}")
            fractions = [k.value for k in spec.keywords if k.arg == "volfrac"]
            if len(fractions) != 1:
                raise ValueError(f"Expected one explicit volfrac for {name!r}")
            fraction = ast.literal_eval(fractions[0])
            if type(fraction) not in (int, float) or not 0 < fraction < 1:
                raise ValueError(f"Invalid literal volfrac for {name!r}")
            table[name] = fraction
        tables[target.id] = table
    if "ALL_PRESETS" not in tables:
        raise ValueError("Missing ALL_PRESETS dictionary")
    return tables["ALL_PRESETS"]
