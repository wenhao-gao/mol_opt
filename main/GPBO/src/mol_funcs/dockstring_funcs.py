import dockstring


# Protected dockstring function
def safe_dock_function(smiles, target_name, **dock_kwargs):
    target = dockstring.load_target(target_name)
    try:
        docking_output = target.dock(smiles, **dock_kwargs)
        score = docking_output[0]
    except dockstring.DockstringError:
        score = float("nan")
    return score
