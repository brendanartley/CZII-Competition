from types import SimpleNamespace

def update_cfg(cfg: SimpleNamespace, other_args: str, log: bool = False):
    """
    Helper to update cfg paramaters.
    """

    # Update params
    for key in other_args:

        # Nested config
        keys = key.split(".")
        if "." in key:
            if log:
                print(f'overwriting cfg.{keys[0]}.{keys[1]}: {cfg.__dict__[keys[0]].__dict__[keys[1]]} -> {other_args[key]}')
            assert len(keys) == 2

            cfg_type = type(cfg.__dict__[keys[0]].__dict__[keys[1]])
            if other_args[key] == 'None':
                cfg.__dict__[keys[0]].__dict__[keys[1]] = None
            elif cfg_type == bool:
                cfg.__dict__[keys[0]].__dict__[keys[1]] = other_args[key] == 'True'
            elif cfg_type == type(None):
                cfg.__dict__[keys[0]].__dict__[keys[1]] = other_args[key]
            else:
                cfg.__dict__[keys[0]].__dict__[keys[1]] = cfg_type(other_args[key])

        # Main config
        elif key in cfg.__dict__:
            if log:
                print(f'overwriting cfg.{key}: {cfg.__dict__[key]} -> {other_args[key]}')

            cfg_type = type(cfg.__dict__[key])
            if other_args[key] == 'None':
                cfg.__dict__[key] = None
            elif cfg_type == bool:
                cfg.__dict__[key] = other_args[key] == 'True'
            elif cfg_type == type(None):
                cfg.__dict__[key] = other_args[key]
            else:
                cfg.__dict__[key] = cfg_type(other_args[key])

    return cfg
