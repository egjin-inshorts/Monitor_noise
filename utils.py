class Option:
    def __init__(self, **kwargs):
        self._options = kwargs
        for key, value in kwargs.items():
            if isinstance(value, dict):
                self._options[key] = Option(**value)
            else:
                self._options[key] = value


    def __setattr__(self, name, value):
        if name != "_options":
            if isinstance(value, dict):
                value = Option(**value)
            self._options[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        if name in self._options:
            return self._options[name]
        else:
            raise AttributeError(f"Option '{name}' not found.")

    def __getitem__(self, key):
        return self._options[key]

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            self._options[key] = Option(**value)
        else:
            self._options[key] = value

    def __repr__(self):
        return f"Option({self._options})"
