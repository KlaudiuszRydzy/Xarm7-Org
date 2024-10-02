class ScriptedController:
    def __init__(self):
        self.action = None

    def __call__(self, obs):
        return self.action

    def update(self, obs, reward, truncated, terminated, info):
        self.action = 1


class ModelController:
    def __init__(self):
        self.model = None

    def __call__(self, obs):
        return self.model.predict(obs)


def build_controller(mode="scripted"):
    if mode == "scripted":
        return ScriptedController()
    elif mode == "model":
        return ModelController()
    else:
        raise ValueError(f"Invalid mode: {mode}")
