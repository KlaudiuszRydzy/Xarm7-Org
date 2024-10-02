import gymnasium as gym
from controllers import ModelController, ScriptedController, build_controller


def main():
    env = gym.make("luc-pickup-v0.0")
    obs = env.reset()
    controller = build_controller(mode="scripted")

    for _ in range(1000):
        action = controller(obs)
        obs, reward, truncated, terminated, info = env.step(env.action_space.sample())
        env.render()
