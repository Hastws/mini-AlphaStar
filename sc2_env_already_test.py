from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env
from absl import app, flags, logging

import random

from pysc2.maps.ladder import players


class PyAgent(base_agent.BaseAgent):
    def step(self, obs):
        super(PyAgent, self).step(obs)

        return actions.FUNCTIONS.no_op()


def main_function(unused_agent):
    agent = PyAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                    map_name="Simple64",
                    players=[sc2_env.Agent(sc2_env.Race.protoss),
                             sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_easy)],
                    agent_interface_format=features.AgentInterfaceFormat(features.Dimensions(256, 256),
                                                                         use_feature_units=True),
                    step_mul=1, game_steps_per_episode=0,
                    visualize=True) as env:
                agent.setup(env.observation_spec(), env.action_spec())
                time_steps = env.reset()
                agent.reset()

                while True:
                    step_action = [agent.step(time_steps[0])]
                    if time_steps[0].last():
                        break
                    time_steps = env.step(step_action)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    app.run(main_function)
