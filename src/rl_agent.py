
# RL agent placeholder - tries to use stable-baselines3 if installed.
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    import gym
except Exception as e:
    PPO = None
    make_vec_env = None
    gym = None

class RLAgent:
    def __init__(self, env_id='CartPole-v1', model_path='models/rl_agent.zip'):
        self.env_id = env_id
        self.model_path = model_path
        self.model = None
        if PPO is None:
            print('stable-baselines3 not available. RLAgent will be a placeholder.')
        else:
            self.env = make_vec_env(env_id, n_envs=1)

    def train(self, total_timesteps=10000):
        if PPO is None:
            raise RuntimeError('stable-baselines3 not installed')
        self.model = PPO('MlpPolicy', self.env, verbose=1)
        self.model.learn(total_timesteps=total_timesteps)
        self.model.save(self.model_path)

    def load(self):
        if PPO is None:
            return None
        self.model = PPO.load(self.model_path)
        return self.model
