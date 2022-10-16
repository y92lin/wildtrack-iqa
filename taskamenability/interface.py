from taskamenability.envs.task_amenability import TaskAmenability

import numpy as np

from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv

from keras import layers
import keras
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from keras.models import load_model


class PPOInterface():
    def __init__(self, x_train, y_train, x_val, y_val, x_holdout, y_holdout, task_predictor, img_shape,
                 load_models=False, controller_save_path=None, task_predictor_save_path=None):

        self.x_holdout, self.y_holdout = x_holdout, y_holdout

        if load_models:
            self.task_predictor = load_model(task_predictor_save_path)
        else:
            self.task_predictor = task_predictor

        def make_env():
            return TaskAmenability(x_train, y_train, x_val, y_val, x_holdout, y_holdout, task_predictor, img_shape)

        self.env = DummyVecEnv([make_env])

        def get_from_env(env, parameter):
            return env.get_attr(parameter)[0]

        self.n_rollout_steps = get_from_env(self.env, 'controller_batch_size') + len(get_from_env(self.env,
                                                                                                  'x_val'))  # number of steps per episode (controller_batch_size + val_set_len) multiply by an integer to do multiple episodes before controller update

        if load_models:
            assert isinstance(controller_save_path, str)
            self.load(save_path=controller_save_path)
        else:
            self.model = PPO2('CnnPolicy',
                              self.env,
                              nminibatches=1,
                              n_steps=self.n_rollout_steps,
                              gamma=1.0,
                              verbose=2,
                              seed=None)

    def train(self, num_episodes):
        time_steps = int(num_episodes * self.n_rollout_steps)

        print(f'Training started for {num_episodes} episodes:')

        self.model.learn(total_timesteps=time_steps)

    def get_controller_preds_on_holdout(self):
        actions = []
        for i in range(len(self.x_holdout)):
            pred = self.model.predict(self.x_holdout[i, :, :, :])[0][0]
            actions.append(pred)

        return np.array(actions)

    def save(self, controller_save_path, task_predictor_save_path):
        self.model.save(controller_save_path)
        #task_predictor_copy = self.env.get_attr('task_predictor')[0]
        #task_predictor_copy.save(task_predictor_save_path)

    def load(self, save_path):
        self.model = PPO2.load(save_path)
        self.model.set_env(self.env)

    # implement saving and loading task predictor
