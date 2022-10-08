import torch
import gym
import numpy as np
from gym import spaces

from keras import layers
import keras
from sklearn.metrics import accuracy_score


class TaskAmenability(gym.Env):

    def __init__(self, x_train, y_train, x_val, y_val, x_holdout, y_holdout, task_predictor, img_shape):

        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val
        self.x_holdout, self.y_holdout = x_holdout, y_holdout

        self.img_shape = img_shape
        self.task_predictor = task_predictor

        self.controller_batch_size = 8
        self.task_predictor_batch_size = 4
        self.epochs_per_batch = 2

        self.img_shape = img_shape

        self.num_val = len(self.x_val)

        self.observation_space = spaces.Box(low=0, high=1, shape=self.img_shape, dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self.actions_list = []
        self.val_metric_list = [0.5] * 10

        self.sample_num_count = 0

    def get_batch(self):
        shuffle_inds = np.random.permutation(len(self.y_train)).astype(int)
        self.x_train, self.y_train = self.x_train[shuffle_inds], self.y_train[shuffle_inds]
        return self.x_train[:self.controller_batch_size], self.y_train[:self.controller_batch_size]

    def compute_moving_avg(self):
        self.val_metric_list = self.val_metric_list[-10:]
        moving_avg = np.mean(self.val_metric_list)
        return moving_avg

    def get_val_acc_vec(self):
        val_acc_vec = []
        for i in range(len(self.y_val)):
            with torch.no_grad():
                obs = self.x_val[i].permute(2, 0, 1).float()
                y_pred = self.task_predictor(obs)
                print(y_pred)
                if  len(y_pred['pred_classes']) == 0:
                  y_pred = 0
                else:
                  to_keep = (y_pred['scores'] > 0.5).nonzero(as_tuple=True)
                  if len(y_pred['pred_classes'][to_keep]) > 0:
                    y_pred = 1
                  else:
                    y_pred = 0

                print((y_pred, self.y_val[i:i+1]))
                val_metric = accuracy_score(self.y_val[i:i+1], np.array([y_pred]))
                val_acc_vec.append(val_metric)
        print(val_acc_vec)
        return np.array(val_acc_vec)

    def step(self, action):
        self.actions_list.append(action)
        self.sample_num_count += 1

        if self.sample_num_count < self.controller_batch_size + self.num_val:
            reward = 0
            done = False
            return self.x_data[self.sample_num_count], reward, done, {}

        else:
            moving_avg = self.compute_moving_avg()
            val_acc_vec = self.get_val_acc_vec()
            val_sel_vec = self.actions_list[self.controller_batch_size:]
            val_sel_vec_normalised = np.array(val_sel_vec) / np.mean(val_sel_vec)

            val_metric = np.mean(np.multiply(val_sel_vec_normalised, np.array(val_acc_vec)))

            self.val_metric_list.append(val_metric)
            reward = val_metric - moving_avg
            done = True
            return np.random.rand(self.img_shape[0], self.img_shape[1], self.img_shape[2]), reward, done, {}

    def reset(self):
        print("test this out")
        self.x_train_batch, self.y_train_batch = self.get_batch()

        self.x_data = np.concatenate((self.x_train_batch, self.x_val), axis=0)
        self.y_data = np.concatenate((self.y_train_batch, self.y_val), axis=0)

        self.actions_list = []
        self.sample_num_count = 0

        return self.x_train_batch[self.sample_num_count]

    def save_task_predictor(self, task_predictor_save_path):
        self.task_predictor.save(task_predictor_save_path)
