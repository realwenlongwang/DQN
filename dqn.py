import numpy as np
import tensorflow as tf
from enum import IntEnum


class Experience:
    CURRENT_STATE = 0
    ACTION = 1
    REWARD = 2
    NEXT_STATE = 3
    DONE = 4


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, random_probability=0.1, mem_size=300,
                 batch_size=32, update_epoch=300):
        self.actions = actions  # number of legal actions
        self.alpha = learning_rate
        self.gamma = reward_decay
        self.epsilon = 1 - random_probability
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.update_epoch = update_epoch
        self.epoch_counter = 0

        self.input_state = tf.placeholder(shape=(None, 1), dtype=tf.float32)
        self.predict_layer_list = []
        self.q_predicts = self.create_net("predict_net", self.predict_layer_list)
        # with tf.variable_scope("net_predict"):
        #     self.predict_layer1 = tf.layers.Dense(units=10, activation=tf.nn.relu, name="layer")
        #     self.predict_layer_list.append(self.predict_layer1)
        #     self.predict_layer2 = tf.layers.Dense(units=10, activation=tf.nn.relu, name="layer")
        #     self.predict_layer_list.append(self.predict_layer2)
        #     self.predict_output_layer = tf.layers.Dense(units=self.actions, name="layer")
        #     self.predict_layer_list.append(self.predict_output_layer)
        self.q_greatest_predict = tf.reduce_max(self.q_predicts)

        self.target_layer_list = []
        self.q_nexts = self.create_net("target_net", self.target_layer_list)
        # with tf.variable_scope("net_target"):
        #     self.target_layer1 = tf.layers.Dense(units=10, activation=tf.nn.relu, name="layer")
        #     self.predict_layer_list.append(self.target_layer1)
        #     self.target_layer2 = tf.layers.Dense(units=10, activation=tf.nn.relu, name="layer")
        #     self.predict_layer_list.append(self.target_layer2)
        #     self.target_output_layer = tf.layers.Dense(units=self.actions, name="layer")
        #     self.predict_layer_list.append(self.target_output_layer)
        self.q_greatest_next = tf.reduce_max(self.q_nexts)

        self.q_target = tf.placeholder(dtype=tf.float32, shape=(), name="Q_target")  # A scalar placeholder for y

        # TODO: looks like
        with tf.variable_scope("loss"):
            self.loss = tf.squared_difference(self.q_target, self.q_greatest_predict)
        with tf.variable_scope("train"):
            self.train_op = tf.train.GradientDescentOptimizer(self.alpha).minimize(self.loss)
        init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init_op)

        # A list of memory for experience
        self.mem = np.zeros((self.mem_size, 5))  # current state, action, reward, next state, done
        self.current_idx = None
        self.first_start = True

    def create_net(self, variable_scope, layer_list):
        with tf.variable_scope(variable_scope):
            layer1 = tf.layers.Dense(units=10, activation=tf.nn.relu, name="layer")
            layer_list.append(layer1)
            layer1_output = layer1(self.input_state)
            layer2 = tf.layers.Dense(units=10, activation=tf.nn.relu, name="layer")
            layer_list.append(layer2)
            layer2_output = layer2(layer1_output)
            output_layer = tf.layers.Dense(units=self.actions, name="layer")
            layer_list.append(output_layer)
            output = output_layer(layer2_output)

            return output

    def choose_action(self, current_state):
        # With probability epsilon select a random weights
        if np.random.uniform() > self.epsilon:
            print("Random choice")
            action = np.random.choice(self.actions)
        else:  # Otherwise select action by dqn
            print("Greedy choice")
            # Find the index of action bounded with the greatest q value
            idx_most = tf.argmax(self.q_predicts, axis=1)

            # Find the largest q value bounded action
            [action] = self.sess.run(idx_most, feed_dict={self.input_state: current_state})

        return action

    def store_experience(self, current_state, action, reward, next_state, done):
        # index point to the oldest experience
        if self.current_idx is None:
            self.current_idx = 0

        """
        Memory is full, replace the oldest one

        The complexity of fetching a element from deque is horrible,
        while pop the front of list is also slow(everything after 
        that must move), thus we keep track the oldest value and
        replace it since, sequence is not important(random pick)
        """
        self.mem[self.current_idx] = [current_state, action, reward, next_state, done]
        if self.current_idx == (self.mem_size - 1):  # Reach the max size
            self.first_start = False
            self.current_idx = 0
        else:
            self.current_idx += 1

    def learn(self, current_state, action, reward, next_state, done):

        self.store_experience(current_state, action, reward, next_state, done)

        # Batch experience from memory to learn
        if self.first_start:
            if self.current_idx < self.batch_size:
                indices = range(self.current_idx)
            else:
                indices = np.random.choice(self.current_idx, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.mem_size, self.batch_size, replace=False)

        batch_mem = self.mem[indices, :]

        for index in indices:
            if not self.mem[index][Experience.DONE]:
               [ max_next_q_value] = self.sess.run(self.q_greatest_next,
                                                   feed_dict={self.input_state: [[self.mem[index][Experience.NEXT_STATE]]]}
                                                   )
                q_learned_value = self.mem[index][Experience.REWARD] + self.gamma * max_next_q_value
            else:
                q_learned_value = self.mem[index][Experience.REWARD]  # next state is done

            _, loss = self.sess.run([self.train_op, self.loss],
                                    feed_dict={self.input_state: self.mem[index][Experience.CURRENT_STATE],
                                               self.q_target: q_learned_value}
                                    )

            print(loss)
