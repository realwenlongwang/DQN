import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing


class Experience:
    CURRENT_STATE = 0
    ACTION = 1
    REWARD = 2
    NEXT_STATE = 3
    DONE = 4


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, random_probability=1.0, mem_size=3000,
                 batch_size=32, update_epoch=100, episodes=10000, observation_space=500):
        self.actions = actions  # number of legal actions
        self.alpha = learning_rate
        self.gamma = reward_decay
        self.random_probability = random_probability
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.update_epoch = update_epoch
        self.epoch_counter = 0
        # self.episodes = episodes
        self.observation_space = observation_space

        self.predict_input_state = tf.placeholder(shape=(None, self.observation_space), dtype=tf.float32,
                                                  name="predict_input_state")
        self.target_input_state = tf.placeholder(shape=(None, self.observation_space), dtype=tf.float32,
                                                 name="target_input_state")
        # ================= State action input ========================
        # self.predict_input_source = tf.placeholder(shape=(None, 2), dtype=tf.float32, name="predict_input_source")
        # self.target_input_source = tf.placeholder(shape=(None, 2), dtype=tf.float32, name="target_input_source")
        # self.input_mat = np.zeros(shape=(self.actions, 2), dtype=np.float32)

        self.predict_layer_list = []
        self.q_predict = self.create_net(self.predict_input_state,
                                         "predict_net", self.predict_layer_list, True)
        # self.q_predict = self.create_net(self.predict_input_source, "predict_net", self.predict_layer_list, True)

        # with tf.variable_scope("net_predict"):
        #     self.predict_layer1 = tf.layers.Dense(units=10, activation=tf.nn.relu, name="layer")
        #     self.predict_layer_list.append(self.predict_layer1)
        #     self.predict_layer2 = tf.layers.Dense(units=10, activation=tf.nn.relu, name="layer")
        #     self.predict_layer_list.append(self.predict_layer2)
        #     self.predict_output_layer = tf.layers.Dense(units=self.actions, name="layer")
        #     self.predict_layer_list.append(self.predict_output_layer)

        self.target_layer_list = []
        self.q_target = self.create_net(self.target_input_state, "target_net", self.target_layer_list, False)

        # self.q_target = self.create_net(self.target_input_source, "target_net", self.target_layer_list, False)

        # with tf.variable_scope("net_target"):
        #     self.target_layer1 = tf.layers.Dense(units=10, activation=tf.nn.relu, name="layer")
        #     self.predict_layer_list.append(self.target_layer1)
        #     self.target_layer2 = tf.layers.Dense(units=10, activation=tf.nn.relu, name="layer")
        #     self.predict_layer_list.append(self.target_layer2)
        #     self.target_output_layer = tf.layers.Dense(units=self.actions, name="layer")
        #     self.predict_layer_list.append(self.target_output_layer)

        self.target_max_q_value = tf.reduce_max(self.q_target)
        self.predict_max_q_value = tf.reduce_max(self.q_predict)
        # Depends on the action of that experience, other action y value set to zero
        # self.y = tf.placeholder(shape=(None, 1), dtype=tf.float32, name="y")
        self.y = tf.placeholder(shape=[None, self.actions], dtype=tf.float32, name="y")

        with tf.variable_scope("loss"):
            # self.loss = tf.squared_difference(self.y, self.q_predict)
            self.loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.q_predict)
        with tf.variable_scope("train"):
            self.train_op = tf.train.GradientDescentOptimizer(self.alpha).minimize(self.loss)

        # Evaluate the learning
        self.loss_list = []
        init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init_op)

        self.copy_net()
        # A list of memory for experience
        self.mem = np.zeros((self.mem_size, 5), dtype=np.int32)  # current state, action, reward, next state, done
        self.current_idx = None
        self.first_start = True

    def create_net(self, input_state, variable_scope, layer_list, trainable):
        with tf.variable_scope(variable_scope):
            layer1 = tf.layers.Dense(units=5, activation=tf.nn.relu,
                                     name="layer", trainable=trainable)
            layer_list.append(layer1)
            layer1_output = layer1(input_state)
            layer2 = tf.layers.Dense(units=5, activation=tf.nn.relu,
                                     name="layer", trainable=trainable)
            layer_list.append(layer2)
            layer2_output = layer2(layer1_output)
            output_layer = tf.layers.Dense(units=self.actions, name="layer", trainable=trainable)
            layer_list.append(output_layer)
            output = output_layer(layer2_output)

            return output

    # def create_net(self, input_source, variable_scope, layer_list, trainable):
    #     with tf.variable_scope(variable_scope):
    #         layer1 = tf.layers.Dense(units=5, activation=tf.nn.relu,
    #                                  trainable=trainable, name="layer")
    #         layer_list.append(layer1)
    #         # Predict net takes 2 inputs, current state and action
    #         layer1_output = layer1(input_source)
    #
    #         layer2 = tf.layers.Dense(units=5, activation=tf.nn.relu,
    #                                  trainable=trainable, name="layer")
    #         layer_list.append(layer2)
    #         layer2_output = layer2(layer1_output)
    #
    #         output_layer = tf.layers.Dense(units=1, activation=None,
    #                                        trainable=trainable, name="layer")
    #         layer_list.append(output_layer)
    #         output = output_layer(layer2_output)
    #
    #         return output

    def copy_net(self):
        for predict_layer, target_layer in zip(self.predict_layer_list, self.target_layer_list):
            predict_variable_list = predict_layer.variables
            target_variable_list = target_layer.variables
            for predict_variable, target_variable in zip(predict_variable_list, target_variable_list):
                assign_op = tf.assign(ref=target_variable, value=predict_variable)
                self.sess.run(assign_op)
                # DEBUG: Print out the variable and the frequency that variable get copied
                # print("Predict_variable: \n {} \n Target_variable: \n {} \n".format(
                #     self.sess.run(predict_variable), self.sess.run(target_variable))
                # )

    def choose_action(self, current_state):
        # With probability epsilon select a random weights
        if np.random.uniform() <= self.random_probability:
            # print("Random choice")
            action = np.random.choice(self.actions)
        else:  # Otherwise select action by dqn
            # print("Greedy choice")

            # Find the index of action bounded with the greatest q value
            idx_most = tf.argmax(self.q_predict, axis=1)
            [action] = self.sess.run(idx_most, feed_dict={self.predict_input_state: self._one_hot_input(current_state)})

            # ======================
            # for i in range(self.actions):
            #     self.input_mat[i] = (current_state / self.observation_space, (i + 1) / self.actions)
            #
            # predict_q_value = self.sess.run(self.q_predict, feed_dict={self.predict_input_source: self.input_mat})
            #
            # action = np.argmax(predict_q_value)
        return action

    def _one_hot_input(self, state):
        state = int(state)
        one_hot_state = np.identity(self.observation_space)[state: state + 1]
        return one_hot_state

    def store_experience(self, current_state, action, reward, next_state, done):
        # current_state = current_state / self.observation_space
        # next_state = next_state / self.observation_space
        # action = (action + 1) / self.actions

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

    def learn(self):

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
            if self.mem[index][Experience.DONE]:

                y = np.zeros((1, self.actions))
                action = self.mem[index][Experience.ACTION]
                y[0][action] = self.mem[index][Experience.REWARD]

                _, loss = self.sess.run([self.train_op, self.loss],
                                        feed_dict={
                                            # self.predict_input_source: [[self.mem[index][Experience.CURRENT_STATE],
                                            #                              self.mem[index][Experience.ACTION]]],
                                            self.predict_input_state: self._one_hot_input(
                                                self.mem[index][Experience.CURRENT_STATE]),
                                            self.y: y
                                        }
                                        )
            else:
                # for i in range(self.actions):
                #     self.input_mat[i] = (self.mem[index][Experience.NEXT_STATE], (i + 1) / self.actions)

                target_q_values = self.sess.run(self.q_target, feed_dict={
                    # self.target_input_source: self.input_mat
                    self.target_input_state: self._one_hot_input(self.mem[index][Experience.NEXT_STATE])
                })

                y = target_q_values
                target_max_q_value = np.amax(target_q_values)
                action = np.argmax(target_max_q_value)

                y[0][action] = self.mem[index][Experience.REWARD] + self.gamma * target_max_q_value

                _, loss = self.sess.run((self.train_op, self.loss),
                                        feed_dict={
                                            # self.predict_input_source: [[self.mem[index][Experience.CURRENT_STATE],
                                            #                              self.mem[index][Experience.ACTION]]],
                                            self.predict_input_state: self._one_hot_input(
                                                self.mem[index][Experience.CURRENT_STATE]),
                                            self.y: y,
                                        }
                                        )

            self.loss_list.append(loss)

        self.epoch_counter += 1

        if self.epoch_counter % self.update_epoch == 0:
            self.copy_net()

    def plot_loss(self):
        plt.plot(np.arange(len(self.loss_list)), self.loss_list)
        plt.ylabel("loss")
        plt.xlabel("training steps")
        plt.show()

    def set_random_probability(self, random_probability):
        self.random_probability = random_probability

    def save_net(self):
        saver = tf.train.Saver()
        # Save variable to the disk
        save_path = saver.save(self.sess, "tmp/model.ckpt")
        print("Model saved in path: %s" % save_path)
