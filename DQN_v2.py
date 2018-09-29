import gym
import tensorflow as tf
import matplotlib.pyplot as plt

from pylab import rcParams

rcParams['figure.figsize'] = 15, 10

import numpy as np
import timeit

env = gym.make("Taxi-v2")


# In[8]:


class Experience:
    CURRENT_STATE = 0
    ACTION = 1
    REWARD = 2
    NEXT_STATE = 3
    DONE = 4


# In[9]:


# def create_net(input_state, variable_scope, layer_list, trainable, action_num):
#     with tf.variable_scope(variable_scope):
#         w_initialiser = tf.random_normal_initializer(mean=0., stddev=0.01)
# #         layer1 = tf.layers.Dense(units=500, activation=tf.nn.relu,
# #                                  name="layer", trainable=trainable,
# #                                 kernel_initializer=w_initialiser)
# #         layer_list.append(layer1)
# #         layer1_output = layer1(input_state)
# #         layer2 = tf.layers.Dense(units=5, activation=tf.nn.relu,
# #                                  name="layer", trainable=trainable)
# #         layer_list.append(layer2)
# #         layer2_output = layer2(layer1_output)
#         output_layer = tf.layers.Dense(units=action_num, name="layer", 
#                                        trainable=trainable, kernel_initializer=w_initialiser,
#                                        activation=None
#                                       )
#         layer_list.append(output_layer)
#         output = output_layer(input_state)

#         return output

def create_net(input_state, variable_scope, variable_list, trainable):
    with tf.variable_scope(variable_scope):
        # =========== layer 1 ===========
        weights_1 = tf.get_variable(name="w1",
                                    #                                     initializer=tf.random_uniform([env.observation_space.n, env.action_space.n],0,0.01),
                                    initializer=tf.random_normal(shape=[env.observation_space.n, 6],
                                                                 stddev=(1 / env.observation_space.n)),
                                    # To fight vanish gradient
                                    trainable=trainable
                                    )
        variable_list.append(weights_1)

        #         ========== layer 2 =============
        #         weights_2 = tf.get_variable(name="w2",
        # #                                     initializer=tf.random_uniform([6, env.action_space.n],0,0.01),
        #                                     initializer=tf.random_normal(shape=[6, env.action_space.n], stddev= (1/ (env.observation_space.n ** 2))),
        #                                     trainable=trainable
        #                                    )
        #         variable_list.append(weights_2)

        #         bias = tf.get_variable(name="b1",
        #                                shape=[6],
        #                                initializer=tf.constant_initializer(0.01),
        #                                trainable=trainable
        #                               )
        #         variable_list.append(bias)

        #         layer1_output = tf.nn.elu(tf.nn.bias_add(tf.matmul(input_state, weights_1), bias)) # relu
        #         output = tf.matmul(layer1_output, weights_2)

        output = tf.matmul(input_state, weights_1)

        return output


# In[10]:


def copy_net(dest_list, src_list, sess):
    for predict_variable, target_variable in zip(src_list, dest_list):
        assign_op = tf.assign(ref=target_variable, value=predict_variable)
        sess.run(assign_op)


# ## Create the neural network
# ---

# In[29]:


tf.reset_default_graph()

# In[30]:


predict_input_state_tf = tf.placeholder(shape=(None, env.observation_space.n), dtype=tf.float32,
                                        name="predict_input_state")
target_input_state_tf = tf.placeholder(shape=(None, env.observation_space.n), dtype=tf.float32,
                                       name="target_input_state")

# In[31]:


predict_variable_list = []
predict_q_values_tf = create_net(predict_input_state_tf, variable_scope="predict_net",
                                 variable_list=predict_variable_list, trainable=True)

predict_action_tf = tf.argmax(predict_q_values_tf, axis=1)

# In[32]:


target_variable_list = []
target_q_values_tf = create_net(target_input_state_tf, variable_scope="target_net",
                                variable_list=target_variable_list, trainable=False)

# In[33]:


y_tf = tf.placeholder(shape=(None, env.action_space.n), dtype=tf.float32)

with tf.variable_scope("loss"):
    #     loss_tf = tf.losses.mean_squared_error(labels=y_tf, predictions=predict_q_values_tf)
    loss_tf = tf.losses.huber_loss(labels=y_tf,
                                   predictions=predict_q_values_tf)  # Clip the error according to the journal

#     loss_tf = tf.reduce_sum(tf.squared_difference(y_tf, predict_q_values_tf))
with tf.variable_scope("train"):
    global_step = tf.Variable(initial_value=0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        learning_rate=0.8,  # Starting learning rate
        global_step=global_step,
        decay_steps=5000,
        decay_rate=0.95,
    )

    momentum = tf.train.exponential_decay(
        learning_rate=0.01,  # Starting momentum
        global_step=global_step,
        decay_steps=5000,
        decay_rate=1.95,
    )

    #     my_optimiser = tf.train.GradientDescentOptimizer(2.2) # Use mean square error
    my_optimiser = tf.train.RMSPropOptimizer(learning_rate=0.4)  # momentum=0.95, epsilon=0.01)
    #     my_optimiser = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum=0.9, use_nesterov=True) # huber_loss
    train_op = my_optimiser.minimize(loss_tf, global_step=global_step)
    #     for variables in predict_variable_list:
    #         gradients_op_list.append(my_optimiser.compute_gradients(loss_tf, var_list=variables))
    grads_and_vars_tf = my_optimiser.compute_gradients(loss_tf, var_list=predict_variable_list)
    accum_vars = [tf.Variable(tv.initialized_value(), trainable=False) for tv in predict_variable_list]
    # Clean up accumulate gradients
    zero_ops = [var.assign(tf.zeros_like(var)) for var in accum_vars]
    accum_ops = [var.assign_add(pair[0]) for var, pair in zip(accum_vars, grads_and_vars_tf)]
    apply_ops = my_optimiser.apply_gradients(
        [(accum_grads, tv) for accum_grads, tv in zip(accum_vars, predict_variable_list)])

# In[34]:


init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

# ## Set up hyperparameters

# In[35]:


max_random_probability = 1.0
min_random_probability = 0
mem_size = 3000
update_epoches = 100
batch_size = 32
gamma = 0.99
one_hot_state = np.identity(env.observation_space.n)

# ## Experience replay
# Create the memory

# In[36]:


mem = np.zeros((mem_size, 5), dtype=np.int32)


# In[37]:


def store_experience(first_start, current_idx, current_state, action, reward, next_state, done):
    # index point to the oldest experience
    if first_start:
        current_idx = 0

    """
    Memory is full, replace the oldest one

    The complexity of fetching a element from deque is horrible,
    while pop the front of list is also slow(everything after 
    that must move), thus we keep track the oldest value and
    replace it since, sequence is not important(random pick)
    """
    mem[current_idx] = [current_state, action, reward, next_state, done]

    if current_idx == (mem_size - 1):  # Reach the max size
        current_idx = 0
    else:
        current_idx += 1

    return current_idx


# ## Start training the network

# In[38]:


sess.run(loss_tf, feed_dict={
    y_tf: [[1, 2, 3, 4, 5, 6]],
    predict_input_state_tf: np.identity(500)[0:1]
})

# In[39]:


loss_list = []
reward_list = []
step_list = []

gradient_list = []
mean_variable_list = []
std_variable_list = []
min_variable_list = []
max_variable_list = []
low_quater_variable_list = []
upper_quater_variable_list = []
# Always point to the oldest experience in mem
learning_rate_list = []
momentum_list = []

training_time_list = []

# Pointer towards the current position of memory
current_idx = 0
first_start = True

episode_num = 1000
random_probability_distribution = np.linspace(max_random_probability, min_random_probability, episode_num * 0.9)

start = timeit.default_timer()
for i in range(episode_num):
    done = False
    episode_reward = 0
    current_state = env.reset()
    step_counter = 0

    random_probability = random_probability_distribution[i] if i < len(random_probability_distribution) else 0

    while step_counter < 100:  # If agent cannot find the solution in 50 steps reset the environment
        [action], predict_q_values = sess.run([predict_action_tf, predict_q_values_tf],
                                              feed_dict={
                                                  predict_input_state_tf: one_hot_state[
                                                                          current_state: current_state + 1]
                                              })

        if np.random.uniform() <= random_probability:
            # Randomly explore the world
            action = env.action_space.sample()

        # Execute the action and observe the reward
        next_state, reward, done, info = env.step(action)

        # ================= Store the experience to memory ================
        # index point to the oldest experience
        """
        Memory is full, replace the oldest one

        The complexity of fetching a element from deque is horrible,
        while pop the front of list is also slow(everything after 
        that must move), thus we keep track the oldest value and
        replace it since, sequence is not important(random pick)
        """
        mem[current_idx] = [current_state, action, reward, next_state, done]

        if current_idx == (mem_size - 1):  # Reach the max size
            current_idx = 0
        else:
            current_idx += 1

        if current_idx > batch_size:
            first_start = False

        # Randomly select batch size experiences from memory
        if first_start:
            indices = range(current_idx)
        else:
            indices = np.random.choice(mem_size, batch_size, replace=False)

        # Get the batch size experiences out of memory
        batch_mem = mem[indices, :]
        next_state_indices = batch_mem[:, Experience.NEXT_STATE]
        current_state_indices = batch_mem[:, Experience.CURRENT_STATE]

        # If we don't use the target network
        next_q_values = sess.run(predict_q_values_tf, feed_dict={
            predict_input_state_tf: one_hot_state[next_state_indices]
        })

        # Pick the max q value for next state
        max_next_q_value = np.max(next_q_values, axis=1)
        # Take the other actions q values
        y = predict_q_values
        # Update the right we performed only
        # y[np.arange(len(y)), batch_mem[:, Experience.ACTION]] = reward + gamma * max_next_q_value
        y[np.arange(len(y)), batch_mem[:, Experience.ACTION]] = batch_mem[:, Experience.REWARD] + (
                    1 - batch_mem[:, Experience.DONE]) * gamma * max_next_q_value

        training_start = timeit.default_timer()
        # Learn
        _, grads_and_vars, loss = sess.run([train_op, grads_and_vars_tf, loss_tf], feed_dict={
            predict_input_state_tf: one_hot_state[current_state_indices],
            y_tf: y
        })
        training_end = timeit.default_timer()

        training_time_list.append(training_end - training_start)

        learning_rate_list.append(sess.run(learning_rate))
        momentum_list.append(sess.run(momentum))

        loss_list.append(loss)
        for pair in grads_and_vars:
            gradient_list.append(np.mean(pair[0]))
            mean_variable_list.append(np.mean(pair[1]))
            low_quater_variable_list.append(np.percentile(pair[1], 25, interpolation='midpoint'))
            upper_quater_variable_list.append(np.percentile(pair[1], 75, interpolation='midpoint'))
            max_variable_list.append(np.max(pair[1]))
            min_variable_list.append(np.min(pair[1]))
        #             std_variable_list.append(np.std(pair[1]))

        current_state = next_state

        episode_reward += reward
        step_counter += 1

        if i >= (episode_num - 1):
            env.render()
            print(loss)

        if done:
            break

    reward_list.append(episode_reward)
    step_list.append(step_counter)

    if i and i % 200 == 0:
        print("Episode: {0},  step: {1}, random: {2}".format(i, step_counter, random_probability))

end = timeit.default_timer()
print("Time usage: ", end - start, " seconds")
print("Training time average usage: ", np.sum(training_time_list), " seconds")

# In[40]:


mem

# ## Evaluate the performance

# In[41]:


print(np.mean(reward_list))
plt.plot(reward_list)

# In[42]:


print(np.mean(step_list))
plt.plot(step_list)

# In[43]:


plt.plot(loss_list)

# In[26]:


plt.plot(learning_rate_list)

# In[27]:


plt.plot(momentum_list)

# In[28]:


plt.plot(gradient_list)

# In[ ]:


plt.plot(mean_variable_list)

# In[ ]:


plt.plot(min_variable_list)

# In[ ]:


plt.plot(max_variable_list)

# In[ ]:


plt.plot(low_quater_variable_list)

# In[ ]:


plt.plot(upper_quater_variable_list)

# In[ ]:


# ## Evaluate with the real q table

# In[ ]:


import pickle

# In[ ]:


q_table = pickle.load(open("obj/q_table.pkl", "rb"))

# Using the `DQN` predicts the whole table

# In[ ]:


predict_q_table = sess.run(predict_q_values_tf, feed_dict={predict_input_state_tf: one_hot_state})

# In[ ]:


np.mean(np.absolute(predict_q_table - q_table))

# In[ ]:
