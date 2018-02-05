import json

import matplotlib.pyplot as plt

# Test set rmse in function of the iterations per processes counts

log_filename = "log/Mon Feb  5 04:05:23 2018-simulation-data.json"

simulation_data = json.load(open(log_filename))

max_iter = simulation_data["logs"][0][0]["max_iter"]
eval_it = simulation_data["logs"][0][1]["eval_it"]
eval_list = [simulation_data["logs"][i][1]["test_perf"] for i in range(4)]

t = [eval_it * i for i in range(1, int(max_iter / eval_it) + 1)]

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.set_title('Test RMSE through iterations for different processes')
plt.plot(t, eval_list[0], label='1 process')
plt.plot(t, eval_list[1], label='2 processes')
plt.plot(t, eval_list[2], label='3 processes')
plt.plot(t, eval_list[3], label='4 processes')
ax.set_xlabel('Iterations')
ax.set_ylabel('RMSE')
ax.legend(loc='best')
# plt.show()
fig.savefig('log/rmse_iter_proc.png')
plt.close(fig)

# Plot test set rmse over the number of observations per processes
log_filename = "log/" + "Mon Feb  5 03:23:25 2018-simulation-data.json"
# log_filename = "log/" + "Mon Feb  5 03:42:46 2018-simulation-data.json"

simulation_data = json.load(open(log_filename))

max_iter = simulation_data["logs"][0][0]["max_iter"]
eval_it = simulation_data["logs"][0][1]["eval_it"]
eval_list = [simulation_data["logs"][i][1]["test_perf"] for i in range(4)]
observations_list = [simulation_data["logs"][i][1]["observations"] for i in range(4)]

observations = observations_list[0]

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.set_title('Test RMSE through iterations for different processes')
plt.plot(observations, eval_list[0], label='1 process')
plt.plot(observations, eval_list[1], label='2 processes')
plt.plot(observations, eval_list[3], label='4 processes')
ax.set_xlabel('Observations')
ax.set_ylabel('RMSE')
ax.legend(loc='best')
plt.show()
# fig.savefig('log/rmse_obs_proc.png')   # save the figure to file
# plt.close(fig)


# Plot conflicts over the number of observations per processes
# log_filename = "log/" + "Mon Feb  5 03:23:25 2018-simulation-data.json"
# log_filename = "log/" + "Mon Feb  5 03:42:46 2018-simulation-data.json"
log_filename = "log/Mon Feb  5 09:13:25 2018-simulation-data.json"

simulation_data = json.load(open(log_filename))

conflicts_list = [simulation_data["logs"][i][1]["conflicts"] for i in range(4)]
observations_list = [simulation_data["logs"][i][1]["observations"] for i in range(4)]

observations = [i * 100 for i in range(60)]

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.set_title('Conflicts through iterations for different processes')
plt.plot(observations, conflicts_list[0], label='1 process')
plt.plot(observations, conflicts_list[1], label='2 processes')
plt.plot(observations, conflicts_list[3], label='4 processes')
ax.set_xlabel('Observations')
ax.set_ylabel('Conflicts')
ax.legend(loc='best')
# plt.show()
fig.savefig('log/conflicts_obs_proc.png')
plt.close(fig)

# Plot conflicts over the number of observations per processes
# log_filename = "log/" + "Mon Feb  5 03:23:25 2018-simulation-data.json"
# log_filename = "log/" + "Mon Feb  5 03:42:46 2018-simulation-data.json"
log_filename = "log/Mon Feb  5 09:13:25 2018-simulation-data.json"

simulation_data = json.load(open(log_filename))

ratings_cts_list = [simulation_data["logs"][i][1]["ratings_counts"] for i in range(4)]
observations_list = [simulation_data["logs"][i][1]["observations"] for i in range(4)]

observations = [i * 100 for i in range(60)]

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.set_title('Ratings counts through iterations for different processes')
plt.plot(observations, ratings_cts_list[0], label='1 process')
plt.plot(observations, ratings_cts_list[1], label='2 processes')
plt.plot(observations, ratings_cts_list[3], label='4 processes')
ax.set_xlabel('Observations')
ax.set_ylabel('Ratings counts')
ax.legend(loc='best')
# plt.show()
fig.savefig('log/ratings_cts_obs_proc.png')
plt.close(fig)
