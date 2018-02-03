import json
import matplotlib.pyplot as plt

# Test set rmse in function of the iterations per processes counts

# log_filename = "log/Sat Feb  3 18:16:41 2018-simulation-data.json"
# log_filename = "log/Sat Feb  3 18:59:00 2018-simulation-data.json"
log_filename = "log/Sat Feb  3 19:02:21 2018-simulation-data.json"

simulation_data = json.load(open(log_filename))

max_iter = simulation_data["logs"][0][0]["max_iter"]
eval_it = simulation_data["logs"][0][1]["eval_it"]
eval_list = [simulation_data["logs"][i][1]["test_perf"] for i in range(4)]

t = [eval_it * i for i in range(1, int(max_iter / eval_it) + 1)]

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.set_title('Test RMSE through iterations for different processes')
plt.plot(t, eval_list[0], label = '1 process')
plt.plot(t, eval_list[1], label = '2 processes')
plt.plot(t, eval_list[2], label = '3 processes')
plt.plot(t, eval_list[3], label = '4 processes')
ax.set_xlabel('Iterations')
ax.set_ylabel('RMSE')
ax.legend(loc='best')
# plt.show()
fig.savefig('log/rmse_iter_proc.png')   # save the figure to file
plt.close(fig)


# Plot test set rmse over the number of observations per processes
log_filename = "log/Sat Feb  3 19:02:21 2018-simulation-data.json"

simulation_data = json.load(open(log_filename))

max_iter = simulation_data["logs"][0][0]["max_iter"]
eval_it = simulation_data["logs"][0][1]["eval_it"]
eval_list = [simulation_data["logs"][i][1]["test_perf"] for i in range(4)]
observations_list = [simulation_data["logs"][i][1]["observations"] for i in range(4)]

observations = observations_list[0]

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.set_title('Test RMSE through iterations for different processes')
plt.plot(observations, eval_list[0], label = '1 process')
plt.plot(observations, eval_list[1], label = '2 processes')
plt.plot(observations, eval_list[3], label = '4 processes')
ax.set_xlabel('Observations')
ax.set_ylabel('RMSE')
ax.legend(loc='best')
plt.show()
fig.savefig('log/rmse_obs_proc.png')   # save the figure to file
plt.close(fig)