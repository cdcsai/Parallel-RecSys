import json

from time import time
from time import strftime


def write_to_json(logs):
    """ Writes the simulation data to a json file in the logs directory"""
    simulation_dict = {"logs": logs}

    now = strftime("%c")
    log_dir = "log/" + now.format(time())

    json_file = open(log_dir + '-simulation-data.json', 'w+')
    json.dump(simulation_dict, json_file, sort_keys=True, indent=4)
    json_file.close()