import yaml

def read_yaml(file_name):
    yaml_file = open("config/" + file_name)
    data = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return data

def write_yaml(data: dict):
    data = wrapper_config(data)
    path = data['path']
    file = open(path + "/" + "config.yml", "w")
    yaml.dump(data, file)
    file.close()

def wrapper_config(data: dict):
    wrapper = str(data['env_params']['wrapper'])
    data["env_params"]["wrapper"] = wrapper
    return data

