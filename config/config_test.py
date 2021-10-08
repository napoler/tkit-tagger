import yaml

with open("config_demo.yaml", 'r') as stream:
    try:
#         print(yaml.safe_load(stream))
        conf=yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
    print(conf)