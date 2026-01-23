import os
import json
def db_config():
    current_script_dir = os.getcwd()
    database_config_file_path = os.path.join(current_script_dir,"bin", "database-config.json")
    database_config = json.load(open(database_config_file_path))
    connection_string = f"host='{database_config['host']}' port='{database_config['port']}' dbname='{database_config['dbname']}' user='{database_config['user']}' password='{database_config['password']}'"
    sqlalchemy_url = database_config["sqlalchemy.url"]

    return connection_string, sqlalchemy_url