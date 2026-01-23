import psycopg2
import os
from datetime import datetime

def quote_names(connection):
    cursor = connection.cursor()
    cursor.execute("SELECT \"TICKER\", COUNT(*) FROM QUOTE GROUP BY \"TICKER\"")
    records = cursor.fetchall()
    cursor.close()
    return [record[0] for record in records]


def create_control_table(connection_string):
    connection = psycopg2.connect(connection_string)
    cursor = connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'flyway_migration'")
    records = cursor.fetchall()
    count = records[0][0]
    cursor.close()


    if count == 0:
        cursor = connection.cursor()
        cursor.execute("CREATE TABLE flyway_migration (version INT NOT NULL PRIMARY KEY,  file_name VARCHAR(255) NOT NULL, executed_at TIMESTAMP NOT NULL DEFAULT NOW(), CONSTRAINT unique_file_name UNIQUE (file_name ))")
        connection.commit()
        cursor.close()
    
    connection.close()

def file_exists(connection, file_name):
    cursor = connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM flyway_migration WHERE file_name = %s", (file_name,))
    records = cursor.fetchall()
    count = records[0][0]
    cursor.close()
    return count

def update_structure(connection_string):
    module_path = os.path.dirname(os.path.abspath(__file__))
    database_scripts_path = os.path.join(module_path, "database")
    create_control_table(connection_string)    
    database_scripts = [os.path.join(database_scripts_path, script) for script in os.listdir(database_scripts_path)]
    database_scripts = list(sorted(database_scripts, key=lambda x: os.path.basename(x)))
    
    with psycopg2.connect(connection_string) as connection:
        for script in database_scripts:        
                file_name = os.path.basename(script)
                if file_exists(connection, file_name):
                    print(f"Skipping {file_name} as it has already been executed")
                    continue

                with open(script, 'r') as file, connection.cursor() as script_cursor, connection.cursor() as flyway_cursor:
                    sql_script = file.read()            
                    script_cursor.execute(sql_script)
                    
                    version = int(file_name.split("_")[0][1:])
                    flyway_cursor.execute("INSERT INTO flyway_migration (version, file_name) VALUES (%s, %s)", (int(version), file_name))
                    
                    print(f"Executed {file_name}")
                
                

    