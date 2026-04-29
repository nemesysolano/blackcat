import psycopg2
import os

def quote_names(connection):
    with connection.cursor() as cursor:
        cursor.execute('SELECT ticker, COUNT(*) FROM QUOTE GROUP BY ticker')
        records = cursor.fetchall()
    return [record[0] for record in records]

def create_control_table(connection_string):
    with psycopg2.connect(connection_string) as connection:
        with connection.cursor() as cursor:
            # Check if table exists
            cursor.execute("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'flyway_migration'
            """)
            if cursor.fetchone()[0] == 0:
                print("Creating control table...")
                cursor.execute("""
                    CREATE TABLE flyway_migration (
                        version INT NOT NULL PRIMARY KEY,
                        file_name VARCHAR(255) NOT NULL,
                        executed_at TIMESTAMP NOT NULL DEFAULT NOW(),
                        CONSTRAINT unique_file_name UNIQUE (file_name)
                    )
                """)
                connection.commit()

def file_exists(connection, file_name):
    with connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM flyway_migration WHERE file_name = %s", (file_name,))
        return cursor.fetchone()[0] > 0

def update_structure(connection_string):
    module_path = os.path.dirname(os.path.abspath(__file__))
    database_scripts_path = os.path.join(module_path, "database")
    
    # Ensure control table exists before starting
    create_control_table(connection_string)    
    
    # Gather and sort scripts
    if not os.path.exists(database_scripts_path):
        print(f"Error: Directory {database_scripts_path} not found.")
        return

    scripts = [f for f in os.listdir(database_scripts_path) if f.endswith('.sql')]
    scripts.sort() 
    
    with psycopg2.connect(connection_string) as connection:
        for script_name in scripts:
            if file_exists(connection, script_name):
                print(f"Skipping {script_name} (already executed)")
                continue

            script_path = os.path.join(database_scripts_path, script_name)
            
            try:
                with open(script_path, 'r') as f:
                    sql_script = f.read()
                
                with connection.cursor() as cursor:
                    # Execute migration script
                    cursor.execute(sql_script)
                    
                    # Parse version (handles 'V1_...' or '1_...')
                    version_part = script_name.split("_")[0]
                    version = int(''.join(filter(str.isdigit, version_part)))
                    
                    # Log migration
                    cursor.execute(
                        "INSERT INTO flyway_migration (version, file_name) VALUES (%s, %s)",
                        (version, script_name)
                    )
                
                # CRITICAL: Commit each script individually
                connection.commit()
                print(f"Successfully executed {script_name}")
                
            except Exception as e:
                connection.rollback()
                print(f"Error executing {script_name}: {e}")
                break # Stop migration on failure