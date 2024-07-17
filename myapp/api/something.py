import pandas as pd
import mysql.connector
from mysql.connector import Error


def create_database(cursor, database_name):
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name}")
    cursor.execute(f"USE {database_name}")


def create_table(cursor, table_name, columns):
    # Clean column names to ensure they are valid SQL identifiers
    cleaned_columns = [f"`{col}` VARCHAR(255)" for col in columns]
    columns_with_types = ", ".join(cleaned_columns)
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        {columns_with_types}
    )
    """
    cursor.execute(create_table_query)


def import_csv_to_mysql(cursor, table_name, csv_file_path):
    data = pd.read_csv(csv_file_path)
    data = data.applymap(lambda x: x if pd.notnull(x) else None)
    columns = ", ".join([f"`{col}`" for col in data.columns])
    values = ", ".join(["%s"] * len(data.columns))

    insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({values})"

    for row in data.itertuples(index=False):
        cursor.execute(insert_query, tuple(row))


def main(csv_file_path, database_name, table_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host="localhost", user="root", password="root"
        )
        if connection.is_connected():
            cursor = connection.cursor()

            # Create database and table
            create_database(cursor, database_name)
            data = pd.read_csv(csv_file_path)
            create_table(cursor, table_name, data.columns)

            # Import CSV data into the table
            import_csv_to_mysql(cursor, table_name, csv_file_path)

            # Commit the transaction
            connection.commit()
            print(f"Data imported successfully into {database_name}.{table_name}")

    except Error as e:
        print(f"Error: {e}")
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()


if __name__ == "__main__":
    csv_file_path = "C:/Users/samya/agen/django/myapp/api/ONE PIECE.csv"
    database_name = "anime_db"
    table_name = "episodes"
    main(csv_file_path, database_name, table_name)
