from dotenv import load_dotenv
import os
import google.generativeai as genai
from sqlalchemy import create_engine
import pandas as pd
from langchain_community.utilities import SQLDatabase

# Load environment variables from .env file
load_dotenv()

# Set your Google API key
GOOGLE_API_KEY = "AIzaSyB5XLKf8Kg5uYk1EBMjjPVzk4G99MSCkpQ"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)


def get_gemini_response(question, prompt):
    llm = genai.GenerativeModel("gemini-pro")
    response = llm.generate_content([prompt + "\n\n" + question])
    return response.text


def read_sql_query(sql, db):
    db_uri = f"mysql+pymysql://root:root@localhost:3306/{db}"
    db = SQLDatabase.from_uri(db_uri)
    df = db.run(sql)
    return df


def clean_response(response):
    # Remove Markdown formatting and check for valid SQL
    # clean_sql = response.replace("sql\n", "").replace("\n", " ").strip()
    clean_sql = (
        response.replace("sql.\n", "")
        .replace(".", "")
        .replace("=", "LIKE")
        .replace("\n", " ")
        .replace("sql", "")
        .replace("'''", "")
        .replace("```", "")
        .strip()
    )
    clean_sql = " ".join(clean_sql.split())  # Remove extra spaces

    # print(clean_sql)
    # Check if the response looks like a valid SQL query
    # if not clean_sql.lower().startswith("select"):
    # raise ValueError("Invalid SQL query generated.")
    # clean_sql = "SELECT name FROM episodes WHERE name = 'Luffy'"
    return clean_sql


# prompt = """
# You are an expert in converting natural language questions to MYSQL queries.Your Query is directly used in production.
# The SQL database has the table named 'episodes' which has the columns: rank, trend, season, episode, name, start, total_votes, average_rating.
# Generate only SQL queries and nothing else.
# """

# question = "when was Luffy first intoduced?"

# # Get the response from the Gemini model
# response = get_gemini_response(question, prompt)
# # print("Generated SQL Query:", response)
# # Clean the response to remove any Markdown formatting
# cleaned_response = clean_response(response)
# # print("Cleaned SQL Query:", cleaned_response)
# data = read_sql_query(cleaned_response, "anime_db")
# print(data)


def get_schema_info(db):
    db_uri = f"mysql+pymysql://root:root@localhost:3306/{db}"
    engine = create_engine(db_uri)
    query = """
    SELECT table_name, column_name
    FROM information_schema.columns
    WHERE table_schema = %s
    """
    schema_info = pd.read_sql(query, engine, params=(db,))
    print("Schema Info DataFrame:\n", schema_info)  # Debugging statement
    # Rename columns to lowercase for consistency
    schema_info.columns = schema_info.columns.str.lower()
    if (
        "table_name" not in schema_info.columns
        or "column_name" not in schema_info.columns
    ):
        raise KeyError(
            "Expected columns 'table_name' and 'column_name' not found in the schema information."
        )
    return schema_info


def generate_prompt(schema_info):
    tables = schema_info["table_name"].unique()
    prompt = "You are an expert in converting natural language questions to MYSQL queries. Your Query is directly used in production.\n"
    for table in tables:
        columns = schema_info[schema_info["table_name"] == table][
            "column_name"
        ].tolist()
        prompt += f"The SQL database has the table named '{table}' which has the columns: {', '.join(columns)}.\n"
    prompt += "Generate only SQL queries and nothing else."
    return prompt


db_name = "anime_db"
schema_info = get_schema_info(db_name)
prompt = generate_prompt(schema_info)
question = "which epicose was luffy first introduced ?"

# Get the response from the Gemini model
response = get_gemini_response(question, prompt)

# Clean the response to remove any Markdown formatting
cleaned_response = clean_response(response)
# print(cleaned_response)
# Execute the query and get the results
# cleaned_response='''SELECT name FROM episodes WHERE name LIKE "%%Luffy%%";'''
data = read_sql_query(cleaned_response, db_name)
print(data)
