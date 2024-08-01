from datetime import datetime, timedelta
from django.core.files.storage import default_storage
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import Chroma
import matplotlib.pyplot as plt
from fpdf import FPDF
import pandas as pd
import os
from dotenv import load_dotenv
import os
import google.generativeai as genai
from sqlalchemy import create_engine
import pandas as pd
from langchain_community.utilities import SQLDatabase

# import mysql.connector
from django.views.decorators.csrf import csrf_exempt

GOOGLE_API_KEY = "aIzaSyB5XLKf8Kg5uYk1EBMjjPVzk4G99MSCkpQ09"
os.environ["GOOGLE_API_KEY"] = "IzaSyB5XLKf8Kg5uYk1EBMjjPVzk4G99MSCkpQ09"
genai.configure(api_key=GOOGLE_API_KEY)


class GenAIView(APIView):
    @csrf_exempt
    def post(self, request, *args, **kwargs):
        # Assuming request.data contains the JSON payload sent from React
        query = request.data.get(
            "query", ""
        )  # Extract the 'query' parameter from request data
        if not query:
            return Response(
                {"error": "No query provided"}, status=status.HTTP_400_BAD_REQUEST
            )

        # Example usage of the AI model with the extracted query
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(query)
        reply = response.text

        return Response({"message": reply}, status=status.HTTP_200_OK)


class ChatWithPdf(APIView):
    file_path = None  # Class-level variable to store file path temporarily

    def post(self, request, *args, **kwargs):
        query = request.data.get("query", "")

        if not query:
            return Response(
                {"error": "No query provided"}, status=status.HTTP_400_BAD_REQUEST
            )

        try:
            if not self.file_path:  # If file path not set or expired
                file = request.FILES.get("file")

                if not file:
                    return Response(
                        {"error": "No file provided"},
                        status=status.HTTP_400_BAD_REQUEST,
                    )

                # Save the uploaded file temporarily
                filename = default_storage.save(file.name, file)
                self.file_path = os.path.join(settings.MEDIA_ROOT, filename)
                # Log file path for debugging
                logging.debug(f"File saved at: {self.file_path}")
                print(f"File saved at: {self.file_path}")
                # Schedule file deletion after 5 minutes
                deletion_time = datetime.now() + timedelta(minutes=5)
                deletion_task = {
                    "file_path": self.file_path,
                    "deletion_time": deletion_time,
                }
                self.schedule_file_deletion(deletion_task)
            # Process the uploaded file and respond to query
            llm = ChatGoogleGenerativeAI(model="gemini-pro")
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

            # Load the PDF and create chunks
            if self.file_path:
                print(self.file_path)
            else:
                print("none")
            loader = PyPDFLoader(self.file_path)
            print("isse")

            text_splitter = CharacterTextSplitter(
                separator=".",
                chunk_size=250,
                chunk_overlap=50,
                length_function=len,
                is_separator_regex=False,
            )
            pages = loader.load_and_split(text_splitter)
            # Turn the chunks into embeddings and store them in Chroma
            vectordb = Chroma.from_documents(pages, embeddings)

            # Configure Chroma as a retriever with top_k=5
            retriever = vectordb.as_retriever(search_kwargs={"k": 5})

            # Create the retrieval chain
            template = """
            You are a helpful AI assistant.
            Answer based on the context provided. 
            context: {context}
            input: {input}
            answer:
            """
            prompt = PromptTemplate.from_template(template)
            combine_docs_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

            # Invoke the retrieval chain
            response = retrieval_chain.invoke({"input": query})

            return Response({"message": response["answer"]}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def schedule_file_deletion(self, deletion_task):
        # This method would typically queue or schedule the deletion task
        # For demonstration, we will delete the file directly after 5 minutes
        file_path = deletion_task["file_path"]
        deletion_time = deletion_task["deletion_time"]

        def delete_file():
            if default_storage.exists(file_path):
                default_storage.delete(file_path)
                print(f"Deleted file: {file_path}")

        # Use a delayed task scheduler like Celery or Django Channels in production
        # For simplicity, we schedule the deletion using a timer
        delay_seconds = (deletion_time - datetime.now()).total_seconds()
        if delay_seconds > 0:
            import threading

            threading.Timer(delay_seconds, delete_file).start()
        else:
            delete_file()  # Execute immediately if delay is negative (shouldn't happen)

    def csv_excel_to_pdf(self, input_file):
        if input_file.endswith(".csv"):
            df = pd.read_csv(input_file)
        else:
            df = pd.read_excel(input_file)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis("tight")
        ax.axis("off")
        table = ax.table(
            cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        plt.savefig("table.png", bbox_inches="tight")

        pdf = FPDF()
        pdf.add_page()
        pdf.image("table.png", x=10, y=10, w=190)
        pdf_output = input_file.replace(os.path.splitext(input_file)[1], ".pdf")
        pdf.output(pdf_output)

        return pdf_output


from django.conf import settings
import logging

logger = logging.getLogger(__name__)

import os
import threading
from time import sleep

from django.conf import settings
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView


# class ChatWithDB(APIView):
#     def post(self, request, *args, **kwargs):
#         question = request.data.get("question", "")

#         if not question:
#             return Response(
#                 {"error": "No question provided"}, status=status.HTTP_400_BAD_REQUEST
#             )

#         try:
#             # Get the schema info
#             db_name = "anime_db"
#             schema_info = get_schema_info(db_name)

#             # Generate the prompt
#             prompt = generate_prompt(schema_info)

#             # Get the response from the Gemini model
#             response = get_gemini_response(question, prompt)

#             # Clean the response to remove any Markdown formatting
#             cleaned_response = clean_response(response)

#             # Execute the query and get the results
#             data = read_sql_query(cleaned_response, db_name)

#             # Save the DataFrame to a CSV file
#             csv_filename = "query_results.csv"
#             csv_filepath = "C:/Users/samya/agen/django/myapp/api/query_results.csv"
#             data.to_csv(csv_filepath, index=False)

#             # Construct the CSV URL
#             csv_url = request.build_absolute_uri(settings.MEDIA_URL + csv_filename)

#             # Start a thread to delete the file after 2 minutes
#             def delete_file(csv_filepath):
#                 sleep(120)  # Wait for 2 minutes
#                 try:
#                     os.remove(csv_filepath)
#                     logger.info(f"Deleted CSV file: {csv_filepath}")
#                 except Exception as e:
#                     logger.error(f"Error deleting CSV file: {e}")

#             delete_thread = threading.Thread(target=delete_file, args=(csv_filepath,))
#             delete_thread.start()

#             return Response(
#                 {
#                     "message": data.to_dict(orient="records"),
#                     "csv_url": csv_url,
#                 },
#                 status=status.HTTP_200_OK,
#             )

#         except Exception as e:
#             logger.error(f"An error occurred: {e}")
#             return Response(
#                 {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
#             )


# def get_gemini_response(question, prompt):
#     llm = genai.GenerativeModel("gemini-pro")
#     response = llm.generate_content([prompt + "\n\n" + question])
#     return response.text


# import pandas as pd
# from sqlalchemy import create_engine, text


# def read_sql_query(sql, db):
#     try:
#         # Establish database connection
#         db_uri = f"mysql+pymysql://root:root@localhost:3306/{db}"
#         engine = create_engine(db_uri)

#         # Use a connection object to execute the SQL query
#         with engine.connect() as connection:
#             result = connection.execute(text(sql))

#             # Check if the query returned rows
#             if result.returns_rows:
#                 df = pd.DataFrame(result.fetchall(), columns=result.keys())
#                 return df
#             else:
#                 # Handle queries that don't return rows (like INSERT, UPDATE)
#                 return pd.DataFrame()

#     except Exception as e:
#         # Handle any exceptions during SQL query execution
#         raise ValueError(f"Error executing SQL query: {str(e)}")


# def clean_response(response):
#     clean_sql = (
#         response.replace("sql.\n", "")
#         .replace(".", "")
#         .replace("=", "LIKE")
#         .replace("\n", " ")
#         .replace("sql", "")
#         .replace("'''", "")
#         .replace("```", "")
#         .strip()
#     )
#     clean_sql = " ".join(clean_sql.split())  # Remove extra spaces
#     return clean_sql


# def get_schema_info(db):
#     db_uri = f"mysql+pymysql://root:root@localhost:3306/{db}"
#     engine = create_engine(db_uri)
#     query = """
#     SELECT table_name, column_name
#     FROM information_schema.columns
#     WHERE table_schema = %s
#     """
#     schema_info = pd.read_sql(query, engine, params=(db,))
#     print("Schema Info DataFrame:\n", schema_info)  # Debugging statement
#     # Rename columns to lowercase for consistency
#     schema_info.columns = schema_info.columns.str.lower()
#     if (
#         "table_name" not in schema_info.columns
#         or "column_name" not in schema_info.columns
#     ):
#         raise KeyError(
#             "Expected columns 'table_name' and 'column_name' not found in the schema information."
#         )
#     return schema_info


# def generate_prompt(schema_info):
#     tables = schema_info["table_name"].unique()
#     prompt = "You are an expert in converting natural language questions to MYSQL queries. Your Query is directly used in production.\n"
#     for table in tables:
#         columns = schema_info[schema_info["table_name"] == table][
#             "column_name"
#         ].tolist()
#         prompt += f"The SQL database has the table named '{table}' which has the columns: {', '.join(columns)}.\n"
#     prompt += "Generate only SQL queries and nothing else."
#     return prompt


from langchain.schema import Document
import requests
import pymysql
import json


# class ChatWithDBCJ(APIView):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.GOOGLE_API_KEY = "aIzaSyB5XLKf8Kg5uYk1EBMjjPVzk4G99MSCkpQ0909"
#         os.environ["GOOGLE_API_KEY"] = self.GOOGLE_API_KEY
#         genai.configure(api_key=self.GOOGLE_API_KEY)
#         self.db_config = {
#             "user": "root",
#             "password": "root",
#             "host": "localhost",
#             "database": "anime_db",
#         }
#         self.embedding_file = "embeddings.json"
#         self.embeddings = self.load_embeddings()

#     def load_embeddings(self):
#         if os.path.exists(self.embedding_file):
#             with open(self.embedding_file, "r") as f:
#                 return json.load(f)
#         return {}

#     def save_embeddings(self):
#         with open(self.embedding_file, "w") as f:
#             json.dump(self.embeddings, f)

#     def connect_db(self):
#         self.conn = pymysql.connect(**self.db_config)
#         self.cursor = self.conn.cursor()

#     def close_db(self):
#         self.cursor.close()
#         self.conn.close()

#     def format_row(self, row, column_names):
#         return ", ".join(f"{col}={val}" for col, val in zip(column_names, row))

#     def get_embedding(self, content, i):
#         if content in self.embeddings:
#             return self.embeddings[content]
#         response = genai.embed_content(
#             model="models/text-embedding-004",
#             content=content,
#             task_type="retrieval_document",
#             title=f"Embedding of row: {i}",
#         )
#         embedding = response["embedding"]
#         self.embeddings[content] = embedding
#         return embedding

#     def generate_embeddings(self, rows, column_names):
#         row_embeddings = []
#         for i, row in enumerate(rows):
#             formatted_row = self.format_row(row, column_names)
#             if len(formatted_row.split()) <= 2048:
#                 embedding = self.get_embedding(formatted_row, i)
#                 row_embeddings.append((row, embedding))
#             else:
#                 print("Exceeded token size")
#         self.save_embeddings()
#         return row_embeddings

#     def convert_to_documents(self, rows, column_names):
#         text_splitter = CharacterTextSplitter(
#             separator=".",
#             chunk_size=250,
#             chunk_overlap=50,
#             length_function=len,
#             is_separator_regex=False,
#         )

#         documents = []
#         for i, row in enumerate(rows):
#             formatted_row = self.format_row(row, column_names)
#             chunks = text_splitter.split_text(formatted_row)
#             for chunk in chunks:
#                 documents.append(
#                     Document(page_content=chunk, metadata={"id": f"row_{i}"})
#                 )
#         return documents

#     def send_message(self, message="code has ran"):
#         api_url = "https://api.pushover.net/1/messages.json"
#         api_token = "avmodo3phfx3j4drgjoycsrpb9wmx6"  # Replace with your application's API token
#         user_key = "ungwjkqedrmvx664rvq5qb8ip1her7"  # Replace with your user key
#         title = "Test Notification"
#         device = "pixel6a"  # Optional: Specify the device name if you want to send to a specific device

#         data = {
#             "token": api_token,
#             "user": user_key,
#             "message": message,
#             "title": title,
#             "device": device,
#         }

#         response = requests.post(api_url, data=data)

#         if response.status_code == 200:
#             print("Notification sent successfully!")
#         else:
#             print(f"Failed to send notification. Status code: {response.status_code}")
#             print(response.text)

#     @csrf_exempt
#     def post(self, request, *args, **kwargs):
#         query = request.data.get(
#             "query", "Which season, episode was Luffy first introduced?"
#         )
#         embedding1 = True
#         # Check if the embedding for the query is already available
#         if embedding1:
#             llm = ChatGoogleGenerativeAI(model="gemini-pro")
#             embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#             with open("embeddings.json", "r") as f:
#                 embedding_data = json.load(f)

#             # Assuming embedding_data is a list of dictionaries with keys 'document' and 'embedding'
#             documents = [item["document"] for item in embedding_data]
#             embeddings = [item["embedding"] for item in embedding_data]

#             # Initialize vector database with precomputed embeddings
#             vectordb = Chroma.from_embeddings(documents, embeddings)
#             retriever = vectordb.as_retriever(search_kwargs={"k": 5})

#             # Define the prompt template
#             template = """
#             You are a helpful AI assistant.
#             Provide a natural language answer based on the context provided.
#             Context: {context}
#             Input: {input}
#             Answer:
#             """
#             prompt = PromptTemplate.from_template(template)

#             # Create the document combination and retrieval chains
#             llm = ChatGoogleGenerativeAI(model="gemini-pro")
#             combine_docs_chain = create_stuff_documents_chain(llm, prompt)
#             retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

#             # Process the query
#             query = request.data.get(
#                 "query", "Which season, episode was Luffy first introduced?"
#             )
#             response = retrieval_chain.invoke({"input": query})
#         else:
#             # Proceed with the original logic if embeddings are not found
#             self.connect_db()
#             self.cursor.execute("SELECT * FROM episodes")
#             rows = self.cursor.fetchall()
#             column_names = [desc[0] for desc in self.cursor.description]

#             row_embeddings = self.generate_embeddings(rows, column_names)
#             documents = self.convert_to_documents(rows, column_names)

#             llm = ChatGoogleGenerativeAI(model="gemini-pro")
#             embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#             vectordb = Chroma.from_documents(documents, embeddings)
#             retriever = vectordb.as_retriever(search_kwargs={"k": 5})

#             template = """
#             You are a helpful AI assistant.
#             Provide a natural language answer based on the context provided.
#             Context: {context}
#             Input: {input}
#             Answer:
#             """
#             prompt = PromptTemplate.from_template(template)
#             combine_docs_chain = create_stuff_documents_chain(llm, prompt)
#             retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

#             response = retrieval_chain.invoke({"input": query})
#             embedding = response["answer"]  # Save the new embedding
#             self.embeddings[query] = embedding
#             self.save_embeddings()  # Save updated embeddings

#             self.close_db()
#         return Response({"message": response["answer"]}, status=status.HTTP_200_OK)


import json


import json


# class ChatWithDBCJ(APIView):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.GOOGLE_API_KEY = "aIzaSyB5XLKf8Kg5uYk1EBMjjPVzk4G99MSCkpQ09"
#         os.environ["GOOGLE_API_KEY"] = self.GOOGLE_API_KEY
#         genai.configure(api_key=self.GOOGLE_API_KEY)
#         self.db_config = {
#             "user": "root",
#             "password": "root",
#             "host": "localhost",
#             "database": "ms_dataa",
#         }
#         self.embedding_file = "embeddings0.json"
#         self.embeddings = self.load_embeddings()

#     def load_embeddings(self):
#         if os.path.exists(self.embedding_file):
#             with open(self.embedding_file, "r") as f:
#                 return json.load(f)
#         return {}

#     def save_embeddings(self):
#         with open(self.embedding_file, "w") as f:
#             json.dump(self.embeddings, f)

#     def connect_db(self):
#         self.conn = pymysql.connect(**self.db_config)
#         self.cursor = self.conn.cursor()

#     def close_db(self):
#         self.cursor.close()
#         self.conn.close()

#     def format_row(self, row, column_names):
#         return ", ".join(f"{col}={val}" for col, val in zip(column_names, row))

#     def get_embedding(self, content, i):
#         if content in self.embeddings:
#             return self.embeddings[content]
#         response = genai.embed_content(
#             model="models/text-embedding-004",
#             content=content,
#             task_type="retrieval_document",
#             title=f"Embedding of row: {i}",
#         )
#         embedding = response["embedding"]
#         self.embeddings[content] = embedding
#         return embedding

#     def generate_embeddings(self, rows, column_names):
#         row_embeddings = []
#         for i, row in enumerate(rows):
#             formatted_row = self.format_row(row, column_names)
#             if len(formatted_row.split()) <= 2048:
#                 embedding = self.get_embedding(formatted_row, i)
#                 row_embeddings.append((row, embedding))
#             else:
#                 print("Exceeded token size")
#         self.save_embeddings()
#         return row_embeddings

#     def convert_to_documents(self, rows, column_names):
#         text_splitter = CharacterTextSplitter(
#             separator=".",
#             chunk_size=250,
#             chunk_overlap=50,
#             length_function=len,
#             is_separator_regex=False,
#         )

#         documents = []
#         for i, row in enumerate(rows):
#             formatted_row = self.format_row(row, column_names)
#             chunks = text_splitter.split_text(formatted_row)
#             for chunk in chunks:
#                 documents.append(
#                     Document(page_content=chunk, metadata={"id": f"row_{i}"})
#                 )
#         return documents

#     def send_message(self, message="code has ran"):
#         api_url = "https://api.pushover.net/1/messages.json"
#         api_token = "avmodo3phfx3j4drgjoycsrpb9wmx6"  # Replace with your application's API token
#         user_key = "ungwjkqedrmvx664rvq5qb8ip1her7"  # Replace with your user key
#         title = "Test Notification"
#         device = "pixel6a"  # Optional: Specify the device name if you want to send to a specific device

#         data = {
#             "token": api_token,
#             "user": user_key,
#             "message": message,
#             "title": title,
#             "device": device,
#         }

#         response = requests.post(api_url, data=data)

#         if response.status_code == 200:
#             print("Notification sent successfully!")
#         else:
#             print(f"Failed to send notification. Status code: {response.status_code}")
#             print(response.text)

#     def post(self, request, *args, **kwargs):
#         emb = True
#         if emb:
#             llm = ChatGoogleGenerativeAI(model="gemini-pro")
#             embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#             vectordb = Chroma(
#                 persist_directory="./chroma_db", embedding_function=embeddings
#             )
#             vectordb.persist()
#             retriever = vectordb.as_retriever(search_kwargs={"k": 5})

#             template = """
#             You are a helpful AI assistant.
#             Provide a natural language answer based on the context provided.
#             Context: {context}
#             Input: {input}
#             Answer:
#             """
#             prompt = PromptTemplate.from_template(template)
#             combine_docs_chain = create_stuff_documents_chain(llm, prompt)
#             retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

#             query = request.data.get(
#                 "query", "Which season, episode was Luffy first introduced?"
#             )
#             response = retrieval_chain.invoke({"input": query})
#         else:
#             self.connect_db()
#             self.cursor.execute("SELECT * FROM episodes1")
#             rows = self.cursor.fetchall()
#             column_names = [desc[0] for desc in self.cursor.description]
#             print(column_names)
#             row_embeddings = self.generate_embeddings(rows, column_names)
#             documents = self.convert_to_documents(rows, column_names)

#             llm = ChatGoogleGenerativeAI(model="gemini-pro")
#             embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#             vectordb = Chroma.from_documents(
#                 documents, embeddings, persist_directory="./chroma_db"
#             )
#             vectordb.persist()
#             retriever = vectordb.as_retriever(search_kwargs={"k": 5})

#             template = """
#             You are a helpful AI assistant.
#             Provide a natural language answer based on the context provided.
#             Context: {context}
#             Input: {input}
#             Answer:
#             """
#             prompt = PromptTemplate.from_template(template)
#             combine_docs_chain = create_stuff_documents_chain(llm, prompt)
#             retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

#             query = request.data.get(
#                 "query", "Which season, episode was Luffy first introduced?"
#             )
#             response = retrieval_chain.invoke({"input": query})

#             self.close_db()

#         return Response({"message": response["answer"]}, status=status.HTTP_200_OK)


import sys


logger = logging.getLogger(__name__)


class ChatWithDBCJ(APIView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.GOOGLE_API_KEY = "aIzaSyB5XLKf8Kg5uYk1EBMjjPVzk4G99MSCkpQ09"
        os.environ["GOOGLE_API_KEY"] = self.GOOGLE_API_KEY
        genai.configure(api_key=self.GOOGLE_API_KEY)
        self.db_config = {
            "user": "root",
            "password": "root",
            "host": "localhost",
            "database": "ms_dataa",
        }
        self.chroma_db_path = "./chroma_db"
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    def save_embeddings(self):
        with open(self.embedding_file, "w") as f:
            json.dump(self.embeddings, f)

    def connect_db(self):
        self.conn = pymysql.connect(**self.db_config)
        self.cursor = self.conn.cursor()

    def close_db(self):
        self.cursor.close()
        self.conn.close()

    def format_row(self, row, column_names):
        return ", ".join(f"{col}={val}" for col, val in zip(column_names, row))

    def get_embedding(self, content):
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=content,
            task_type="retrieval_document",
        )
        embedding = response["embedding"]
        return embedding

    def generate_database_embeddings(self, rows, column_names, max_bytes=8000):
        all_text = " ".join(self.format_row(row, column_names) for row in rows)
        words = all_text.split()

        chunked_embeddings = []
        current_chunk = []

        for word in words:
            if current_chunk:
                potential_chunk = current_chunk + [word]
            else:
                potential_chunk = [word]
            current_chunk_bytes = sys.getsizeof(
                " ".join(potential_chunk).encode("utf-8")
            )

            if current_chunk_bytes > max_bytes:
                if current_chunk:
                    embedding = self.get_embedding(" ".join(current_chunk))
                    chunked_embeddings.append(embedding)
                current_chunk = [word]
            else:
                current_chunk.append(word)

        if current_chunk:
            embedding = self.get_embedding(" ".join(current_chunk))
            chunked_embeddings.append(embedding)

        return chunked_embeddings

    def convert_to_documents(self, rows, column_names):
        documents = []
        for row in rows:
            formatted_text = self.format_row(row, column_names)
            metadata = {
                col: (val if val is not None else "")
                for col, val in zip(column_names, row)
            }
            documents.append(Document(page_content=formatted_text, metadata=metadata))
        return documents

    def post(self, request, *args, **kwargs):
        if os.path.exists(self.chroma_db_path):
            vectordb = Chroma(
                persist_directory=self.chroma_db_path,
                embedding_function=self.embeddings,
            )
            retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        else:
            self.connect_db()
            self.cursor.execute("SELECT * FROM episodes1")
            rows = self.cursor.fetchall()
            column_names = [desc[0] for desc in self.cursor.description]
            documents = self.convert_to_documents(rows, column_names)
            self.close_db()

            vectordb = Chroma.from_documents(
                documents, self.embeddings, persist_directory=self.chroma_db_path
            )
            vectordb.persist()
            retriever = vectordb.as_retriever(search_kwargs={"k": 5})

        template = """
        You are a helpful AI assistant.
        Provide a natural language answer based on the context provided.
        Context: {context}
        Input: {input}
        Answer:
        """
        prompt = PromptTemplate.from_template(template)
        combine_docs_chain = create_stuff_documents_chain(
            ChatGoogleGenerativeAI(model="gemini-pro"), prompt
        )
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

        query = request.data.get(
            "query", "Which season, episode was Luffy first introduced?"
        )
        response = retrieval_chain.invoke({"input": query})

        return Response({"message": response["answer"]}, status=status.HTTP_200_OK)
