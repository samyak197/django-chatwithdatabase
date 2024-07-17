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

GOOGLE_API_KEY = "AIzaSyB5XLKf8Kg5uYk1EBMjjPVzk4G99MSCkpQ"
os.environ["GOOGLE_API_KEY"] = "AIzaSyB5XLKf8Kg5uYk1EBMjjPVzk4G99MSCkpQ"
genai.configure(api_key=GOOGLE_API_KEY)


class GenAIView(APIView):
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
                self.file_path = default_storage.save(file.name, file)
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
            loader = PyPDFLoader(self.file_path)
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


class ChatWithDB(APIView):
    def post(self, request, *args, **kwargs):
        question = request.data.get("question", "")

        if not question:
            return Response(
                {"error": "No question provided"}, status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Get the schema info
            db_name = "anime_db"
            schema_info = get_schema_info(db_name)

            # Generate the prompt
            prompt = generate_prompt(schema_info)

            # Get the response from the Gemini model
            response = get_gemini_response(question, prompt)

            # Clean the response to remove any Markdown formatting
            cleaned_response = clean_response(response)

            # Execute the query and get the results
            data = read_sql_query(cleaned_response, db_name)

            # Save the DataFrame to a CSV file
            csv_filename = "query_results.csv"
            csv_filepath = "C:/Users/samya/agen/django/myapp/api/query_results.csv"
            data.to_csv(csv_filepath, index=False)

            # Construct the CSV URL
            csv_url = request.build_absolute_uri(settings.MEDIA_URL + csv_filename)

            # Start a thread to delete the file after 2 minutes
            def delete_file(csv_filepath):
                sleep(120)  # Wait for 2 minutes
                try:
                    os.remove(csv_filepath)
                    logger.info(f"Deleted CSV file: {csv_filepath}")
                except Exception as e:
                    logger.error(f"Error deleting CSV file: {e}")

            delete_thread = threading.Thread(target=delete_file, args=(csv_filepath,))
            delete_thread.start()

            return Response(
                {
                    "message": data.to_dict(orient="records"),
                    "csv_url": csv_url,
                },
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


def get_gemini_response(question, prompt):
    llm = genai.GenerativeModel("gemini-pro")
    response = llm.generate_content([prompt + "\n\n" + question])
    return response.text


import pandas as pd
from sqlalchemy import create_engine, text


def read_sql_query(sql, db):
    try:
        # Establish database connection
        db_uri = f"mysql+pymysql://root:root@localhost:3306/{db}"
        engine = create_engine(db_uri)

        # Use a connection object to execute the SQL query
        with engine.connect() as connection:
            result = connection.execute(text(sql))

            # Check if the query returned rows
            if result.returns_rows:
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
            else:
                # Handle queries that don't return rows (like INSERT, UPDATE)
                return pd.DataFrame()

    except Exception as e:
        # Handle any exceptions during SQL query execution
        raise ValueError(f"Error executing SQL query: {str(e)}")


def clean_response(response):
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
    return clean_sql


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
