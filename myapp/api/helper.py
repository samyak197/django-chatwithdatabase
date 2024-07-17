from fpdf import FPDF
import pandas as pd


def csv_excel_to_pdf(input_file, output_file):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    if input_file.endswith(".csv"):
        # Read CSV file
        df = pd.read_csv(input_file)
    elif input_file.endswith(".xlsx") or input_file.endswith(".xls"):
        # Read Excel file
        df = pd.read_excel(input_file)
    else:
        raise ValueError(
            "Unsupported file format. Only CSV and Excel files are supported."
        )

    # Convert DataFrame to string and add to PDF
    pdf.set_left_margin(10)
    pdf.set_right_margin(10)
    pdf.multi_cell(0, 10, df.to_string(index=False))

    # Save PDF to output file
    pdf.output(output_file)


# Example usage:
csv_excel_to_pdf("C:/Users/samya/agen/django/myapp/api/data.csv", "output.pdf")
# csv_excel_to_pdf("data.xlsx", "output.pdf")
