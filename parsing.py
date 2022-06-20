import pdfplumber

def extractPDFText(route):
    """ Function defined to extract the info contained inside the PDF
    of interest.

    input:
        -route: (String) route where the file is located

    output:
        -output: (String) string compiling all the info extracted
                from the file"""

    try:
        output = ""
        with pdfplumber.open(route) as pdf:
            for page in pdf.pages:
                output += page.extract_text()
                
    except FileNotFoundError:
        print("The file was not found!")

    return output
    

