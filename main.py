import pyttsx3
from PyPDF2 import PdfReader
from tkinter.filedialog import askopenfilename

# Open file dialog to select PDF
book = askopenfilename(filetypes=[("PDF files", "*.pdf")])
if not book:
    exit("No file selected.")

# Initialize PyPDF2 PdfReader
pdfreader = PdfReader(book)
pages = pdfreader.pages  # list of pages

# Initialize pyttsx3 once
player = pyttsx3.init()

# Loop through all pages
for page in pages:
    text = page.extract_text()  # updated method
    if text:  # make sure page isn't empty
        player.say(text)

# Play all speech at once
player.runAndWait()