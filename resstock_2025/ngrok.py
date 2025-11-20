import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

import socketserver
import http.server

port = 8000
handler = "https://cinderlike-enrique-magnetically.ngrok-free.dev"

with socketserver.TCPServer(("", port), handler) as httpd:
    print()

# hostname = 

# UPLOAD_FOLDER = r"/path/to/upload/folder"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# app = Flask(__name__)
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# HTML_FORM = """
# <h1>Upload File</h1>
# <form method="POST" enctype="multipart/form-data">
#   <input type="file" name="file">
#   <input type="submit" value="Upload">
# </form>
# """

# @app.route("/", methods=["GET", "POST"])
# def upload():
#     if request.method == "POST":
#         f = request.files.get("file")
#         if f:
#             save_path = os.path.join(app.config["UPLOAD_FOLDER"], f.filename)
#             f.save(save_path)
#             return f"Uploaded {f.filename} to server."
#         return "No file", 400
#     return HTML_FORM

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)
