import os
import torch

from datetime import datetime
from random import randint
from google.cloud import datastore

from forms import UploadForm
from flask import Flask, request, render_template, url_for, redirect, flash

app = Flask(__name__)

app.config.update(
    {
        "SECRET_KEY": os.environ.get("SECRET_KEY"),
        "WTF_CSRF_SECRET_KEY": os.environ.get("WTF_CSRF_SECRET_KEY"),
        "BASIC_AUTH_USERNAME": os.environ.get("BASIC_AUTH_USERNAME"),
        "BASIC_AUTH_PASSWORD": os.environ.get("BASIC_AUTH_PASSWORD")
    }
)

model = torch.load("trained_trouble.pt", map_location="cpu")

client = datastore.Client("artificial-trouble")

@app.route("/")
def index():
    form = UploadForm()

    return render_template(
        "index.html", form=form
    )


@app.route("/predict", methods=["POST"])
def predict():
    form = UploadForm(request.form)
    if form.validate():
        return render_template("prediction.html", form=form)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
