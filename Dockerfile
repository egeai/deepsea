FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . /app
EXPOSE 8501

LABEL authors="User"

ENTRYPOINT ["streamlit", "run", "src/deepsea/train_prep_front.py", "--server.port=8501", "--server.address=0.0.0.0"]
# ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]