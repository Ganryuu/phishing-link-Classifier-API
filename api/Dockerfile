FROM python:3.9.12 

COPY requirements.txt . 

ADD app.py handle.py phishing_classifier.pkl /root/

RUN pip install -r requirements.txt 

WORKDIR /root

CMD ["python", "app.py"]