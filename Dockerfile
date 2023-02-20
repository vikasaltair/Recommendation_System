FROM python:3.9

CMD mkdir /docker_test

COPY . /docker_test

WORKDIR /docker_test

EXPOSE 8080

#RUN echo "nameserver 8.8.8.8" > /etc/resolv.conf && pip install pipenv && pipenv install --system

RUN pip install -r requirements.txt

#RUN pip3 install --upgrade protobuf
RUN pip install protobuf==3.20.*


CMD streamlit run --server.port 8080 --server.enableCORS false Recommendation_Home.py 

#CMD ["python",  "./run_final.py"]