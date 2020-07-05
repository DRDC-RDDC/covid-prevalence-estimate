FROM python:3.6.10-buster

COPY requirements.txt /tmp/

RUN pip install -r /tmp/requirements.txt

RUN useradd --create-home simuser

# This is where the simulation data will be put
RUN mkdir -p /content
RUN chown simuser /content

WORKDIR /home/simuser
USER simuser

COPY main.py .

CMD [ "python", "./main.py" ]