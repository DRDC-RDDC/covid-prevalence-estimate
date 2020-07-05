FROM python:3.6.11-slim-buster
# not 3.6.10-buster  3.6.11-alpine3.12 (fails)

# g++ is needed for compiling optimized model
RUN apt-get update \
    && apt-get install -y g++ \
    && apt-get install -y --no-install-recommends gnutls-bin \
    && apt-get install -y --no-install-recommends git \
    && apt-get purge -y --auto-remove \
    && rm -rf /var/lib/apt/lists/*

# This is needed to fix a git error:  RPC failed; curl 56 GnuTLS recv error (-54): Error in the pull function.
RUN git config --global http.postBuffer 1048576000

COPY requirements.txt /tmp/

RUN pip install -r /tmp/requirements.txt

RUN useradd --create-home simuser

# This is where the simulation data will be placed
RUN mkdir -p /content
RUN chown simuser /content

WORKDIR /home/simuser
USER simuser

COPY controller.py .
COPY rediswq.py .

CMD [ "python", "./controller.py" ]