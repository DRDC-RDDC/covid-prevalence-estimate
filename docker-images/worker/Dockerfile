FROM python:3.6.11-slim-buster
# 3.6.11-alpine3.12 (fails)
# Python 3.6 is used since google colab uses it.  3.8 would be better.

# [optimization] g++ is needed for compiling optimized model
# [optimization] libopenblas provides faster calculations in theano (pymc3)
# kdiff3 is used for git mergetool automation
RUN apt-get update \
    && apt-get install -y apt-utils \
    && apt-get install -y nfs-common \
    && apt-get install -y --no-install-recommends g++ \
    && apt-get install -y --no-install-recommends libopenblas-dev \
    && apt-get install -y --no-install-recommends libblas-dev \
    && apt-get install -y --no-install-recommends liblapack-dev \
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

COPY worker.py .

# Run the main program
CMD [ "python", "./worker.py" ]