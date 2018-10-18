FROM quay.io/fenicsproject/stable:2017.2.0
MAINTAINER Henrik Finsberg <henriknf@simula.no>
USER root

RUN git clone https://github.com/finsberg/pulse.git pulse && \
    cd pulse && \
    sudo pip install -r requirements.txt && \
    sudo pip install .
