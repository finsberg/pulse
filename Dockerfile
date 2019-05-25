FROM finsberg/fenics2017
MAINTAINER Henrik Finsberg <henriknf@simula.no>

RUN sudo pip3 install git+https://github.com/finsberg/pulse.git
RUN sudo pip3 install git+https://github.com/finsberg/ldrb.git

