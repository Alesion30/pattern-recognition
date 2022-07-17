FROM python:3.9

WORKDIR /usr/src/app

# install for ssh server
RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd

# change ssh setting
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
RUN sed -i 's/#Port 22/Port 20022/' /etc/ssh/sshd_config

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

# copy public key to authorized_keys
COPY key/id_rsa.pub /root/authorized_keys

# release port for ssh
EXPOSE 20022

# setting permission
CMD mkdir ~/.ssh && \
    mv ~/authorized_keys ~/.ssh/authorized_keys && \
    chmod 0600 ~/.ssh/authorized_keys &&  \
    /usr/sbin/sshd -D

# install for python
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
