# DockerSetup

This is a short version of howto install Docker on Ubuntu 20.04. It is basically a summary of the following source:\

- https://docs.docker.com/install/linux/docker-ce/ubuntu/#os-requirements

## Install Docker

```bash
sudo apt update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

curl -fsSL https://download.docker.com/linux/ubuntu/gpg |\
    sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo \
    "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] \
    https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install docker-ce
```

## Add user to "docker" group & re-login

Add yourself to the docker group (to get access to the docker deamon socket)\
note: beeing member of docker gives you root access via the docker deamon

```bash
sudo usermod -a -G docker `whoami`
```

Logout / Login to load the new group rights\
Using Ubuntu Gnome it may be required to restart

## Install Docker-Compose

```bash
sudo apt install python3-pip
sudo pip3 install docker-compose
```
