ARG BASE_IMAGE=vectorchai/scalellm_manylinux2_28:cuda12.6
FROM ${BASE_IMAGE}

ARG USER=vscode
ARG UID=1000
ARG GID=1000
ARG TORCH_VERSION=2.8.0
ARG CUDA_VERSION=12.8
ARG PYTHON_VERSION=3.12

# Run as non-root user
RUN groupadd --gid ${GID} ${USER} \
    && useradd --uid ${UID} --gid ${GID} -m ${USER} --shell /bin/bash
RUN echo ${USER} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USER} \
    && chmod 0440 /etc/sudoers.d/${USER}

# Install tools
RUN yum -y update && yum -y install \
    htop \
    vim \
    jq \
    gdb

# Choose python 3.12
ENV PATH="/opt/python/cp312-cp312/bin:$PATH"

# Install torch and other dependencies
RUN python -m pip install --upgrade pip
RUN pip install --upgrade pip
RUN pip install torch==$TORCH_VERSION -i "https://download.pytorch.org/whl/cu${CUDA_VERSION//./}"
RUN pip install numpy jinja2
RUN pip install --upgrade setuptools wheel

# Set up bash prompt
RUN cat <<EOT >> /home/${USER}/.bashrc
parse_git_branch() {
  git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/ (\1)/'
}
export PS1="\[\e[0;33m\]\u@\h \w\[\e[m\]\[\033[32m\]\$(parse_git_branch)\[\033[00m\]$ "
export CLICOLOR=1
export LSCOLORS=ExFxCxDxBxegedabagacad
EOT

ENTRYPOINT [ "/usr/bin/bash" ]
