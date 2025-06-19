ARG BASE_IMAGE=vectorchai/scalellm_devel:cuda12.9
FROM ${BASE_IMAGE}

ARG USER=vscode
ARG UID=1000
ARG GID=1000

# Run as non-root user
RUN groupadd --gid ${GID} ${USER} \
    && useradd --uid ${UID} --gid ${GID} -m ${USER} --shell /bin/bash
RUN echo ${USER} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USER} \
    && chmod 0440 /etc/sudoers.d/${USER}

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
