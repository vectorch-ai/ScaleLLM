ARG BASE_IMAGE=vectorchai/scalellm_devel:cuda12.4
FROM ${BASE_IMAGE}

ARG USER=vscode
# Run as non-root user
RUN useradd -m ${USER}
RUN echo ${USER} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USER} \
    && chmod 0440 /etc/sudoers.d/${USER}

ENTRYPOINT [ "/usr/bin/bash" ]