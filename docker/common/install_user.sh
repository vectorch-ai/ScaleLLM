#!/bin/bash

set -ex

# mirror jenkins user in container
echo "jenkins:x:1000:1000::/var/lib/jenkins:" >> /etc/passwd
echo "jenkins:x:1000:" >> /etc/group
# needed on focal or newer
echo "jenkins:*:19110:0:99999:7:::" >>/etc/shadow

# allow sudo
echo 'jenkins ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/jenkins