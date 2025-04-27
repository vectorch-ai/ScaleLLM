#/usr/bin/env sh
# This script installs zsh and oh-my-zsh, and sets up the powerlevel10k theme
# and some plugins. It also modifies the ~/.zshrc file to use the powerlevel10k theme
# and adds some key bindings for zsh-autosuggestions.
# Usage: curl -fsSL https://raw.githubusercontent.com/vectorch-ai/ScaleLLM/main/tools/install_zsh.sh | sh

GREEN=$(echo -en '\033[00;32m')
RESTORE=$(echo -en '\033[0m')

echo "Installing zsh"
sudo apt install -y zsh
echo "zsh is now installed"

echo "Installing oh-my-zsh"
curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh | sh

echo "Installing powerlevel10k theme"
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
echo "Theme powerlevel10k installed"

echo "Modifying ~/.zshrc to use powerlevel10k"
sed -i 's+ZSH_THEME="robbyrussell"+ZSH_THEME="powerlevel10k/powerlevel10k"+' ~/.zshrc

echo "Continuing with plugins"
git clone https://github.com/zsh-users/zsh-autosuggestions.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
echo "Plugin zsh-autosuggestions installed"

git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
echo "Plugin zsh-syntax-highlighting installed"

git clone https://github.com/MichaelAquilina/zsh-you-should-use.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/you-should-use
echo "Plugin you-should-use installed"

echo "Add the plugins to the plugins array in ~/.zshrc"
sed -i "s+plugins=(git+plugins=(git zsh-autosuggestions zsh-syntax-highlighting you-should-use npm kubectl web-search copyfile+" ~/.zshrc

echo "Adding key bindings to ~/.zshrc"
cat <<EOT >> ~/.zshrc
# Key bindings for zsh-autosuggestions
bindkey '^I'   complete-word       # tab          | complete
bindkey '^[[Z' autosuggest-accept  # shift + tab  | autosuggest
EOT

echo "Changing default shell to zsh"
sudo chsh -s $(which zsh) $(whoami)

exec zsh
# the powerlevel10k config should launch at this point.
# To customize prompt, run `p10k configure` or edit ~/.p10k.zsh.
