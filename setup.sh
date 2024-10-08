#!/bin/bash

setup_hf() {
    echo "Please enter your Hugging Face token (press Enter to skip):"
    read -r token
    if [ -n "$token" ]; then
        echo "Storing HF_TOKEN in .env file..."
        echo "HF_TOKEN=$token" >> .env
        
        echo "Installing Hugging Face CLI..."
        yes | pip install --upgrade huggingface_hub
        echo "Logging in to Hugging Face CLI..."
        huggingface-cli login --token $token
    else
        echo "No token entered. Skipping..."
    fi
}

setup_python() {
    echo "Checking Python version..."

    # Get the current Python version
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    REQUIRED_VERSION="3.11"

    # Compare the current version with the required version
    if [[ $(echo -e "$PYTHON_VERSION\n$REQUIRED_VERSION" | sort -V | head -n1) == "$REQUIRED_VERSION" ]]; then
        echo "Python version is $PYTHON_VERSION, which meets the requirement."
    else
        echo "Python version is $PYTHON_VERSION, which does not meet the requirement. Installing Python $REQUIRED_VERSION using pyenv..."

        echo "Installing developer tools"
        apt-get update
        apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

        # Install pyenv if not already installed
        if ! command -v pyenv &> /dev/null; then
            curl https://pyenv.run | bash
            export PATH="$HOME/.pyenv/bin:$PATH"
            eval "$(pyenv init --path)"
            eval "$(pyenv init -)"
        fi

        # Add pyenv initialization to .bashrc, .zshrc, .bash_profile, .profile
        echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
        echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
        echo 'eval "$(pyenv init -)"' >> ~/.bashrc
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
        echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
        echo 'eval "$(pyenv init -)"' >> ~/.profile
        echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bash_profile
        echo 'eval "$(pyenv init --path)"' >> ~/.bash_profile
        echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
        echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.zshrc
        echo 'eval "$(pyenv init --path)"' >> ~/.zshrc
        echo 'eval "$(pyenv init -)"' >> ~/.zshrc

        # Install the required Python version and set it as global
        pyenv install $REQUIRED_VERSION
        pyenv global $REQUIRED_VERSION

        echo "Python $REQUIRED_VERSION installed and set as global version."
    fi
}

setup_venv() {
    echo "Setting up venv..."

    python -m venv venv
    source venv/bin/activate

    echo "source ~/mats_hallucinations/venv/bin/activate" >> ~/.bashrc
    echo "source ~/mats_hallucinations/venv/bin/activate" >> ~/.zshrc
    echo "source ~/mats_hallucinations/venv/bin/activate" >> ~/.bash_profile
    echo "source ~/mats_hallucinations/venv/bin/activate" >> ~/.profile

    echo "Done setting up venv!"
}

install_requirements() {
    echo "Installing requirements..."

    yes | pip install -r requirements.txt --upgrade

    echo "Done installing requirements!"
}

setup_vscode() {
    echo "Setting up VSCode..."

    # Defining the vscode variable with the path to the VSCode executable
    vscode_path=$(ls -td ~/.vscode-server/cli/servers/*/server/bin/remote-cli/code | head -1)
    cursor_path=$(ls -td ~/.cursor-server/bin/*/bin/remote-cli/cursor | head -1)
    vscode="$vscode_path"
    cursor="$cursor_path"

    # Append vscode path to .bashrc for future use
    echo 'alias code="'$vscode'"' >> ~/.bashrc
    echo 'alias cursor="'$cursor'"' >> ~/.bashrc
    echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc

    # Update the system and install jq
    apt-get update
    apt-get install -y jq

    # Install recommended VSCode extensions
    jq -r '.recommendations[]' .vscode/extensions.json | while read extension; do "$vscode" --install-extension "$extension"; done
    jq -r '.recommendations[]' .vscode/extensions.json | while read extension; do "$cursor" --install-extension "$extension"; done
}


echo "Running set up..."

echo "" > .env
setup_hf
setup_python
setup_venv
install_requirements

echo "All set up!"