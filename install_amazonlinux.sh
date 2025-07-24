#!/bin/bash

# Activate the Neuron environment - using the PyTorch 2.6 environment
env="source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate"
name=$(hostname)

if ! grep -q "$env" ~/.bashrc; then
    echo $env | sudo tee -a ~/.bashrc > /dev/null
fi

source ~/.bashrc

# Install InfluxDB on Amazon Linux 2023
if ! command -v influx &> /dev/null; then
    # Try method 1: Use RHEL 9 repository (compatible with Amazon Linux 2023)
    cat <<EOF | sudo tee /etc/yum.repos.d/influxdb.repo
[influxdb]
name = InfluxDB Repository - RHEL 9
baseurl = https://repos.influxdata.com/rhel/9/x86_64/stable
enabled = 1
gpgcheck = 1
gpgkey = https://repos.influxdata.com/influxdata-archive_compat.key
EOF

    # Try to install from repository
    sudo dnf update -y
    if ! sudo dnf install -y influxdb2 influxdb2-cli 2>/dev/null; then
        echo "Repository method failed, trying direct download..."
        
        # Method 2: Direct download if repository fails
        sudo rm -f /etc/yum.repos.d/influxdb.repo
        
        # Download and install InfluxDB directly
        cd /tmp
        wget https://dl.influxdata.com/influxdb/releases/influxdb2-2.7.6-1.x86_64.rpm
        wget https://dl.influxdata.com/influxdb/releases/influxdb2-cli-2.7.6-1.x86_64.rpm
        
        sudo dnf install -y ./influxdb2-2.7.6-1.x86_64.rpm ./influxdb2-cli-2.7.6-1.x86_64.rpm
        cd -
    fi
    
    # Start and enable InfluxDB service
    sudo systemctl start influxdb
    sudo systemctl enable influxdb
    
    # Wait a moment for InfluxDB to start
    sleep 5
    
    # Setup InfluxDB with initial configuration
    influx setup \
      --username $name \
      --org "Stanford" \
      --bucket "Asst4" \
      --force
fi

echo "Setup complete! InfluxDB should be running on port 8086"
echo "Remember to use port forwarding when SSH'ing: ssh -L 3001:localhost:3001 -L 8086:localhost:8086"
