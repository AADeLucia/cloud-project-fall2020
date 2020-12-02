#!/bin/bash

# Download data
mkdir -p "flux/data"
cd "flux/data"
unzip flux_data.zip

cd ../../
mkdir -p "other_flows"
cd "other_flows"
wget --user=alexandra --password=ejch5fd1az http://netweb.ing.unibs.it/~ntw/tools/traces/download/unibs20090930.anon.pcap.tar.bz2 .
wget --user=alexandra --password=ejch5fd1az http://netweb.ing.unibs.it/~ntw/tools/traces/download/unibs20091001.anon.pcap.tar.bz2 .
wget --user=alexandra --password=ejch5fd1az http://netweb.ing.unibs.it/~ntw/tools/traces/download/unibs20091002.anon.pcap.tar.bz2 .
wget --user=alexandra --password=ejch5fd1az http://netweb.ing.unibs.it/~ntw/tools/traces/download/groundtruth.log .
tar -xvjf unibs20090930.anon.pcap.tar.bz2
tar -xvjf unibs20091001.anon.pcap.tar.bz2
tar -xvjf unibs20091002.anon.pcap.tar.bz2

# Set up Python
conda create --name cloud python=3.6
conda install pip
pip install -r requirements.txt
conda activate cloud

# Install Tshark
sudo apt-get update
sudo apt-get install tshark
