wget --output-document=data.zip https://www.research.ibm.com/haifa/dept/vst/files/IBM_Debater_\(R\)_EviConv-ACL-2019.v1.zip
unzip data.zip
mkdir data
mv IBM_Debater_\(R\)_EviConv-ACL-2019.v1/*.csv data
rm -r IBM_Debater_\(R\)_EviConv-ACL-2019.v1
rm data.zip