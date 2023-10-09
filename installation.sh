cd AlternativeModels\-SC2

cd SC_Fault_injections
find . -name "*.sh" | xargs chmod +x

# pytorchfi_SC 

# create the sc2-benchmark environmet and install the required dependencies
# if you already crerated the sc2-benchmark please first remove it and then create it again as follows
cp environment.yaml ../environment.yaml
cd ..
conda deactivate

conda env create -f environment.yaml
conda deactivate
source ~/miniconda3/bin/activate sc2-benchmark

python -m pip install -e .

python -m pip install -e ./SC_Fault_injections/pytorchfi_SC/