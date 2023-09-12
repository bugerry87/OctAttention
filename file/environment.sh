conda create -n $1 python=3.7
conda activate $1
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install hdf5storage
pip install Ninja
pip install tensorboard
pip install h5py
pip install tqdm
pip install matplotlib
pip install plyfile
