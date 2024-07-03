# Prenatal Diagnosis of Cerebellar Hypoplasia in Fetal Ultrasound Using Deep Learning under the Constraint of the Anatomical Structures of the Cerebellum and Cistern
by Xiaoxiao Wu, Fu Liu, Guoping Xu, Chen Cheng, Yilin Ma, Ruifan He, Aoxiang Yang, Jiayi Gan, Jiajun Liang, Xinglong Wu, and Sheng Zhao. This is a code repo of the paper.

## __Installation__
Ubuntu 18.04   
__Conda Environment Setup__  
Create your own conda environment
```shell
conda create -n ASCNet python=3.8
conda activate ASCNet
```

Install Pytorch == 2.0.0(depends on your NVIDIA driver and you can see your compatible CUDA version at the right hand corner in nvidia-smi)
```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
Install monai == 1.1.0
```shell
pip install monai[all]
```
other python required packages 
```
pip install -r requirements.txt
```
## __Run__
train command
```shell
python train.py configs/ascnet/epoch-350_ascnet_224x224.py
```
test command
```shell
python test.py work_dirs/epoch-350_ascnet_224x2244/time_prefix/checkpoints/xxx.ckpt
```
