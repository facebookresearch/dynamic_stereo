# [CVPR 2023] DynamicStereo: Consistent Dynamic Depth from Stereo Videos.

**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**; **[University of Oxford, VGG](https://www.robots.ox.ac.uk/~vgg/)**

[Nikita Karaev](https://nikitakaraevv.github.io/), [Ignacio Rocco](https://www.irocco.info/), [Benjamin Graham](https://ai.facebook.com/people/benjamin-graham/), [Natalia Neverova](https://nneverova.github.io/), [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi/), [Christian Rupprecht](https://chrirupp.github.io/)

[[`Paper`](https://research.facebook.com/publications/dynamicstereo-consistent-dynamic-depth-from-stereo-videos/)] [[`Project`](https://dynamic-stereo.github.io/)] [[`BibTeX`](#citing-dynamicstereo)]

![nikita-reading](https://user-images.githubusercontent.com/37815420/236242052-e72d5605-1ab2-426c-ae8d-5c8a86d5252c.gif)

**DynamicStereo** is a transformer-based architecture for temporally consistent depth estimation from stereo videos. It has been trained on a combination of two datasets: [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) and **Dynamic Replica** that we present below.

## Dataset

https://user-images.githubusercontent.com/37815420/236239579-7877623c-716b-4074-a14e-944d095f1419.mp4

The dataset consists of 145200 *stereo* frames (524 videos) with humans and animals in motion. 

We provide annotations for both *left and right* views, see [this notebook](https://github.com/facebookresearch/dynamic_stereo/blob/main/notebooks/Dynamic_Replica_demo.ipynb):  
- camera intrinsics and extrinsics
- image depth (can be converted to disparity with intrinsics)
- instance segmentation masks
- binary foreground / background segmentation masks
- optical flow (released!)
- long-range pixel trajectories (released!)


### Download the Dynamic Replica dataset
Download `links.json` from the *data* tab on the [project website](https://dynamic-stereo.github.io/) after accepting the license agreement.
```
git clone https://github.com/facebookresearch/dynamic_stereo
cd dynamic_stereo
export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH
```
Add the downloaded `links.json` file to the project folder. Use flag `download_splits` to choose dataset splits that you want to download: 
```
python ./scripts/download_dynamic_replica.py --link_list_file links.json \
--download_folder ./dynamic_replica_data --download_splits real valid test train
```

Memory requirements for dataset splits after unpacking (with all the annotations):
- train - 1.8T
- test - 328G
- valid - 106G
- real - 152M

You can use [this PyTorch dataset class](https://github.com/facebookresearch/dynamic_stereo/blob/dfe2907faf41b810e6bb0c146777d81cb48cb4f5/datasets/dynamic_stereo_datasets.py#L287) to iterate over the dataset.

## Installation

Describes installation of DynamicStereo with the latest PyTorch3D, PyTorch 1.12.1 & cuda 11.3

### Setup the root for all source files:
```
git clone https://github.com/facebookresearch/dynamic_stereo
cd dynamic_stereo
export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH
```
### Create a conda env:
```
conda create -n dynamicstereo python=3.8
conda activate dynamicstereo
```
### Install requirements
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
# It will require some time to install PyTorch3D. In the meantime, you may want to take a break and enjoy a cup of coffee.
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install -r requirements.txt
```

### (Optional) Install RAFT-Stereo
```
mkdir third_party
cd third_party
git clone https://github.com/princeton-vl/RAFT-Stereo
cd RAFT-Stereo
bash download_models.sh
cd ../..
```



## Evaluation
To download the checkpoints, you can follow the below instructions:
```
mkdir checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/dynamic_replica_v1/dynamic_stereo_sf.pth 
wget https://dl.fbaipublicfiles.com/dynamic_replica_v1/dynamic_stereo_dr_sf.pth 
cd ..
```
You can also download the checkpoints manually by clicking the links below. Copy the checkpoints to `./dynamic_stereo/checkpoints`.

- [DynamicStereo](https://dl.fbaipublicfiles.com/dynamic_replica_v1/dynamic_stereo_sf.pth) trained on SceneFlow
- [DynamicStereo](https://dl.fbaipublicfiles.com/dynamic_replica_v1/dynamic_stereo_dr_sf.pth) trained on SceneFlow and *Dynamic Replica*

To evaluate DynamicStereo:
```
python ./evaluation/evaluate.py --config-name eval_dynamic_replica_40_frames \
 MODEL.model_name=DynamicStereoModel exp_dir=./outputs/test_dynamic_replica_ds \
 MODEL.DynamicStereoModel.model_weights=./checkpoints/dynamic_stereo_sf.pth 
```
Due to the high image resolution, evaluation on *Dynamic Replica* requires a 32GB GPU. If you don't have enough GPU memory, you can decrease `kernel_size` from 20 to 10 by adding `MODEL.DynamicStereoModel.kernel_size=10` to the above python command. Another option is to decrease the dataset resolution.

As a result, you should see the numbers from *Table 5* in the [paper](https://arxiv.org/pdf/2305.02296.pdf). (for this, you need `kernel_size=20`)

Reconstructions of all the *Dynamic Replica* splits (including *real*) will be visualized and saved to `exp_dir`.

If you installed [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo), you can run:
```
python ./evaluation/evaluate.py --config-name eval_dynamic_replica_40_frames \
  MODEL.model_name=RAFTStereoModel exp_dir=./outputs/test_dynamic_replica_raft
```

Other public datasets we use: 
 - [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
 - [Sintel](http://sintel.is.tue.mpg.de/stereo)
 - [Middlebury](https://vision.middlebury.edu/stereo/data/)
 - [ETH3D](https://www.eth3d.net/datasets#low-res-two-view-training-data)
 - [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_stereo.php) 

## Training
Training requires a 32GB GPU. You can decrease `image_size` and / or `sample_len` if you don't have enough GPU memory.
You need to donwload SceneFlow before training. Alternatively, you can only train on *Dynamic Replica*.
```
python train.py --batch_size 1 \
 --spatial_scale -0.2 0.4 --image_size 384 512 --saturation_range 0 1.4 --num_steps 200000  \
 --ckpt_path dynamicstereo_sf_dr  \
  --sample_len 5 --lr 0.0003 --train_iters 10 --valid_iters 20    \
  --num_workers 28 --save_freq 100  --update_block_3d --different_update_blocks \
  --attention_type self_stereo_temporal_update_time_update_space --train_datasets dynamic_replica things monkaa driving
```
If you want to train on SceneFlow only, remove the flag `dynamic_replica` from `train_datasets`.



## License
The majority of dynamic_stereo is licensed under CC-BY-NC, however portions of the project are available under separate license terms: [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) is licensed under the MIT license, [LoFTR](https://github.com/zju3dv/LoFTR) and [CREStereo](https://github.com/megvii-research/CREStereo) are licensed under the Apache 2.0 license.


## Citing DynamicStereo
If you use DynamicStereo or Dynamic Replica in your research, please use the following BibTeX entry.
```
@article{karaev2023dynamicstereo,
  title={DynamicStereo: Consistent Dynamic Depth from Stereo Videos},
  author={Nikita Karaev and Ignacio Rocco and Benjamin Graham and Natalia Neverova and Andrea Vedaldi and Christian Rupprecht},
  journal={CVPR},
  year={2023}
}
```
