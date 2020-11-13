# FrontNet on PULP-shield

The goal of this project is to port face-following capabilities to the PULP-shield, and run it on the Crazyflie 2.0. Images captured by the mounted Himax camera, are fed to the neural network suggested in [Dario Mantegazza's thesis](https://github.com/idsia-robotics/proximity-quadrotor-learning), which provides the drone's control variables as output. The original network was written in Keras and designed to run on a desktop. The adaptations and optimizations performed on the network were done according to the work published by [Daniele Palossi](https://github.com/pulp-platform/pulp-dronet) and with his generous help.

<p align="center">
<img src="/resources/crazyflie2.0.jpg" alt="drawing" width="500"/>
</p>


### Milestones
  - Porting Dario's NN from Keras to PyTorch
  - Writing PyTorch-compatible tools for performance analysis and visualization
  - Retraining the network from scratch and comparing the results with the original 
  - Applying the quantization procedure developed in ETH while maintaining the quality
  - Getting familiar with the [GAPuino](https://greenwaves-technologies.com/product/gapuino/)
  - Implementing image capture and viewing abilities for the GAPuino
  - Augmenting the original dataset with images taken with the GAPuino's Himax camera
  - Retraining the network and estimating the performance 
  - Porting the network to the crazyflie 2.0 and live testing

# From Keras to PyTorch 
The PyTorch implementation followed closely on the network architecture suggested by Dario:

<img src="/resources/NN.png" alt="drawing" width="1000"/>

However, everything that wrapped the NN, from data loading to performance analysis had to be re-written completely to fit the PyTorch API. In Dario's work, he examines two networks- both take as input video frames, but the first outputs the pose variables (x,y,z,yaw) of the person relative to the drone, and the second one outputs the control variables of the drone (steering angle, speed). In the data collection process, each frame has the GT for the pose variables, provided by a MoCap system. Since this GT labels were readily available and easy to compare, I started by porting the first network.

<p align="center">
<img src="/resources/learning_curves.png" alt="drawing" width="500"/>
<p/>

The performance was evaluated in several ways:
* Inspecting the trend in the training and validation loss, to ensure we do not overfit
* MSE
* MAE
* R2 score

<p align="center">
<img src="/resources/stats.gif" alt="drawing"/>
</p>

Two visualization tools are implemented:
* Pose variables - Prediction vs. GT
![viz1](/resources/viz1.gif)
* Bird's-eye-view presentation with actual values
![viz2](/resources/viz2.gif)

# DNN: Desktop vs. PULP Platrform 
Working on the GAPuino introduced many challenges. The Himax camera accompanying the board is a grey-scale, low-resolution device, chosen for its ability to maintain a reasonable fps while consuming very little power. We were hoping to re-use the lovely dataset collected by Dario, for which he used the Parrot 2. But the images from the Himax camera differed greatly from the ones captured by the high-resolution, RGB camera of the Parrot 2. 
So in order to still re-use the dataset, it was decided to adapt the original images to look more like those of the target camera. For that I hacked a setup where the Himax camera was attached to the drone, close to its camera, and image streams of both cameras was recorded into a rosbag. 
<p align="center">
<img src="/resources/setup.png" alt="drawing" width="500"/>
<p/>
I attached a time-stamp to each stamped image, so I can sync between the two streams. The Parrot 2 delivered images at roughly 30 fps, while capturing from the Himax was dramatically slower at around 0.4 fps. This is due to the fact that images from the Himax camera were written to the board's memory, and then had to be transferred through a bridge to my PC. The bridge is intended mainly for debugging purposes and not for image streaming, so it was expected that the performance would be lacking. 
To orchestrate this recording setup, I wrote c code to run on the embedded device, which captures the images from the Himax camera and writes them into a named pipe on the host (using the bridge). There is also a ROS node running on my PC, that reads from that pipe, and publishes it as ROS image messages that can be recorded. 
<p align="center">
<img src="/resources/Gapuino.png" alt="drawing" width="500"/>
<p/>
After recording a few sessions, I focused on synching the images from both cameras. Of course, the fps was different, but there was also the need to account for the Himax's delay - the image is stamped when it reached the ROS node on the host, but it is actually captured awhile earlier. To calculate the delay I had one recording session where I captured my phone's timer, this allowed me to estimate the delay with an error of a few milliseconds. 
Once I had the frames synced, I transformed the images to be more alike - gamma correction, FOV cropping, RGB->gray, and resized them both to the input resolution of the network.
<p align="center">
<img src="/resources/sync.gif" alt="drawing"/>
</p>



# Real-time, Real-life 
In order to quantize the NN you will need the secret nemo folder, which is not public yet. Once the weights were quantized, I created a folder structure identical to that of pulp-dronet. Note that if you clone my repo, you will have to download and compile the autotiler yourself, as explained beautifully in pulp-dronet [documentation](https://github.com/pulp-platform/pulp-dronet#23-install-the-autotiler). When you retrain a model, keeping the same architecture, you can simply replace the weights and biases, and manually update the checksums in the config.h. If you make changes to the architecture, it gets a little messy. To run the project I use the same commands as in the pulp-dronet.
If you wish to simply capture images from the pulp-shield, you can run the MasterScript.py, just use the bottom part. 
<p align="center">
<img src="/resources/pulpdrone.jpg" alt="drawing" width="500"/>
<p/>
To ease the deployment and reduce possible bugs, the first NN I converted from PyTorch to its c equivalent was very similar to the original pulp-dronet. This allowed me to reuse the majority of the layer definitions from the pulp-dronet project. The performance of the netwrok was tested, and it gave satisfying results, performing at the level of FrontNet. 
<p align="center">
<img src="/resources/dronetarch.png" alt="drawing" width="1000"/>
<p/>

# Second Generation Pipeline
In the current deploymnt pipelin, a new version, 0.0.7, of [nemo](https://github.com/pulp-platform/nemo) is used. 
PyToch version should be 1.3 or more. 
To quantize and fine-tune the NN, Run
```bash
#CUDA_VISIBLE_DEVICES= python ETHQuantize.py --regime regime.json --epochs 5 --gray 1 --load-trainset "path-to-trainset.pickle" --load-model "your-pytorch-mmodel.pt" --quantiz --trainq
```
Make sure it runs on CPU, because weird things happened in GPU. The result of this step is a intermediate .pth model file, in the checkpoint folder, with the name "your-model-final.pth". To export the model as onnx file run
```bash
#CUDA_VISIBLE_DEVICES= python ETHQuantize.py --regime regime.json --epochs 10 --gray 1 --load-trainset "path-to-trainset.pickle" --load-model "checkpoint/your-model-final.pth --quantiz 
```
Once you have the .onnx file, simply follow th DORY documentation to get the auto-generatd c code, as explained [here](https://iis-git.ee.ethz.ch/alessio.burrello/dory_example). 

### Project Structure
The project has the following directory structure. Change it at your own risk.
```bash
.
├── config.py                                   #global information
├── data                                        #rosbags, calibration, dump folders
│   ├── bebop_calibration.yaml
│   └── calibration.yaml
├── DataProcessing                  
│   ├── Syncing scripts
│   ├── Dataset creation scripts
│   └── Images augmentation scripts
├── MasterScript.py                             #himax capture->bridge->rosnode-> display & record
├── pulp                                        #c code for the embedded device and python ROS nodes
│   ├── Calibrator.py
│   ├── CameraCalibration.py                    #calibrating the himax camera
│   ├── camera_to_fifo
│   │   ├── ..
│   ├── camera_to_file
│   │   ├── ..
│   ├── fifo_to_disk
│   │   ├── ...
│   ├── HimaxPublisher.py                       #publishing to rosnode images transferred from the device
│   └── Visualizer.py                           #broadcasts bebop and himax images concatenated
├── PyTorch                                     #everything related to the NN
|   ├── nemo 
|   |   └── secrets!
│   ├── Loading scripts
│   ├── Training scripts
│   └── Visualization scripts
```
### Installation
This code run on Ubuntou 16.04. If it happens to run on any other OS, consider it a miracle.
The following dependencies are needed:
* ROS kinetic - http://wiki.ros.org/kinetic/Installation
* PULP-sdk - https://github.com/pulp-platform/pulp-sdk
* PULP Toolchain - https://github.com/pulp-platform/riscv
* PULP-dronet - https://github.com/pulp-platform/pulp-dronet
* PyTorch 1.4

Following the installation instruction on the [PULP-dronet](https://github.com/pulp-platform/pulp-dronet) would be the easiest, as they cover all the PULP-related dependencies.

### How-To Guide
* Recording - In order to redcord, you may use the MasterScript.py. This will run the c code needed to transfer images from the device to the pipe, the ROS node that reads from the pipe and broadcasts the topic and viewers that allow you to see the streams. 
* Creating Dataset from Rosbags - Once you have rosbags recorded, you can converted it to a .pickle, with the proper format for training, using DataProcessing/Main.py. 
* Training, Testing, Infering - Examples for how to execute tasks related to PyTorch can be found in PyTorch/Main.py
* Visualization - To visualize a standard dataset, tou can use the DatasetVisualizer class. For displaying the prediction along with the GT, you can use another script, the purpose of the script will be indicated by its name. Don't use scripts that include "Hand" in their names, they are specifically for a side quest called Scarlet Witch, where I also recorded the poses of the hand. 

### Datasets
Beware that there are two different types of rosbags, Dario's rosbags, which can be found [here](https://drive.switch.ch/index.php/s/1Q0zN0XDzyRxug4), and my rosbags that I shall upload once the lame-ass OneDrive will stop malfunctioning. Each rosbag type needs to be handled differently, so choose the right method from the DatasetCreator class.
An example for a test.pickle after processing can be downloaded from [here](https://usi365-my.sharepoint.com/:u:/g/personal/zimmen_usi_ch/Ea4eoe34Y_JKot_Mzbh8ZCgBLA_9lWrycZwdCeGwKfpE0A?e=w4rNSr).

### Development

Want to contribute? Great!
Send me six-packs of diet coke.

License
----

MIT

