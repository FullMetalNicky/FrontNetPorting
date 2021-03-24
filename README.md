# FrontNet on PULP-shield/AI-Deck

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

# The Birth of PenguiNet
As a risk minimization step towards deployment, I also ported the Dronet architecture from TensorFlow to PyTorch. I explored several architectures, trying to find one that is accurate enough to perform the task, but also small and ULP-friendly. This search brought to life PenguiNet - the skinny nephew of Dronet. With PenguiNet I removed the residual connections from the Dronet architecture, while maintaining the same accuracy. Due to its efficiency, PenguiNet was chosen for the final task evaluation on the nano-drone.

<img src="/resources/Pingu.png" alt="drawing" width="1000"/>

# Real-time, Real-life 
## Manual Pipeline
The first attempt at deployment was based on the method used for PULP-Dronet. Note that if you clone my repo, you will have to download and compile the autotiler yourself, as explained beautifully in PULP-Dronet [documentation](https://github.com/pulp-platform/pulp-dronet#23-install-the-autotiler). When you retrain a model, keeping the same architecture, you can simply replace the weights and biases, and manually update the checksums in the config.h. If you make changes to the architecture, it gets a little messy. To run the project I use the same commands as in the pulp-dronet.

To ease the deployment and reduce possible bugs, the first NN I converted from PyTorch to its c equivalent was very similar to the original PULP-Dronet. This allowed me to reuse the majority of the layer definitions from the PULP-Dronet project. The performance of the netwrok was tested, and it gave satisfying results, performing at the level of FrontNet. 
<p align="center">
<img src="/resources/dronetarch.png" alt="drawing" width="1000"/>
<p/>

## Auto-Generation Pipeline

The second, and more friendly deployment pipeline is based on two libraries:
* [NEMO (NEural Minimizer for pytOrch)](https://github.com/pulp-platform/nemo)
* [DORY: Deployment ORiented to memorY](https://github.com/pulp-platform/dory)
 
Trained, full precision PyTorch models are quantized and fine-tuned using NEMO. The result is an .onnx file which is a graph representation of the model. The steps for producing this are detailed in the How-To guide below. The .onnx is then fed to DORY, which generated PULP-optimized c code desscribing the network's functionality. This code can be used for inference on PULP chips, and specifically on the AI Deck. This code will be made public soon.  
This pipeline was used for the quantitative evaluation and user experiments, using the PenguiNet architecture.

<p align="center">
<img src="/resources/pulpdrone.jpg" alt="drawing" width="500"/>
<p/>

### Project Structure
The project has the following directory structure. Change it at your own risk.
```bash
.
├── data                                        #datasets
│   ├── trainset.pickle
│   └── testset.pickle
├── DataProcessing                  
│   ├── Syncing scripts
│   ├── Dataset creation scripts
│   └── Images augmentation scripts
├── PyTorch                                     #everything related to the NN
│   ├── Testing scripts
│   ├── Training scripts
│   └── Visualization scripts
├── Models  
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
* Recording is handled in this dedicated repo - [PULP-Streamer](https://github.com/FullMetalNicky/PULP-Streamer)
* Creating Dataset from Rosbags - Once you have rosbags recorded, you can converted it to a .pickle, with the proper format for training, using DataProcessing/Main.py. 
* Visualization - To visualize a standard dataset, tou can use the DatasetVisualizer class. For displaying the prediction along with the GT, you can use another script, the purpose of the script will be indicated by its name. Don't use scripts that include "Hand" in their names, they are specifically for a side quest called Scarlet Witch, where I also recorded the poses of the hand. 
#### Training 
For training full precision models use the FPTraining.py. For quantizing and fine-tuning an already trained PyTorch model, use QTraining.py
#### Deployment 
To deploy the quantized PyTorch model into an .onnx file that is compatible with DORY, run QDeploy.py
#### Testing
Testing full precision model can be done using FPTesting.py, and for quantized models - QTesting.py
#### Configuration
For all Q scripts, cmd arguments are required for the configuration of the quantization, deployment and testing. See examples in ExampleScript.sh


### Datasets
Beware that there are two different types of rosbags, Dario's rosbags, which can be found [here](https://drive.switch.ch/index.php/s/1Q0zN0XDzyRxug4), and my rosbags that I shall upload once the lame-ass OneDrive will stop malfunctioning. Each rosbag type needs to be handled differently, so choose the right method from the DatasetCreator class.
An example for a test.pickle after processing can be downloaded from [here](https://usi365-my.sharepoint.com/:u:/g/personal/zimmen_usi_ch/Ea4eoe34Y_JKot_Mzbh8ZCgBLA_9lWrycZwdCeGwKfpE0A?e=w4rNSr).

### Development

Want to contribute? Great!
Send me six-packs of diet coke.

License
----

MIT

