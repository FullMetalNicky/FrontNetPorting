# FrontNet on PULP-shield

The goal of this project is to port face-following capabilities to the PULP-shield, and run it on the Crazyflie 2.0. Images captured by the mounted Himax camera, are fed to the neural network suggested in [Dario Mantegazza's thesis](https://github.com/idsia-robotics/proximity-quadrotor-learning), which provides the drone's control variables as output. The original network was written in Keras and designed to run on a desktop. The adaptations and optimizations performed on the network were done according to the work published by [Daniele Palossi](https://github.com/pulp-platform/pulp-dronet) and with his generous help.

<p align="center">
<img src="/resources/crazyflie2.0.jpg" alt="drawing" width="500"/>
</p>


### Milestones
  - Porting Dario's NN from Keras to PyTorch
  - Writing PyTorch-compatible tools for performance analysis and visualizatrion
  - Retraining the network from scratch and comparing the results with the original 
  - Applying the quantization procedure developed in ETH whille maintaining the quality
  - Getting familiar with the [GAPuino](https://greenwaves-technologies.com/product/gapuino/)
  - Implementing image capture and viewing abilities for the GAPuino
  - Augmenting the original dataset with images taken with the GAPuino's Himax camera
  - Retraining the network and estimating the performance 
  - Porting the network to the crazyflie 2.0 and live testing

# From Keras to PyTorch 
The PyTorch implementation followed closely on the network architecture suggested by Dario:

<img src="/resources/NN.png" alt="drawing" width="1000"/>

However, everything that wrapped the NN, from data loading to performance analysis had to be re-written completely to fit the PyTorch API. In Dario's work, he examines two networks- both take as input video frames, but the first outputs the pose variables (x,y,z,yaw) of the person relative to the drone, and the second one outputs the control variables of the drone (steering angle, speed). In the data collection process, each frames has the GT for the pose variables, provided by a MoCap system. Since this GT labels were readily available and easy to compare, I started by porting the first network.

<p align="center">
<img src="/resources/learning_curves.png" alt="drawing" width="500"/>
<p/>

The performance was evaluated in several ways:
* Inspecting the trend in the training and validation loss, to ensure we do not overfit
* MSE
* MAE
* R2 score

<p align="center">
<img src="/resources/train.gif" alt="drawing"/>
</p>

Two visualization tools are implemented:
* Pose variables - Prediction vs. GT
![viz1](/resources/viz1.gif)
* Bird's-eye-view presentation with actual values
![viz2](/resources/viz2.gif)

# DNN: Desktop vs. ULP Platrform 
<p align="center">
<img src="/resources/Gapuino.png" alt="drawing" width="500"/>
<p/>


# Real-time, Real-life 


### Installation
This code run on Ubuntou 16.04. If it happens to run on any other OS, consider it a miracle.
The following dependencies are needed:
* ROS kinetic - http://wiki.ros.org/kinetic/Installation
* PULP-sdk - https://github.com/pulp-platform/pulp-sdk
* PULP Toolchain - https://github.com/pulp-platform/riscv
* PULP-dronet - https://github.com/pulp-platform/pulp-dronet

Following the installation instruction on the [PULP-dronet](https://github.com/pulp-platform/pulp-dronet) would be the easiest, as they cover all the PULP-related dependencies.

### Datasets

### Development

Want to contribute? Great!
Send me six-packs of diet coke.

License
----

MIT

