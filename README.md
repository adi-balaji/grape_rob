# GrapeJuice: Grape Localization for Automated Harvesting

<div align="center">
  <img src="stem_detection/images/grape_harvest.gif" alt="Robotic harvesting on fetch" width="1200"/><br>
  <sub><em>Fig: Robotic Grape Harvesting on Fetch</em></sub>
</div>

<br>

Hello! Welcome to GrapeJuice, a grape localization pipeline designed for automated harvesting. To run the masking and stem detection scripts, you need the dataset, which is too large to store on a git repo. Email me at *advaithb@umich.edu* for the dataset zip file!

*This project is in progress!* 

## Overview

<div align="center">
  <img src="stem_detection/images/grapesmodel.png" alt="GrapeJuice model architecture" width="1000"/><br>
  <sub><em>Fig: GrapeJuice model architecture</em></sub>
</div>

<br>

GrapeJuice proposes a localization system for grape bunches to support robotic harvesting. The main objective is to localize cutting points and estimate the pose of grape bunches to provide waypoints to a robotic manipulator. 

## Features

- Localization of grape bunches
- Estimation of cutting points
- Pose estimation for robotic manipulation
- Generalizable to almost all fruits with labeled data

## Grape Masking

*grape_masking* contains an implementation of a MaskRCNN for grape bunch masking using the WGISD Dataset. We report an average test set IoU of 72% on just 100 training images!

<div align="center">
  <img src="grape_masking/images/masking_eg2.png" alt="GrapeJuice model architecture" width="400"/><br>
  <sub><em>Fig: Predicted Grape Mask Example </em></sub>
</div>

<br>

## Stem Detection

*stem_detection* contains an implementation of a MaskRCNN for stem masking using a custom grape stem dataset. The stem masks will be used to localize the stems for grasping.

<div align="center">
  <img src="stem_detection/images/stem_example1.png" alt="Predicted Stem Mask Example 1" width="400" style="margin-right: 50;"/>
  <img src="stem_detection/images/stem_example2.png" alt="Predicted Stem Mask Example 2" width="400" style="margin-left: 50;"/><br>
  <sub><em>Fig: Predicted Stem Mask Examples </em></sub>
</div>



## Status

This is an ongoing project. We hope to test on the Fetch robot by Summer 2024!

