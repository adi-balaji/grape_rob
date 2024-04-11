# GrapeJuice: Grape Localization for Automated Harvesting

<div align="center">
  <img src="stem_detection/images/grapesmodel.png" alt="GrapeJuice model architecture" width="1600"/><br>
  <sub><em>Fig: GrapeJuice model architecture</em></sub>
</div>

<br>

Hello! Welcome to GrapeJuice, a grape localization pipeline designed for automated harvesting. To run the masking and stem detection scripts, you need the dataset, which is too large to store on a git repo. Email me at *advaithb@umich.edu* for the dataset zip file!

*This project is in progress!* 

## Overview

GrapeJuice proposes a localization system for grape bunches to support robotic harvesting. The main objective is to localize cutting points and estimate the pose of grape bunches to provide waypoints to a robotic manipulator.

## Features

- Localization of grape bunches
- Estimation of cutting points
- Pose estimation for robotic manipulation
- Generalizable to almost all fruits with labeled data

## Masking

*grape_masking* contains an implementation of a MaskRCNN for grape bunch masking using the WGISD Dataset. We report an average test set IoU of 72% on just 100 training images!

<div align="center">
  <img src="grape_masking/images/masking_eg2.png" alt="GrapeJuice model architecture" width="1600"/><br>
  <sub><em>Fig: Predicted Grape Mask Example </em></sub>
</div>

<br>

## Status

This is an ongoing project. We hope to test on the Fetch robot by Summer 2024!

