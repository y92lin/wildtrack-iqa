# WildTrack Image Quality Assessment
The goal of this project is to provide a model which emulates expert assessment of images for WildTrack species detection tasks.

## Data Processing:

[cropping_script.ipynb](cropping_script.ipynb) - This notebook was used to generated cropped images using either exported annotations or from a footprint object detectiom model

[generate_image_feature_set.ipynb ](generate_image_feature_set.ipynb)- This notebook was used to generate the data set containing natural scene statistics from images.
That data was then leveraged in baseline model development

## Data Analysis:

[cluster_analysis.ipynb](cluster_analysis.ipynb) - Used to analyze any trends in natural scene statistic features of images which could be used in modeling

## Data Modeling and Experimentation:

[wildtrack_multitask_train.ipynb](wildtrack_multitask_train.ipynb) - Baseline model development notebook (attempted multitaks and basic CNN)

[task_amenablity_species.ipynb](task_amenability_species.ipynb) - Reinforcement Learning model development which will score images based on task amenability (classifying image baed on success downstream, e.g. species classification)

## Proposed Final Model
[IQA.ipynb](IQA.ipynb) - Notebook supporting final model evaluation, with inference example using sampled image data. 