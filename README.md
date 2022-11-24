# WildTrack Image Quality Assessment
The goal of this project is to provide a model which emulates expert assessment of images for WildTrack species detection tasks.

## Data Processing:

[cropping_script.ipynb](cropping_script.ipynb) - This notebook was used to generated cropped images using either exported annotations or from a footprint object detectiom model

[generate_image_feature_set.ipynb ](generate_image_feature_set.ipynb)- This notebook was used to generate the data set containing natural scene statistics from images.
That data was then leveraged in baseline model development

## Data Analysis:

[subjective_scoring_analysis.ipynb](subjective_scoring_analysis.ipynb) - Detailed nalysis of subjective scoring performance against existing species classification model

## Data Modeling and Experimentation:

[brisque_downstream_model.ipynb](brisque_downstream_model.ipynb) - Baseline model development notebook leverating natural statistics measures to predict image quality based on downstream classification performance

[variational_autoencoder_quality_assessment.ipynb](variational_autoencoder_quality_assessment.ipynb) - Variational Autoencoder method model development notebook. This model focusing on training a model on classifable images in order to detect unclassifiable images as anomalies

[task_amenablity_species.ipynb](task_amenability_species.ipynb) - Reinforcement Learning model development which will and agent to score images based on downstream classification task  (classifying image baed on success downstream, e.g. species classification)

## Proposed Final Model
[IQA.ipynb](IQA.ipynb) - Notebook supporting final model evaluation, with inference example using sampled image data. 