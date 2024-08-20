---
layout: tutorial_hands_on
level: Intermediate
title: Create and run a Deep learning image classifier
zenodo_link: 
- 'https://zenodo.org/records/13292895/files/mnist_images.zip'
- 'https://zenodo.org/records/13292924/files/config.yaml'
- 'https://zenodo.org/records/13292929/files/mnist_dataset.csv'
questions:
- How do I use the Ludwig tool to create and run a deep learning image classifier? 
- What files do I need to run Ludwig? 
- How do I build and evaluate the image classification model? 
objectives:
- Upload image files and Ludwig files to Galaxy. 
- Execute a Ludwig experiment and evaluate the results. 
- Compare different configurations to improve model performance. 
time_estimation: 40M
key_points:
- Use Galaxy-Ludwig Experiment to solve an image classification problem
- Using CNN on MNIST dataset. 
contributors:
  autorship:
  - Paulo Cilas M Lyra-Jr 
  - Junhao Qiu
  - Jeremy Goecks
tags:
- MNIST
- Deep learning
- Ludwig
---

Using the MNIST image dataset of handwritten digits as input, we will build an image recognition model with the Galaxy-Ludwig tool. The goal is to classify the digit handwritten in each image.

To accomplish this, three steps are needed: (i) upload Ludwig files and image files to Galaxy (ii) setting up and running the Ludwig experiment function on Galaxy, and (iii) evaluate the image classification model. As a bonus step, we'll also explore (iv) improving the model's classificationperformance (Figure 1).

![alt text] (../../images/model_schema.png "create and run image classifier model")
Figure 1 - Overview of the steps process to obtain the handwritten classification model and testing it.

<!-- You may want to cite some publications; this can be done by adding citations to the
bibliography file (`tutorial.bib` file next to your `tutorial.md` file). These citations
must be in bibtex format. If you have the DOI for the paper you wish to cite, you can
get the corresponding bibtex entry using [doi2bib.org](https://doi2bib.org).

With the example you will find in the `tutorial.bib` file, you can add a citation to
this article here in your tutorial like this:
{% raw %} `{% cite Batut2018 %}`{% endraw %}.
This will be rendered like this: {% cite Batut2018 %}, and links to a
[bibliography section](#bibliography) which will automatically be created at the end of the
tutorial.-->

> <agenda-title></agenda-title>
>
> In this tutorial, we will cover:
>
> 1. TOC
> {:toc}
>
{: .agenda}


> <comment-title>Background</comment-title>
>
> [MNIST dataset](https://yann.lecun.com/exdb/mnist/) provides the database of handwritten digits
> the database has a training set of 60,000 examples, and a test set of 10,000 examples.
> at our planet and its environment for the ultimate benefit of all European citizens.
> It is a subset of a larger set available from NIST.
> The digist have been size-normalized and centered in a fixed-size image.
{:  .comment}

# 1 FILES FORMAT
Before starting our hands-on, here is a brief explanation of the three files generated for the Ludwig Experiment tool.

## 1.1 Image_Files.zip 

Since our model will learn directly from the PNG files containing the handwritten numbers, we need to upload these files as well. The images are stored in two main folders: one for training and one for testing. Another folder, named after the figure label, is used to store the figures (Figure 2).

Notice that we are going to work with compacted file (.zip), Galaxy-Ludwig knows how to decompress the files and have them ready.

![alt text] (../../images/images_file.png "folder tree")
Figure 2 - File tree containing the images used to training and testing the model

## 1.2 MNIST_dataset.csv

The MNIST dataset consists of images and their corresponding labels. For the purpose of this tutorial, mnist_dataset.csv file is created and contains three columns: image_path, label, and, split.

Briefly, the image_path column provides the file paths to the images that will be fed into the deep learning algorithm. The label column contains the correct classifications, ranging from 0 to 9, for the handwritten digits in the images. The split column indicates whether the data should be used for training (0) or testing (2) the model.

![alt text] (../../images/dataset_format.png "dataset.csv file snapshot")
Figure 3 - Dataset.csv file format snapshot. features in order: file_path, label, split.

## 1.3 Config.yaml


The config.yaml file is crucial as it defines the entire structure of your machine learning experiment. This configuration file tells Ludwig how to process your data, what model to use, how to train it, and what outputs to generate.

The rational on how this file was construct for this dataset is the following:
i) The model takes images as input and uses a stacked convolutional neural network (CNN) to extract features.
ii) It consists of two convolutional layers followed by a fully connected layer, with dropout applied to both the second convolutional layer and the fully connected layer to reduce overfitting.
iii) The model is trained to classify images into categories (e.g., recognizing digits), and it will train for 20 epochs.

![alt text] (../../images/config.png "config file snapshot")
Figure 4 - Config.yaml file snapshot.

# 2 GALAXY-LUDWIG 

Ludwig simplifies the complexities of machine learning by automating essential steps such as data preprocessing, model architecture selection, hyperparameter tuning, and device management. This streamlined approach is particularly beneficial for Galaxy users who are more interested in addressing their scientific questions than in navigating the intricacies of machine learning workflows.

## 2.1 Prepare environment and get the data 

> <hands-on-title> Environment and Data Upload </hands-on-title>
>
> 1. Create a new history for this tutorial. If you are not inspired, you can name it *climate101*.
>
>    {% snippet faqs/galaxy/histories_create_new.md %}
>
> 2. Import the files from Zenodo or from the shared data library
>
>    ```
>    https://zenodo.org/records/13292895/files/mnist_images.zip
>    https://zenodo.org/records/13292929/files/mnist_dataset.csv 
>    https://zenodo.org/records/13292924/files/config.yaml 
>    ```
>
>    {% snippet faqs/galaxy/datasets_import_via_link.md %}
>
> 3. Check that the data format assigned for each file is **zip**, **csv**, and **yaml**, respectively.
>    If it is not, follow the Changing the datatype tip.
>
>    {% snippet faqs/galaxy/datasets_change_datatype.md datatype="datatypes" %}
>
> 5. Add a tag (`Stacked CNN`) to the dataset corresponding to `config.yaml`
>    This is important to trace back on what configurations the model was built on.
>
>    {% snippet faqs/galaxy/datasets_add_tag.md %}
>
{: .hands_on}

## 2.2 Ludwig Experiment

> <hands-on-title> Task description </hands-on-title>
>
> 1. {% tool [Ludwig Experiment](ludwig_experiment) %} with the following parameters:
>    - {% icon param-file %} *"Select the dataset containing model configuration"*: `output` (Input dataset)
>    - {% icon param-file %} *"Input dataset"*: `output` (Input dataset)
>    - {% icon param-file %} *"Raw data"*: `output` (Input dataset)
>
>    > <comment-title> short description </comment-title>
>    >
>    > A comment about the tool or something else. This box can also be in the main text
>    {: .comment}
>
{: .hands_on}

***TODO***: *Consider adding a question to test the learners understanding of the previous exercise*

> <question-title></question-title>
>
> 1. Question1?
> 2. Question2?
>
> > <solution-title></solution-title>
> >
> > 1. Answer for question1
> > 2. Answer for question2
> >
> {: .solution}
>
{: .question}


## Re-arrange

To create the template, each step of the workflow had its own subsection.

***TODO***: *Re-arrange the generated subsections into sections or other subsections.
Consider merging some hands-on boxes to have a meaningful flow of the analyses*

# Conclusion

Sum up the tutorial and the key takeaways here. We encourage adding an overview image of the
pipeline used.
