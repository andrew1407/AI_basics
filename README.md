# AI_basics

A collection of artificial intelligence laboratory works covering fuzzy logic, classic neural networks, evolutionary computation, and deep learning, plus a final exam project that wraps a custom LSTM sentiment model in a desktop GUI. Each lab is provided as a runnable Python script, with several labs also available as Jupyter notebooks.

## Overview

The labs progress from fuzzy systems and unsupervised clustering through hand-built neural networks and genetic algorithms to TensorFlow/Keras deep-learning models for image classification and text sentiment analysis. The exam task packages a trained recurrent model behind a Tkinter interface.

## Labs

| Lab | Topic | Key implementation |
| --- | --- | --- |
| [`1-lab`](1-lab) | Fuzzy control system | Mamdani-style fuzzy controller over a target function using `skfuzzy.control` (antecedents, consequents, rule generation, simulation) |
| [`2-lab`](2-lab) | Fuzzy membership functions | Plots and fuzzy AND/OR operators across `skfuzzy` membership functions (`trimf`, `trapmf`, `gaussmf`, `gbellmf`, `sigmf`, `pimf`, etc.) |
| [`3-lab`](3-lab) | Fuzzy clustering | Fuzzy c-means clustering (`skfuzzy.cmeans`) on a 2D dataset with objective-function convergence plot |
| [`4-lab`](4-lab) | Backpropagation networks | Feed-forward, cascade-forward, and Elman recurrent networks built with `neurolab`, `theano`, and `lasagne` to approximate a function |
| [`5-lab`](5-lab) | Hebbian learning | Hebb-rule network that trains on and identifies BMP letter images |
| [`6-lab`](6-lab) | ANFIS | Adaptive neuro-fuzzy inference system (TensorFlow/Keras) regressing concrete compressive strength from the [`Concrete_Data.xls`](6-lab/Concrete_Data.xls) dataset |
| [`7-lab`](7-lab) | Evolutionary optimization | Genetic-algorithm function minimization (1D and 2D) using the `evol` library with custom selection, crossover, and mutation |
| [`8-lab`](8-lab) | Genetic packing | Chromosome-based optimization placing rectangular shapes within a bounded area, visualized with Matplotlib patches |
| [`9-lab`](9-lab) | MNIST classification | Keras dense network for handwritten-digit recognition, predicting on custom BMP examples via OpenCV |
| [`10-lab`](10-lab) | AlexNet (CIFAR-10) | AlexNet-style convolutional network in Keras with image standardization/resizing and `tf.data` pipelines |
| [`11-lab`](11-lab) | GoogLeNet (CIFAR-10) | Inception/GoogLeNet architecture with convolution, inception, and downsample modules and data augmentation |
| [`12-lab`](12-lab) | Bidirectional LSTM sentiment | Text sentiment classifier on the `imdb_reviews` dataset using `TextVectorization`, embeddings, and a bidirectional LSTM |

Notebook versions are provided for labs 10, 11, and 12 (`*.ipynb` alongside the scripts).

## Exam task

[`exam-task/`](exam-task) contains a sentiment-analysis application:

- [`app/sentiment_analyzer/`](exam-task/app/sentiment_analyzer) — a [`SentimentAnalyzer`](exam-task/app/sentiment_analyzer/analyzer.py) facade and a custom [`ELSTMCell`](exam-task/app/sentiment_analyzer/elstm.py) Keras layer (an extended LSTM cell), loading a saved model to classify text as positive or negative.
- [`app/gui/`](exam-task/app/gui) — a Tkinter [`WindowClient`](exam-task/app/gui/window_client.py) providing text input and prediction output.
- [`app/main.py`](exam-task/app/main.py) — entry point wiring the analyzer to the GUI.
- [`model/`](exam-task/model) — training notebooks ([`ELSTM.ipynb`](exam-task/model/ELSTM.ipynb), [`ELSTM_EXTRA.ipynb`](exam-task/model/ELSTM_EXTRA.ipynb), [`12-lab.ipynb`](exam-task/model/12-lab.ipynb)) and the exported SavedModel ([`model1/`](exam-task/model/model1)).

## Tech stack

- **Language:** Python (scripts and Jupyter notebooks)
- **Deep learning:** TensorFlow / Keras, TensorFlow Datasets
- **Classic neural networks:** neurolab, Theano, Lasagne
- **Fuzzy logic:** scikit-fuzzy (`skfuzzy`)
- **Evolutionary computation:** evol
- **Scientific / data:** NumPy, pandas, scikit-learn, Matplotlib
- **Imaging:** Pillow (PIL), OpenCV (`cv2`)
- **GUI:** Tkinter

## How to run

Each lab is self-contained and is run from its own directory so that relative data paths (datasets, image folders) resolve correctly. For example:

```bash
cd 9-lab
python 9-lab.py
```

Some labs read local assets:

- [`3-lab`](3-lab) reads [`dataset.txt`](3-lab/dataset.txt)
- [`5-lab`](5-lab) reads images from [`letters/`](5-lab/letters)
- [`6-lab`](6-lab) reads [`Concrete_Data.xls`](6-lab/Concrete_Data.xls)
- [`9-lab`](9-lab) reads BMP images from [`examples/`](9-lab/examples)

Labs that write plots expect an `out/` directory in the lab folder (e.g., [`2-lab`](2-lab), [`4-lab`](4-lab)).

To run the exam application:

```bash
cd exam-task/app
python main.py
```

Install the relevant libraries for the lab you intend to run (TensorFlow, scikit-fuzzy, neurolab, Theano, Lasagne, evol, NumPy, pandas, scikit-learn, Matplotlib, Pillow, OpenCV). No dependency manifest is included in the repository.

## Project structure

```
AI_basics/
├── 1-lab/        # Fuzzy control system (skfuzzy.control)
├── 2-lab/        # Fuzzy membership functions and operators
├── 3-lab/        # Fuzzy c-means clustering (+ dataset.txt)
├── 4-lab/        # Backprop / cascade / Elman networks
├── 5-lab/        # Hebbian learning (+ letters/)
├── 6-lab/        # ANFIS (+ Concrete_Data.xls)
├── 7-lab/        # Genetic-algorithm optimization
├── 8-lab/        # Genetic shape packing
├── 9-lab/        # MNIST digit recognition (+ examples/)
├── 10-lab/       # AlexNet on CIFAR-10 (.py + .ipynb)
├── 11-lab/       # GoogLeNet on CIFAR-10 (.py + .ipynb)
├── 12-lab/       # Bidirectional LSTM sentiment (.py + .ipynb)
└── exam-task/    # ELSTM sentiment analyzer + Tkinter GUI + saved model
```
