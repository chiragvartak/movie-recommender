# Movie Recommender

This repository contains the code implementation and demonstration of our movie recommending service. Two algorithms,
corresponding to two different approaches have been implemented:  
1) Hybrid  
2) Neural Collaborative Filtering (NCF)

## Installation & Setup

### Environment

We are using Python 3 (more specifically, Python 3.8). Nevertheless, any Python>=3.5 should work fine since we are not
using any version-specific functionalities.

### Install the dependencies

```bash
pip install matplotlib numpy pandas scikit-learn scikit-surprise Flask flask-restful flask-cors torch torchvision pytorch-lightning
```

The above command will install the most recent version of the Python dependencies. If that creates any problems, we have
provided a `requirements.txt` file so that the exact environment can be emulated.

### Download data and model files

1) Download the data files - movies.csv, ratings.csv, tags.csv - from the link below, and place it in the project root
   folder:  
   [Data files](https://drive.google.com/drive/folders/1rg6QNPmEz1cIvhiSyNRUmFbtcKOr0CxW)  
   You can also download these files directly from the GroupLens official website:  
   [GroupLens MovieLens Dataset](https://grouplens.org/datasets/movielens/25m/)  
   Just unzip the files and place them in the project root folder.
   
2) (Optional) If you do not want to train the model yourself, we have provided pre-trained model so that you can quickly
   get started. Place the `trained.model` in the `code/ncf` folder, and the `hybrid.model` in the `code/hybrid`
   folder.  
   [trained.model (size ~800 MB)](https://drive.google.com/file/d/1k-Wgvwbo4qydm8YFha2KOc_r7FvsGgl7/view?usp=sharing)  
   [hybrid.model (size ~130 MB)](https://drive.google.com/file/d/19ZfMdmH9bQlblxn-yRUoho9kgSwK4LdE/view?usp=sharing)
   
## Train the models

1) To train the NCF model simply run the `code/ncf/Train.py` file. This will take some time. The model that will be
generate will be placed in the same folder.
   
2) To train the Hybrid model, run the `code/hybrid/train.py` file. The trained model will be generated and placed in the
same folder.
   
## Run the application

To run the flask application, execute the main method present in `code/ncf/main.py`.

This should start the flask application on Port:5000 of your system. 

## Caveats

1) We observed that PyTorch installation using pip sometimes causes problems. In that case, follow the instructions on
the [PyTorch Official Website](https://pytorch.org/) to install the version that is right for you.
   
2) PyTorch installation using pip sometimes gives some `setup.py` errors. Just verify that your Python installation is
a 64-bit installation. PyTorch does not work on a 32-bit Python installation on Windows.

## UI Application Installation and Setup

### Install the dependencies

1. Install Node.js from: https://nodejs.org/en/download/
2. Open terminal in `code/UI` folder and run the command `npm install` to install all the dependent packages.

### Run the app

1. Open terminal in the `code/UI` folder and run the command `ng serve`, this will run the application and serve it on Port : 4200
2. Open your browser and navigate to http://localhost:4200

### User Id's for testing

When you open the UI application, you will be required to enter the User Id for the user you want to get the recommendations for.

If you use the pre-trained model files, you could use the following user id's to get both NCF and Hybrid recommendations: `2103, 5413, 12, 18768`.
