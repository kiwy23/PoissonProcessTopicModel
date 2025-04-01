# PoissonProcessTopicModel

This repository contains data and a saved model associated with our paper:

**"A Poisson-process topic model for integrating knowledge from pre-trained language models"**

You can access the data and saved model via the following link:
- [Google Drive Folder](https://drive.google.com/drive/folders/15O9BDdzeqRgnyL_HNLyLOq6ozFdqhoJJ?dmr=1&ec=wgc-drive-globalnav-goto)


To reproduce the results presented in our paper from scratch, follow these steps:

- **Install Dependencies**
  - Run the following command to install all necessary packages:
    ```bash
    pip install -r requirements.txt
    ```

- **Data Preprocessing**
  - Run the command below to preprocess the data:
    ```bash
    python process_ap_c.py
    ```
  - This will create a folder named `./npz/datetime`.

- **Reproducing Results**
  - Open and run the Jupyter notebook `reproducibility_ap.ipynb` for step-by-step instructions.

For further details, please refer to the accompanying documentation or contact the authors for support.

Happy modeling!