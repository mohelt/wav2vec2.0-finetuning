# FYP_wav2vec2.0Finetuning

wav2vec 2.0 has emerged as a new self supervised deep learning model which has been applied to many speech signal applications such as speech recognition, language identification, speaker verification and emotion recognition. I will be finetuning it.

## Introduction
This will act as boilerplate template for wav2vec 2.0 fine-tuning. Please add your own files for training, testing and validation. Please add these in the data/DATA/ under /sets for your .txt files containing the filename and MOS scores for training, testing and validation. Add your files in this directory also.

#Prerequisites

1.download the repo

2. unzip wav2vec_small from the fairseq folder

# Steps for running on google colab

1. First we have to install a conda so we can create a virtual enviroment using the enviroment.yml specified. Type in the following command. It should restart your kernel.
```
!pip install -q condacolab
import condacolab
condacolab.install()
```

2. Type this to import conda for colab: 
```
import condacolab
condacolab.check()
```

3. Your files should be stored in google drive. The next code is used to mount drive so we can access the files.
```
from google.colab import drive
drive.mount('/content/drive')
```

4. Type (ignore any errors):
```
!conda env update -n base -f /content/drive/MyDrive/mos/environment.yml
```

5. Now its time to install any dependencies that may not have been mentioned:

```
pip install fairseq
!pip install tensorboardX
!pip install torch
!pip install numpy
!pip install scipy
!pip install torchaudio
!pip install fairseq
```

You must then download w2v_small.pt and also ckpt_w2vsmall:
i.e:

```
Here is some python code to do that:
    ## 1. download the base model from fairseq
    if not os.path.exists('fairseq/wav2vec_small.pt'):
        os.system('mkdir -p fairseq')
        os.system('wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt -P fairseq')
        os.system('wget https://raw.githubusercontent.com/pytorch/fairseq/main/LICENSE -P fairseq/')

    ## 2. download the finetuned checkpoint
    if not os.path.exists('pretrained/ckpt_w2vsmall'):
        os.system('mkdir -p pretrained')
        os.system('wget https://zenodo.org/record/6785056/files/ckpt_w2vsmall.tar.gz')
        os.system('tar -zxvf ckpt_w2vsmall.tar.gz')
        os.system('mv ckpt_w2vsmall pretrained/')
        os.system('rm ckpt_w2vsmall.tar.gz')
        os.system('cp fairseq/LICENSE pretrained/')
```

6. To run finetuning:
Make sure your training, testing and validation datasets are located in the correct place (data/DATA/) along with the .txt files which contain the filename and Actual MOS score.Download the mosfairseq.py file and edit the locations of 'traindir', 'wavdir' and 'validlist' variables. 

Optional: Change the patience value from 7 to whatever you would like.

Run:
```!python /content/drive/MyDrive/mosfairseq.py --datadir /content/drive/MyDrive/data/DATA/ --fairseq_base_model /content/drive/MyDrive/fairseq/wav2vec_small.pt --outdir /content/drive/MyDrive/mos/DirectoryToOutputTo --finetune_from_checkpoint /content/drive/MyDrive/pretrained/ckpt_w2vsmall```

You should then see it load the model and begin decreasing the loss function. If you have cuda capable device you should see "cuda" otherwise "cpu"

e.g
```
2023-03-16 10:51:30 | INFO | fairseq.tasks.text_to_speech | Please install tensorboardX: pip install tensorboardX
DEVICE: cuda
EPOCH: 1
AVG EPOCH TRAIN LOSS: 0.44713365010023115
EPOCH VAL LOSS: 0.36651407808065417
Loss has decreased
EPOCH: 2
AVG EPOCH TRAIN LOSS: 0.38450653619766234
EPOCH VAL LOSS: 0.4922252687215805
EPOCH: 3
```

With google colab pro using High RAM and Premium GPU. Finetuning with 5000 10 second samples with patience 7 takes 5375.0 secs. 1000 10 second samples with patience 7 takes 1105.0 seconds.

For prediction:
Download the prediction.py file and edit the locations of 'wavdir' and 'validlist' variables.

Then type:
```
!python /content/drive/MyDrive/predictN.py  --fairseq_base_model /content/drive/MyDrive/fairseq/wav2vec_small.pt --outfile answer_values_mos.txt --finetuned_checkpoint  /content/drive/MyDrive/pretrained/ckpt_w2vsmall --datadir  /content/drive/MyDrive/data/DATA
```
 You should then see for example after a few minutes:
 ```
 Loading checkpoint
Loading data
Starting prediction
[UTTERANCE] Test error= 0.623658
[UTTERANCE] Linear correlation coefficient= 0.801685
[UTTERANCE] Spearman rank correlation coefficient= 0.802130
[UTTERANCE] Kendall Tau rank correlation coefficient= 0.612214
```

7. I have also included two files predictionClassifier.py and fairseqClassifier.py which are for using wav2vec 2.0 for binary classification, in our case prediction if an audio file is urban or rural. Running the code is the exact same for prediction and for training, just change the file names to the new names, predictionClassifier.py and fairseqClassifier.py.

8. Once you have the output answers, create a csv with the first column predicted answers and the second column actual answers like "outputandactual.csv" (you need to round the values to the nearest value either 0 or 1 to complete the binary classification), then you can obtain the f1, recall, support results, here is python code showing you how to do that:

```
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load CSV file into a pandas DataFrame
data = pd.read_csv('/content/drive/MyDrive/mos/outputandactual.csv')

# Extract the predicted and actual scores as NumPy arrays
y_pred = data.iloc[:, 0].values
y_true = data.iloc[:, 1].values

# Create the confusion matrix using scikit-learn
cm = confusion_matrix(y_true, y_pred)

# Print the confusion matrix
print("Confusion matrix:\n", cm)

report = classification_report(y_true, y_pred)
print("Classification report:\n", report)
```
Thats it!
If you have any questions you can contact me at mohamed.l.eltayeb@gmail.com .


