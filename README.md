

# RGN

The source code of RGN:Residual based graph attention and convolutional network for protein-protein interaction site prediction.

# Data:

Since our dataset exceeds the maximum upload capacity of Github, please download our processed dataset at this link(https://drive.google.com/drive/folders/1KoxQs4c4iZg4EqTn8SfC0FXqVnRW1Ihm?usp=sharing).
There are three files, named Dset_1291, Dset_315 and Dset_395 respectively.
All protein sequences are stored in PKL file format.

If you want to test the performance on your own data, please make sure you install the following software and download the corresponding databases:

(1) PSSM(L*20) can be obtained by BLAST("https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/") and UNIREF("https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz").

(2) HHM(L*20) can be obtained by HH-suite("https://github.com/soedinglab/hh-suite") and 	Uniclust("https://gwdu111.gwdg.de/~compbiol/uniclust/2022_02/UniRef30_2022_02_hhsuite.tar.gz").

(3) DSSP(L*14) can be obtained by ("https://github.com/cmbi/dssp").

(4)Protbert(L*1024) can be obtained by "https://huggingface.co/Rostlab/prot_bert".



# Model

The source code of the model is saved in the Model/layers.py. 

# Predicting the PPI by the Model

Since our pre-trained model exceeds the maximum upload capacity of Github, we upload all the pre-trained model in this link. 

There are 252_63, 335_60, 335_315,1215_315 respectively in (“https://drive.google.com/drive/folders/1_baPkIS5kQQ3kSlg23_ozlGNKhus4ssW?usp=sharing“), which means the model training on the Dset_252 and testing on Dset_63 as so on.

Please download all the pre-trained model and place it in saved file.

```python
python3 Predict_PPI.py -pre_model your_saved/335_60 -dataset your_Dset_60
```

```python
python3 Predict_PPI.py -pre_model your_saved/252_63 -dataset your_Dset_63
```

```python
python3 Predict_PPI.py -pre_model your_saved/1215_315 -dataset your_Dset_315
```

# Example

We give an example case in the Example file and it can be download in https://drive.google.com/drive/folders/1jo_QNMIcjEDwNv1jKUVCeeRN7UQRGyEi?usp=sharing.
In the Example file, just execute:
```
python3 example.py 
```



# contact

If you have any questions, please contact Wenqi Chen(Email:[demainchen@gmail.com](mailto:demainchen@gmail.com)).
