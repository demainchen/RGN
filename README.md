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


# contact
If you have any questions, please contact Wenqi Chen(Email:demainchen@gmail.com).
