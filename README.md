# Leveraging Mutants for Automatic Prediction of Metamorphic Relations using Machine Learning 

Official python implementation for the paper "Leveraging Mutants for Automatic Prediction of MetamorphicRelations using Machine Learning" (Maltesque 2019)

----------------------
Code Functionalities
----------------------

createPickle.py: Takes the Dot files and their corresponding class labels of a corresponding MR as input and generates a graph pickle object out of it. This graph pickle could be loaded by other programs for applying graph algorithms on it.

get_ROC-py: Takes the graph pickle as input and perform graph ML algorihtms on it to classiyfing it to its MR class. Later it provides a ROC metric containing the classifier accuracy details.

my_functions.py: Used to calculate the Random Walk Kernel (RWK) between two graphs.

----------------------
Acknowledgement
----------------------

The researchers gratefully acknowledge the support from the ITEA3 TESTOMAT Project, KTH Royal Institute of Technology and  Ericsson AB.

----------------------
Cite
----------------------
Please cite our paper if you use this code in your own work:

```
@inproceedings{nair2019leveraging,
  title={Leveraging mutants for automatic prediction of metamorphic relations using machine learning},
  author={Nair, Aravind and Meinke, Karl and Eldh, Sigrid},
  booktitle={Proceedings of the 3rd ACM SIGSOFT International Workshop on Machine Learning Techniques for Software Quality Evaluation},
  pages={1--6},
  year={2019},
  organization={ACM}
}
```



