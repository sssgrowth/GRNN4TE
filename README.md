## GRNN4TE ([Project Tutorial](https://sssgrowth.github.io/GRNN4TE/))  
#### As long as this paper is accepted, this project is publicly available. 

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)

GRNN4TE is a framework to accelerate graph data processing and triple extraction research. It is a preview. The detailed descriptions are still in the making.  

<span><img src="pic/1563456690.840768.png" width="10%" alt="Dimensionality reduction with PCA"/></span>  

### Contents

* [Basics](#basics)
  * [Installation](#installation)
  * [Addition](#addition)
    * [Elastic weight consolidation (for lifelong learning)](#elastic-weight-consolidation)
    * [Conditional random field](#conditional-random-field)
    * [Model visualization](#model-visualization)
    * [Data visualization analysis](#data-visualization-analysis)
    * [Object detection papers](#object-detection-papers)
* [Citation](#citation) 
* [Updating](#updating)

## Contents
### Addition
#### Model visualization
<span><img src="pic/pcaboth-1.png" width="90%" alt="Dimensionality reduction with PCA"/></span>  
Figure 1. Dimensionality reduction for vertices with PCA  
Different colors denote different relation types. As shown in Figure 1, PCA tends to reveal the semantic relationship between relation types. We first observe that there are two main clusters, where the cluster of “contains” and “neighbor_of ” relations describes the relationship between locations, while the other cluster describes the relations about person and organization/location. The lower part of the diagram shows the relations of “place lived”, “company” and “nationality” are closely related, but the relations are mixed. t-SNE tends to show a more accurate relationship between relations. As shown in Figure 2, the “placed lived” is closer to “nationality”. “nationality” is closely related to “company”, while the “place lived” is not always close to “company”. This approach is instructive for exploring interesting phenomena in social and economic development. This GRNN, by leveraging vertex embeddings, provides better understanding about relations. This means the vertex embeddings encode the semantic relationship between relation types.  
<span><img src="pic/tsneboth-1.png" width="90%" alt="Dimensionality reduction with t-SNE"/></span>  
Figure 2. Dimensionality reduction for vertices with t-SNE

#### Object detection papers
Get more inspiration or evolution from papers for object detection or face detection recognition.  
```
@inproceedings{girshick2014rich,
  title={Rich feature hierarchies for accurate object detection and semantic segmentation},
  author={Girshick, Ross and Donahue, Jeff and Darrell, Trevor and Malik, Jitendra},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={580--587},
  year={2014}
}
@inproceedings{girshick2015fast,
  title={Fast r-cnn},
  author={Girshick, Ross},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={1440--1448},
  year={2015}
}
@inproceedings{ren2015faster,
  title={Faster r-cnn: Towards real-time object detection with region proposal networks},
  author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
  booktitle={Advances in neural information processing systems},
  pages={91--99},
  year={2015}
}
@inproceedings{redmon2016you,
  title={You only look once: Unified, real-time object detection},
  author={Redmon, Joseph and Divvala, Santosh and Girshick, Ross and Farhadi, Ali},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={779--788},
  year={2016}
}
```  
Papers about face detection.
```
@inproceedings{schroff2015facenet,
  title={Facenet: A unified embedding for face recognition and clustering},
  author={Schroff, Florian and Kalenichenko, Dmitry and Philbin, James},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={815--823},
  year={2015}
}
@article{viola2001rapid,
  title={Rapid object detection using a boosted cascade of simple features},
  author={Viola, Paul and Jones, Michael and others},
  journal={CVPR (1)},
  volume={1},
  number={511-518},
  pages={3},
  year={2001}
}
@article{viola2004robust,
  title={Robust real-time face detection},
  author={Viola, Paul and Jones, Michael J},
  journal={International journal of computer vision},
  volume={57},
  number={2},
  pages={137--154},
  year={2004},
  publisher={Springer}
}
```  
## Updating...
* 2019-Oct-15, CIL4TE v0.3.0, introduce EOIs and POEOIs and establish a direct link between relation extraction and computer vision studies  
* 2019-Aug-10, CIL4TE v0.2.4, enhance the cascade inference learning paradigm for other models  
* 2019-Aug-09, GRNN4TE v0.2.3, introduce ranking mechanism for candidate selection
* 2019-Mar-20, GRNN4TE v0.2.2, support vertex visualization for interpretability and adding optimized graph search
* 2019-Jan-20, GRNN4TE v0.2.1, compatible with pre-trained BERT
* 2018-Nov-03, GRNN4TE v0.2.0, enhance joint representation learning
* 2018-Jun-10, GRNN v0.1, initial version, build a new framework, graph recursive neural network, for graph data processing and lifelong relation extraction

