# Network-Embedding
Network embedding, which is similar to graph embedding and network representation learning(NRL).
All related open resources about network embedding are listed including published benchmark datasets, evaluation methods, and open source algorithms.

## Reference Papers<br>
* Network Representation Learning: A Survey(2018 arXiv)
* Representation Learning on Graphs: Methods and Applications(2017 arXiv)
* A Survey on Network Embedding（2017 arXiv)
* Graph Embedding Techniques, Applications, and Performance: A Survey（2017 arXiv）
* A Comprehensive Survey of Graph Embedding-problems, Techniques and Applications（2017 TKDE）

## NRL Alorithms
* Matrix Factorization
  * Social Dim
  * GraRep 
  * HOPE
  * GraphWave
  * M-NMF
  * TADW 
  * HSCA 
  * MMDW 
  * DMF
  * LANE
* Random Walk
  * DeepWalk
  * node2vec
  * APP
  * DDRW
  * GENE
  * TriDNR
  * UPP-SNE
  * struct2vec
  * SNS
  * PPNE
  * SemiNE
* Edge Modeling
  * LINE
  * TLINE
  * LDE
  * pRBM
  * GraphGAN
* Deep Learning
  * DNGR
  * SDNE
* Hybrid
  * DP
  * HARP
  * Planetoid

## Evaluation Method
* Network Reconstruction
* Vertex Classification
* Link Prediction
* Visualization

## Optimization Method
* Eigen Decomposition
* Stochastic Gradient Descent
* Alternative Optimization
* Gradient Descent

## Source Code<br>
* [DeepWalk](https://github.com/phanein/deepwalk "DeepWalk")
* [LINE](https://github.com/tangjianpku/LINE "LINE")
* [GraRep](https://github.com/ShelsonCao/GraRep "GraRep")
* [DNGR](https://github.com/ShelsonCao/DNGR "DNGR")
* [SDNE](https://github.com/suanrong/SDNE "SDNE")
* [node2vec](https://github.com/aditya-grover/node2vec "nodd2vec")
* [M-NMF](http://git.thumedia.org/embedding/M-NMF "M-NMF")
* [GraphWave](http://snap.stanford.edu/graphwave "GraphWave")
* [TADW](https://github.com/thunlp/tadw "TADW")
* [HSCA](https://github.com/daokunzhang/HSCA "HSCA")
* [MMDW](https://github.com/thunlp/MMDW "MMDW")
* [TriDNR](https://github.com/shiruipan/TriDNR "TriDNR")
* [DMF_CC](https://github.com/daokunzhang/DMF_CC "DMF_CC")
* [Palnetoid](https://github.com/kimiyoung/planetoid "Palnetoid")

## Benchmark Dataset<br>
* Social Network
  * [BlogCatalog](http://www.public.asu.edu/~ltang9/ "BlogCatalog")
  * [Flickr](http://socialnetworks.mpi-sws.org/data-imc2007.html "Flickr")
  * [YouTube](http://socialnetworks.mpi-sws.org/data-imc2007.html "YouTube")
  * [FaceBook](https://snap.stanford.edu/data/egonets-Facebook.html "FaceBook")
  * [Amherst](https://escience.rpi.edu/data/DA/fb100/ "Amherst"]
  * [Hamilton](https://escience.rpi.edu/data/DA/fb100/ "Hamilton"]
  * [Mich](https://escience.rpi.edu/data/DA/fb100/ "Mich"]
  * [Rochester](https://escience.rpi.edu/data/DA/fb100/ "Rochester"]
* Language Network
  * Wikipedia(LINE:Large-scale information network embedding)
* Citation Network
  * DBLP(PsperCitation)(LINE:Large-scale information network embedding && Arnetminer:extraction and mining of academic social networks)
  * DBLP(AuthorCitation)(LINE:Large-scale information network embedding && Arnetminer:extraction and mining of academic social networks)
  * [Cora](https://linqs.soe.ucsc.edu/data "Cora")
  * [Citeseer](https://linqs.soe.ucsc.edu/data "Citeseer")
  * [PubMed](https://linqs.soe.ucsc.edu/data "PubMed")
  * [Citeseer-M10](http://citeseerx.ist.psu.edu/ "Citeseer-M10")
* Collaboration network
  * Arxiv GR-QC(Graph evolution:densification and shrinking diameters)
* Webpage Network
  * [Wikipedia](https://linqs.soe.ucsc.edu/data "Wikipedia")
  * [WebKB](https://linqs.soe.ucsc.edu/data "WebKB")
  * Political Blog(The political blogosphere and the 2004 US election: divided they blog)
* Biological Network
  * Protein-Protein Interaction(The BioGRID interaction database: 2008 update)
* Communication Network
  * [Enron Email Networkg](https://snap.stanford.edu/data/email-Enron.html "Enron Email Networkg")
* Traffic Network
  * [European Airline Networksh](http://complex.unizar.es/~atnmultiplex/ "European Airline Networksh")

## Future Directions
* Task-dependence
* Theory
* Dynamics
* Scalability
* Heterogeneity and semantics
* Robustness


