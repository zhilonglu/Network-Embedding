# Network-Embedding
Network embedding, which is similar to graph embedding and network representation learning(NRL).
All related open resources about network embedding are listed including published benchmark datasets, evaluation methods, and open source algorithms.

## Reference Papers<br>
* Network Representation Learning: A Survey(2018 arXiv)
* Representation Learning on Graphs: Methods and Applications(2017 arXiv)
* A Survey on Network Embedding（2017 arXiv)
* Graph Embedding Techniques, Applications, and Performance: A Survey（2017 arXiv）
* A Comprehensive Survey of Graph Embedding-problems, Techniques and Applications（2017 TKDE）

## Challenges
* High computational complexity
* Low parallelizability
* Inapplicability of machine learning methods


## NRL Alorithms
* Matrix Factorization
  * Social Dim
  * Laplacian Eigenmaps
  * Graph Factorization
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

## Optimization Method
* Eigen Decomposition
* Stochastic Gradient Descent
* Alternative Optimization
* Gradient Descent

## Application
* Knowledge graph related
  * Learning structured embeddings of knowledge bases,AAAI 2011
  * Knowledge graph embedding by translating on hyperplanes,AAAI 2014
  * A semantic matching energy function for learning with multi-relational data application to word-sense disambiguation,Machine Learning 2014
  * Structured embedding via pairwise relations and long-range interactions in knowledge base,AAAI 2015
  * Learning entity and relation embeddings for knowledge graph completion,AAAI 2015
  * Rdf2vec: Rdf graph embeddings for data mining,ISWC 2016
  * Multi-modal bayesian embeddings for learning social knowledge graphs,IJCAI 2016
  * Label noise reduction in entity typing by heterogeneous partial-label embedding,KDD 2016
  * GAKE: Graph aware knowledge embedding,COLING 2016
  * Revisiting semisupervised learning with graph embeddings,ICML 2016
  * Explicit semantic ranking for academic search via knowledge graph embedding,WWW 2017
  * Proje: Embedding projection for knowledge graph completion,AAAI 2017
* Multimedia network related
* Information propagation related
* Social networks alignment
* Recommender Systems
* Image related
* Natural Language Processing
* Network Reconstruction
* Node Classification
  * A comparison of event models for naive bayes text classification,AAAI-98 workshop on learning for text categorization 1998
  * Iterative classification in relational data,Workshop on Learning Statistical Models from Relational Data 2000
  * Link-based classification,ICML 2003
  * Applying link-based classification to label blogs,WebKDD 2007
  * The rendezvous algorithm: Multiclass semi-supervised learning with markov random walks,ICML 2007
  * Combining content and link for classification using matrix factorization,SIGIR 2007
  * Video suggestion and discovery for youtube:taking random walks through the view graph,WWW 2008
  * Collective classification in network data,AI Magazine 2008
  * Hypergraph spectral learning for multilabel classification,KDD 2008
  * Node classification in social networks,Social network data analytics,Springer 2011
  * Label-dependent node classification in the network,Neurocomputing 2012
  * Applied logistic regression,John Wiley & Sons 2013
  * Probabilistic latent document network embedding,ICDM 2014
  * Spherical and hyperbolic embeddings of data,TPAMI 2014
  * DeepWalk: Online learning of social representations, KDD 2014
  * Heterogeneous network embedding via deep architectures,KDD 2015
  * LINE:Large-scale information network embedding,WWW 2015
  * Heterogeneous network embedding via deep architectures,KDD 2015
  * Network representation learning with rich text information,IJCAI 2015
  * Grarep: Learning graph representations with global structural information,CIKM 2015
  * Pte: Predictive text embedding through large-scale heterogeneous text networks,KDD 2015
  * Homophily, structure,and content augmented network representation learning,ICDM 2016
  * node2vec: Scalable feature learning for networks,KDD 2016
  * Collective classification via discriminative matrix factorization on sparsely labeled networks,CIKM 2016
  * Learning from collective intelligence: Feature learning using social images and tags,TOMCCAP 2016
  * Large-scale embedding learning in heterogeneous event data,ICDM 2016
  * Discriminative deep random walk for network classification,ACL 2016
  * Max-Margin DeepWalk:discriminative learning of network representation,IJCAI 2016
  * TLINE: scalable transductive network embedding,Information Retrieval Technology 2016
  * Revisiting semisupervised learning with graph embeddings,ICML 2016
  * Structural deep network embedding,KDD 2016
  * Tri-party deep network representation,IJCAI 2016
  * Partially supervised graph embedding for positive unlabelled feature selection,IJCAI 2016
  * Predicting user’s multi-interests with network embedding in health-related topics,IJCNN 2016
  * Inductive representation learning on large graphs,arXiv 2017
  * Semi-supervised classification with graph convolutional networks,ICLR 2017
  * Community preserving network embedding,AAAI 2017
  * Label informed attributed network embedding,WSDM 2017
  * Incorporating knowledge graph embeddings into topic modeling,AAAI 2017
  * Unsupervised and scalable algorithm for learning node representations,ICLR 2017
  * SSP: semantic space projection for knowledge graph embedding with text descriptions,AAAI 2017
  * Geometric deep learning on graphs and manifolds using mixture model cnns,CVPR 2017
  * RSDNE: Exploring Relaxed Similarity and Dissimilarity from Completely-imbalanced Labels for Network Embedding, AAAI 2018
  * Network Embedding with Completely-imbalanced Labels, TKDE 2020.
* Link Prediction
* Node Clustering
* Network Visualization
* traffic related
  * Sparse-Representation-Based Graph Embedding for Traffic Sign Recognition,TITS 2012
  * Representation learning for geospatial areas using large-scale mobility data from smart card,UbiComp 2016
  * Geospatial area embedding based on the movement purpose hypothesis using large-scale mobility data from smart card,IJCNS 2016


## Source Code<br>
* [DeepWalk](https://github.com/phanein/deepwalk "DeepWalk")
* [LINE](https://github.com/tangjianpku/LINE "LINE")
* [GraRep](https://github.com/ShelsonCao/GraRep "GraRep")
* [DNGR](https://github.com/ShelsonCao/DNGR "DNGR")
* [SDNE](https://github.com/suanrong/SDNE "SDNE")
* [node2vec](https://github.com/aditya-grover/node2vec "nodd2vec")
* [RSDNE](https://github.com/zhengwang100/RSDNE-python)
* [RECT](https://github.com/zhengwang100/RECT)
* [M-NMF](http://git.thumedia.org/embedding/M-NMF "M-NMF")
* [GED](https://users.ece.cmu.edu/˜sihengc/publications.html "GED")
* [Ou](http://nrl.thumedia.org/non-transitive-hashing-with-latent-similarity-components "Ou")
* [HOPE](http://nrl.thumedia.org/asymmetric-transitivity-preserving-graph-embedding "HOPE")
* [GraphWave](http://snap.stanford.edu/graphwave "GraphWave")
* [TADW](https://github.com/thunlp/tadw "TADW")
* [HSCA](https://github.com/daokunzhang/HSCA "HSCA")
* [MMDW](https://github.com/thunlp/MMDW "MMDW")
* [TriDNR](https://github.com/shiruipan/TriDNR "TriDNR")
* [DMF_CC](https://github.com/daokunzhang/DMF_CC "DMF_CC")
* [Palnetoid](https://github.com/kimiyoung/planetoid "Palnetoid")
* [Information diffusion](https://github.com/ludc/social_network_diffusion_embeddings "Information diffusion")
* [Cascade prediction](https://github.com/chengli-um/DeepCas "Cascade prediction")
* [Anomaly detection](https://github.com/hurenjun/EmbeddingAnomalyDetection "Anomaly detection")
* [Collaboration prediction](https://github.com/chentingpc/GuidedHeteEmbedding "Collaboration prediction")
* [open-source Python library GEM](https://github.com/palash1992/GEM "GEM")

## Benchmark Dataset<br>
* Synthetic
  * SYN-SBM(Stochastic blockmodels for directed graphs Journal of the American Statistical Association 82 (397) (1987) 8–19.) 
  
* Social Network
  * [BlogCatalog](http://socialcomputing.asu.edu/datasets/BlogCatalog3 "BlogCatalog")
  * KARATE(An information flow model for conflict and fission in small groups, Journal of anthropological research 33 (4) (1977).)
  * [Flickr](http://socialcomputing.asu.edu/datasets/Flickr "Flickr")
  * [YouTube](http://socialcomputing.asu.edu/datasets/YouTube2 "YouTube")
  * [Twitter](http://socialcomputing.asu.edu/datasets/Twitter "Twitter")
  * [FaceBook](https://snap.stanford.edu/data/egonets-Facebook.html "FaceBook")
  * [Amherst](https://escience.rpi.edu/data/DA/fb100/ "Amherst")
  * [Hamilton](https://escience.rpi.edu/data/DA/fb100/ "Hamilton")
  * [Mich](https://escience.rpi.edu/data/DA/fb100/ "Mich")
  * [Rochester](https://escience.rpi.edu/data/DA/fb100/ "Rochester")
* Language Network
  * [Wikipedia](http://www.mattmahoney.net/dc/textdata "Wikipedia")
* Citation Network
  * [DBLP](http://arnetminer.org/citation "DBLP")
  * DBLP(PsperCitation)(LINE:Large-scale information network embedding && Arnetminer:extraction and mining of academic social networks)
  * DBLP(AuthorCitation)(LINE:Large-scale information network embedding && Arnetminer:extraction and mining of academic social networks)
  * [Cora](https://linqs.soe.ucsc.edu/data "Cora")
  * [Citeseer](https://linqs.soe.ucsc.edu/data "Citeseer")
  * [ArXiv](http://snap.stanford.edu/data/ca-AstroPh.html "ArXiv")
  * [PubMed](https://linqs.soe.ucsc.edu/data "PubMed")
  * [Citeseer-M10](http://citeseerx.ist.psu.edu/ "Citeseer-M10")
* Collaboration network
  * Arxiv GR-QC(Graph evolution:densification and shrinking diameters)
  * HEP-TH(Overview of the 2003 kdd cup, ACM SIGKDD Explorations 5 (2).)
  * [ASTRO-PH](http://snap.stanford.edu/data "ASTRO-PH")
* Webpage Network
  * [Wikipedia](https://linqs.soe.ucsc.edu/data "Wikipedia")
  * [WebKB](https://linqs.soe.ucsc.edu/data "WebKB")
  * Political Blog(The political blogosphere and the 2004 US election: divided they blog)
* Biological Network
  * [Protein-Protein Interaction(PPI)](http://konect.uni-koblenz.de/networks/maayan-vidal "PPI")
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
* More Structures and Properties
* The Effect of Side Information
* More Advanced Information and Tasks
* Dynamic Network Embedding
* More embedding spaces
* Decoding higher-order motifs
* Reasoning about large sets of candidate subgraphs
* Improving interpretability

## Related List
* [awesome-network-embedding](https://github.com/chihming/awesome-network-embedding)
* [awesome-graph-embedding](https://github.com/benedekrozemberczki/awesome-graph-embedding)
* [awesome-embedding-models](https://github.com/Hironsan/awesome-embedding-models)
* [Must-read papers on network representation learning (NRL) / network embedding (NE)](https://github.com/thunlp/NRLPapers)
* [Must-read papers on knowledge representation learning (KRL) / knowledge embedding (KE)](https://github.com/thunlp/KRLPapers)
* [Network Embedding Resources](https://github.com/nate-russell/Network-Embedding-Resources)
* [2vec-type embedding models](https://github.com/MaxwellRebo/awesome-2vec)
* [An Open-Source Package for Network Embedding (OpenNE)](https://github.com/thunlp/OpenNE)
* [An Open-Source Package for Knowledge Embedding (KE)](https://github.com/thunlp/OpenKE)
