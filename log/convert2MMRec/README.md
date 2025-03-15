<h2>🚀Note: </h2>
Existing multimodal recommendation systems are generally implemented based on two frameworks. 
One category includes LATTICE [[1](https://dl.acm.org/doi/abs/10.1145/3474085.3475259)] and MMSSL [[2](https://arxiv.org/pdf/2104.09036)] (our method is also built upon this framework), 
while the other is MMRec [[3](https://dl.acm.org/doi/pdf/10.1145/3611380.3628561 )] (an open-source framework integrating various multimodal recommendation algorithms, which LGMRec[[4](https://ojs.aaai.org/index.php/AAAI/article/view/28688)] is based). 

The fundamental difference between these two frameworks lies in their underlying data processing strategies. Specifically, upon examining their open-source implementations, we observe that LATTICE, as the earliest multimodal algorithm framework, does not sort interactions between users and items when processing the Amazon dataset. Instead, it uses random sampling to partition the dataset into training, validation, and test sets with a ratio of 8:1:1, subsequently organizing the processed data in JSON format. On the other hand, MMRec sorts user-item interactions according to timestamps and partitions the dataset chronologically, structuring the data into ".inter" files. In this ".inter" format, each data entry consists of ["userID", "itemID", "x_label"], where "userID" and "itemID" indicate the identifiers for the user and the item, respectively, and "x_label" serves as a label (with 0 indicating training data, 1 indicating validation data, and 2 indicating test data). 

Consequently, before conducting experiments, it is necessary to convert datasets processed by LATTICE and MMSSL frameworks into the corresponding format required by MMRec to utilize all the methods included therein. Through multiple experiments, we found that frameworks based on MMRec methods generally outperform those built on LATTICE-based methods, which is understandable due to the chronological sorting of the dataset. In our experiment, since we initially built our framework based on the MMSSL model, data conversion was essential. 

The specific conversion steps and corresponding code are available in the notebook "convert.ipynb", which converts JSON-formatted data into the ".inter" format required by the MMRec framework. For more details, please refer to the code. If you have any questions, feel free to discuss or contact me.

[1] Zhang J, Zhu Y, Liu Q, et al. Mining latent structures for multimedia recommendation[C]//Proceedings of the 29th ACM international conference on multimedia. 2021: 3872-3880.

paper: https://arxiv.org/pdf/2104.09036

code: https://github.com/CRIPAC-DIG/LATTICE

[2] Wei W, Huang C, Xia L, et al. Multi-modal self-supervised learning for recommendation[C]//Proceedings of the ACM Web Conference 2023. 2023: 790-800.


paper: https://arxiv.org/pdf/2302.10632

code: https://github.com/HKUDS/MMSSL

[3] Zhou X. Mmrec: Simplifying multimodal recommendation[C]//Proceedings of the 5th ACM International Conference on Multimedia in Asia Workshops. 2023: 1-2.

paper: https://dl.acm.org/doi/pdf/10.1145/3611380.3628561 

code: https://github.com/enoche/MMRec

[4] Guo Z, Li J, Li G, et al. LGMRec: local and global graph learning for multimodal recommendation[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2024, 38(8): 8454-8462.

paper: https://ojs.aaai.org/index.php/AAAI/article/view/28688

code: https://github.com/enoche/MMRec/blob/master/src/models/lgmrec.py (based MMRec)
