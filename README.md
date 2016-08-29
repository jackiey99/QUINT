#  QUINT: On Query-Specific Optimal Networks
## Overview

The package contains the following files:
- Astro50.mat: Astro dataset, 50% of edges as training, the rest as testing (see astro_train.txt and astro_test.txt for the edge list)
- Astro_seeds5p5n_negposnonbr.mat: the query nodes and their positive and negative nodes
- QUINT_Basic1st.m: QUINT basic version with first-order Taylor expansion
- QUINT_rankOne.m: QUINT rank-one version
- FB_LinkPred_ProSIN.m: comparison baseline -- ProSIN
- computeAP.m: compute the average precision
- computeHLU.m: compute HLU score
- computePR.m: compute percentile ranking score
- precisionAtK.m: compute precision@K
- recallAtK.m: compute recall@K
- extractQColst.m: extract columns of Q using Taylor expansion
- BLin_W2P.m: normalize the matrix

## Usage
Please refer to the demo code demo.m and comments in each file for the detailed information

## Refereces
Liangyue Li, Yuan Yao, Jie Tang, Wei Fan, Hanghang Tong. QUINT: On Query-Specific Optimal Networks. 22nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (**KDD**), 2016. (*Full Presentation*)
