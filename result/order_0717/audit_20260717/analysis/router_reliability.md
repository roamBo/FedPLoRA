# FedPLoRA router reliability audit

- root: `/data/yaominghao/gb/result/FedPLoRA`
- runs: 18

## Run-level summary

| method | seed | Local | NMI | ARI | K | sil | mixed_clusters | minority_clients | domain_cluster_counts |
|---|---|---|---|---|---|---|---|---|---|
| NX1_v13a_os_split43_train43 | 42 | 0.5998 | 0.9311 | 0.8561 | 8 | 0.1752 | 1 | 2 | {"code": 1, "education": 1, "finance": 1, "general": 3, "legal": 1, "math": 1, "medical": 1} |
| NX1_v13a_os_split44_train44 | 42 | 0.6032 | 0.9820 | 0.9668 | 8 | 0.1665 | 0 | 0 | {"code": 1, "education": 1, "finance": 1, "general": 2, "legal": 1, "math": 1, "medical": 1} |
| NX1_v13b_bonly_split43_train43 | 42 | 0.5941 | 0.8651 | 0.6932 | 8 | 0.2386 | 2 | 7 | {"code": 1, "education": 1, "finance": 1, "general": 4, "legal": 1, "math": 1, "medical": 1} |
| NX1_v13b_bonly_split44_train44 | 42 | 0.5996 | 0.9820 | 0.9668 | 8 | 0.1982 | 0 | 0 | {"code": 1, "education": 1, "finance": 1, "general": 2, "legal": 1, "math": 1, "medical": 1} |
| NX1_v13a_os_split43_train43 | 43 | 0.6002 | 0.9260 | 0.8303 | 8 | 0.1940 | 1 | 3 | {"code": 1, "education": 1, "finance": 1, "general": 3, "legal": 1, "math": 1, "medical": 1} |
| NX1_v13b_bonly_split43_train43 | 43 | 0.5941 | 0.8651 | 0.6932 | 8 | 0.2313 | 2 | 7 | {"code": 1, "education": 1, "finance": 1, "general": 4, "legal": 1, "math": 1, "medical": 1} |
| NX1_v13a_os_split44_train44 | 44 | 0.6083 | 0.9820 | 0.9668 | 8 | 0.1598 | 0 | 0 | {"code": 1, "education": 1, "finance": 1, "general": 2, "legal": 1, "math": 1, "medical": 1} |
| NX1_v13b_bonly_split44_train44 | 44 | 0.6086 | 0.9820 | 0.9668 | 8 | 0.6512 | 0 | 0 | {"code": 1, "education": 1, "finance": 1, "general": 2, "legal": 1, "math": 1, "medical": 1} |
| NX3_v11c_mu020_split42_train42 | 42 | 0.6050 | 0.9820 | 0.9668 | 8 | 0.1742 | 0 | 0 | {"code": 1, "education": 1, "finance": 1, "general": 2, "legal": 1, "math": 1, "medical": 1} |
| NX3_v11a_alpha100_split42_train43 | 43 | 0.6082 | 0.9820 | 0.9668 | 8 | 0.1727 | 0 | 0 | {"code": 1, "education": 1, "finance": 1, "general": 2, "legal": 1, "math": 1, "medical": 1} |
| NX3_v11c_mu020_split42_train44 | 44 | 0.6054 | 0.9820 | 0.9668 | 8 | 0.1675 | 0 | 0 | {"code": 1, "education": 1, "finance": 2, "general": 1, "legal": 1, "math": 1, "medical": 1} |
| smoke_v13a_os | 42 | 0.5049 | 0.7114 | 0.4492 | 8 | 0.1208 | 3 | 12 | {"code": 2, "education": 2, "finance": 2, "general": 4, "legal": 1, "math": 1, "medical": 1} |
| smoke_v13b_os_bonly | 42 | 0.5046 | 0.7114 | 0.4492 | 8 | 0.1208 | 3 | 12 | {"code": 2, "education": 2, "finance": 2, "general": 4, "legal": 1, "math": 1, "medical": 1} |
| NX0_v13a_os_split42_train42 | 42 | 0.6076 | 0.9820 | 0.9668 | 8 | 0.1742 | 0 | 0 | {"code": 1, "education": 1, "finance": 1, "general": 2, "legal": 1, "math": 1, "medical": 1} |
| NX0_v13b_bonly_split42_train42 | 42 | 0.6017 | 0.9820 | 0.9668 | 8 | 0.2077 | 0 | 0 | {"code": 1, "education": 1, "finance": 1, "general": 2, "legal": 1, "math": 1, "medical": 1} |
| smoke_ecolora | 42 | 0.5786 |  |  | 1 |  | 1 | 30 | {"code": 1, "education": 1, "finance": 1, "general": 1, "legal": 1, "math": 1, "medical": 1} |
| smoke_fedlease | 42 | 0.5772 | 0.1049 | -0.0005 | 3 | 0.1110 | 1 | 28 | {"code": 2, "education": 1, "finance": 1, "general": 2, "legal": 1, "math": 1, "medical": 1} |
| smoke_hydralora | 42 | 0.5757 | 0.5006 | 0.1386 | 8 | 0.1473 | 2 | 18 | {"code": 3, "education": 1, "finance": 3, "general": 4, "legal": 1, "math": 1, "medical": 1} |

## Suspicious clients

| method | seed | cid | domain | cluster | cluster_size | cluster_majority | purity | domain_K |
|---|---|---|---|---|---|---|---|---|
| NX1_v13a_os_split43_train43 | 42 | 15 | general | 3 | 7 | math | 0.7143 | 3 |
| NX1_v13a_os_split43_train43 | 42 | 16 | general | 4 | 2 | general | 1.0000 | 3 |
| NX1_v13a_os_split43_train43 | 42 | 17 | general | 3 | 7 | math | 0.7143 | 3 |
| NX1_v13a_os_split43_train43 | 42 | 18 | general | 5 | 1 | general | 1.0000 | 3 |
| NX1_v13a_os_split43_train43 | 42 | 19 | general | 4 | 2 | general | 1.0000 | 3 |
| NX1_v13a_os_split44_train44 | 42 | 15 | general | 3 | 1 | general | 1.0000 | 2 |
| NX1_v13a_os_split44_train44 | 42 | 16 | general | 4 | 4 | general | 1.0000 | 2 |
| NX1_v13a_os_split44_train44 | 42 | 17 | general | 4 | 4 | general | 1.0000 | 2 |
| NX1_v13a_os_split44_train44 | 42 | 18 | general | 4 | 4 | general | 1.0000 | 2 |
| NX1_v13a_os_split44_train44 | 42 | 19 | general | 4 | 4 | general | 1.0000 | 2 |
| NX1_v13b_bonly_split43_train43 | 42 | 15 | general | 3 | 7 | math | 0.7143 | 4 |
| NX1_v13b_bonly_split43_train43 | 42 | 16 | general | 4 | 1 | general | 1.0000 | 4 |
| NX1_v13b_bonly_split43_train43 | 42 | 17 | general | 3 | 7 | math | 0.7143 | 4 |
| NX1_v13b_bonly_split43_train43 | 42 | 18 | general | 5 | 1 | general | 1.0000 | 4 |
| NX1_v13b_bonly_split43_train43 | 42 | 19 | general | 6 | 1 | general | 1.0000 | 4 |
| NX1_v13b_bonly_split43_train43 | 42 | 30 | medical | 2 | 10 | finance | 0.5000 | 1 |
| NX1_v13b_bonly_split43_train43 | 42 | 31 | medical | 2 | 10 | finance | 0.5000 | 1 |
| NX1_v13b_bonly_split43_train43 | 42 | 32 | medical | 2 | 10 | finance | 0.5000 | 1 |
| NX1_v13b_bonly_split43_train43 | 42 | 33 | medical | 2 | 10 | finance | 0.5000 | 1 |
| NX1_v13b_bonly_split43_train43 | 42 | 34 | medical | 2 | 10 | finance | 0.5000 | 1 |
| NX1_v13b_bonly_split44_train44 | 42 | 15 | general | 3 | 1 | general | 1.0000 | 2 |
| NX1_v13b_bonly_split44_train44 | 42 | 16 | general | 4 | 4 | general | 1.0000 | 2 |
| NX1_v13b_bonly_split44_train44 | 42 | 17 | general | 4 | 4 | general | 1.0000 | 2 |
| NX1_v13b_bonly_split44_train44 | 42 | 18 | general | 4 | 4 | general | 1.0000 | 2 |
| NX1_v13b_bonly_split44_train44 | 42 | 19 | general | 4 | 4 | general | 1.0000 | 2 |
| NX1_v13a_os_split43_train43 | 43 | 15 | general | 3 | 8 | math | 0.6250 | 3 |
| NX1_v13a_os_split43_train43 | 43 | 16 | general | 4 | 1 | general | 1.0000 | 3 |
| NX1_v13a_os_split43_train43 | 43 | 17 | general | 3 | 8 | math | 0.6250 | 3 |
| NX1_v13a_os_split43_train43 | 43 | 18 | general | 3 | 8 | math | 0.6250 | 3 |
| NX1_v13a_os_split43_train43 | 43 | 19 | general | 5 | 1 | general | 1.0000 | 3 |
| NX1_v13b_bonly_split43_train43 | 43 | 15 | general | 3 | 7 | math | 0.7143 | 4 |
| NX1_v13b_bonly_split43_train43 | 43 | 16 | general | 4 | 1 | general | 1.0000 | 4 |
| NX1_v13b_bonly_split43_train43 | 43 | 17 | general | 3 | 7 | math | 0.7143 | 4 |
| NX1_v13b_bonly_split43_train43 | 43 | 18 | general | 5 | 1 | general | 1.0000 | 4 |
| NX1_v13b_bonly_split43_train43 | 43 | 19 | general | 6 | 1 | general | 1.0000 | 4 |
| NX1_v13b_bonly_split43_train43 | 43 | 30 | medical | 2 | 10 | finance | 0.5000 | 1 |
| NX1_v13b_bonly_split43_train43 | 43 | 31 | medical | 2 | 10 | finance | 0.5000 | 1 |
| NX1_v13b_bonly_split43_train43 | 43 | 32 | medical | 2 | 10 | finance | 0.5000 | 1 |
| NX1_v13b_bonly_split43_train43 | 43 | 33 | medical | 2 | 10 | finance | 0.5000 | 1 |
| NX1_v13b_bonly_split43_train43 | 43 | 34 | medical | 2 | 10 | finance | 0.5000 | 1 |
| NX1_v13a_os_split44_train44 | 44 | 15 | general | 3 | 1 | general | 1.0000 | 2 |
| NX1_v13a_os_split44_train44 | 44 | 16 | general | 4 | 4 | general | 1.0000 | 2 |
| NX1_v13a_os_split44_train44 | 44 | 17 | general | 4 | 4 | general | 1.0000 | 2 |
| NX1_v13a_os_split44_train44 | 44 | 18 | general | 4 | 4 | general | 1.0000 | 2 |
| NX1_v13a_os_split44_train44 | 44 | 19 | general | 4 | 4 | general | 1.0000 | 2 |
| NX1_v13b_bonly_split44_train44 | 44 | 15 | general | 3 | 1 | general | 1.0000 | 2 |
| NX1_v13b_bonly_split44_train44 | 44 | 16 | general | 4 | 4 | general | 1.0000 | 2 |
| NX1_v13b_bonly_split44_train44 | 44 | 17 | general | 4 | 4 | general | 1.0000 | 2 |
| NX1_v13b_bonly_split44_train44 | 44 | 18 | general | 4 | 4 | general | 1.0000 | 2 |
| NX1_v13b_bonly_split44_train44 | 44 | 19 | general | 4 | 4 | general | 1.0000 | 2 |
| NX3_v11c_mu020_split42_train42 | 42 | 15 | general | 3 | 4 | general | 1.0000 | 2 |
| NX3_v11c_mu020_split42_train42 | 42 | 16 | general | 3 | 4 | general | 1.0000 | 2 |
| NX3_v11c_mu020_split42_train42 | 42 | 17 | general | 3 | 4 | general | 1.0000 | 2 |
| NX3_v11c_mu020_split42_train42 | 42 | 18 | general | 4 | 1 | general | 1.0000 | 2 |
| NX3_v11c_mu020_split42_train42 | 42 | 19 | general | 3 | 4 | general | 1.0000 | 2 |
| NX3_v11a_alpha100_split42_train43 | 43 | 15 | general | 3 | 4 | general | 1.0000 | 2 |
| NX3_v11a_alpha100_split42_train43 | 43 | 16 | general | 3 | 4 | general | 1.0000 | 2 |
| NX3_v11a_alpha100_split42_train43 | 43 | 17 | general | 4 | 1 | general | 1.0000 | 2 |
| NX3_v11a_alpha100_split42_train43 | 43 | 18 | general | 3 | 4 | general | 1.0000 | 2 |
| NX3_v11a_alpha100_split42_train43 | 43 | 19 | general | 3 | 4 | general | 1.0000 | 2 |
| NX3_v11c_mu020_split42_train44 | 44 | 10 | finance | 2 | 4 | finance | 1.0000 | 2 |
| NX3_v11c_mu020_split42_train44 | 44 | 11 | finance | 2 | 4 | finance | 1.0000 | 2 |
| NX3_v11c_mu020_split42_train44 | 44 | 12 | finance | 3 | 1 | finance | 1.0000 | 2 |
| NX3_v11c_mu020_split42_train44 | 44 | 13 | finance | 2 | 4 | finance | 1.0000 | 2 |
| NX3_v11c_mu020_split42_train44 | 44 | 14 | finance | 2 | 4 | finance | 1.0000 | 2 |
| smoke_v13a_os | 42 | 0 | code | 0 | 6 | code | 0.6667 | 2 |
| smoke_v13a_os | 42 | 1 | code | 0 | 6 | code | 0.6667 | 2 |
| smoke_v13a_os | 42 | 2 | code | 0 | 6 | code | 0.6667 | 2 |
| smoke_v13a_os | 42 | 3 | code | 0 | 6 | code | 0.6667 | 2 |
| smoke_v13a_os | 42 | 4 | code | 1 | 1 | code | 1.0000 | 2 |
| smoke_v13a_os | 42 | 5 | education | 2 | 10 | math | 0.5000 | 2 |
| smoke_v13a_os | 42 | 6 | education | 2 | 10 | math | 0.5000 | 2 |
| smoke_v13a_os | 42 | 7 | education | 2 | 10 | math | 0.5000 | 2 |
| smoke_v13a_os | 42 | 8 | education | 2 | 10 | math | 0.5000 | 2 |
| smoke_v13a_os | 42 | 9 | education | 3 | 1 | education | 1.0000 | 2 |
| smoke_v13a_os | 42 | 10 | finance | 4 | 10 | medical | 0.5000 | 2 |
| smoke_v13a_os | 42 | 11 | finance | 4 | 10 | medical | 0.5000 | 2 |
| smoke_v13a_os | 42 | 12 | finance | 5 | 1 | finance | 1.0000 | 2 |
| smoke_v13a_os | 42 | 13 | finance | 4 | 10 | medical | 0.5000 | 2 |
| smoke_v13a_os | 42 | 14 | finance | 4 | 10 | medical | 0.5000 | 2 |
| smoke_v13a_os | 42 | 15 | general | 2 | 10 | math | 0.5000 | 4 |
| smoke_v13a_os | 42 | 16 | general | 0 | 6 | code | 0.6667 | 4 |
| smoke_v13a_os | 42 | 17 | general | 6 | 1 | general | 1.0000 | 4 |
| smoke_v13a_os | 42 | 18 | general | 0 | 6 | code | 0.6667 | 4 |
| smoke_v13a_os | 42 | 19 | general | 4 | 10 | medical | 0.5000 | 4 |
| smoke_v13b_os_bonly | 42 | 0 | code | 0 | 6 | code | 0.6667 | 2 |
| smoke_v13b_os_bonly | 42 | 1 | code | 0 | 6 | code | 0.6667 | 2 |
| smoke_v13b_os_bonly | 42 | 2 | code | 0 | 6 | code | 0.6667 | 2 |
| smoke_v13b_os_bonly | 42 | 3 | code | 0 | 6 | code | 0.6667 | 2 |
| smoke_v13b_os_bonly | 42 | 4 | code | 1 | 1 | code | 1.0000 | 2 |
| smoke_v13b_os_bonly | 42 | 5 | education | 2 | 10 | math | 0.5000 | 2 |
| smoke_v13b_os_bonly | 42 | 6 | education | 2 | 10 | math | 0.5000 | 2 |
| smoke_v13b_os_bonly | 42 | 7 | education | 2 | 10 | math | 0.5000 | 2 |
| smoke_v13b_os_bonly | 42 | 8 | education | 2 | 10 | math | 0.5000 | 2 |
| smoke_v13b_os_bonly | 42 | 9 | education | 3 | 1 | education | 1.0000 | 2 |
| smoke_v13b_os_bonly | 42 | 10 | finance | 4 | 10 | medical | 0.5000 | 2 |
| smoke_v13b_os_bonly | 42 | 11 | finance | 4 | 10 | medical | 0.5000 | 2 |
| smoke_v13b_os_bonly | 42 | 12 | finance | 5 | 1 | finance | 1.0000 | 2 |
| smoke_v13b_os_bonly | 42 | 13 | finance | 4 | 10 | medical | 0.5000 | 2 |
| smoke_v13b_os_bonly | 42 | 14 | finance | 4 | 10 | medical | 0.5000 | 2 |
| smoke_v13b_os_bonly | 42 | 15 | general | 2 | 10 | math | 0.5000 | 4 |
| smoke_v13b_os_bonly | 42 | 16 | general | 0 | 6 | code | 0.6667 | 4 |
| smoke_v13b_os_bonly | 42 | 17 | general | 6 | 1 | general | 1.0000 | 4 |
| smoke_v13b_os_bonly | 42 | 18 | general | 0 | 6 | code | 0.6667 | 4 |
| smoke_v13b_os_bonly | 42 | 19 | general | 4 | 10 | medical | 0.5000 | 4 |
| NX0_v13a_os_split42_train42 | 42 | 15 | general | 3 | 4 | general | 1.0000 | 2 |
| NX0_v13a_os_split42_train42 | 42 | 16 | general | 3 | 4 | general | 1.0000 | 2 |
| NX0_v13a_os_split42_train42 | 42 | 17 | general | 3 | 4 | general | 1.0000 | 2 |
| NX0_v13a_os_split42_train42 | 42 | 18 | general | 4 | 1 | general | 1.0000 | 2 |
| NX0_v13a_os_split42_train42 | 42 | 19 | general | 3 | 4 | general | 1.0000 | 2 |
| NX0_v13b_bonly_split42_train42 | 42 | 15 | general | 3 | 4 | general | 1.0000 | 2 |
| NX0_v13b_bonly_split42_train42 | 42 | 16 | general | 3 | 4 | general | 1.0000 | 2 |
| NX0_v13b_bonly_split42_train42 | 42 | 17 | general | 3 | 4 | general | 1.0000 | 2 |
| NX0_v13b_bonly_split42_train42 | 42 | 18 | general | 4 | 1 | general | 1.0000 | 2 |
| NX0_v13b_bonly_split42_train42 | 42 | 19 | general | 3 | 4 | general | 1.0000 | 2 |
| smoke_ecolora | 42 | 5 | education | 0 | 35 | code | 0.1429 | 1 |
| smoke_ecolora | 42 | 6 | education | 0 | 35 | code | 0.1429 | 1 |
| smoke_ecolora | 42 | 7 | education | 0 | 35 | code | 0.1429 | 1 |
| smoke_ecolora | 42 | 8 | education | 0 | 35 | code | 0.1429 | 1 |
| smoke_ecolora | 42 | 9 | education | 0 | 35 | code | 0.1429 | 1 |
| smoke_ecolora | 42 | 10 | finance | 0 | 35 | code | 0.1429 | 1 |
| smoke_ecolora | 42 | 11 | finance | 0 | 35 | code | 0.1429 | 1 |
| smoke_ecolora | 42 | 12 | finance | 0 | 35 | code | 0.1429 | 1 |
| smoke_ecolora | 42 | 13 | finance | 0 | 35 | code | 0.1429 | 1 |
| smoke_ecolora | 42 | 14 | finance | 0 | 35 | code | 0.1429 | 1 |
| smoke_ecolora | 42 | 15 | general | 0 | 35 | code | 0.1429 | 1 |
| smoke_ecolora | 42 | 16 | general | 0 | 35 | code | 0.1429 | 1 |
| smoke_ecolora | 42 | 17 | general | 0 | 35 | code | 0.1429 | 1 |
| smoke_ecolora | 42 | 18 | general | 0 | 35 | code | 0.1429 | 1 |
| smoke_ecolora | 42 | 19 | general | 0 | 35 | code | 0.1429 | 1 |
| smoke_ecolora | 42 | 20 | legal | 0 | 35 | code | 0.1429 | 1 |
| smoke_ecolora | 42 | 21 | legal | 0 | 35 | code | 0.1429 | 1 |
| smoke_ecolora | 42 | 22 | legal | 0 | 35 | code | 0.1429 | 1 |
| smoke_ecolora | 42 | 23 | legal | 0 | 35 | code | 0.1429 | 1 |
| smoke_ecolora | 42 | 24 | legal | 0 | 35 | code | 0.1429 | 1 |
| smoke_ecolora | 42 | 25 | math | 0 | 35 | code | 0.1429 | 1 |
| smoke_ecolora | 42 | 26 | math | 0 | 35 | code | 0.1429 | 1 |
| smoke_ecolora | 42 | 27 | math | 0 | 35 | code | 0.1429 | 1 |
| smoke_ecolora | 42 | 28 | math | 0 | 35 | code | 0.1429 | 1 |
| smoke_ecolora | 42 | 29 | math | 0 | 35 | code | 0.1429 | 1 |
| smoke_ecolora | 42 | 30 | medical | 0 | 35 | code | 0.1429 | 1 |
| smoke_ecolora | 42 | 31 | medical | 0 | 35 | code | 0.1429 | 1 |
| smoke_ecolora | 42 | 32 | medical | 0 | 35 | code | 0.1429 | 1 |
| smoke_ecolora | 42 | 33 | medical | 0 | 35 | code | 0.1429 | 1 |
| smoke_ecolora | 42 | 34 | medical | 0 | 35 | code | 0.1429 | 1 |
| smoke_fedlease | 42 | 0 | code | 0 | 33 | education | 0.1515 | 2 |
| smoke_fedlease | 42 | 1 | code | 0 | 33 | education | 0.1515 | 2 |
| smoke_fedlease | 42 | 2 | code | 0 | 33 | education | 0.1515 | 2 |
| smoke_fedlease | 42 | 3 | code | 1 | 1 | code | 1.0000 | 2 |
| smoke_fedlease | 42 | 4 | code | 0 | 33 | education | 0.1515 | 2 |
| smoke_fedlease | 42 | 10 | finance | 0 | 33 | education | 0.1515 | 1 |
| smoke_fedlease | 42 | 11 | finance | 0 | 33 | education | 0.1515 | 1 |
| smoke_fedlease | 42 | 12 | finance | 0 | 33 | education | 0.1515 | 1 |
| smoke_fedlease | 42 | 13 | finance | 0 | 33 | education | 0.1515 | 1 |
| smoke_fedlease | 42 | 14 | finance | 0 | 33 | education | 0.1515 | 1 |
| smoke_fedlease | 42 | 15 | general | 0 | 33 | education | 0.1515 | 2 |
| smoke_fedlease | 42 | 16 | general | 2 | 1 | general | 1.0000 | 2 |
| smoke_fedlease | 42 | 17 | general | 0 | 33 | education | 0.1515 | 2 |
| smoke_fedlease | 42 | 18 | general | 0 | 33 | education | 0.1515 | 2 |
| smoke_fedlease | 42 | 19 | general | 0 | 33 | education | 0.1515 | 2 |
| smoke_fedlease | 42 | 20 | legal | 0 | 33 | education | 0.1515 | 1 |
| smoke_fedlease | 42 | 21 | legal | 0 | 33 | education | 0.1515 | 1 |
| smoke_fedlease | 42 | 22 | legal | 0 | 33 | education | 0.1515 | 1 |
| smoke_fedlease | 42 | 23 | legal | 0 | 33 | education | 0.1515 | 1 |
| smoke_fedlease | 42 | 24 | legal | 0 | 33 | education | 0.1515 | 1 |
| smoke_fedlease | 42 | 25 | math | 0 | 33 | education | 0.1515 | 1 |
| smoke_fedlease | 42 | 26 | math | 0 | 33 | education | 0.1515 | 1 |
| smoke_fedlease | 42 | 27 | math | 0 | 33 | education | 0.1515 | 1 |
| smoke_fedlease | 42 | 28 | math | 0 | 33 | education | 0.1515 | 1 |
| smoke_fedlease | 42 | 29 | math | 0 | 33 | education | 0.1515 | 1 |
| smoke_fedlease | 42 | 30 | medical | 0 | 33 | education | 0.1515 | 1 |
| smoke_fedlease | 42 | 31 | medical | 0 | 33 | education | 0.1515 | 1 |
| smoke_fedlease | 42 | 32 | medical | 0 | 33 | education | 0.1515 | 1 |
| smoke_fedlease | 42 | 33 | medical | 0 | 33 | education | 0.1515 | 1 |
| smoke_fedlease | 42 | 34 | medical | 0 | 33 | education | 0.1515 | 1 |
| smoke_hydralora | 42 | 0 | code | 0 | 22 | education | 0.2273 | 3 |
| smoke_hydralora | 42 | 1 | code | 1 | 6 | legal | 0.8333 | 3 |
| smoke_hydralora | 42 | 2 | code | 0 | 22 | education | 0.2273 | 3 |
| smoke_hydralora | 42 | 3 | code | 2 | 1 | code | 1.0000 | 3 |
| smoke_hydralora | 42 | 4 | code | 0 | 22 | education | 0.2273 | 3 |
| smoke_hydralora | 42 | 10 | finance | 3 | 1 | finance | 1.0000 | 3 |
| smoke_hydralora | 42 | 11 | finance | 0 | 22 | education | 0.2273 | 3 |
| smoke_hydralora | 42 | 12 | finance | 4 | 1 | finance | 1.0000 | 3 |
| smoke_hydralora | 42 | 13 | finance | 0 | 22 | education | 0.2273 | 3 |
| smoke_hydralora | 42 | 14 | finance | 0 | 22 | education | 0.2273 | 3 |
| smoke_hydralora | 42 | 15 | general | 0 | 22 | education | 0.2273 | 4 |
| smoke_hydralora | 42 | 16 | general | 5 | 1 | general | 1.0000 | 4 |
| smoke_hydralora | 42 | 17 | general | 6 | 1 | general | 1.0000 | 4 |
| smoke_hydralora | 42 | 18 | general | 7 | 2 | general | 1.0000 | 4 |
| smoke_hydralora | 42 | 19 | general | 7 | 2 | general | 1.0000 | 4 |
| smoke_hydralora | 42 | 25 | math | 0 | 22 | education | 0.2273 | 1 |
| smoke_hydralora | 42 | 26 | math | 0 | 22 | education | 0.2273 | 1 |
| smoke_hydralora | 42 | 27 | math | 0 | 22 | education | 0.2273 | 1 |
| smoke_hydralora | 42 | 28 | math | 0 | 22 | education | 0.2273 | 1 |
| smoke_hydralora | 42 | 29 | math | 0 | 22 | education | 0.2273 | 1 |
| smoke_hydralora | 42 | 30 | medical | 0 | 22 | education | 0.2273 | 1 |
| smoke_hydralora | 42 | 31 | medical | 0 | 22 | education | 0.2273 | 1 |
| smoke_hydralora | 42 | 32 | medical | 0 | 22 | education | 0.2273 | 1 |
| smoke_hydralora | 42 | 33 | medical | 0 | 22 | education | 0.2273 | 1 |
| smoke_hydralora | 42 | 34 | medical | 0 | 22 | education | 0.2273 | 1 |

