# ODIS-Text-to-SQL

## Description
This repo contains codes for the paper: [Selective Demonstrations for Cross-domain Text-to-SQL](https://arxiv.org/pdf/2310.06302.pdf).

## Setup
1. Please download the processed data(https://drive.google.com/file/d/1Xp_wSLZFd81gNOo-Z2sRHcVSuUQWeWfd/view?usp=drive_link) and unzip it under the root directory.
2. Install the necessary packages

```
pip install -r requirements.txt
python preprocessing.py
```


## Run ODIS with OpenAI Models
Suport "codex", "chatgpt", "chatgpt16k", "gpt4". 
```
export OPENAI_API_KEY=<your-api-key>
```

For example, to run Codex in Spider in the zero-shot setting
```
python text_to_sql.py --setting zeroshot --dataset spider --model codex
```

Run Codex in Spider with out-of-domain demonstrations
```
python text_to_sql.py --setting outdomain --dataset spider --model codex --retrieval_outdomain simsql_pred
```

Run Codex in Spider with in-domain synthetic demonstrations
```
python text_to_sql.py --setting indomain --dataset spider --model codex --retrieval_indomain covsql --synthetic_data synthetic_ship_codex_verified
```

Run Codex in Spider with both out-of-domain demonstrations and in-domain synthetic demonstrations (ODIS)
```
python text_to_sql.py --setting inoutdomain --dataset spider --model codex --retrieval_outdomain simsql_pred --retrieval_indomain covsql --synthetic_data synthetic_ship_codex_verified
```

The predictions can be found in `outputs/codex/spider/`.

## Run ODIS with Other LLMs 

Similar to above, but add the argument `--save_prompt_only`, for example

```
python text_to_sql.py --setting [setting] --dataset [dataset] --model [model] --save_prompt_only
```
This script will retrieve the demonstration examples using the initial predictions of a LLM (Codex by default but you can replace it with the predictions of another LLM if needed) and save the input prompt in `output/[dataset]/[model]/input_prompt*.json`. You can further run any LLMs with this input.



## Evaluation

We recommend using the official [test-suite evaluation scripts](https://github.com/taoyds/test-suite-sql-eval) for the execution accuracy.

## Citation and Contact

If you use our approaches or data in your work, please cite our paper and the corresponding papers to the data.

```
@article{chang2023selective,
  title={Selective Demonstrations for Cross-domain Text-to-SQL},
  author={Chang, Shuaichen and Fosler-Lussier, Eric},
  journal={Findings of EMNLP},
  year={2023}
}
@inproceedings{yu2018spider,
  title={Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task},
  author={Yu, Tao and Zhang, Rui and Yang, Kai and Yasunaga, Michihiro and Wang, Dongxu and Li, Zifan and Ma, James and Li, Irene and Yao, Qingning and Roman, Shanelle and others},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  pages={3911--3921},
  year={2018}
}
@inproceedings{lee-2021-kaggle-dbqa,
    title = "{KaggleDBQA}: Realistic Evaluation of Text-to-{SQL} Parsers",
    author = "Lee, Chia-Hsuan  and
      Polozov, Oleksandr  and
      Richardson, Matthew",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.176",
    pages = "2261--2273"
}
@article{chang2023dr,
  title={Dr. Spider: A Diagnostic Evaluation Benchmark towards Text-to-SQL Robustness},
  author={Chang, Shuaichen and Wang, Jun and Dong, Mingwen and Pan, Lin and Zhu, Henghui and Li, Alexander Hanbo and Lan, Wuwei and Zhang, Sheng and Jiang, Jiarong and Lilien, Joseph and others},
  journal={arXiv preprint arXiv:2301.08881},
  year={2023}
}
```
Please contact Shuaichen Chang (chang.1692[at]osu.edu) for questions and suggestions. Thank you!

