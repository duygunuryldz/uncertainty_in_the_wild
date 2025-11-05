##  Reconsidering LLM Uncertainty Estimation Methods in the Wild (ACL - 2025) ([Full Paper](https://aclanthology.org/2025.acl-long.1429/))

In this work, we systematically examine four key aspects of deploying UE methods in practical settings:

(1) the sensitivity of UE methods to decision threshold selection, 

(2) their robustness to query transformations such as typos, adversarial prompts, and prior chat history, 

(3) their applicability to long-form generation, and 

(4) strategies for handling multiple UE scores for a single query. 

---

### Implementation Overview

This repository is built using the [TruthTorchLM](https://github.com/Ybakman/TruthTorchLM) library.

__1.__ Sensitivity of threshold selection implementations can be found in ```benchmark.py``` and corresponding commands are in ```run.sh```.

__2.__ Robustness to query transformations:

- Typo implementation is in ```benchmark_with_typo.py``` and corresponding commands are in ```runs_typo.sh```.
- Context implementation is in ```benchmark_with_context.py``` and corresponding commands are in ```runs_benchmark_with_context.sh```.
- Advesarial prompt implementation is in ```benchmark_with_adversarial.py``` and corresponding commands are in ```runs_benchmark_with_adversarial.sh```.

__3.__ Long-form generation experiments are implemented in ```benchmark_long_form.py```.

---

## Citation  

```bibtex
@inproceedings{bakman-etal-2025-reconsidering,
    title = "Reconsidering {LLM} Uncertainty Estimation Methods in the Wild",
    author = "Bakman, Yavuz Faruk  and
      Yaldiz, Duygu Nur  and
      Kang, Sungmin  and
      Zhang, Tuo  and
      Buyukates, Baturalp  and
      Avestimehr, Salman  and
      Karimireddy, Sai Praneeth"
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1429/",
    doi = "10.18653/v1/2025.acl-long.1429",
    pages = "29531--29556",
    ISBN = "979-8-89176-251-0"
}
```

```bibtex
@inproceedings{yaldiz-etal-2025-truthtorchlm,
    title = "{T}ruth{T}orch{LM}: A Comprehensive Library for Predicting Truthfulness in {LLM} Outputs",
    author = {Yaldiz, Duygu Nur  and
      Bakman, Yavuz Faruk  and
      Kang, Sungmin  and
      {\"O}zi{\c{s}}, Alperen  and
      Yildiz, Hayrettin Eren  and
      Shah, Mitash Ashish  and
      Huang, Zhiqi  and
      Kumar, Anoop  and
      Samuel, Alfy  and
      Liu, Daben  and
      Karimireddy, Sai Praneeth  and
      Avestimehr, Salman},
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-demos.54/",
    pages = "717--728",
    ISBN = "979-8-89176-334-0",
}
```


