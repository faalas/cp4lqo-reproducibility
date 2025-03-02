# Conformal Prediction for Verifiable Learned Query Optimization

This link is for the Additional Experiments mentioned in the revision comments: [Additional Experiments Conformal Prediction for Query Optimisation](https://anonymous.4open.science/r/Conformal-Prediction-4-Database-646E/Additional_Experiments_Conformal_Prediction_for_Query_Optimisation.pdf). The PDF preview function in the anonymous repository may display distortions. Please refresh the page or download the file to view it with the correct formatting.

---

This repository contains the source code and commands used for the "Conformal Prediction for Verifiable Learned Query Optimization VLDB submission". It also includes the adapted source code of the main learned query optimizers used in our experiments, including Balsa, Lero, and RTOS.

### Balsa: CP Guided Plan Search

The pre-trained checkpoint is available in the `balsa/train-checkpoints` directory, and all configurations have been correctly set within the source code. To initiate the experiment, execute:

```sh
python run.py --local
```

By default, the experiment runs in the `without cp-guided` mode. To enable CP-guided mode, modify line 2274 by setting `p.cp_guided = True`.

Please refer to [Balsa Original Repo](https://github.com/balsa-project/balsa) to set up the whole environment or retrain the Balsa.

----

### Lero

To start Lero, we should first start the `server`.

```sh
python lero/server.py
```

Then Lero's evaluation can be easily started by the following command:

```sh
python lero/test_sript/train_model.py --query_folder train-query-path --test_query_folder test-query-path --algo lero --output_query_latency_file imdb.log --model_prefix imdb_model --topK 3
```

Please refer to [Lero Original Repo](https://github.com/AlibabaIncubator/Lero-on-PostgreSQL) to set up the whole environment or retrain the Lero.

----

### RTOS

The training process for RTOS involves two stages:

```sh
python CostTraining.py -> CostTraining.pth

python LatencyTuning.py -> LatencyTuning.pth
```

After training, the queries can be evaluated with the following steps:

```sh
python Analysis/convertQueries.py -> Convert the SQLs to executable one-line queries

python Analysis/runAllQueries.py -> Evaluation
```

Please refer to [RTOS Original Repo](https://github.com/TsinghuaDatabaseGroup/AI4DBCode/tree/master/RTOS) to set up the whole environment or retrain the Lero.

---

### Scripts

The `scripts/` directory contains resources for data parsing and processing.

A clear example is `RTOS/4-CP.ipynb`, where we maintain execution logs. This notebook directly allows you to observe results that align with those reported in the paper.