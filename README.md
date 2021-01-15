# **MORE** : a **M**etric learning based framework for **O**pen-domain **R**elation **E**xtraction

This repository is built for the source code of paper -- **MORE: A METRIC LEARNING BASED FRAMEWORK FOR OPEN-DOMAIN RELATION EXTRACTION** . You can follow the steps below to use our code.

## 1. Configure the Environment

To run the code, you need ：

~~~bash
torch>=1.3.0
tensorflow>=1.9.0
keras>=2.2.5
transformers==3.2.0
~~~

You can run the following command to set up a new Anaconda environment:

```bash
conda create -n more python=3.6
pip install -r ./requirements.txt
```

We suggest that you use the same environment as ours to avoid any problems.

## 2. Prepare the Datasets

In this code, we use two real-world RE datasets:

- **FewRel** :  We follow [RSNs](https://github.com/thunlp/RSN). The processed dataset is already in ./data/datasets/fewrel_ori/ .

- **NYT+FB-sup**: We use the original NYT+FB and process it to NYT+FB-sup. The dataset is not open source, but you can get the [sample](https://github.com/diegma/relation-autoencoder/blob/master/data-sample.txt) if you need.  

  To process **nyt_ori.txt** (suppose you already own it and store it in the ./data/datasets/nyt_su/ ), run the following command:

  ```bash
  python ./data/datasets/nyt_su/process2json.py
  python ./data/datasets/nyt_su/nyt_divide_supervision.py
  ```

  then the original **.txt** file will be processed into **.json** format and be divided into train\dev\test(6:2:2).

## 3. Run it

In our experiments, we use CNN and BERT for our extractor. The architecture of CNN is same as [RSNs](https://github.com/thunlp/RSN) used, and the pre-trained language model we exploit is [huggingface transformers](https://huggingface.co/transformers/model_doc/bert.html).

- **On FewRel**:

  - MORE(CNN) 

  ```bash
  python main_cmd.py --dataset fewrel 
  ```

  - MORE(CNN)+VAT

  ```bash
  python main_cmd.py --dataset fewrel --VAT 1 --epoch_num 4 --warm_up 3 --power_iterations 1 --p_mult 0.03 --lambda_V 1 
  ```

  - MORE(BERT)

  ```bash
  python main_cmd.py --dataset fewrel --learning_rate 0.00001 --batch_num 1000 --BERT 1 
  ```

- **On NYT+FB-sup**:

  - MORE(CNN)

  ```bash
  python main_cmd.py --dataset nyt
  ```

  - MORE(CNN)+VAT

  ```bash
  python main_cmd.py --dataset nyt --VAT 1 --epoch_num 6 --warm_up 4 --power_iterations 1 --p_mult 0.5 --lambda_V 1.5
  ```

  - MORE(BERT)

  ```bash
  python main_cmd.py --dataset nyt --learning_rate 0.00001 --batch_num 1000 --BERT 1 
  ```

*Note that if you have enough computing resources, you can try to use **MORE(BERT)+VAT** (We didn't list this result on paper  due to the limitation of GPU memory) :*

```bash
python main_cmd.py --dataset fewrel --VAT 1 --epoch_num 4 --warm_up 0 --power_iterations 1 --p_mult 0.03 --lambda_V 1 --learning_rate 0.00001 --batch_num 1000 --BERT 1
python main_cmd.py --dataset nyt --VAT 1 --epoch_num 4 --warm_up 0 --power_iterations 1 --p_mult 0.5 --lambda_V 1.5 --learning_rate 0.00001 --batch_num 1000 --BERT 1
```

## 4. Future Work

- Optimize virtual adversarial training.
- Complete MORE(BERT)+VAT.

 

