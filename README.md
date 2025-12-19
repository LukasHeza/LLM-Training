# **Introduction**
Multi-GPU training of Gemma 2 9b with FSDP and QLoRA for Kaggle competition [WSDM Cup - Multilingual Chatbot Arena](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena).

# **Requirements**
## **Packages**
    transformers==4.57.1  
    bitsandbytes==0.48.1  
    peft==0.17.1  
    accelerate==1.10.1  
    datasets==4.2.0  
    pandas==2.3.3  
    scikit-learn==1.7

## **System requirements**
Used vast.ai as a computing resource:

    GPU: RTSX A6000(45GB) 2x
    Image: Pytorch  
        - cuda: 12.9.1
    Runtime: ~15 hours  

# **How to use**
First download the competition data into the `Data` folder.

[Gemma 2](https://huggingface.co/google/gemma-2-9b-it) is a gated model so you will need to request 
access and generate access token on huggingface. Save it as environment variable `HF_TOKEN`. 

Download the repository: 

```
git clone https://github.com/LukasHeza/LLM-Training.git
cd LLM-Training
python -m pip install -r requirements.txt
```
    
Run training:
    
```
accelerate launch --config_file './Config/FSDP_config.yaml' train.py
```

The trained adapters will be in the `Output` folder.

# **Inference**
The inference notebook:  
https://www.kaggle.com/code/lukasheza/wsdm-inference
