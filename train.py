import os
import datasets as ds
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    BitsAndBytesConfig
)
from peft import get_peft_model,LoraConfig,TaskType,prepare_model_for_kbit_training
from peft.utils.other import fsdp_auto_wrap_policy
import torch
import numpy as np
from sklearn.metrics import accuracy_score,log_loss
import argparse
from huggingface_hub import login
import yaml
from accelerate import PartialState


class WSDM_tokenizer:
    
    def __init__(self,tokenizer,max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self,batch):
        prompt = ['<prompt>: '+p for p in batch['prompt']]
        response_a = ['\n\n'+'<response_a>: '+a for a in batch['response_a']]
        response_b = ['\n\n'+'<response_b>: '+b for b in batch['response_b']]
        text_input = [p+a+b for p,a,b in zip(prompt,response_a,response_b)]
        tokenized_input = self.tokenizer(text_input, max_length = self.max_length, truncation = True)
        
        labels = []
        for winner in batch['winner']:
            if winner == 'model_a':
                labels.append(0)
            else:
                labels.append(1)
                
        return {**tokenized_input,'labels':labels}

def compute_metric(eval_preds):
    logits = eval_preds.predictions
    labels = eval_preds.label_ids
    predictions = logits.argmax(axis=1)
    probabilities = torch.from_numpy(logits).float().softmax(1).numpy()
    return {'accuracy':accuracy_score(predictions, labels),
           'loss': log_loss(labels,probabilities)}
    

def main(config):
    login(token = config.HF_TOKEN)
    
    if torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.bfloat16
        #attn_implementation = 'flash_attention_2'
    else:
        dtype = torch.float16
       # attn_implementation = 'sdpa'
    attn_implementation = 'sdpa'
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_compute_dtype = dtype,
        bnb_4bit_quant_storage = dtype, 
        bnb_4bit_quant_type = 'nf4',
        bnb_4bit_use_double_quant = True
    ) 

    lora_config = LoraConfig(
        r = config.lora_r,
        target_modules = config.lora_target_modules,
        lora_alpha = config.lora_alpha,
        lora_dropout = config.lora_dropout,
       # layers_to_transform = [i for i in range(42) if i > config.lora_layers_threshold],
        task_type = TaskType.SEQ_CLS,
        bias="none"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.model) # token = config.HF_token
    tokenizer.add_eos_token = True
    tokenizer.padding_side = 'right'

    #from accelerate import Accelerator
   # device_index = Accelerator().process_index
    #device_map = {"": device_index}

    device_map={"": PartialState().process_index}
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model,
        num_labels = 2,
        quantization_config = quantization_config,
        device_map = device_map,
        use_cache = False,
        dtype = dtype,
        attn_implementation = attn_implementation
        #token = config.HF_token
    )
    
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing =True,
        gradient_checkpointing_kwargs = {'use_reentrant': True})
   # for name,param in model.named_parameters():
  #      param.requires_grad = False
   # model.enable_input_require_grads()
  #  model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {'use_reentrant': True})
    model = get_peft_model(model,lora_config)

    for param in model.parameters():
         if (param.dtype == torch.float32):
             param.data = param.data.to(dtype)
 
    training_hyperparameters = TrainingArguments(
        per_device_train_batch_size = config.per_device_train_batch_size,
        per_device_eval_batch_size = config.per_device_eval_batch_size,
        gradient_accumulation_steps = config.gradient_accumulation_steps,
        optim = config.optimizer,
        learning_rate = config.learning_rate,
        num_train_epochs = config.num_train_epochs,
        lr_scheduler_type = 'linear',
        warmup_steps = config.warmup_steps,
        output_dir = config.output_directory,
        overwrite_output_dir = True,
        logging_steps = 10,
        eval_strategy = 'steps',
        save_strategy = 'steps',
        save_steps = 300,
        fp16 = True if dtype == torch.float16 else False,
        bf16 = True if dtype == torch.bfloat16 else False,
        report_to = 'wandb',
        run_name = config.run_name
    )

    tokenize = WSDM_tokenizer(tokenizer,config.max_length)
    
    dataset = ds.load_dataset('parquet', data_files = config.data_directory+'/train.parquet')['train']
    dataset = dataset.select(range(100))
    dataset = dataset.map(tokenize,batched = True)
    
    folds = [    
    ([i for i in range(len(dataset)) if i % config.k_folds != val_fold],
    [i for i in range(len(dataset)) if i % config.k_folds == val_fold])
        
        for val_fold in range(config.k_folds)
    ]
    
    train_idx, val_idx = folds[config.fold]
    
    trainer = Trainer(
        model = model,
        processing_class=tokenizer,
        args = training_hyperparameters,
        data_collator = DataCollatorWithPadding(tokenizer),
        train_dataset = dataset.select(train_idx),
        eval_dataset = dataset.select(val_idx),
        compute_metrics = compute_metric
    )

    fsdp_plugin = trainer.accelerator.state.fsdp_plugin
    fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)
    
    trainer.train()

    if trainer.is_fsdp_enabled:
       trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory', default = 'Data')
    parser.add_argument('--output_directory', default = 'Output')
    parser.add_argument('--model', default = 'google/gemma-2-9b-it')
    parser.add_argument('--HF_TOKEN', default = os.environ['HF_TOKEN'])
    parser.add_argument('--WANDB_API_KEY', default = os.environ['WANDB_API_KEY'])
    parser.add_argument('--max_length',type = int, default = 2048)
    parser.add_argument('--k_folds',type = int, default = 10)
    parser.add_argument('--fold',type = int, default = 0)
    parser.add_argument('--lora_r',type = int, default = 128)
   # parser.add_argument('--lora_target_modules',type = str,\
   #                     nargs = '+',\
   #                     default = ['q_proj','k_proj','v_proj','o_proj',\
   #                               'gate_proj','up_proj','down_proj'])
    parser.add_argument('--lora_target_modules',type = str,\
                        default = 'all-linear')
    parser.add_argument('--lora_alpha',type = int, default = 128*2)
    parser.add_argument('--lora_dropout',type = float, default = 0.05)
    parser.add_argument('--lora_layers_threshold',type = int, default = 10)
    parser.add_argument('--per_device_train_batch_size',type = int, default = 4)
    parser.add_argument('--gradient_accumulation_steps',type = int, default = 2)
    parser.add_argument('--per_device_eval_batch_size',type = int, default = 8)
    parser.add_argument('--learning_rate',type = float, default = 2e-4)
    parser.add_argument('--num_train_epochs',type = int, default = 1)
    parser.add_argument('--optimizer', default = 'adamw_torch')
    parser.add_argument('--warmup_steps', type = int, default = 50)
    parser.add_argument('--run_name',type = str, default = 'Trial_1')
    config = parser.parse_args()

    main(config)
