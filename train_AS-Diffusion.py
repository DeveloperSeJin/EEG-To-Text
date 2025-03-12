import os
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import pickle
import json
import time
import copy
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration
from data import ZuCo_dataset
from model_decoding import LDMTranslator, Rater
from config import get_config
from eval_decoding import eval_model
import wandb
from ldm.autoencoder import AutoencoderKL

loss_fn = torch.nn.BCEWithLogitsLoss()
def compute_reward(discriminator_output, scaling_factor=1.0, penalty_factor=1.0):
    """
    Compute reward based on the output of the discriminator
    """
    sigmoid_output = torch.sigmoid(discriminator_output)
    reward = 1 - sigmoid_output
    
    reward = torch.exp(scaling_factor * reward)
    penalty = penalty_factor * sigmoid_output

    final_reward = reward - penalty
    final_reward = torch.clamp(final_reward, min=0.0)

    return final_reward

def create_labels(n: int, r1: float, r2: float, device: torch.device = None):
    """
    Create smoothed labels
    """
    return torch.empty(n, 1, requires_grad=False, device=device).uniform_(r1, r2).squeeze(1)

# 학습 함수 정의
def d_train(discriminator, batch, device):
    input_ids = batch['input_probs'].float().to(device)
    labels = batch['label'].to(device)
    
    # output = discriminator(input_ids, attention_mask, target_ids_batch, input_masks_invert).squeeze(-1)
    output = discriminator(input_ids)
    
    if output.dim() > 1:  # Check if shape is [batch_size, 1]
        output = output.squeeze(-1)  # Remove the second dimension
    loss = loss_fn(output, labels)
    return output.detach(), loss

def train_model(dataloaders, device, discriminator, discriminator_optimizer, discriminator_scheduler, model, optimizer, scheduler, num_epochs=25, checkpoint_path_best = './checkpoints/decoding/best/temp_decoding.pt', checkpoint_path_last = './checkpoints/decoding/last/temp_decoding.pt', stage = 'second_stage'):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000000

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        lambda_1=0.5
        lambda_2=0.5
        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()  # Set model to training mode
                discriminator.train()
            else:
                model.eval()   # Set model to evaluate mode
                discriminator.eval()

            running_loss = 0.0
            smoothing = 0.05
            
            # Iterate over data.
            for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, rawEEG in tqdm(dataloaders[phase]):
                
                # load in batch
                input_embeddings_batch = input_embeddings.to(device).float()
                # input_embeddings_batch = sent_level_EEG.unsqueeze(1).to(device).float()
                input_masks_batch = input_masks.to(device)
                input_mask_invert_batch = input_mask_invert.to(device)
                target_ids_batch = target_ids.to(device)
                """replace padding ids in target_ids with -100"""
                context = target_ids_batch.clone()
                target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

                # zero the parameter gradients
                optimizer.zero_grad()
                discriminator_optimizer.zero_grad()

                # forward
    	        # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    r_labels = create_labels(batch_size, 1.0-smoothing, 1.0, device=device)
                    f_labels = create_labels(batch_size, 0.0, smoothing, device=device)
                    fake_probs, loss = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch, context, stage = stage)
                    real_encoding = {
                        'input_probs': context,
                        'label': r_labels,
                    }

                    # Prepare fake data for discriminator
                    fake_encoding = {
                        'input_probs': fake_probs.argmax(dim=-1),
                        'label': f_labels,
                    }
                    _, real_loss = d_train(discriminator, real_encoding, device)
                    discriminator_output, fake_loss = d_train(discriminator, fake_encoding, device)
                    d_loss = real_loss + fake_loss

                    if stage == 'first_stage':
                        g_loss = loss
                    else:
                        g_loss = loss * compute_reward(discriminator_output)
                    
                    if phase == 'train':
                        # with torch.autograd.detect_anomaly():
                        g_loss.sum().backward()
                        d_loss.backward()
                        optimizer.step()
                        discriminator_optimizer.step()

                # statistics
                
                running_loss += g_loss.sum().item() * input_embeddings_batch.size()[0] # batch loss
                
                # print('[DEBUG]loss:',loss.item())
                # print('#################################')
                

            if phase == 'train':
                scheduler.step()
                discriminator_scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            wandb.log({
                f'{phase}_loss':epoch_loss,
            }, step=epoch)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            
            # deep copy the model
            if phase == 'dev' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                '''save checkpoint'''
                torch.save(best_model_wts, checkpoint_path_best)
                print(f'update best on dev checkpoint: {checkpoint_path_best}')
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), checkpoint_path_last)
    print(f'update last checkpoint: {checkpoint_path_last}')
    return best_model_wts, model

def show_require_grad_layers(model):
    print()
    print(' require_grad layers:')
    # sanity check
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(' ', name)

if __name__ == '__main__':
    args = get_config('train_decoding')

    ''' config param'''
    dataset_setting = 'unique_sent'
    
    num_epochs_step1 = args['num_epoch_step1']
    num_epochs_step2 = args['num_epoch_step2']
    step1_lr = args['learning_rate_step1']
    step2_lr = args['learning_rate_step2']
    
    batch_size = args['batch_size']
    
    model_name = args['model_name']
    # model_name = 'BrainTranslatorNaive' # with no additional transformers
    # model_name = 'BrainTranslator' 
    
    # task_name = 'task1'
    # task_name = 'task1_task2'
    # task_name = 'task1_task2_task3'
    # task_name = 'task1_task2_taskNRv2'
    task_name = args['task_name']
    train_input = args['train_input']
    print("train_input is:", train_input)
    save_path = args['save_path']
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    skip_step_one = args['skip_step_one']
    print(f'[INFO]using model: {model_name}')
    
    if skip_step_one:
        save_name = f'ldm_{task_name}_skipstep1_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}'
    else:
        save_name = f'ldm_{task_name}_2steptraining_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}'

    

    save_path_best = os.path.join(save_path, 'best')
    if not os.path.exists(save_path_best):
        os.makedirs(save_path_best)

    output_checkpoint_name_best = os.path.join(save_path_best, f'{save_name}.pt')

    save_path_last = os.path.join(save_path, 'last')
    if not os.path.exists(save_path_last):
        os.makedirs(save_path_last)

    output_checkpoint_name_last = os.path.join(save_path_last, f'{save_name}.pt')

    # subject_choice = 'ALL
    subject_choice = args['subjects']
    print(f'![Debug]using {subject_choice}')
    # eeg_type_choice = 'GD
    eeg_type_choice = args['eeg_type']
    print(f'[INFO]eeg type {eeg_type_choice}')
    # bands_choice = ['_t1'] 
    # bands_choice = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] 
    bands_choice = args['eeg_bands']
    print(f'[INFO]using bands {bands_choice}')
    
    ''' set random seeds '''
    # https://pytorch.org/docs/stable/notes/randomness.html
    seed_val = 312

    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        # dev = "cuda:3" 
        dev = args['cuda'] 
    else:  
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3  
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')
    print()

    ''' set up dataloader '''
    whole_dataset_dicts = []
    if 'task1' in task_name:
        dataset_path_task1 = '/home/saul_park/workspace/code/EEG-Diffusion/dataset/ZuCo/task1- SR/pickle/task1- SR-dataset.pickle'
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in task_name:
        dataset_path_task2 = '/home/saul_park/workspace/code/EEG-Diffusion/dataset/ZuCo/task2 - NR/pickle/task2 - NR-dataset.pickle'
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task3' in task_name:
        dataset_path_task3 = '/home/saul_park/workspace/code/EEG-Diffusion/dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset.pickle'
        with open(dataset_path_task3, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in task_name:
        dataset_path_taskNRv2 = '/home/saul_park/workspace/code/EEG-Diffusion/dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle'
        with open(dataset_path_taskNRv2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

    print()

    """save config"""
    cfg_dir = './config/decoding/'

    if not os.path.exists(cfg_dir):
        os.makedirs(cfg_dir)

    with open(os.path.join(cfg_dir,f'{save_name}.json'), 'w') as out_config:
        json.dump(args, out_config, indent = 4)

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    
    # train dataset
    train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=train_input)
    # dev dataset
    dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=train_input)
    # test dataset
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)

    
    dataset_sizes = {'train': len(train_set), 'dev': len(dev_set)}
    print('[INFO]train_set size: ', len(train_set))
    print('[INFO]dev_set size: ', len(dev_set))
    print('[INFO]test_set size: ', len(test_set))
    
    # train dataloader
    train_dataloader = DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers=4)
    # dev dataloader
    val_dataloader = DataLoader(dev_set, batch_size = 1, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_set, batch_size = 1, shuffle=False, num_workers=4)
    # dataloaders
    dataloaders = {'train':train_dataloader, 'dev':val_dataloader, 'test':test_dataloader}

    ''' set up model '''
    pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    vocab_size = tokenizer.vocab_size

    checkpoint_path = args['checkpoint_path']
    pretrained_autoencoder = AutoencoderKL(
    embed_dim=56,
        monitor="val/rec_loss",
        ddconfig={
            "double_z": True,
            "z_channels": 58,
            "resolution": 256,
            "in_channels": 56,
            "out_ch": 56,
            "ch": 128,
            "ch_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0
        },
        lossconfig={
            "target": "torch.nn.Identity"
        },
        vocab_size = vocab_size,
        latent_dim=1024
    ).to(device)
    state_dict = torch.load(checkpoint_path)
    pretrained_autoencoder.load_state_dict(state_dict)

    
    model = LDMTranslator(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048,device = device, pretrained_autoencoder = pretrained_autoencoder)

    model.to(device)
    discriminator = Rater(vocab_size = vocab_size).to(device)
    discriminator_optimizer = optim.SGD(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=step1_lr, momentum=0.9)
    discriminator_scheduler = lr_scheduler.StepLR(discriminator_optimizer, step_size=20, gamma=0.1)

    ''' training loop '''
    ######################################################
    '''step one trainig: freeze most of BART params'''
    ######################################################

    # closely follow BART paper
    for name, param in model.named_parameters():
        if param.requires_grad and 'pretrained' in name:
            if ('shared' in name) or ('embed_positions' in name) or ('encoder.layers.0' in name):
                continue
            else:
                param.requires_grad = False

    if skip_step_one:
        print('skip step one, start from scratch at step two')
    else:

        ''' set up optimizer and scheduler'''
        print('=== start Step1 training ... ===')
        # print training layers
        show_require_grad_layers(model)
        optimizer_step1 = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=step1_lr, momentum=0.9)
        exp_lr_scheduler_step1 = lr_scheduler.StepLR(optimizer_step1, step_size=20, gamma=0.1)
        # return best loss model from step1 training
        wandb.init(project='EEG-Diffusion-beta v0.4', name = 'first_stage'+save_name)
        _, model = train_model(dataloaders, device, discriminator, discriminator_optimizer, discriminator_scheduler, model, optimizer_step1, exp_lr_scheduler_step1, num_epochs=num_epochs_step1, 
                            checkpoint_path_best = output_checkpoint_name_best, checkpoint_path_last = output_checkpoint_name_last, stage = 'first_stage')
        wandb.finish()
    ######################################################
    '''step two trainig: update whole model for a few iterations'''
    ######################################################
    for name, param in model.named_parameters():
        param.requires_grad = True

    ''' set up optimizer and scheduler'''
    optimizer_step2 = optim.SGD(model.parameters(), lr=step2_lr, momentum=0.9)
    exp_lr_scheduler_step2 = lr_scheduler.StepLR(optimizer_step2, step_size=30, gamma=0.1)

    print()
    print('=== start Step2 training ... ===')
    # print training layers
    show_require_grad_layers(model)
    
    '''main loop'''
    wandb.init(project='EEG-Diffusion-beta v0.4', name = 'second_stage'+save_name)
    best_wts, last_model = train_model(dataloaders, device, discriminator, discriminator_optimizer, discriminator_scheduler, model, optimizer_step2, exp_lr_scheduler_step2, num_epochs=num_epochs_step2, 
                                       checkpoint_path_best = output_checkpoint_name_best, checkpoint_path_last = output_checkpoint_name_last, stage = 'second_stage')
    wandb.finish()

    eval_model(dataloaders, device, tokenizer, last_model, output_all_results_path = f'./results/last_{save_name}.txt' , score_results=f'./score_results/last_{save_name}.txt')

    best_model = LDMTranslator(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048,device = device, pretrained_autoencoder = pretrained_autoencoder)
    best_model.load_state_dict(best_wts)
    eval_model(dataloaders, device, tokenizer, best_model, output_all_results_path = f'./results/best_{save_name}.txt' , score_results=f'./score_results/best_{save_name}.txt')

    