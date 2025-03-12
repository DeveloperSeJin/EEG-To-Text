import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import torch
from ldm.diffusion_module import register_schedule
from ldm.util import q_sample, extract_into_tensor
from ldm.openaimodel import UNetModel


class LDMTranslator(nn.Module):
    def __init__(self, pretrained_layers, in_feature=840, decoder_embedding_size=1024, 
                 additional_encoder_nhead=8, additional_encoder_dim_feedforward=2048, latent_dim=1024, device = 'cuda', pretrained_autoencoder = None):
        super(LDMTranslator, self).__init__()
        self.unet = UNetModel(
            image_size=850,
            dims = 1, 
            in_channels=56, 
            out_channels=56, 
            model_channels=320, 
            attention_resolutions=[4, 2, 1], 
            num_res_blocks=2, 
            channel_mult=[1, 2, 4, 4], 
            num_heads=8, 
            use_spatial_transformer=True, 
            transformer_depth=1, 
            context_dim=1024, 
            # context_dim=512, 
            use_checkpoint=True, 
            legacy=False
            ).to(device)

        # text: [batch_size, seq_len, 1024]
        self.autoencoder = pretrained_autoencoder
        self.pretrained = pretrained_layers

        layers = []
        filters = 8
        prev_filters = 56
        # Build a sequence of 6 convolution layers
        for _ in range(6):
            layers.append(
                nn.Conv1d(
                    in_channels=prev_filters,
                    out_channels=filters,
                    kernel_size=3,
                    padding=1,
                    stride=1
                )
            )
            layers.append(nn.ReLU(inplace=True))
            prev_filters = filters
            # filters = filters // 2
            # Depending on your design choice, you could double filters or keep it constant:
            filters *= 2  # Optionally increase filter size each layer

        self.conv_stack = nn.Sequential(*layers)
        self.conv_1x1 = nn.Conv1d(in_channels=prev_filters, out_channels=56, kernel_size=1)

        # Additional transformer encoder for EEG feature extraction
        self.additional_encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_feature, nhead=additional_encoder_nhead, 
            dim_feedforward=additional_encoder_dim_feedforward, batch_first=True
        )
        self.additional_encoder = nn.TransformerEncoder(self.additional_encoder_layer, num_layers=6)

        # for addin forward
        self.fc1 = nn.Linear(in_feature, latent_dim)

        self.fc2 = nn.Linear(64, latent_dim)
        # self.eeglayer = nn.Linear(840, 512)
        self.schedule = register_schedule(device = device)

    def condition_embedding(self,input_embeddings_batch,  input_masks_invert):
        """input_embeddings_batch: batch_size*Seq_len*840"""
        """input_mask: 1 is not masked, 0 is masked"""
        """input_masks_invert: 1 is masked, 0 is not masked"""

        # input_embeddings_batch = self.positional_embedding(input_embeddings_batch)
        # use src_key_padding_masks
        conv_out = self.conv_stack(input_embeddings_batch)
        conv_out = self.conv_1x1(conv_out)
        encoded_embedding = self.additional_encoder(conv_out, src_key_padding_mask=input_masks_invert)
        encoded_embedding = self.fc1(encoded_embedding)
        
        return encoded_embedding
    
    @torch.no_grad()
    def generate(
            self,
            input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted,
            generation_config = None,
            logits_processor = None,
            stopping_criteria = None,
            prefix_allowed_tokens_fn= None,
            synced_gpus= None,
            assistant_model = None,
            streamer= None,
            negative_prompt_ids= None,
            negative_prompt_attention_mask = None,
            device = 'cuda',
            **kwargs,
    ):
        z = torch.rand(target_ids_batch_converted.shape[0], target_ids_batch_converted.shape[1], 64, device = device)  # (16, 1280)
        noise = torch.randn_like(z, device=device)
        timesteps = torch.randint(0, 1000, (z.shape[0],), device=device).long()
        
        x_noisy = q_sample(z, timesteps, self.schedule['sqrt_alphas_cumprod'], self.schedule['sqrt_one_minus_alphas_cumprod'], noise=noise)
        condition=self.condition_embedding(input_embeddings_batch, input_masks_invert)

        predicted_noise = self.unet(x_noisy, timesteps, condition)
        sqrt_one_minus_alphas_cumprod = extract_into_tensor(self.schedule['sqrt_one_minus_alphas_cumprod'], timesteps, predicted_noise.shape)    
        sqrt_alphas_cumprod = extract_into_tensor(self.schedule['sqrt_alphas_cumprod'], timesteps, predicted_noise.shape)

        x_recon = (x_noisy - (sqrt_one_minus_alphas_cumprod * predicted_noise)) / sqrt_alphas_cumprod

        encoded_embedding = F.relu(self.fc2(x_recon))
        output=self.pretrained.generate(
            inputs_embeds = encoded_embedding,
            attention_mask = input_masks_batch[:,:encoded_embedding.shape[1]],
            labels = target_ids_batch_converted,
            # return_dict = True,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs,)

        return output
    
    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted, context, stage = 'second_stage'):
        device = target_ids_batch_converted.device
        posterior = self.autoencoder.encode(context)# (B, 56, 105)
        z = posterior.sample()
        
        noise = torch.randn_like(z, device=device)
        timesteps = torch.randint(0, 1000, (z.shape[0],), device=device).long() # (B,)
        
        x_noisy = q_sample(z, timesteps, self.schedule['sqrt_alphas_cumprod'], self.schedule['sqrt_one_minus_alphas_cumprod'], noise=noise) # (B, 56, 105)
        condition = self.condition_embedding(input_embeddings_batch, input_masks_invert) # (B, 56, 1024)
        
        predicted_noise = self.unet(x_noisy, timesteps, condition)

        sqrt_one_minus_alphas_cumprod = extract_into_tensor(self.schedule['sqrt_one_minus_alphas_cumprod'], timesteps, predicted_noise.shape)    
        sqrt_alphas_cumprod = extract_into_tensor(self.schedule['sqrt_alphas_cumprod'], timesteps, predicted_noise.shape)

        x_recon = (x_noisy - (sqrt_one_minus_alphas_cumprod * predicted_noise)) / sqrt_alphas_cumprod

        encoded_embedding = F.relu(self.fc2(x_recon))
        out = self.pretrained(inputs_embeds = encoded_embedding, attention_mask = input_masks_batch,
                            return_dict = True, labels = target_ids_batch_converted)
        # Diffusion process
        if stage == 'first_stage':
            loss = F.mse_loss(predicted_noise, noise)
        elif stage == 'second_stage':
            loss = out.loss

        return out, loss
        
    def evaluation(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted, context):
        device = target_ids_batch_converted.device
        
        z = torch.rand(target_ids_batch_converted.shape[0], target_ids_batch_converted.shape[1], 64, device = device)  # (16, 1280)
        noise = torch.randn_like(z, device=device)
        timesteps = torch.randint(0, 1000, (z.shape[0],), device=device).long() # (B,)
        
        x_noisy = q_sample(z, timesteps, self.schedule['sqrt_alphas_cumprod'], self.schedule['sqrt_one_minus_alphas_cumprod'], noise=noise) # (B, 56, 105)
        condition = self.condition_embedding(input_embeddings_batch, input_masks_invert) # (B, 56, 1024)
        
        predicted_noise = self.unet(x_noisy, timesteps, condition)
        sqrt_one_minus_alphas_cumprod = extract_into_tensor(self.schedule['sqrt_one_minus_alphas_cumprod'], timesteps, predicted_noise.shape)    
        sqrt_alphas_cumprod = extract_into_tensor(self.schedule['sqrt_alphas_cumprod'], timesteps, predicted_noise.shape)

        x_recon = (x_noisy - (sqrt_one_minus_alphas_cumprod * predicted_noise)) / sqrt_alphas_cumprod

        encoded_embedding = F.relu(self.fc2(x_recon))
        out = self.pretrained(inputs_embeds = encoded_embedding, attention_mask = input_masks_batch,
                            return_dict = True, labels = target_ids_batch_converted)

        return out
    
class Rater(nn.Module):
    def __init__(self, in_feature=840, additional_encoder_nhead=8, 
                 additional_encoder_dim_feedforward=2048, vocab_size = 100):
        
        self.embedding_layer = nn.Embedding(vocab_size, in_feature)
        self.additional_encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_feature, nhead=additional_encoder_nhead, 
            dim_feedforward=additional_encoder_dim_feedforward, batch_first=True
        )
        self.additional_encoder = nn.TransformerEncoder(self.additional_encoder_layer, num_layers=6)
        self.fc1 = nn.Linear(1024, 1)

    def forward(self, x):
        
        embedding_inputs = self.embedding_layer(x)
        encoded_output = self.additional_encoder(embedding_inputs)
        out = self.fc1(encoded_output)
        return out