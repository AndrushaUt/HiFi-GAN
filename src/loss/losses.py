import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneratorLoss(nn.Module):
    def forward(self, disc_outputs):
        loss = 0
        for output in disc_outputs:
            loss += torch.mean((output - 1) ** 2)
        return loss

class DiscriminatorLoss(nn.Module):
    def forward(self, disc_real_outputs, disc_generated_outputs):
        loss = 0
        for real_output, gen_output in zip(disc_real_outputs, disc_generated_outputs):
            loss += torch.mean((real_output - 1) ** 2) + torch.mean(gen_output ** 2)
        return loss

class FeatureLoss(nn.Module):
    def forward(self, features_real, features_generated):
        loss = 0
        for real_fmaps, gen_fmaps in zip(features_real, features_generated):
            for real, gen in zip(real_fmaps, gen_fmaps):
                loss += torch.mean(torch.abs(real - gen))
        return loss

class MelSpecLoss(nn.Module):
    def forward(self, mel_real, mel_fake):
        loss = F.l1_loss(mel_real, mel_fake)
        return loss
