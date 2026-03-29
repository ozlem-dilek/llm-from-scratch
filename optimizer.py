import torch.optim as optim
def configure_optimizers(model, weight_decay, learning_rate, device_type):
    #sadece 2D matrislere weight decay uygula, bias ve RMSNorm(1D) muaf tutulur.

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >=2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {'params' : decay_params, 'weight_decay' : weight_decay},
        {'params' : nodecay_params, 'weight_decay' : 0.0 }
    ]

    use_fused = (device_type=='cuda')
    return optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), fused=use_fused)