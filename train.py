import torch
from LLM import LLM
from optimizer import configure_optimizers
import math
import os
# * - * - * HIPERPARAMETERS * - * - *

vocab_size = 50257
d_model = 768
n_layers = 12
n_heads = 12
n_kv_heads = 4 
hidden_dim = 2048
max_seq_len = 1024

batch_size = 8 # gpu'nun fiziksel alabildiği batch
grad_accum_steps = 8 # sanal batch size = 8 * 8 = 64
max_iters = 50000
learning_rate = 6e-4
min_lr = 6e-5
warmup_iters = 2000

device = "cpu"
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
torch.backends.cuda.matmul.allow_tf32 = True


# * - * - * INITS * - * - *

model = LLM(vocab_size, d_model, n_layers, n_heads, n_kv_heads, hidden_dim, max_seq_len).to(device)
optimizer =  configure_optimizers(model, weight_decay=0.1, learning_rate = learning_rate, device_type=device)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype==torch.float16))

#test için dummy dataset oluşturuyoruz. (gerçekte kendi .bin dosyamızı vereceğiz.)

#dataloader = DataLoader(LLMDataset("train.bin", max_seq_len), batch_size=batch_size, shuffle=True, num_workers = 4, pin_memory_True)

dummy_x = torch.randint(0, vocab_size, (batch_size, max_seq_len)).to(device=device)
dummy_y = torch.randint(0, vocab_size, (batch_size, max_seq_len)).to(device=device)

def get_lr(it):
    if it<warmup_iters:
        return learning_rate*it/warmup_iters
    
    if it>max_iters:
        return min_lr
    
    decay_ratio= (it-warmup_iters) / (max_iters -warmup_iters)
    coeff = 0.5*(1.0 +math.cos(math.pi*decay_ratio))
    return min_lr + coeff*(learning_rate- min_lr)

# * - * - * END-TO-END TRAINING LOOP * - * - *
model.train()
optimizer.zero_grad(set_to_none=True) #ram / vram tasarrufu

for step in range(1, max_iters+1):
    #   X, Y = next(iter(dataloader)); X = X.to(device), Y = Y.to(device)
    X, Y = dummy_x, dummy_y

    #LR güncellemesi
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    #forward pass (mixed-precision ve tensor cores devrede)
    with torch.autocast(device_type=device, dtype=dtype):
        logits, loss = model(X, targets=Y)

        #gradient accumulation içib ort. al (loss scaling)
        loss = loss/grad_accum_steps

    scaler.scale(loss).backward() #backward pass -> scaler ile underflow!a karşı koruyoruz
    
    if step % grad_accum_steps == 0:
        #clipping yapmadan önce scaler'ı geri al!!
        scaler.unscale_(optimizer)

        #exploding engellemek için L2Norm clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        #adamw ile weights güncelle
        scaler.step(optimizer)
        scaler.update()

        #gelecek birikim için gradyanları ram'den sil
        optimizer.zero_grad(set_to_none=True)

        #performans ve loglama
        actual_loss = loss.item() * grad_accum_steps
        perplexity = math.exp(actual_loss)

        if step%100 == 0:
            print(f"Adım: {step} | Loss:{actual_loss:.4f} | PPL:{perplexity:.2f} | LR:{lr:.2e}")

    #FAULT TOLERENCE
    
    if step % 5000 == 0:
        checkpoint = {
            'step' : step,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'scaler_state_dict' : scaler.state_dict(),
            'rng_state_cpu' : torch.get_rng_state(),
            'rng_state_cuda' : torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        }
        os.makedirs("checkpoints", exist_ok = True) 
        torch.save(checkpoint, f"checkpoints/llm_step_{step}.pt")
        
print("Training tamamlandı. Model prod'a hazır!")