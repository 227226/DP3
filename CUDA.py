import torch

if torch.cuda.is_available():
    free_memory, total_memory = torch.cuda.mem_get_info()
    print(f"Volná RAM: {free_memory / 1024 ** 2:.2f} MB")
    print(f"Celková RAM: {total_memory / 1024 ** 2:.2f} MB")
else:
    print("CUDA není dostupná.")
