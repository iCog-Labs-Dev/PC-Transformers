device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = GPTConfig(
    vocab_size= tokenizer.get_vocab_size(),
    block_size=64,
    n_embed=64,
    dropout=0.1,
    energy_fn=energy_fn,
    x_init=x_init,
    local_learning_rate=1e-2,
    T=2,
    is_holding_error=True,
    num_heads=2,
    n_blocks=2,
    num_epochs = 3,   
)

def train(model, dataloader, device):
    model.train()
    total_energy = 0.0
    batch_count = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)

        logits = model(input_ids)

        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=0
        )

        # Collect energy from all PCLayer instances
        layer_energies = []
        for module in model.modules():
            if isinstance(module, PCLayer) and hasattr(module, "get_energy"):
                energy = module.get_energy()
               
                if energy is not None:
                    layer_energies.append(energy)


        # Compute average energy for this batch
        batch_energy = ce_loss.item() if not layer_energies else sum(layer_energies) / len(layer_energies)
        total_energy += batch_energy
        batch_count += 1


        # Clear energy and errors for all PCLayer instances
        for module in model.modules():
            if hasattr(module, "clear_energy"):
                module.clear_energy()
            if hasattr(module, "clear_errors"):
                module.clear_errors()

    avg_energy = total_energy / batch_count if batch_count > 0 else 0.0
    return avg_energy
