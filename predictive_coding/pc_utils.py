import torch

def step_embed(self, t, target, x_word, x_pos, layer):
        word_layer = layer["word"]
        pos_layer = layer["pos"]

        mu_word = word_layer(x_word)
        mu_pos = pos_layer(x_pos)

        mu = mu_word + mu_pos
        error = target - mu

        word_update = error @ word_layer.weight.T
        delta_word_W = self.local_lr * torch.einsum("bsh,bsv->vh", error, x_word)
        word_layer.weight.data.add_(delta_word_W)
        x_word = torch.clamp(x_word + self.local_lr * word_update, -self.clamp_value, self.clamp_value)

        pos_update = error @ pos_layer.weight.T
        delta_pos_W = self.local_lr * torch.einsum("bsh,bsv->vh", error, x_pos)
        pos_layer.weight.data.add_(delta_pos_W)
        x_pos = torch.clamp(x_pos + self.local_lr * pos_update, -self.clamp_value, self.clamp_value)

        if t == self.T - 1:
            self._finalize_step(mu, target, error, t, "embed")

        return x_word, x_pos
