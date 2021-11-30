import torch
import torch.nn as nn


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, num_layers, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        linears = [nn.Linear(num_inputs, hidden_dim)] +\
                  [nn.Linear(hidden_dim, hidden_dim)] * (num_layers - 2) +\
                  [nn.Linear(hidden_dim, num_actions)]
        self.linears = nn.ModuleList(linears)
        self.noise = torch.Tensor(num_actions)
        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, state):
        h1 = nn.functional.relu(self.linears[0](state))
        h2 = nn.functional.relu(self.linears[1](h1))
        mean = torch.tanh(self.linears[2](h2)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        # WARNING: Do not use sampling as we are behavior cloning and this can get us to weird places
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
