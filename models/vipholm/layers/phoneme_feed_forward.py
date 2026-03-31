from torch import nn

class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.hidden_size = config.hidden_size
        self.norm = nn.LayerNorm(self.hidden_size)
        self.linear_projection_1 = nn.Linear(self.hidden_size, \
                                            self.hidden_size * 2)
        self.ReLu = nn.ReLU()
        self.linear_projection_2 = nn.Linear(self.hidden_size*2,\
                                            self.hidden_size)

    def forward(self, x):
        x = self.norm(x)
        x_rest = x
        x = self.ReLu(self.linear_projection_1(x))
        x = self.linear_projection_2(x)
        x += x_rest 
        return x