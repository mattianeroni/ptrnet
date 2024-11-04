import torch 
from torch import nn 
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchsummary import summary


class Encoder(nn.Module):
  def __init__(self, input_size, num_layers, hidden_size, bidirectional=False):
    super(Encoder, self).__init__()
    self.input_size = input_size
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.lstm = nn.LSTM(input_size=input_size, num_layers=num_layers, hidden_size=hidden_size, batch_first=True, bidirectional=bidirectional)
  
  def forward(self, x):
    x, (h, c) = self.lstm(x)
    return x, (h, c)


class Decoder(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, bidirectional=False):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, x, hidden=None):
        x, (h, c) = self.lstm(x, hidden)
        return x, (h, c)
    

class Attention(nn.Module):
  def __init__(self, hidden_size, attention_units=None):
    super(Attention, self).__init__()
    self.hidden_size = hidden_size
    self.attention_units = attention_units or hidden_size
    self.w1 = nn.Linear(hidden_size, attention_units, bias=False)
    self.w2 = nn.Linear(hidden_size, attention_units, bias=False)
    self.v =  nn.Linear(attention_units, 1, bias=False)

  def forward(self, encoder_out: torch.Tensor, decoder_hidden: torch.Tensor):
    """ 
    Compute the attention between each decoder hidden state and the encoder outputs.

    param encoder_out: (batch, sequence_len, hidden_size)
    param decoder_hidden: (batch, hidden_size)
    """
    # Add time axis to decoder hidden state in order to make operations compatible with encoder_out
    # decoder_hidden_time: (batch, 1, hidden_size)
    decoder_hidden_time = decoder_hidden.unsqueeze(1)
    # uj: (batch, sequence_len, attention_units) this is the so called energy vector
    # NOTE: we can add the both linear outputs thanks to broadcasting
    uj = self.w1(encoder_out) + self.w2(decoder_hidden_time)
    uj = torch.tanh(uj)
    # uj: (batch, sequnence_len, 1)
    uj = self.v(uj)
    # Attention mask over inputs pointing to input sequence
    # aj: (batch, sequence_len, 1)
    aj = f.softmax(uj, dim=1)
    return aj


class PtrNet(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, attention_units=None):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.attention_units = attention_units or hidden_size
        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False)
        self.decoder = Decoder(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False)
        self.attention = Attention(hidden_size=hidden_size, attention_units=self.attention_units)

    def forward(self, x, y=None, teacher_force_ratio=0.0):
      batch_size, sequence_size = x.size(0), x.size(1)

      # Encode the inputs 
      # encoded: (batch, sequence_len, hidden_size)
      # he, ce: (num_layers, batch, hidden_size)
      encoded, (he, ce) = self.encoder(x)

      # First decoder input is encoder output
      # decoder_in: (batch, sequence_len, hidden_size)
      decoder_in = encoded
      hd, cd = he, ce 

      for t in range(sequence_size):
          aj = self.attention(encoded, hd[-1])

          # di_prime: (batch, hidden_size)
          di_prime = aj * decoder_in
          prediction = aj.argmax(1)

          decoded, (hd, cd) = self.decoder(di_prime, (hd, cd))
          decoder_in = decoded
          print(prediction)
          #decoder_in = torch.stack([x[b, predictions[b].item()] for b in range(x.size(0))])
          #decoder_in = decoder_in.view(encoded.size(0), 1, 1).type(torch.float32)