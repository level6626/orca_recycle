import numpy as np

import torch
from torch import nn
import pdb


def marginal_prob_std(t, sigma):
  """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

  Args:    
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.  
  
  Returns:
    The standard deviation.
  """    
  t = torch.tensor(t, device='cuda')
  return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))


def diffusion_coeff(t, sigma):
  """Compute the diffusion coefficient of our SDE.

  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.
  
  Returns:
    The vector of diffusion coefficients.
  """
  return torch.tensor(sigma**t, device='cuda')


def loss_fn(model, x, y, marginal_prob_std, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a 
      time-dependent score-based model.
    x: A mini-batch of training data.    
    marginal_prob_std: A function that gives the standard deviation of 
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  mask_y = torch.isnan(y)
  torch.nan_to_num_(y, nan=0.0)
  random_t = torch.rand(y.shape[0], device=y.device) * (1. - eps) + eps  
  z = torch.randn_like(y)
  std = marginal_prob_std(random_t)
  perturbed_y = y + z * std[:, None, None, None]
  score = model(x, perturbed_y, random_t)
  loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2 * (~mask_y), dim=(1,2,3)))
  return loss


def Euler_Maruyama_sampler(score_model,
                           x,
                           length,
                           marginal_prob_std,
                           diffusion_coeff, 
                           batch_size=32, 
                           num_steps=500, 
                           device='cuda', 
                           eps=1e-3):
  """Generate samples from score-based models with the Euler-Maruyama solver.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
      Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.
  
  Returns:
    Samples.    
  """
  t = torch.ones(batch_size, device=device)
  init_y = torch.randn(batch_size, 1, length, length, device=device) \
    * marginal_prob_std(t)[:, None, None, None]
  time_steps = torch.linspace(1., eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  y = init_y
  with torch.no_grad():
    for time_step in time_steps:      
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      g = diffusion_coeff(batch_time_step)
      mean_y = y + (g**2)[:, None, None, None] * score_model(x, y, batch_time_step) * step_size
      y = mean_y + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(y)      
  # Do not include any noise in the last sampling step.
  return mean_y