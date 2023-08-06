# attacks.py

import torch

def no_attack(x_original, epsilon, f_gradient):
    return x_original

def fgsm_attack(x_original, epsilon, f_gradient):
    return x_original + epsilon * (f_gradient(x_original).sign())

def pgd_attack(x_original, epsilon, f_gradient):
    iter = 20
    alpha = 2/255
    x_perturbed = x_original.clone()
    for i in range(iter) :
        x_perturbed = x_perturbed + alpha * (f_gradient ( x_perturbed ).sign())
        eta = torch.clamp ( x_perturbed - x_original, min=-epsilon, max=+epsilon )
        x_perturbed = x_original + eta
    return x_perturbed
