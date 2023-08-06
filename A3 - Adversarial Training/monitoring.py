import torch
import torch.nn.functional as F

import numpy as np

def fmt(x):
    if x is None : return '?'
    return "{:.2f}%".format(100 * x)

def get_accuracy(model, dataset, max_n=None):
    model.eval()
    datas = dataset.datas
    res, n = 0., 0
    with torch.no_grad():
        for x, y in datas:
            y_hat = model(x.view(1, *x.shape))
            if torch.argmax(y_hat[0]).detach().item() == y: res += 1
            n += 1
            if (max_n is not None) and (n >= max_n) : break
    if n != 0: return res / n

def get_accuracy_after_attacks ( model, dataset, attacks = None, eps = 1e-2, max_n = None ) :
    model.eval ( )
    datas = dataset.datas
    
    
    if attacks is None : 
        attacks = [ lambda img, epsilon, grad : img ]
    
    r_init, r_atk, n = 0, [0 for i in range(len(attacks))], 0
    
    for x, y in datas :
        def f_grad ( inp ) :
            model.zero_grad ( )
            inp = inp.view ( 1, *inp.shape )
            inp.requires_grad = True
            y_hat = F.log_softmax ( model ( inp )[0], dim = 0 )
            return torch.autograd.grad ( outputs = y_hat.sum() - 2 * y_hat[y], inputs = inp )[0][0]
        inp = x.view ( 1, *x.shape )
        inp.requires_grad = True
        y_hat = F.log_softmax ( model ( inp )[0], dim = 0 )
        y_1 = torch.argmax ( y_hat )
        
        if y_1.detach().item() == y :
            r_init += 1
            
            for i in range(len(attacks)) :
                x_prime = attacks[i] ( x, eps, f_grad )
                if torch.max(torch.abs(x_prime-x)) > eps+1e-7 : 
                    print ( f"norme infinie supérieure à {eps} pour l'attaque {i}, norm = ", torch.max(torch.abs(x_prime-x)) )
                with torch.no_grad ( ) :
                    z_hat = model ( x_prime.view ( 1, *x_prime.shape ) )
                    if torch.argmax ( z_hat[0] ).detach().item() == y :
                        r_atk[i] += 1
        n += 1
        if (max_n is not None) and (n >= max_n) : break
    
    if n != 0 : return ( r_init, r_atk, n )



def get_all_after_attacks ( model, dataset, attacks = None, eps = 1e-2, max_n = None ) :
    model.eval ( )
    datas = dataset.datas
    
    
    if attacks is None : 
        attacks = [ lambda img, epsilon, grad : img ]
    
    alldatas, n = [[] for i in range(len(attacks))], 0
    
    for x, y in datas :
        def f_grad ( inp ) :
            model.zero_grad ( )
            inp = inp.view ( 1, *inp.shape )
            inp.requires_grad = True
            y_hat = F.log_softmax ( model ( inp )[0], dim = 0 )
            return torch.autograd.grad ( outputs = y_hat.sum() - 2 * y_hat[y], inputs = inp )[0][0]
        inp = x.view ( 1, *x.shape )
        inp.requires_grad = True
        y_hat = F.log_softmax ( model ( inp )[0], dim = 0 )
        y_1 = torch.argmax ( y_hat )
        
        if y_1.detach().item() == y :
            for i in range(len(attacks)) :
                x_prime = attacks[i] ( x, eps, f_grad )
                if torch.max(torch.abs(x_prime-x)) > eps+1e-7 : 
                    print ( f"norme infinie supérieure à {eps} pour l'attaque {i}, norm = ", torch.max(torch.abs(x_prime-x)) )
                with torch.no_grad ( ) :
                    z_hat = model ( x_prime.view ( 1, *x_prime.shape ) )
                    alldatas[i].append ( (y,torch.argmax(z_hat[0]).detach().item()) )
        n += 1
        if (max_n is not None) and (n >= max_n) : break
    
    if n != 0 : return ( alldatas, n )

def getcont_accuracy_after_attacks ( model, dataset, attacks = None, eps = 1e-2, max_n = None ) :
    model.eval ( )
    datas = dataset.datas 
    
    if attacks is None :
        attacks = [ lambda img, epsilon, grad : img ]
    
    r_init, r_atk, n = 0, [0 for i in range(len(attacks))], 0
    
    k, mk = 5, 0
    examples = { z : [] for z in range(10) }
    model.set_latent(True)
    
    for x, z in datas :
        if len(examples[z.cpu().item()]) < k : 
            yh, zh = model ( x.view ( 1, *x.shape ) )
            examples[z.cpu().item()].append ( yh[0].cpu().detach().numpy() )
            mk += 1
        if mk == 50 : break
    
    
    for x, z in datas :
        def f_grad ( inp ) :
            model.zero_grad ( )
            model.set_latent ( False )
            inp = inp.view ( 1, *inp.shape )
            inp.requires_grad = True
            y_hat = F.log_softmax ( model ( inp )[0], dim = 0 )
            return torch.autograd.grad ( outputs = y_hat.sum() - 2 * y_hat[z], inputs = inp )[0][0]
        inp = x.view ( 1, *x.shape )
        inp.requires_grad = True
        model.set_latent(True)
        y_hat = model ( inp )[0][0].cpu().detach().numpy()
        z_hat = min ( examples, key = lambda kz : sum(np.sum((examples[kz][ik]-y_hat)**2) for ik in range(k)) )
        
        if z.cpu().item() == z_hat :
            r_init += 1
            for i in range(len(attacks)) :
                x_prime = attacks[i] ( x, eps, f_grad )
                if torch.max(torch.abs(x_prime-x)) > eps+1e-7 : 
                    print ( f"norme infinie supérieure à {eps} pour l'attaque {i}, norm = ", torch.max(torch.abs(x_prime-x)) )
                with torch.no_grad ( ) :
                    model.set_latent(True)
                    yy_hat = model ( x_prime.view ( 1, *x_prime.shape ) )[0][0].cpu().detach().numpy()
                    zz_hat = min ( examples, key = lambda kz : sum(np.sum((examples[kz][ik]-yy_hat)**2) for ik in range(k)) )
                    if zz_hat == z :
                        r_atk[i] += 1
        n += 1
        if (max_n is not None) and (n >= max_n) : break
    model.set_latent(False)
    if n != 0 : return ( r_init, r_atk, n )
            
            