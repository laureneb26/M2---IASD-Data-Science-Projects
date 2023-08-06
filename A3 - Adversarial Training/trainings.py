import sys
import time

import torch
import torch.nn.functional as F
import numpy as np

def train_simple(model, dataset, batch_size=32, epochs=100, learning_rate=1e-3, log_enabled=True):
    model.train()
    model.zero_grad()
    
    datas = dataset.make(batch_size)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(epochs):
        m_loss = 0.
        n = 0
        t0 = time.perf_counter()

        for x, y in datas:
            optimizer.zero_grad()

            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            m_loss += loss.detach().item()
            n += 1

        if log_enabled:
            log_str = "epoch {}/{}:".format(epoch, epochs) + " >>> loss = {:.5f} | time per epoch = {:.2f}sec(n:{})".format(m_loss, time.perf_counter() - t0, n)
            print(log_str, end = '\r')
            sys.stdout.flush()
    print('')

def get_grad ( model, x, y ) :
    x.requires_grad = True
    y_hat = F.log_softmax ( model ( x ), dim = 1 )
    y0 = torch.Tensor ( range(y.shape[0]) ).to ( device = 'cuda', dtype = torch.long )
    otp = y_hat.sum() - 2*y_hat[y0,y].sum()
    otp.backward ( )
    return x.grad
    # return torch.autograd.grad ( outputs = otp, inputs = x )[0][0]

def get_grad2 ( model, x, y ) :
    loss_fn = torch.nn.CrossEntropyLoss ( )
    x.requires_grad = True
    y_hat = model ( x )
    loss = loss_fn ( y_hat, y )
    loss.backward ( )
    return x.grad

def train_adv(model, dataset, batch_size=32, epochs=100, learning_rate=1e-3, log_enabled=True):
    model.train()
    model.zero_grad()
    
    datas = dataset.make(batch_size)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    for epoch in range(epochs):
        m_loss = 0.
        n = 0
        t0 = time.perf_counter()
        
        for xo, y in datas :
            optimizer.zero_grad()
            xp = xo + np.random.uniform(0.,0.25) * get_grad ( model, xo, y ).sign()
            model.zero_grad()
            yo_hat = model ( xo )
            yp_hat = model ( xp )
            losso = loss_fn ( yo_hat, y )
            lossp = loss_fn ( yp_hat, y )
            loss = losso + lossp
            loss.backward()
            optimizer.step()
            
            m_loss += loss.detach().item()
            n += 1
        
        if log_enabled:
            log_str = "epoch {}/{}:".format(epoch, epochs) + " >>> loss = {:.5f} | time per epoch = {:.2f}sec(n:{})".format(m_loss, time.perf_counter() - t0, n)
            print(log_str, end = '\r')
            sys.stdout.flush()
    print('')

def train_advpgd(model, dataset, batch_size=32, epochs=100, learning_rate=1e-3, log_enabled=True):
    model.train()
    model.zero_grad()
    
    datas = dataset.make(batch_size)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    for epoch in range(epochs):
        m_loss = 0.
        n = 0
        t0 = time.perf_counter()
        
        for xo, y in datas :
            optimizer.zero_grad()
            xp = xo.clone()
            for iter in range(4):
                xp = xo + 5/255 * get_grad ( model, xo, y ).sign()
                ep = torch.clamp ( xp - xo, min=-0.05, max=+0.05 )
                xp = xo + ep
            model.zero_grad()
            yo_hat = model ( xo )
            yp_hat = model ( xp )
            losso = loss_fn ( yo_hat, y )
            lossp = loss_fn ( yp_hat, y )
            loss = losso + lossp
            loss.backward()
            optimizer.step()
            
            m_loss += loss.detach().item()
            n += 1
        
        if log_enabled:
            log_str = "epoch {}/{}:".format(epoch, epochs) + " >>> loss = {:.5f} | time per epoch = {:.2f}sec(n:{})".format(m_loss, time.perf_counter() - t0, n)
            print(log_str, end = '\r')
            sys.stdout.flush()
    print('')

def train_cont(model, dataset, batch_size=32, epochs=100, learning_rate=1e-3, log_enabled=True):
    model.train()
    model.zero_grad()
    model.set_latent(True)
    
    m = 2.
    
    datas_1 = dataset.make(batch_size)
    datas_2 = dataset.make(batch_size)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    for epoch in range(epochs):
        m_loss = 0.
        n = 0
        t0 = time.perf_counter()
        
        for (x1,z1), (x2,z2) in zip ( datas_1, datas_2 ) :
            optimizer.zero_grad()
            y1h, z1h = model ( x1 )
            y2h, z2h = model ( x2 )
            loss_1 = loss_fn ( z1h, z1 )
            loss_2 = loss_fn ( z2h, z2 )
            distances = torch.norm ( y1h - y2h, dim = 1 )
            loss_cl = torch.where ( z1 == z2, torch.square ( distances ), torch.square ( F.relu ( m - distances ) ) ).mean ( )

            loss = loss_1 + loss_2 + loss_cl
            loss.backward()
            optimizer.step()
            
            m_loss += loss.detach().item()
            n += 1
        
        if log_enabled:
            log_str = "epoch {}/{}:".format(epoch, epochs) + " >>> loss = {:.5f} | time per epoch = {:.2f}sec(n:{})".format(m_loss, time.perf_counter() - t0, n)
            print(log_str, end = '\r')
            sys.stdout.flush()
    print('')
    model.set_latent(False)

def train_advtriplet(model, dataset, batch_size=32, epochs=100, learning_rate=1e-3, log_enabled=True):
    model.train()
    model.zero_grad()
    
    m = 3.
    
    datas_1 = dataset.make(batch_size)
    datas_2 = dataset.make(batch_size)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    for epoch in range(epochs):
        m_loss = 0.
        m_loss_cl = 0.
        n = 0
        t0 = time.perf_counter()
        
        for (x1,z1), (x2,z2) in zip ( datas_1, datas_2 ) :
            optimizer.zero_grad()
            model.set_latent(False)
            x3 = x1 + 0.05 * get_grad ( model, x1, z1 ).sign()
            model.set_latent(True)
            optimizer.zero_grad()
            y1h, z1h = model ( x1 )
            y2h, z2h = model ( x2 )
            y3h, z3h = model ( x3 )
            loss_1 = loss_fn ( z1h, z1 )
            loss_2 = loss_fn ( z2h, z2 )
            loss_3 = loss_fn ( z3h, z1 )
            d12 = torch.square ( y1h - y2h ).sum(axis=1)
            d13 = torch.square ( y1h - y3h ).sum(axis=1)
            
            loss_cl = F.relu ( d13 - 10/9 * torch.where ( z1 != z2, d12.double(), 0. ) + m ).mean()
            
            loss = loss_1 + loss_2 + loss_3 + loss_cl
            loss.backward()
            optimizer.step()
            
            m_loss_cl += loss_cl.detach().item()
            m_loss += loss.detach().item()
            n += 1
        
        if log_enabled:
            log_str = "epoch {}/{}:".format(epoch, epochs) + " >>> loss = {:.5f} | time per epoch = {:.2f}sec(n:{},cl:{:.4f}%)".format(m_loss, time.perf_counter() - t0, n, 100*m_loss_cl/m_loss)
            print(log_str, end = '\r')
            sys.stdout.flush()
    print('')
    model.set_latent(False)
