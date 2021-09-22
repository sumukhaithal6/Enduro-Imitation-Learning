#!/usr/bin/env python3
"""
Model train routines.

Authors:
LICENCE:
"""

import torch
import torch.nn as nn

def trainer(model, dataloader, optimizer, args):

    loss_fn = nn.CrossEntropyLoss()
    log_interval = 50

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}")

        for batch_idx, data in enumerate(dataloader):
            states, action = data
            
            states = states.to(args.device)
            action = action.to(args.device, dtype=torch.long)
            
            optimizer.zero_grad()
            
            output = model(states)
            loss = loss_fn(output, action)

            if((batch_idx+1)%log_interval==0):
                print(f"Batch {batch_idx} Loss: {loss}")
            
            loss.backward()
            optimizer.step()

    

        

