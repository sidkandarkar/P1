from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        if torch.cuda.is_available():
            model.cuda()

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################

        # Epochs
        for epoch in range(num_epochs):
            # Iterations
            iteration = 0
            for imgs, lbls in train_loader:
                iteration += 1

                # Get images and labels
                images = Variable(imgs)
                labels = Variable(lbls)

                # Forward pass
                optim.zero_grad()               # zero the gradient buffers
                output = model(images)          # forward pass

                # Computation of loss and accuracy
                loss = self.loss_func(output, labels)

                # Backward pass
                loss.backward()

                # Optimization
                optim.step()

                # Log outcome every log_nth iteration
                if (iteration+1) % log_nth == 0:

                    # Log training loss
                    self.train_loss_history.append(loss.data[0])

                    print('Epoch [%d/%d], Iteration[%d/%d] Loss: %.4f'
                          %(epoch+1, num_epochs, iteration+1, iter_per_epoch,
                            loss.data[0]))

                # Log training accuracy
                _, predicted = torch.max(output, 1)
                correct = (predicted == labels).data.cpu().numpy()
                #num_correct = (predicted.int() == lbls.int()).sum()
                acc_train = np.mean(correct)

            self.train_acc_history.append(acc_train)

            # validate model
            num_labels = 0
            num_correct = 0
            scores = []
            for imgs, lbls in val_loader:

                # get images and labels of test data
                images = Variable(imgs)
                labels = Variable(lbls)

                # forward pass on test data
                output = model(images)

                # Compute loss
                loss_val = self.loss_func(output, labels)

                # Count number of correct and total labels
                _, predicted = torch.max(output, 1)
                scores.extend((predicted == labels).data.cpu().numpy())
                #num_labels += lbls.size()[0]
                #num_correct += (predicted == lbls).sum()

            # Compute val accuracy
            acc_val = np.mean(scores) #num_correct / num_labels
            self.val_loss_history.append(loss_val.data[0])
            self.val_acc_history.append(acc_val)

            print('Epoch [%d/%d] TRAIN loss: %.4f, acc: %.4f'
                  %(epoch+1, num_epochs, loss.data[0], acc_train))
            print('Epoch [%d/%d] VALID loss: %.4f, acc: %.4f'
                  %(epoch+1, num_epochs, loss_val.data[0], acc_val))

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')
