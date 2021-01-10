import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import numpy as np
import math


class EarlyStoppingCallback:
    def __init__(self, early_stopping_patience=100000):
        self._early_stopping_patience = early_stopping_patience
        self.save_loss = 1000
        self.sum = -1

    def step(self,validation_loss):
        if validation_loss < self.save_loss:
            self.save_loss = validation_loss
            self.sum += 0
        else:
            self.sum += 1
        
        if self.sum > self._early_stopping_patience:
            return True
        else:
            return False


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience = 20):  # The patience for early stopping

        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._early_stopping_patience = early_stopping_patience
        self._early_stopping_callback = EarlyStoppingCallback()
        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
        self.counter = 0
            
    def save_checkpoint(self, epoch):
        # why don't we direct save the modul and added a path to the location
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
                                         # 将训练后的参数权重存储在模型文件中
              opset_version=10,          # the ONNX version to export the model to 用于将模型导出到的ONNX版本
              do_constant_folding=True,  # whether to execute constant folding for optimization #是否执行常量折叠以进行优化
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients
        # optimizer.zero_grad()
        self._optim.zero_grad()
        # -propagate through the network
        output = self._model(x)
        # -calculate the loss
        loss = self._crit(output,y)
        # -compute gradient by backward propagation
        loss.backward()
        # optimizer.step()   
        self._optim.step()   
        # -update weights
        # -return the loss
        return loss
        
    def val_test_step(self, x, y):
        
        # predict
        # propagate through the network and calculate the loss and predictions
        output = self._model(x)
        loss = self._crit(output,y)
        output[output>=0.5]=1
        output[output<0.5]=0
        # output = np.around(output)
        # return the loss and the predictions
        # print("loss: " + loss)
        # print("output: " + output)
        return loss,output
        
    def train_epoch(self):
        # set training mode
        self._model.train()
        # iterate through the training set#遍历训练集

        # train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
        # for epoch in range(EPOCH):
        #     for step, (x, y) in enumerate(train_loader):   
        for epoch in range(2):#  loop over the dataset multiple times
            running_loss = 0.0
            for i, (x, y) in enumerate(self._train_dl):
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()
                
                running_loss +=self.train_step(x,y).item()
        return running_loss/(self._train_dl.__len__())
                
        # gives batch data
        # transfer the batch to "cuda()" -> the gpu if a gpu is given＃将批处理转移到“ cuda（）”->如果已指定gpu，则将gpu
        # perform a training step执行训练步骤
        # calculate the average loss for the epoch and return it计算该时期的平均损失并返回
        #TODO

    def val_test(self):
        # set eval mode＃设置评估模式
        self._model.eval()
        running_loss = 0.0
        inactive_len = 0
        inactive_sum=0
        crack_len=0
        crack_sum=0
        inactive_accuracy=0
        crack_accuracy=0
        # disable gradient computation＃禁用梯度计算
        pred_list=[]
        label_list=[]
        y_true = []
        y_pred = []
        with t.no_grad():
            for i, (x, y) in enumerate(self._val_test_dl ):
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()
                loss,pred =self.val_test_step(x,y)
                pred = pred.cpu()#返回cpu计算
                y = y.cpu()
                for m in y:
                    y_true.append(transform1d(m))
                
                for m in pred:                    
                    y_pred.append(transform1d(m))
                running_loss +=loss.item()
                average_loss = running_loss/self._val_test_dl.__len__()
                # score = model(images)
                # prediction = torch.argmax(score, dim=1)
                # num_correct = torch.sum(prediction == labels).item()
                # accuruacy = num_correct / labels.size(0)

                # prediction = t.argmax(x, dim=1)
                # num_correct = t.sum(prediction == y).item()
                # accuracy += num_correct / y.size(0)

                # transform the label_list to 1 Dimension

            for pred_batch in pred_list:
                for pred in pred_batch:                    
                    y_pred.append(transform1d(pred))
                
            for i in range(len(y_pred)):
                if y_true[i]==1 or y_true[i]==3:
                    inactive_len += 1
                    if y_pred[i]== y_true[i]:
                        inactive_sum += 1
                if y_true[i]==2 or y_true[i]==3:
                    crack_len += 1
                    if y_pred[i]== y_true[i]:
                        crack_sum += 1
            mean_F1_score= f1_score(y_true=y_true,y_pred=y_pred,average ='micro')
            crack_accuracy = crack_sum/inactive_len
            inactive_accuracy = inactive_sum/inactive_len
            print("crack_accuracy: %.3f%%" %(crack_accuracy*100))
            print("inactive_accuracy: %.3f%%" %(inactive_accuracy*100))
            print("mean_F1_score %.2f" %(mean_F1_score))
        return average_loss

        # iterate through the validation set ＃遍历验证集
        # transfer the batch to the gpu if given＃如果有批处理，将批次转移到GPU
        # perform a validation step＃执行验证步骤
        # save the predictions and the labels for each batch＃保存每个批次的预测和标签
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        #＃计算您选择的平均损失和平均指标。您可能要在指定的函数中计算这些指标
        # return the loss and print the calculated metrics＃返回损失并打印计算出的指标
        #TODO
    
    def fit(self, epochs):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        #TODO
        train_loss_list = []
        validation_loss_list = []
        # train_loss_list.append(self.val_test())
        # validation_loss_list.append(self.train_epoch())
        
        while True:

            if  self.counter >= epochs:
                return train_loss_list,validation_loss_list
            
            else:

                train_loss = self.train_epoch()
                validation_loss = self.val_test()
                # stop by epoch number＃按纪元编号停止
                # train for a epoch and then calculate the loss and metrics on the validation set＃训练一个时期，然后根据验证集计算损失和指标

                # append the losses to the respective lists＃将损失附加到各自的列表中
                train_loss_list.append(train_loss)
                validation_loss_list.append(validation_loss)
                # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)＃使用save_checkpoint函数保存模型（可以限制在改进的时期）
                self.save_checkpoint(self.counter)
                # early_stopping(valid_loss, model)
                # self._early_stopping_callback()
                 
                if self._early_stopping_callback.step(validation_loss):
                    return train_loss_list, validation_loss_list
            
                else:               
                    self.counter += 1
                    print("counter: ", self.counter)
                    print("elp: ", self._early_stopping_patience)

            # check whether early stopping should be performed using the early stopping criterion and stop if so＃检查是否应使用提前停止标准执行提前停止，如果是，则停止
   
            # return the losses for both training and validation＃返回培训和验证的损失
        #TODO

def transform1d(x):

    if x[0] == 0 and x[1] == 0:
        return 0
    elif x[0] == 0 and x[1] == 1:
        return 1
    elif x[1] == 1 and x[1] == 0:
        return 2
    else:
        return 3


        
        
        