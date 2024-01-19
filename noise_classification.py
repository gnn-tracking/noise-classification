import torch
import torch_geometric as pyg
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, accuracy_score, roc_curve, roc_auc_score
from torch_geometric.data import Data
from torch import nn
import os

"""
Data aggregation
"""
data = pyg.data.Data()

for f_str in os.listdir('./data'):
    tmp = torch.load('./data/' + f_str)
    if data.x is None:
        data.x = tmp.x
        data.pt = tmp.pt
        data.particle_id = tmp.pt
    else:
        data.x = torch.concat((data.x, tmp.x))
        data.pt = torch.concat((data.pt, tmp.pt))
        data.particle_id = torch.concat((data.particle_id, tmp.particle_id))


"""
Trivial, ground-truth 'classifier'
"""
def ground_truth_classifier(data: Data, prob: float):
    gt_mask = data.particle_id == 0
    gt_mask =
    return (data.particle_id == 0)

class NoiseClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data: Data):
        return ground_truth_classifier(data)


class WithNoiseClassification(nn.Module, HyperparametersMixin):
    def __init__(self, noise_model, normal_model):
        super().__init__()
        self.noise_model = noise_model
        self.normal_model = normal_model

    def forward(self, data: Data):
        mask = self.noise_model(data)
        masked_data = data.subgraph(mask)
        out = self.normal_model(masked_data)
        out["hit_mask"] = mask
        return out

WithNoiseClassification(noise_model,
                        graph_construction_model)

# torch.count_nonzero(data.particle_id[data.pt < 0.9] == 0)
torch.count_nonzero(data.particle_id == 0)


X = data.x
y_noise = data.particle_id == 0
X_train, X_test, y_noise_train, y_noise_test = train_test_split(X, y_noise,
                                                    test_size=0.2)

"""
XGBoost, noise classification training
"""
num_pos = torch.count_nonzero(data.particle_id == 0)
scale_pos_weight = (data.particle_id.shape[0] - num_pos)/num_pos
bst = XGBClassifier(n_estimators=15, max_depth=5,
                    learning_rate=1, scale_pos_weight=float(scale_pos_weight), objective='binary:logistic')
bst.fit(X_train, y_noise_train)
preds = torch.tensor(bst.predict(X_test))


roc_curve(y_noise_test, preds)
roc_auc_score(y_noise_test, preds)
accuracy_score(y_noise_test, preds)
jaccard_score(y_noise_test, preds)

torch.sum(torch.count_nonzero(preds == y_noise_test))/y_noise_test.shape[0]



"""
XGBoost, low-pt training
"""


"""
Simple FNN, noise classification training
"""

"""
Simple FNN, low-pt training
"""


"""
Comparison
"""


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch_geometric as tg
# from sklearn.model_selection import train_test_split

# class ClassificationNetwork(nn.Module):
#     """
#     A very simple FCN
#     """""
#     def __init__(self, indim, size1, size2):
#         super(ClassificationNetwork,self).__init__()
#         self.l1 = nn.Linear(indim, size1)
#         self.l2 = nn.Linear(size2,10)
#         self.l3 = nn.Linear(10,1)

#     def forward(self,x):
#         x = F.tanh(self.l1(x))
#         x = F.tanh(self.l2(x))
#         x = F.sigmoid(self.l3(x))
#         return x

# data_init = torch.load("data21000_s0.pt")

# # solving the de-noising problem:
# X = data_init.x
# y = data_init.particle_id == 0

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# TrainDataset = torch.utils.data.TensorDataset(X_train, y_train)
# TestDataset = torch.utils.data.TensorDataset(X_test, y_test)

# TrainLoader = torch.utils.data.DataLoader(
#     TrainDataset,
#     batch_size=64,
#     shuffle=True,
# )

# TestLoader = torch.utils.data.DataLoader(
#     TestDataset,
#     batch_size=64,
#     shuffle=True,
# )

# def trainCLRclassification(model, trainLoader, valLoader, optimizer, criterion, tau, epochs, ls_list, valList, acc_list, loss_name= "sBQC", device= "cuda"):
#     """
#     Training loop used for CLR training
#     """
#     for epoch in range(epochs):
#         epoch_loss= 0.0
#         # training loop
#         model.train()
#         for inputs, labels in trainLoader:
#             inputs= inputs.to(device)
#             labels= labels.to(device)
#             optimizer.zero_grad()
#             outputs= model(inputs)
#             if loss_name== "BCE":
#                 loss= criterion(outputs.view(outputs.shape[0],), labels) # For BCE
#             elif loss_name== "sBQC":
#                 loss= criterion(labels, outputs.view(outputs.shape[0],), tau) # For sBQC
#             loss.backward()
#             optimizer.step()
#             epoch_loss+= loss.item()
#         ls_list.append(epoch_loss/len(trainLoader))

#         # validation loop
#         val_loss= 0.0
#         num_correct= 0
#         total= 0
#         model.eval()
#         for inputs, labels in valLoader:
#             inputs= inputs.to(device)
#             labels= labels.to(device)
#             outputs= model(inputs)
#             if loss_name== "BCE":
#                 loss= criterion(outputs.view(outputs.shape[0],), labels) # For BCE
#             elif loss_name== "sBQC":
#                 loss= criterion(labels, outputs.view(outputs.shape[0],), tau) # For sBQC
#             val_loss+= loss.item()
#             x= torch.where(outputs.view(outputs.shape[0]) > 0.5, 1, 0)
#             num_correct += (x==labels).sum()
#             total += labels.size(0)
#         valList.append(val_loss/len(valLoader))
#         acc_list.append(float(num_correct)/float(total)*100)
#         print("Epoch: {} Training Loss: {} Validation loss: {} Accuracy: {}".format(epoch, epoch_loss/len(trainLoader), val_loss/len(valLoader),
#          float(num_correct)/float(total)*100))
