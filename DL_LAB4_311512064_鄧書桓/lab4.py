import torch
import torch.nn as nn
from dataloader import RetinopathyLoader
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import copy
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse

Batch_size_train = 4
Batch_size_test = 4
Learning_rate = 1e-3
Epochs_resnet18 = 10
Epochs_resnet50 = 5
Momentum = 0.9
use_pretrained = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_train = RetinopathyLoader("./dataset/train_resize", "train",
                                  transform=True)

dataset_test = RetinopathyLoader("./dataset/test_resize", "test",
                                 transform=False)

train_loader = DataLoader(
    dataset_train, batch_size=Batch_size_train, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=Batch_size_test)
train_len = len(train_loader)
print(train_len)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained):

    model_ft = None
    # input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        if use_pretrained:
            model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            model_ft = models.resnet18()
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        # input_size = 521

    elif model_name == "resnet50":
        """ Resnet50
        """
        if use_pretrained:
            model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            model_ft = models.resnet50()
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        # input_size = 521

    return model_ft


def evaluate(model, Loader, epoch):
    test_accuracy = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(Loader, desc='Epoch %d (eval)' % (epoch + 1)):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            output = model(inputs)

            _, Predicted = torch.max(output, 1)

            test_accuracy += ((Predicted == labels).sum().item()
                              ) / len(Loader.dataset)
        print('eval Acc: {:.4f}'.format(test_accuracy))
    return test_accuracy


def train_model(model_name, model, dataloaders, dataloaders_eval, optimizer, criterion, num_epochs=5, phase='train'):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_acc_history = []
    eval_acc_history = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs ))
        print('-' * 10)

        # Each epoch has a training and validation phase
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode or test mode

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(dataloaders, desc='Epoch (train) %d' % (epoch + 1)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders.dataset)
        epoch_acc = running_corrects.double() / len(dataloaders.dataset)
        print("acc = :", epoch_acc)
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))

        # evaluaing test data
        eval_accurarcy = evaluate(model, dataloaders_eval, epoch)
        eval_acc_history.append(eval_accurarcy)

        # deep copy the model
        if phase == 'train' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        #     train_acc_history.append(epoch_acc)
        # elif phase == 'test':
        #     train_acc_history.append(epoch_acc)
        train_acc_history.append(epoch_acc)
    model.load_state_dict(best_model_wts)

    if use_pretrained == True:
        torch.save(model.state_dict(), 'pretrain_' + model_name + '_train.pt')
        return model, train_acc_history, eval_acc_history
    else:
        torch.save(model.state_dict(), model_name + '_train.pt')
        return model, train_acc_history, eval_acc_history

    # if phase == 'train':
    #     if use_pretrained == True:
    #         torch.save(model.state_dict(), 'pretrain_' + model_name+ '_train.pt')
    #         return model, train_acc_history
    #     else:
    #         torch.save(model.state_dict(), model_name+ '_train.pt')
    #         return model, train_acc_history
    # else:
    #     if use_pretrained == True:
    #         torch.save(model.state_dict(), 'pretrain_' + model_name+ '_test.pt')
    #         return model, train_acc_history
    #     else:
    #         torch.save(model.state_dict(), model_name+ '_test.pt')
    #         return train_acc_history


def create_confusionmatrix(model, dataloaders):
    y_pred = []
    y_true = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloaders, desc="create coonfusion matrix"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # .view(-1)展平成一維張量
            # .detch()為斷開張量與計算圖的連接, 用來避免該張量後續進行反向傳播(去梯度操作)
            # confusion matrix 計算是使用numpy計算(使用cpu)
            y_pred.extend(preds.view(-1).detach().cpu().numpy())
            y_true.extend(labels.view(-1).detach().cpu().numpy())
    cf_matrix = confusion_matrix(y_true, y_pred)
    cf_matrix_norm = confusion_matrix(y_true, y_pred, normalize='true')
    return cf_matrix, cf_matrix_norm


def plot_confusion_matrix(cf_matrix, name):
    class_names = ['no DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    df_cm = pd.DataFrame(cf_matrix, class_names, class_names)
    sns.heatmap(df_cm, annot=True, cmap='Oranges')
    plt.title(name)
    plt.xlabel("prediction")
    plt.ylabel("laTbel (ground truth)")
    plt.savefig('Confusion_matrix' + name + '.png')
    plt.clf()


if __name__ == "__main__":

    # initial parameters
    parser = argparse.ArgumentParser(description="select the model")
    parser.add_argument('--demo', type=bool, default=False)
    args = parser.parse_args()
    args.demo_mode = args.demo

    # initial loss function
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # initial model
    model_name = "resnet18"
    use_feature_extract = False
    use_pretrained = True
    model_ft = initialize_model(
        model_name, 5, use_feature_extract, use_pretrained)
    model_ft = model_ft.to(device)

    # get the parameters list that need to update
    if use_feature_extract:
        params_to_update = []
        for param in model_ft.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
    else:
        params_to_update = model_ft.parameters()
    optimizer_ft = torch.optim.SGD(
        params_to_update, lr=Learning_rate, momentum=Momentum)

    # train/test model with pretrain and unpretrain weight
    model_after_pretrain, pretrain_hist_train, pretrain_hist_test = train_model(
        model_name, model_ft, train_loader, test_loader, optimizer_ft, criterion,  num_epochs=Epochs_resnet18)

    print("resnet 18 pretrain is done!!!!!!!!")
    # weight_18_pre = torch.load("pretrain_resnet18_train.pt")

    # no pretrain
    # initial model
    model_name = "resnet18"
    use_feature_extract = False
    use_pretrained = False
    model_ft = initialize_model(
        model_name, 5, use_feature_extract, use_pretrained)
    model_ft = model_ft.to(device)

    # get the parameters list that need to update
    if use_feature_extract:
        params_to_update = []
        for param in model_ft.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
    else:
        params_to_update = model_ft.parameters()
    optimizer_ft = torch.optim.SGD(
        params_to_update, lr=Learning_rate, momentum=Momentum)

    # train/test model with no pretrain and unpretrain weight
    model_after_train, hist_train, hist_test = train_model(
        model_name, model_ft, train_loader, test_loader, optimizer_ft, criterion,  num_epochs=Epochs_resnet18)

    print("resnet 18 is done!!!!!!!!")

    # plot accurarcy
    pretrain_hist_train = np.array(pretrain_hist_train)
    pretrain_hist_test = np.array(pretrain_hist_test)
    hist_train = np.array(hist_train)
    hist_test = np.array(hist_test)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Result Comparasion({})'.format(model_name))
    plt.plot(range(10), pretrain_hist_train, label='Train(with pretrained')
    plt.plot(range(10), pretrain_hist_test, label='Test(with pretrained)')
    plt.plot(range(10), hist_train, marker='o', label='Train(w/o pretrained)')
    plt.plot(range(10), hist_test, marker='o', label='Test(w/o pretrained)')
    plt.legend()
    plt.grid()
    plt.savefig('resnet18_Accurarcy.png')
    print('max accuracy', max(hist_test)*100, '%')
    print("plot resnet 18 is done!!!!!!!!")
    plt.clf()

    # weight_18 = torch.load("pretrain_resnet18_train.pt")
    # model_ft.load_state_dict(weight_18)
    # cf_matrix, cf_matrix_norm = create_confusionmatrix(
    #     model_ft, test_loader)
    # plot_confusion_matrix(cf_matrix_norm, model_name)

    cf_matrix, cf_matrix_norm = create_confusionmatrix(
        model_after_pretrain, test_loader)
    plot_confusion_matrix(cf_matrix_norm, model_name)

    # resnet 50
    # initial loss function
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # initial model
    model_name = "resnet50"
    use_feature_extract = False
    use_pretrained = True
    model_ft = initialize_model(
        model_name, 5, use_feature_extract, use_pretrained)
    model_ft = model_ft.to(device)

    # get the parameters list that need to update
    if use_feature_extract:
        params_to_update = []
        for param in model_ft.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
    else:
        params_to_update = model_ft.parameters()
    optimizer_ft = torch.optim.SGD(
        params_to_update, lr=Learning_rate, momentum=Momentum)

    # train/test model with pretrain and unpretrain weight
    model_after_pretrain, pretrain_hist_train, pretrain_hist_test = train_model(
        model_name, model_ft, train_loader, test_loader, optimizer_ft, criterion,  num_epochs=Epochs_resnet50)
    print("resnet 50 pretrain is done!!!!!!!!")


    # initial model(no pretrain)
    model_name = "resnet50"
    use_feature_extract = False
    use_pretrained = False
    model_ft = initialize_model(
        model_name, 5, use_feature_extract, use_pretrained)
    model_ft = model_ft.to(device)

    # get the parameters list that need to update
    if use_feature_extract:
        params_to_update = []
        for param in model_ft.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
    else:
        params_to_update = model_ft.parameters()
    optimizer_ft = torch.optim.SGD(
        params_to_update, lr=Learning_rate, momentum=Momentum)

    # train/test model with no pretrain and unpretrain weight
    model_after_train, hist_train, hist_test = train_model(
        model_name, model_ft, train_loader, test_loader, optimizer_ft, criterion,  num_epochs=Epochs_resnet50)

    print("resnet 50 is done!!!!!!!!")

    # plot accurarcy
    pretrain_hist_train = np.array(pretrain_hist_train)
    pretrain_hist_test = np.array(pretrain_hist_test)
    hist_train = np.array(hist_train)
    hist_test = np.array(hist_test)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Result Comparasion({})'.format(model_name))
    plt.plot(range(Epochs_resnet50), pretrain_hist_train, label='Train(with pretrained')
    plt.plot(range(Epochs_resnet50), pretrain_hist_test, label='Test(with pretrained)')
    plt.plot(range(Epochs_resnet50), hist_train, marker='o', label='Train(w/o pretrained)')
    plt.plot(range(Epochs_resnet50), hist_test, marker='o', label='Test(w/o pretrained)')
    plt.legend()
    plt.grid()
    plt.savefig('resnet50_Accurarcy.png')
    print('max accuracy', max(hist_test)*100, '%')
    plt.clf()

    cf_matrix, cf_matrix_norm = create_confusionmatrix(
        model_after_pretrain, test_loader)
    plot_confusion_matrix(cf_matrix_norm, model_name)
