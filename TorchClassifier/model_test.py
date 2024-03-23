from TorchClassifier.Datasetutil.Torchdatasetutil import loadTorchdataset
from TorchClassifier.myTorchModels.TorchCNNmodels import createTorchCNNmodel
import torch
import numpy as np
from collections import Counter

path = './plants'


class ModelTester():
    def __init__(self, data_name: str, data_path: str, data_type: str, model_path: str, model_name: str):
          
        self.dataloaders, self.dataset_sizes, self.class_names, self.imageshape = loadTorchdataset(name=data_name, 
                                                                                                   type=data_type, 
                                                                                                   path=data_path,
                                                                                                   img_height=244, 
                                                                                                   img_width=244, 
                                                                                                   batch_size=128)


        self.numclasses = 30
        self.checkpoint = torch.load(model_path)
        self.model_ft = createTorchCNNmodel(model_name, self.numclasses, img_shape=(244,244), pretrained=True).cuda()

        print(self.model_ft)
        self.model_ft.load_state_dict(self.checkpoint['state_dict'])
        self.model_ft.eval()

        self.corrects = 0
        self.incorrects = 0

        self.actual = np.array([])
        self.predicted = np.array([])

        self.per_class_result = {}

        for name in self.class_names:
            self.per_class_result[name] = [0,0]


        with torch.no_grad():
            for data in self.dataloaders['val']:
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                outputs = self.model_ft(images)

                    
                _, pred = torch.max(outputs, 1) 

                self.actual = np.append(self.actual, labels.cpu())
                self.predicted = np.append(self.predicted, pred.cpu())
                for i in range(len(labels)):
                    classname = self.class_names[labels[i]]
                    if pred[i] == labels[i]:
                        self.corrects += 1
                        self.per_class_result[classname][0] += 1
                    else:
                        self.incorrects += 1

                    # add result for each class
                    
                    self.per_class_result[classname][1] += 1
                
    def get_correct_incorrect(self):
        return self.corrects, self.incorrects
    
    def get_accuracy(self):
        return self.corrects / (self.corrects + self.incorrects)

    def get_class_results(self):
        return self.per_class_result
    

class EnsembleTester():
    def __init__(self, name: str, path: str, type: str, img_height: int, img_width: int, batch_size: int,
                 model_paths: list[str], model_names: list[str]):

        self.dataloaders, self.dataset_sizes, self.class_names, self.imageshape = loadTorchdataset(name=name, path=path, type=type, img_height=img_height, img_width=img_width, batch_size=batch_size)

        # load3 models
        self.list_of_models = []

        self.models_path = model_paths

        self.model_names = model_names

        self.list_of_models = self.intialize_models(self.models_path, self.model_names)

        self.correct = 0
        self.incorrect = 0

        self.model_predictions = [[] for i in range(len(self.list_of_models))]
        self.actual_label = []


        with torch.no_grad():
            for data in self.dataloaders['val']:
                images, labels = data
                images, labels = images.cuda(), labels.cuda()

                predictions = []

                for model in self.list_of_models:
                    output = model(images)
                    _, pred = torch.max(output, 1)
                    predictions.append(pred)


                majority_vote = []

                for i in range(len(labels)):
                    votes = [prediction[i].item() for prediction in predictions]
                    # votes = [pred_1[i].item(), pred_2[i].item(), pred_3[i].item()]
                    for j, vote in enumerate(votes):
                        self.model_predictions[j].append(vote)

                    
                    counter = Counter(votes)
                    
                    majority = counter.most_common(1)[0][0]

                    majority_vote.append(majority)

                # compare with real labels
                labels = labels.cpu()
                for i in range(len(labels)):
                    if pred[i] == labels[i]:
                        self.correct += 1
                    else:
                        self.incorrect += 1
                    self.actual_label.append(labels[i])

    def get_ensemble_acc(self):
        return self.correct / (self.correct + self.incorrect)

    def get_model_acc(self, preds, labels):
        correct, incorrect = 0, 0
        for i in range(len(preds)):
            if preds[i] == labels[i]: 
                correct += 1
            else:
                incorrect += 1

        return correct / (correct + incorrect)


    def intialize_models(self, model_paths, model_names):
        list_of_models = []

        for i, path in enumerate(model_paths): 
            model = createTorchCNNmodel(model_names[i], numclasses=3, img_shape=(300,300), pretrained=True).cuda()
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()

            list_of_models.append(model)

        return list_of_models
