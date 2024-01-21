import memnn as m
import numpy as np
import torch

lr = 0.001
margin = torch.tensor(0.1, dtype=torch.float32)
total_epochs = 100
embedding_dimension = 100
data_files = ["/home/celhage/Documents/ei2/DEEP/presentation/MNN/data/en-valid/qa1_test.txt", "/home/celhage/Documents/ei2/DEEP/presentation/MNN/data/en-valid/qa1_train.txt" ]

training_data = data_files[0]
test_data = data_files[1]

vocabDict = m.create_dict(data_files)

feature_space = 3*len(vocabDict)

out = m.I("John went to the hallway. John went to the garden.", vocabDict, torch.float32)

memory = []
memory = m.G(out, memory)

print(memory)

model = m.init_weights(np.float32, feature_space, embedding_dimension, 0.1)
training_avg_loss = m.train(data_files[1], model['u0'], model['ur'],vocabDict , lr, margin, torch.float32)
# o_accuracy, r_accuracy = m.trainingAccuracy()