import pandas as pd
import numpy as np
from torchvision.transforms.functional import vflip,hflip
import torchvision
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torch.autograd import Variable

batch_size = 128
max=0
device = torch.device('cuda')
device2=torch.device('cpu')
# 设置numpy随机种子
seed=42
np.random.seed(seed)
# 设置Python内置随机数生成器的种子
random.seed(seed)
# 设置PyTorch的随机种子
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
def generate_flip_grid(w, h, device):
    # used to flip attention maps
    x_ = torch.arange(w).view(1, -1).expand(h, -1)
    y_ = torch.arange(h).view(-1, 1).expand(-1, w)
    grid = torch.stack([x_, y_], dim=0).float().to(device)
    grid = grid.unsqueeze(0).expand(1, -1, -1, -1)
    grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
    grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
    grid[:, 0, :, :] = -grid[:, 0, :, :]
    return grid

def ACLoss(att_map1, att_map2, grid_l, output):
    flip_grid_large = grid_l.expand(output.size(0), -1, -1, -1)
    flip_grid_large = Variable(flip_grid_large, requires_grad = False)
    flip_grid_large = flip_grid_large.permute(0, 2, 3, 1)
    att_map2_flip = F.grid_sample(att_map2, flip_grid_large, mode = 'bilinear', padding_mode = 'border', align_corners=True)
    flip_loss_l = F.mse_loss(att_map1, att_map2_flip)
    return flip_loss_l
def update_teacher_variables(model, teacher_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for teacher_param, param in zip(teacher_model.parameters(), model.parameters()):
        teacher_param.data.mul_(alpha).add_(1 - alpha, param.data)

# 自定义数据增强函数
def add_gaussian_noise(image_array, mean=0.0, var=30):
    std = var**0.5
    noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped

def flip_image(image_array):
    return cv2.flip(image_array, 1)

def color2gray(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    gray_img_3d = image_array.copy()
    gray_img_3d[:, :, 0] = gray
    gray_img_3d[:, :, 1] = gray
    gray_img_3d[:, :, 2] = gray
    return gray_img_3d

class CustomDataset(Dataset):
    def __init__(self, phase,file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = [label - 1 for label in labels]
        
        self.transform = transform
        self.phase=phase

    def __len__(self):
        return len(self.file_paths)
    
    
    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        decoder_label = torch.zeros(7)  # 假设有7个类别
       
        decoder_label[label] = 1
        
        return image, decoder_label,idx

# 定义数据转换
# 新增数据增强的transform
train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
                transforms.RandomRotation(20),
                transforms.RandomCrop(224, padding=32)
            ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02,0.25)),
])


test_transform = transforms.Compose([

    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

# 损失函数部分


# 读取标签的 Excel 表格
file_path = 'train_labels.csv'
labels_df = pd.read_csv(file_path)

# 读取图像文件路径
image_folder_path = 'train'
file_paths = [os.path.join(image_folder_path, filename) for filename in os.listdir(image_folder_path)]
labels = labels_df['label'].astype(int).tolist()  # 假设标签列名为 'Label'
with open('noise02.txt', 'r') as file:
    # 逐行读取文件内容并去除换行符
    lines = file.readlines()

# 初始化空列表用于存储第二列数据
n_labels = []

# 遍历每一行
for line in lines:
    # 使用空格分割每一行，取第二列数据（索引为1），并转换为整数类型后添加到列表中
    n_labels.append(int(line.split()[1]))


# 创建自定义数据集
train_dataset = CustomDataset(phase='train',file_paths=file_paths, labels=labels, transform=train_transform)

# 创建数据加载器

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

file_path = 'test_labels.csv'
t_labels_df = pd.read_csv(file_path)

# 读取测试集图像文件路径
test_image_folder_path = 'test'
test_file_paths = [os.path.join(test_image_folder_path, filename) for filename in os.listdir(test_image_folder_path)]
test_labels = t_labels_df['label'].tolist()  # 假设测试集标签列名为 'Label'

# 创建测试集自定义数据集
test_dataset = CustomDataset(phase='test',file_paths=test_file_paths, labels=test_labels, transform=test_transform)

# 创建测试集数据加载器
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


class res18(nn.Module):
    def __init__(self):
        super(res18, self).__init__()

        # 自定义模型结构
        module = models.resnet18(pretrained=True)
        model_state_dict = module.state_dict()
        model_path = 'mv_epoch_17.pt'
        checkpoint = torch.load(model_path)
        checkpoint = checkpoint['state_dict']

        # 去掉预训练模型中参数名的 "module." 前缀
        checkpoint = {k.replace('feat_net.', ''): v for k, v in checkpoint.items()}
        for key in checkpoint:
            if (key == 'fc.weight') or (key == 'fc.bias'):
                pass
            else:
                model_state_dict[key] = checkpoint[key]

        module.load_state_dict(model_state_dict, strict=False)
        # 添加一个新的输出层
        
        self.features=nn.Sequential(*list(module.children())[:-2])
        self.features2 = nn.Sequential(*list(module.children())[-2:-1])
        self.fc = nn.Linear(512, 7)
    def forward(self, x):
        # 获取卷积部分的特征输出
        x = self.features(x)
        
        #### 1, 2048, 7, 7
        feature = self.features2(x)
        
        #### 1, 2048, 1, 1
        # 接下来的代码可以保持不变，如果需要，你可以在这里添加进一步的处理
        feature = feature.view(feature.size(0), -1)

        #这里补上从self.fc中提取参数
        params = list(self.parameters())
        fc_weights = params[-2].data
        fc_weights = fc_weights.view(1, 7, 512, 1, 1)
        fc_weights = Variable(fc_weights, requires_grad=False)
        

        # attention
        feat = x.unsqueeze(1) # N * 1 * C * H * W
        
        hm = feat * fc_weights
        hm = hm.sum(2) # N * self.num_labels * H * W
        

        out = self.fc(feature)
        return out,hm


model  =res18()
teacher=res18()
teacher.load_state_dict(model.state_dict()) 
model.to(device)
teacher.to(device)
for param in teacher.parameters():
    param.requires_grad = False
num_epochs =40# 定义迭代次数
learning_rate = 1e-3 # 学习率
optimizer =torch.optim.Adam(model.parameters(), lr=learning_rate)# 定义优化器
scheduler =  torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=10,gamma=0.1)
total_step = len(train_loader)
max=0.9041
global_step=0
for epoch in range(num_epochs):
    total_loss = 0
    
    count_samples=0
    for images,labels,indexes in train_loader:
        flipped_images = hflip(images)
        images, labels, indexes,flipped_images = images.to(device), labels.to(device), indexes.to(device) ,flipped_images.to(device)
        optimizer.zero_grad() # 清空上一个batch的梯度信息
        
        outputs,h1=model(images)
        _,h2=model(flipped_images)
        ag1,_=teacher(flipped_images)
        
        b_z = outputs.size(0)
        labels=torch.tensor(labels,requires_grad=False)
        f_outputs=F.softmax(ag1,dim=1)
        # 对模型输出进行softmax操作
        or_outputs=torch.nn.functional.softmax(outputs, dim=1)
        
        ce_loss=-torch.sum(labels*torch.log(or_outputs))/b_z
        flip_loss=-torch.sum(labels*torch.log(or_outputs))/b_z+0.9*F.kl_div(or_outputs.log(), f_outputs, reduction='batchmean')
        grid_l = generate_flip_grid(7, 7, device)
    
        flip_loss_l = ACLoss(h1, h2, grid_l, outputs)
        loss=flip_loss+0.2*flip_loss_l
        
        loss.backward()
         
        total_loss += loss.item()
        optimizer.step()
        global_step += 1
        update_teacher_variables(model, teacher, 0.999, global_step)
      
        total_loss += loss.item()
        
    average_loss = total_loss / len(train_loader)
    scheduler.step()
    print(f'Epoch [{epoch+1}/{40}], Loss: {average_loss}')  
        # 在每个迭代之后调用 scheduler.step() 来更新学习率
    model.eval()
    with torch.no_grad(): # 进行评测的时候网络不更新梯度
        correct = 0
        total = 0
        for images, labels,_ in test_loader:
            images, labels = images.to(device), labels.to(device)
            out= model(images)    
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            _, labels = torch.max(labels, 1)  # 将独热编码转换
            correct += (predicted == labels).sum().item()   
        print('accuarcy: {} %'.format(100 * correct/total))
        
       
    model.train()
    
