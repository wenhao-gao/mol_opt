import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import tensorflow as tf

class Logger(object):

    def __init__(self, log_dir='./logs/'):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def predict(self, x):
        output = self.forward(x)
        _, prediction = torch.max(output, 1)
        return prediction

dataset_dir = './MNIST/'
transform = transforms.Compose([transforms.Resize((32, 32)),
                                transforms.ToTensor()])
batch_size = 64

train_dataset = torchvision.datasets.MNIST(root=dataset_dir, train=True, transform=transform, download=True)
val_dataset = torchvision.datasets.MNIST(root=dataset_dir, train=False, transform=transform, download=True)

print('train dataset: {} \nval dataset: {}'.format(len(train_dataset), len(val_dataset)))

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
net = Net()
net.to(device)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
loss_fc = nn.CrossEntropyLoss()
logger = Logger('./logs')

iter_count = 0
NUM_EPOCH = 300
for epoch in range(NUM_EPOCH):

    running_loss = 0.0
    tr_loss = 0.0
    tr_acc = 0.0
    ts_acc = 0.0
    tr_total = 0
    tr_correct = 0
    ts_total = 0
    ts_correct = 0


    scheduler.step()
    print('EPOCH %i / %i:' %(epoch, NUM_EPOCH))
    tqdm_data = tqdm(train_dataloader, desc='MNIST training (epoch #{})'.format(epoch))
    for i, sample_batch in enumerate(tqdm_data):
        inputs = sample_batch[0].to(device)
        labels = sample_batch[1].to(device)

        # Set up the preperation of network and optimizer
        net.train()
        optimizer.zero_grad()

        outputs = net(inputs)

        loss = loss_fc(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        tr_total += labels.size(0)
        tr_correct += (torch.max(outputs, 1)[1] == labels).sum().item()

        if (i + 1) % 200 == 0:
            # test
            for sample_batch in val_dataloader:
                inputs = sample_batch[0].to(device)
                labels = sample_batch[1].to(device)

                net.eval()
                prediction = net.predict(inputs)
                ts_correct += (prediction == labels).sum().item()
                ts_total += labels.size(0)

            tr_loss = running_loss / 200
            tr_acc = tr_correct / tr_total
            ts_acc = ts_correct / ts_total
            iter_count += 200

            print ('Epoch [{}/{}], Loss: {:.4f}, Train Acc: {:.2f}, Test Acc: {:.2f}'
                   .format(epoch+1, NUM_EPOCH, tr_loss, tr_acc, ts_acc))

            # 1. Log scalar values (scalar summary)
            # 日志输出标量信息（scalar summary）
            info = { 'loss': tr_loss, 'train accuracy': tr_acc, 'test accuracy': ts_acc}

            for tag, value in info.items():
                logger.scalar_summary(tag, value, iter_count)

            # 2. Log values and gradients of the parameters (histogram summary)
            # 日志输出参数值和梯度（histogram summary)
            for tag, value in net.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), iter_count)
                logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), iter_count)

            # 3. Log training images (image summary)
            # 日志输出图像(image summary)
#             info = { 'images': images.view(-1, 28, 28)[:10].cpu().numpy() }

#             for tag, images in info.items():
#                 logger.image_summary(tag, images, iter_count)



            running_loss = 0
            tr_total = 0
            tr_correct = 0
            ts_total = 0
            ts_correct = 0

print('Train finish!')

model_dir = './model/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
torch.save(net.state_dict(), os.path.join(model_dir, 'model_static_dict.pth'))
