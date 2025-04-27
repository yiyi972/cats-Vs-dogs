import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random


# 数据增强函数
def data_augmentation(img):
    img = Image.fromarray((img * 255).astype(np.uint8))
    # 随机水平翻转
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # 随机旋转
    angle = random.randint(-10, 10)
    img = img.rotate(angle)
    # 新增随机亮度调整
    from PIL import ImageEnhance
    if random.random() > 0.5:
        factor = random.uniform(0.8, 1.2)
        img = ImageEnhance.Brightness(img).enhance(factor)
    img = np.array(img) / 255.0
    return img


# ==================== 数据加载 ====================
class DataLoader:
    def __init__(self, image_size=64, train_ratio=0.8, num_samples=2000):
        self.image_size = image_size
        self.train_ratio = train_ratio
        self.train_dir = 'train'
        self.test_dir = 'test1'
        self.num_samples = num_samples

        # 获取训练图片路径
        all_files = [
            os.path.join(self.train_dir, f)
            for f in os.listdir(self.train_dir)
            if f.endswith('.jpg')
        ]

        # 分类猫狗图片
        cat_files = [
            f for f in all_files if 'cat' in os.path.basename(f).lower()
        ]
        dog_files = [
            f for f in all_files if 'dog' in os.path.basename(f).lower()
        ]

        # 限制数据量
        num_each_class = num_samples // 2
        cat_files = cat_files[:num_each_class]
        dog_files = dog_files[:num_each_class]

        # 合并猫狗图片并打乱
        all_selected_files = cat_files + dog_files
        all_labels = [0] * len(cat_files) + [1] * len(dog_files)
        combined = list(zip(all_selected_files, all_labels))
        random.shuffle(combined)
        all_selected_files, all_labels = zip(*combined)

        # 划分训练/验证集
        split_idx = int(len(all_selected_files) * train_ratio)
        self.train_files = all_selected_files[:split_idx]
        self.train_labels = all_labels[:split_idx]
        self.val_files = all_selected_files[split_idx:]
        self.val_labels = all_labels[split_idx:]

        # 测试集
        self.test_files = [
            os.path.join(self.test_dir, f)
            for f in os.listdir(self.test_dir)
            if f.endswith('.jpg')
        ]

    def load_image(self, path, augment=False):
        img = Image.open(path)
        img = img.resize((self.image_size, self.image_size))
        img = np.array(img) / 255.0

        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)

        if augment:
            img = data_augmentation(img)

        return img.transpose(2, 0, 1)

    def get_batch(self, batch_size, mode='train'):
        if mode == 'train':
            files, labels = self.train_files, self.train_labels
            augment = True
        elif mode == 'val':
            files, labels = self.val_files, self.val_labels
            augment = False
        else:
            files, labels = self.test_files, [0] * len(self.test_files)
            augment = False

        indices = np.random.choice(len(files), batch_size, replace=False)
        batch_images = np.array([self.load_image(files[i], augment) for i in indices])
        batch_labels = np.array([labels[i] for i in indices])
        return batch_images, batch_labels


# 批量归一化层（二维）
class BatchNorm2D:
    def __init__(self, num_channels, momentum=0.9, eps=1e-5):
        self.num_channels = num_channels
        self.momentum = momentum
        self.eps = eps
        self.gamma = np.ones(num_channels)
        self.beta = np.zeros(num_channels)
        self.running_mean = np.zeros(num_channels)
        self.running_var = np.ones(num_channels)
        self.cache = None

    def forward(self, x, mode='train'):
        n, c, h, w = x.shape
        x = x.transpose(0, 2, 3, 1).reshape(-1, c)

        if mode == 'train':
            sample_mean = np.mean(x, axis=0)
            sample_var = np.var(x, axis=0)
            x_norm = (x - sample_mean) / np.sqrt(sample_var + self.eps)
            out = self.gamma * x_norm + self.beta

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * sample_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * sample_var
            self.cache = (x, x_norm, sample_mean, sample_var)
        else:
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_norm + self.beta

        out = out.reshape(n, h, w, c).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        n, c, h, w = dout.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, c)
        x, x_norm, sample_mean, sample_var = self.cache

        N = x.shape[0]
        dx_norm = dout * self.gamma
        dvar = np.sum(dx_norm * (x - sample_mean) * (-0.5) * (sample_var + self.eps) ** (-3 / 2), axis=0)
        dmean = np.sum(dx_norm * (-1 / np.sqrt(sample_var + self.eps)), axis=0) + dvar * (-2 / N) * np.sum(
            x - sample_mean, axis=0)
        dx = dx_norm / np.sqrt(sample_var + self.eps) + dvar * (2 / N) * (x - sample_mean) + dmean / N
        dgamma = np.sum(dout * x_norm, axis=0)
        dbeta = np.sum(dout, axis=0)

        dx = dx.reshape(n, h, w, c).transpose(0, 3, 1, 2)
        return dx, dgamma, dbeta

    def update(self, dgamma, dbeta, lr):
        self.gamma -= lr * dgamma
        self.beta -= lr * dbeta


# 批量归一化层（一维）
class BatchNorm1D:
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.cache = None

    def forward(self, x, mode='train'):
        if mode == 'train':
            sample_mean = np.mean(x, axis=0)
            sample_var = np.var(x, axis=0)
            x_norm = (x - sample_mean) / np.sqrt(sample_var + self.eps)
            out = self.gamma * x_norm + self.beta

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * sample_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * sample_var
            self.cache = (x, x_norm, sample_mean, sample_var)
        else:
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_norm + self.beta
        return out

    def backward(self, dout):
        x, x_norm, sample_mean, sample_var = self.cache
        N = x.shape[0]
        dx_norm = dout * self.gamma
        dvar = np.sum(dx_norm * (x - sample_mean) * (-0.5) * (sample_var + self.eps) ** (-3 / 2), axis=0)
        dmean = np.sum(dx_norm * (-1 / np.sqrt(sample_var + self.eps)), axis=0) + dvar * (-2 / N) * np.sum(
            x - sample_mean, axis=0)
        dx = dx_norm / np.sqrt(sample_var + self.eps) + dvar * (2 / N) * (x - sample_mean) + dmean / N
        dgamma = np.sum(dout * x_norm, axis=0)
        dbeta = np.sum(dout, axis=0)
        return dx, dgamma, dbeta

    def update(self, dgamma, dbeta, lr):
        self.gamma -= lr * dgamma
        self.beta -= lr * dbeta


# Dropout 层
class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.cache = None

    def forward(self, x, mode='train'):
        if mode == 'train':
            mask = np.random.binomial(1, 1 - self.p, size=x.shape) / (1 - self.p)
            out = x * mask
            self.cache = mask
        else:
            out = x
        return out

    def backward(self, dout):
        mask = self.cache
        dx = dout * mask
        return dx


# ==================== 网络层实现 ====================
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.biases = np.zeros(out_channels)
        self.cache = None
        # Adam优化器参数
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_biases = np.zeros_like(self.biases)
        self.v_biases = np.zeros_like(self.biases)

    def forward(self, x):
        n, c, h, w = x.shape
        out_h = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kernel_size) // self.stride + 1

        x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))

        output = np.zeros((n, self.out_channels, out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                w_start = j * self.stride
                receptive_field = x_padded[:, :, h_start:h_start + self.kernel_size, w_start:w_start + self.kernel_size]
                output[:, :, i, j] = np.sum(receptive_field[:, None] * self.weights, axis=(2, 3, 4)) + self.biases

        self.cache = (x, x_padded)
        return output

    def backward(self, dout):
        x, x_padded = self.cache
        n, c, h, w = x.shape
        _, _, out_h, out_w = dout.shape

        dw = np.zeros_like(self.weights)
        db = np.sum(dout, axis=(0, 2, 3))
        dx_padded = np.zeros_like(x_padded)

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                w_start = j * self.stride
                receptive_field = x_padded[:, :, h_start:h_start + self.kernel_size, w_start:w_start + self.kernel_size]

                dw += np.sum(receptive_field[:, None] * dout[:, :, i, j][:, :, None, None, None], axis=0)
                dx_padded[:, :, h_start:h_start + self.kernel_size, w_start:w_start + self.kernel_size] += \
                    np.sum(self.weights[None] * dout[:, :, i, j][:, :, None, None, None], axis=1)

        dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding] if self.padding > 0 else dx_padded
        return dx, dw, db

    def update(self, dw, db, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
        # 更新权重
        self.m_weights = beta1 * self.m_weights + (1 - beta1) * dw
        self.v_weights = beta2 * self.v_weights + (1 - beta2) * (dw ** 2)
        m_hat_weights = self.m_weights / (1 - beta1 ** t)
        v_hat_weights = self.v_weights / (1 - beta2 ** t)
        self.weights -= lr * m_hat_weights / (np.sqrt(v_hat_weights) + eps)

        # 更新偏置
        self.m_biases = beta1 * self.m_biases + (1 - beta1) * db
        self.v_biases = beta2 * self.v_biases + (1 - beta2) * (db ** 2)
        m_hat_biases = self.m_biases / (1 - beta1 ** t)
        v_hat_biases = self.v_biases / (1 - beta2 ** t)
        self.biases -= lr * m_hat_biases / (np.sqrt(v_hat_biases) + eps)


class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None
        self.input_shape = None  # 新增：记录输入形状

    def forward(self, x):
        self.input_shape = x.shape  # 记录输入形状
        n, c, h, w = x.shape
        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1

        output = np.zeros((n, c, out_h, out_w))
        max_indices = np.zeros((n, c, out_h, out_w, 2), dtype=int)

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                w_start = j * self.stride
                window = x[:, :, h_start:h_start + self.pool_size, w_start:w_start + self.pool_size]
                output[:, :, i, j] = np.max(window, axis=(2, 3))

                for ni in range(n):
                    for ci in range(c):
                        flat_idx = np.argmax(window[ni, ci])
                        h_idx = flat_idx // self.pool_size
                        w_idx = flat_idx % self.pool_size
                        max_indices[ni, ci, i, j] = [h_start + h_idx, w_start + w_idx]

        self.cache = max_indices
        return output

    def backward(self, dout):
        max_indices = self.cache
        n, c, out_h, out_w = dout.shape
        dx = np.zeros(self.input_shape)  # 使用记录的输入形状初始化dx

        for i in range(out_h):
            for j in range(out_w):
                for ni in range(n):
                    for ci in range(c):
                        h_idx, w_idx = max_indices[ni, ci, i, j]
                        dx[ni, ci, h_idx, w_idx] += dout[ni, ci, i, j]
        return dx


class ReLU:
    def __init__(self):
        self.cache = None

    def forward(self, x):
        self.cache = x
        return np.maximum(0, x)

    def backward(self, dout):
        x = self.cache
        return dout * (x > 0)


class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        scale = np.sqrt(2.0 / in_features)
        self.weights = np.random.randn(in_features, out_features) * scale
        self.biases = np.zeros(out_features)
        self.cache = None
        # Adam优化器参数
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_biases = np.zeros_like(self.biases)
        self.v_biases = np.zeros_like(self.biases)

    def forward(self, x):
        self.cache = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, dout):
        x = self.cache
        dx = np.dot(dout, self.weights.T)
        dw = np.dot(x.T, dout)
        db = np.sum(dout, axis=0)
        return dx, dw, db

    def update(self, dw, db, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
        # 更新权重
        self.m_weights = beta1 * self.m_weights + (1 - beta1) * dw
        self.v_weights = beta2 * self.v_weights + (1 - beta2) * (dw ** 2)
        m_hat_weights = self.m_weights / (1 - beta1 ** t)
        v_hat_weights = self.v_weights / (1 - beta2 ** t)
        self.weights -= lr * m_hat_weights / (np.sqrt(v_hat_weights) + eps)

        # 更新偏置
        self.m_biases = beta1 * self.m_biases + (1 - beta1) * db
        self.v_biases = beta2 * self.v_biases + (1 - beta2) * (db ** 2)
        m_hat_biases = self.m_biases / (1 - beta1 ** t)
        v_hat_biases = self.v_biases / (1 - beta2 ** t)
        self.biases -= lr * m_hat_biases / (np.sqrt(v_hat_biases) + eps)


class SoftmaxCrossEntropy:
    def __init__(self):
        self.cache = None

    def forward(self, x, y):
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        n = y.shape[0]
        log_probs = -np.log(probs[range(n), y])
        loss = np.sum(log_probs) / n

        self.cache = (probs, y)
        return loss

    def backward(self):
        probs, y = self.cache
        n = y.shape[0]
        dx = probs.copy()
        dx[range(n), y] -= 1
        dx /= n
        return dx


# ==================== 网络架构 ====================
class CNN:
    def __init__(self):
        # 3个卷积层，增加卷积核数量
        self.conv1 = Conv2D(3, 64)  # 64->32
        self.bn1 = BatchNorm2D(64)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D()

        self.conv2 = Conv2D(64, 128)  # 32->16
        self.bn2 = BatchNorm2D(128)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D()

        self.conv3 = Conv2D(128, 256)  # 16->8
        self.bn3 = BatchNorm2D(256)
        self.relu3 = ReLU()
        self.pool3 = MaxPool2D()

        # 2个全连接层
        self.fc1 = Linear(256 * 8 * 8, 512)
        self.bn_fc1 = BatchNorm1D(512)
        self.relu4 = ReLU()
        self.dropout1 = Dropout(0.5)
        self.fc2 = Linear(512, 2)

    def forward(self, x, mode='train'):
        # 打印各层尺寸（调试用）
        print(f"Input shape: {x.shape}")

        x = self.conv1.forward(x)
        x = self.bn1.forward(x, mode)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        print(f"After pool1: {x.shape}")

        x = self.conv2.forward(x)
        x = self.bn2.forward(x, mode)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)
        print(f"After pool2: {x.shape}")

        x = self.conv3.forward(x)
        x = self.bn3.forward(x, mode)
        x = self.relu3.forward(x)
        x = self.pool3.forward(x)
        print(f"After pool3: {x.shape}")

        n = x.shape[0]
        x = x.reshape(n, -1)
        print(f"Flattened shape: {x.shape}")

        x = self.fc1.forward(x)
        x = self.bn_fc1.forward(x, mode)
        x = self.relu4.forward(x)
        x = self.dropout1.forward(x, mode)
        x = self.fc2.forward(x)
        return x

    def backward(self, dout):
        dout, dw2, db2 = self.fc2.backward(dout)
        dout = self.dropout1.backward(dout)
        dout = self.relu4.backward(dout)
        dout, dgamma_bn_fc1, dbeta_bn_fc1 = self.bn_fc1.backward(dout)
        dout, dw1, db1 = self.fc1.backward(dout)

        n = dout.shape[0]
        dout = dout.reshape(n, 256, 8, 8)

        dout = self.pool3.backward(dout)
        dout = self.relu3.backward(dout)
        dout, dgamma_bn3, dbeta_bn3 = self.bn3.backward(dout)
        dout, dw_conv3, db_conv3 = self.conv3.backward(dout)

        dout = self.pool2.backward(dout)
        dout = self.relu2.backward(dout)
        dout, dgamma_bn2, dbeta_bn2 = self.bn2.backward(dout)
        dout, dw_conv2, db_conv2 = self.conv2.backward(dout)

        dout = self.pool1.backward(dout)
        dout = self.relu1.backward(dout)
        dout, dgamma_bn1, dbeta_bn1 = self.bn1.backward(dout)
        dout, dw_conv1, db_conv1 = self.conv1.backward(dout)

        grads = {
            'conv1': {'dw': dw_conv1, 'db': db_conv1},
            'bn1': {'dgamma': dgamma_bn1, 'dbeta': dbeta_bn1},
            'conv2': {'dw': dw_conv2, 'db': db_conv2},
            'bn2': {'dgamma': dgamma_bn2, 'dbeta': dbeta_bn2},
            'conv3': {'dw': dw_conv3, 'db': db_conv3},
            'bn3': {'dgamma': dgamma_bn3, 'dbeta': dbeta_bn3},
            'fc1': {'dw': dw1, 'db': db1},
            'bn_fc1': {'dgamma': dgamma_bn_fc1, 'dbeta': dbeta_bn_fc1},
            'fc2': {'dw': dw2, 'db': db2}
        }
        return grads

    def update(self, grads, lr, t):
        self.conv1.update(grads['conv1']['dw'], grads['conv1']['db'], lr, t)
        self.bn1.update(grads['bn1']['dgamma'], grads['bn1']['dbeta'], lr)
        self.conv2.update(grads['conv2']['dw'], grads['conv2']['db'], lr, t)
        self.bn2.update(grads['bn2']['dgamma'], grads['bn2']['dbeta'], lr)
        self.conv3.update(grads['conv3']['dw'], grads['conv3']['db'], lr, t)
        self.bn3.update(grads['bn3']['dgamma'], grads['bn3']['dbeta'], lr)
        self.fc1.update(grads['fc1']['dw'], grads['fc1']['db'], lr, t)
        self.bn_fc1.update(grads['bn_fc1']['dgamma'], grads['bn_fc1']['dbeta'], lr)
        self.fc2.update(grads['fc2']['dw'], grads['fc2']['db'], lr, t)


# 学习率调度器
def lr_scheduler(epoch, lr):
    # 余弦退火学习率调度器
    T_max = 10
    lr = 0.001 * (1 + np.cos(np.pi * epoch / T_max)) / 2
    return lr


# ==================== 训练和测试 ====================
def train(epochs=3 , batch_size=8, lr=0.001, num_samples=2000):
    loader = DataLoader(num_samples=num_samples)
    model = CNN()
    criterion = SoftmaxCrossEntropy()

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    t = 1  # Adam优化器的时间步

    for epoch in range(epochs):
        lr = lr_scheduler(epoch, lr)

        # 训练阶段
        model_loss = 0.0
        correct = 0
        total = 0
        num_batches = len(loader.train_files) // batch_size

        for _ in range(num_batches):
            x, y = loader.get_batch(batch_size, 'train')
            outputs = model.forward(x, 'train')
            loss = criterion.forward(outputs, y)

            dout = criterion.backward()
            grads = model.backward(dout)

            # 更新参数
            model.update(grads, lr, t)
            t += 1

            model_loss += loss
            predicted = np.argmax(outputs, axis=1)
            correct += (predicted == y).sum()
            total += y.size

        train_acc = correct / total
        train_loss = model_loss / num_batches
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 验证阶段
        val_loss, val_acc = evaluate(model, criterion, loader, batch_size)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}\n')

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend()
    # 保存训练曲线图像
    plt.savefig('training_curves.png')
    plt.show()

    return model


def evaluate(model, criterion, loader, batch_size):
    val_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(loader.val_files) // batch_size

    for _ in range(num_batches):
        x, y = loader.get_batch(batch_size, 'val')
        outputs = model.forward(x, 'val')
        loss = criterion.forward(outputs, y)

        val_loss += loss
        predicted = np.argmax(outputs, axis=1)
        correct += (predicted == y).sum()
        total += y.size

    val_acc = correct / total
    val_loss /= num_batches
    return val_loss, val_acc


def show_test_results(model, num_images=2):
    loader = DataLoader()
    x, _ = loader.get_batch(num_images, 'test')
    outputs = model.forward(x, 'val')
    probs = np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)

    plt.figure(figsize=(15, 3))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        img = x[i].transpose(1, 2, 0)
        plt.imshow(img)
        plt.title(f'Cat: {probs[i, 0]:.2f}\nDog: {probs[i, 1]:.2f}')
        plt.axis('off')
    # 保存测试结果图像
    plt.savefig('test_results.png')
    plt.show()


# ==================== 主程序 ====================
if __name__ == '__main__':
    print("开始训练模型...")
    trained_model = train(epochs=10, batch_size=8, lr=0.001, num_samples=2000)

    print("\n测试模型...")
    show_test_results(trained_model, num_images=2)