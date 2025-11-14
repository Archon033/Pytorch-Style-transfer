import time
from torch import torch
from torch import optim
from torch import nn
from torchvision.transforms import transforms
from torchvision.models import vgg19_bn
from PIL import Image
import os

# 指定模型缓存目录
os.environ["TORCH_HOME"] = "./model"


# TV Loss类
class TVLoss(nn.Module):
    def __init__(self, strength=1e-4):
        super().__init__()
        self.strength = strength

    def forward(self, input):
        h_diff = input[:, :, 1:, :] - input[:, :, :-1, :]
        v_diff = input[:, :, :, 1:] - input[:, :, :, :-1]

        # 改进后的归一化方式：按实际差异像素数归一化
        normalization_factor = max(input.shape[2], input.shape[3])  # 使用图像的最大边长
        loss = (
            self.strength
            * (torch.sum(torch.abs(h_diff)) + torch.sum(torch.abs(v_diff)))
            / normalization_factor
        )  # 关键修改点

        return loss


# 加载图像
def load_image(image_path, max_size=400):  # 因设备限制，这里设置最大尺寸为400
    image = Image.open(image_path).convert("RGB")

    # 如果图片过大则调整大小
    size = min(max_size, max(image.size))

    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Normalize(mean=[0.45, 0.4, 0.4], std=[0.2, 0.2, 0.2]),
        ]
    )

    image = transform(image).unsqueeze(0)  # 添加批次维度
    return image


# 保存图像
def save_image(tensor, path):
    image = tensor.clone().detach()
    image = image.squeeze(0)  # 去掉批次维度

    image = transforms.ToPILImage()(image)
    image.save(path)


# 定义VGG19模型，只提取特定层的特征
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = vgg19_bn(weights="DEFAULT").features.eval()

        # 风格层
        self.style_layers = {
            1,
            6,
            11,
            20,
            29,
        }  # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
        # 内容层
        self.content_layers = {21, 30}  # conv4_2 + conv5_2

        # # 实验 A1 - 单个靠前层
        # self.content_layers = {6}  # conv2_1
        # # 风格层 - 偏重浅层特征
        # self.style_layers = {1, 6, 11}  # conv1_1, conv2_1, conv3_1

        # 实验 A2 - 两个靠前层
        # # 内容层 - 两个靠前层
        # self.content_layers = {6, 11}  # conv2_1 + conv3_1
        # # 风格层 - 均衡分布
        # self.style_layers = {1, 6, 11, 15}  # conv1_1, conv2_1, conv3_1, conv3_2

        # 实验 A3 - 三个靠前层
        # 内容层 - 三个靠前层
        # self.content_layers = {1, 6, 11}  # conv1_1 + conv2_1 + conv3_1
        # # 风格层 - 全范围分布
        # self.style_layers = {1, 6, 11, 20, 29}  # conv1_1到conv5_1

        # # # 实验 B1 - 单个靠后层
        # 内容层 - 单个靠后层
        # self.content_layers = {29}  # conv5_1
        # # 风格层 - 偏重深层特征
        # self.style_layers = {15, 20, 29}  # conv3_2, conv4_1, conv5_1

        # # # 实验 B2 - 两个靠后层
        # # 内容层 - 两个靠后层
        # self.content_layers = {20, 29}  # conv4_1 + conv5_1
        # # 风格层 - 均衡分布
        # self.style_layers = {11, 15, 20, 29}  # conv3_1到conv5_1

        # # # # 实验 B3 - 三个靠后层
        # # # 内容层 - 三个靠后层
        # self.content_layers = {15, 20, 29}  # conv3_2 + conv4_1 + conv5_1
        # # 风格层 - 全范围分布
        # self.style_layers = {1, 6, 11, 20, 29}  # conv1_1到conv5_1

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.style_layers or i in self.content_layers:
                features.append(x)
        return features


# 内容损失函数
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        return nn.functional.mse_loss(input, self.target)


# 风格损失函数
class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target).detach()

    def gram_matrix(self, input):
        batch_size, channels, height, width = input.size()
        features = input.view(batch_size * channels, height * width)
        G = torch.mm(features, features.t())
        return G.div(batch_size * channels * height * width)

    def forward(self, input):
        G = self.gram_matrix(input)
        return nn.functional.mse_loss(G, self.target)


def post_process(tensor):
    # 添加轻微的高斯模糊以减少噪点
    blur = transforms.GaussianBlur(kernel_size=3, sigma=0.5)
    return blur(tensor)


# 图像风格迁移
def style_transfer(
    content_img,
    style_img,
    num_steps=700,  # 700轮次
    style_weight=1e9,
    content_weight=1,
    tv_weight=1e-6,  # 设tv_weight为1e-6
    progress_callback=None,  # 新增TV Loss权重参数
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化TV Loss模块
    tv_loss = TVLoss(strength=tv_weight).to(device)

    # 动态权重参数（使用列表实现闭包内修改）
    alpha = [content_weight]
    beta = [style_weight]

    content_img = content_img.to(device)
    style_img = style_img.to(device)

    model = VGG().to(device)

    # 提取风格和内容特征
    style_features = model(style_img)
    content_features = model(content_img)

    # 初始化输入图像（使用内容图像作为初始图像）
    input_img = content_img.clone().requires_grad_(True).to(device)

    # 定义优化器
    optimizer = optim.LBFGS([input_img])

    style_losses = []
    content_losses = []

    # 创建损失模块
    for sf, cf in zip(style_features, content_features):
        content_losses.append(ContentLoss(cf))
        style_losses.append(StyleLoss(sf))

    run = [0]
    while run[0] <= num_steps:

        def closure():
            optimizer.zero_grad()

            # 动态调整权重（每100次迭代）
            if run[0] % 100 == 0 and run[0] != 0:
                alpha[0] *= 0.8  # 内容权重缓慢减小
                beta[0] *= 1.5  # 风格权重缓慢增加
                print(f"\n调整权重：α={alpha[0]:.1e}, β={beta[0]:.1e}")

            input_features = model(input_img)
            content_loss = 0
            style_loss = 0

            # 计算内容损失
            content_loss = sum(
                alpha[0] * cl(input_f)
                for cl, input_f in zip(content_losses, input_features)
            )

            # 计算风格损失
            style_loss = sum(
                beta[0] * sl(input_f)
                for sl, input_f in zip(style_losses, input_features)
            )

            # 新增TV Loss计算
            tv_loss_value = tv_loss(input_img)

            # 组合总损失
            loss = content_loss + style_loss + tv_loss_value
            loss.backward()

            run[0] += 1

            if progress_callback is not None:
                progress = int((run[0] / num_steps) * 100)
                progress_callback(min(100, progress))  # 确保不超过100%
            if run[0] % 50 == 0:
                print(
                    f"Step {run[0]}, "
                    f"Content loss: {content_loss.item():.2f}, "
                    f"Style loss: {style_loss.item():.2f}, "
                    f"TV loss: {tv_loss_value.item():.4f}"  # 新增TV Loss显示
                )

            return loss

        optimizer.step(closure)

    # 取消归一化并返回结果
    # unnormalize = transforms.Normalize(
    #     mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    #     std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    # )
    unnormalize = transforms.Normalize(
        mean=[-0.45 / 0.2, -0.4 / 0.2, -0.4 / 0.2], std=[1 / 0.2, 1 / 0.2, 1 / 0.2]
    )
    result = unnormalize(input_img)
    result = post_process(result)
    return result


if __name__ == "__main__":
    content_image_path = "./images/Mona_lisa.jpg"
    style_image_path = "./images/d.jpg"
    output_image_path = "./output/result_1.jpg"

    content_img = load_image(content_image_path)
    style_img = load_image(style_image_path)
    print("\n开始风格迁移...")
    start_time = time.time()  # 开始计时

    result = style_transfer(content_img, style_img)

    end_time = time.time()  # 结束计时

    save_image(result, output_image_path)
    print(
        f"风格迁移完成，图像已保存为 {output_image_path}"
        f"\t耗时={end_time - start_time:.2f}秒"
    )
