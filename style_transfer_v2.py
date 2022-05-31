# This is a tutorial for style transfer in Sookmyung Wonmen's Univ's
# Deep Learning Course 2022 with Prof. Joo Yong Sim

# download link
# !wget https://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg 
# !wget https://pytorch.org/tutorials/_static/img/neural-style/dancing.jpg

from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models

models.vgg19(pretrained=True).features

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_size = 512
transform = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
])
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), 
            #                         std=(0.229, 0.224, 0.225))
            #                         ])
def load_image(img_path):
    img = Image.open(img_path)
    img = transform(img).unsqueeze(0)
    return img.to(device)

# denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))

def restore(img):
    # img = denorm(img) #.clamp_(0,1)
    img = img.squeeze().cpu().detach().numpy().transpose((1,2,0))
    return img
    
import matplotlib.pyplot as plt
style = load_image('picasso.jpg')
content = load_image('dancing.jpg')

device = torch.device('cuda')

class VGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.sel = ['0','5','10','19','28']
        self.vgg = models.vgg19(pretrained=True).features
    def forward(self,x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.sel:
                features.append(x)
        return features

vggf = VGGFeatures().to(device).eval()
con_features = vggf(content)

for con_feature in con_features:
    print(con_feature.shape)

generated = content.clone().requires_grad_(True)

    
optimizer = torch.optim.Adam([generated],lr = 0.005)
vggf.requires_grad_(False)

for step in range(1000):
    gen_features = vggf(generated)
    con_features = vggf(content)
    sty_features = vggf(style)

    style_loss = 0
    content_loss = 0
    for gen_feature, con_feature, sty_feature in zip(gen_features, con_features,\
                                                     sty_features):
        content_loss += torch.mean((gen_feature-con_feature)**2)

        # Calculate Gram Matrix
        b, c, h, w = gen_feature.shape
        G_gen = gen_feature.reshape(c,h*w).mm(gen_feature.reshape(c,h*w).t())
        G_sty = sty_feature.reshape(c,h*w).mm(sty_feature.reshape(c,h*w).t())
        style_loss += torch.mean((G_gen-G_sty)**2/(c*h*w))

    loss = content_loss + style_loss * 1e6
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step%10 == 0:
        print(f'Step: {step}, C loss: {content_loss}, S loss: {style_loss}')
    if step%50 == 0:
        img = generated.clone().squeeze()
        plt.figure(figsize=[15,5])
        plt.subplot(131)
        plt.imshow(restore(content)); plt.axis('off')
        plt.subplot(132)
        plt.imshow(restore(style)); plt.axis('off')
        plt.subplot(133)
        plt.imshow(restore(img))
        plt.axis('off')
        plt.show()

    
