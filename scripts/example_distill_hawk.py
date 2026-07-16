# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘   Knowledge Distillation: VGG16 â†’ Custom ConvNet                                   â•‘
# â•‘                                                                                     â•‘
# â•‘   Teacher : VGG16 (frozen backbone + trained classifier head)                      â•‘
# â•‘   Student : Pooled-output distillation (predict C-dim vector)                      â•‘
# â•‘   Output dim : 512                                                                  â•‘
# â•‘                                                                                     â•‘
# â•‘   Three-phase pipeline:                                                             â•‘
# â•‘     1. Train the teacher classifier (backbone frozen)                               â•‘
# â•‘     2. Distil into the student:  Loss = Î±Â·MSE + (1-Î±)Â·CE                          â•‘
# â•‘     3. Evaluate the student using the teacher's frozen classifier                   â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

# â”€â”€ Setup and Imports â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

import copy

import numpy as np
import torch
import torch.optim as optim
from glob import glob
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

gpu    = torch.cuda.is_available()
device = torch.device(0) if gpu else torch.device("cpu")
print(f"Using device: {device}")


# â”€â”€ Dataset Configuration (50/50 train/test) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

orig_dir   = "./images/corel"
nclasses   = 6
width, height, nchannels = 224, 224, 3
input_shape = (nchannels, height, width)

train_perc = 0.50
test_perc  = 0.50

data = glob(orig_dir + "/*.png")
num_train_samples = int(len(data) * train_perc)

np.random.seed(42)
np.random.shuffle(data)

trainset_files = data[:num_train_samples]
testset_files  = data[num_train_samples:]

print(f"Training samples : {len(trainset_files)}")
print(f"Test samples     : {len(testset_files)}")


# â”€â”€ Data Transformations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

prep = transforms.Compose([
    transforms.Resize((224, 224),
                      interpolation=transforms.InterpolationMode.BILINEAR,
                      antialias=True),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

aug = transforms.Compose([
    transforms.Resize((300, 300),
                      interpolation=transforms.InterpolationMode.BILINEAR,
                      antialias=True),
    transforms.RandomAffine(
        degrees=10, translate=(0.05, 0.10), scale=(0.9, 1.1), shear=(-2, 2),
        interpolation=transforms.InterpolationMode.BILINEAR, fill=0,
    ),
    transforms.CenterCrop(250),
    transforms.Resize((224, 224),
                      interpolation=transforms.InterpolationMode.BILINEAR,
                      antialias=True),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


class ImageDataset:
    """Loads images with labels encoded in filename (e.g. '1_xxx.png' â†’ label 0)."""

    def __init__(self, dataset, transform=None):
        self.dataset  = dataset
        self.targets  = [int(str(x).split("/")[-1].split("_")[0]) - 1 for x in dataset]
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ix):
        image = Image.open(self.dataset[ix])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[ix]


trainset = ImageDataset(trainset_files, aug)
testset  = ImageDataset(testset_files,  prep)

batchsize  = 32
trainload  = DataLoader(trainset, batch_size=batchsize, shuffle=True)
testload   = DataLoader(testset,  batch_size=batchsize, shuffle=False)
validload  = testload   # no separate validation set in 50/50 split

print(f"Training batches : {len(trainload)}")
print(f"Test batches     : {len(testload)}")


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# â•‘  PHASE 1 â€” Train the Teacher (VGG16)
# â•‘  Backbone is frozen; only the classifier head trains.
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class TeacherVGG16(nn.Module):
    """VGG16-based teacher.

    Interface:
        get_features(x)  â†’ (B, 512, 7, 7) pre-pool feature maps
        get_pooled(x)    â†’ (B, 512)        pooled vector
        forward(x)       â†’ (B, nclasses)   class logits
    """

    FEATURE_CHANNELS = 512

    def __init__(self, nclasses: int):
        super().__init__()
        base = models.vgg16(weights="IMAGENET1K_V1")
        for p in base.parameters():
            p.requires_grad = False
        self.backbone   = base.features
        self.avgpool    = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.FEATURE_CHANNELS, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, nclasses),
        )

    def get_features(self, x):
        return self.backbone(x)

    def get_pooled(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        return self.classifier(x)


teacher_model = TeacherVGG16(nclasses).to(device)


# â”€â”€ Classification helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def classification_criterion(preds, targets):
    ce   = nn.CrossEntropyLoss().to(device)
    loss = ce(preds, targets.long())
    acc  = (torch.max(preds, 1)[1] == targets.data).float().mean()
    return loss, acc


def train_batch_classification(model, data, optimizer):
    model.train()
    ims, targets = data
    ims, targets = ims.to(device), targets.to(device)
    optimizer.zero_grad()
    preds = model(ims)
    loss, acc = classification_criterion(preds, targets)
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()


@torch.no_grad()
def validate_batch_classification(model, data):
    model.eval()
    ims, targets = data
    ims, targets = ims.to(device), targets.to(device)
    preds = model(ims)
    loss, acc = classification_criterion(preds, targets)
    return loss.item(), acc.item()


@torch.no_grad()
def test_model(model, loader):
    losses, accs = [], []
    for data in loader:
        l, a = validate_batch_classification(model, data)
        losses.append(l); accs.append(a)
    return sum(losses) / len(losses), sum(accs) / len(accs)


# â”€â”€ Phase 1 training loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

print("=" * 70)
print("PHASE 1: Training Teacher Classifier")
print("=" * 70)

teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=1e-3)
n_epochs_teacher  = 10

for epoch in range(n_epochs_teacher):
    trn_losses, trn_accs = [], []
    for data in trainload:
        l, a = train_batch_classification(teacher_model, data, teacher_optimizer)
        trn_losses.append(l); trn_accs.append(a)

    val_losses, val_accs = [], []
    for data in validload:
        l, a = validate_batch_classification(teacher_model, data)
        val_losses.append(l); val_accs.append(a)

    if (epoch + 1) % 2 == 0:
        print(
            f"  Epoch {epoch+1:>3}/{n_epochs_teacher}"
            f"  trn_loss={sum(trn_losses)/len(trn_losses):.4f}"
            f"  trn_acc={sum(trn_accs)/len(trn_accs):.4f}"
            f"  val_loss={sum(val_losses)/len(val_losses):.4f}"
            f"  val_acc={sum(val_accs)/len(val_accs):.4f}"
        )

teacher_loss, teacher_acc = test_model(teacher_model, testload)
print(f"\nTeacher Test  loss={teacher_loss:.6f}  acc={teacher_acc:.4f} ({teacher_acc*100:.2f}%)")


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# â•‘  PHASE 2 â€” Train the Student (Pooled-output distillation)
# â•‘
# â•‘  Student architecture:
# â•‘    224Ã—224Ã—3 â†’ conv_block(48) â†’ 112Ã—112Ã—48
# â•‘             â†’ conv_block(512) â†’  56Ã—56Ã—512
# â•‘             â†’ AdaptiveAvgPool2d(1) â†’ Flatten â†’ Linear(512,512)
# â•‘
# â•‘  Loss = Î± Â· MSE(student_pooled, teacher_pooled)
# â•‘       + (1-Î±) Â· CE(classifier(student_pooled), y)
# â•‘
# â•‘  Î± = 0.7  (same as notebook)
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def conv_block(ch_in: int, ch_out: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(ch_in, ch_out, 5, 1, 2, bias=False),
        nn.BatchNorm2d(ch_out),
        nn.ReLU(),
        nn.MaxPool2d(3, stride, padding=1),
    )


class StudentModel(nn.Module):
    """Encoder (2 conv blocks) + predictor head targeting 512-dim teacher space.

    Output: (B, 512) vector matching the teacher's pooled representation.
    """

    def __init__(self, in_shape: tuple, out_channels: int = 512):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(in_shape[0], 48,  stride=2),   # 224 â†’ 112
            conv_block(48,          512, stride=2),   # 112 â†’  56
        )
        self.predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # 56Ã—56Ã—512 â†’ 1Ã—1Ã—512
            nn.Flatten(),              # â†’ 512
            nn.Linear(512, out_channels),
        )
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.features(x))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1); m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()


student_model = StudentModel(input_shape).to(device)
print(f"\nStudent output shape: {student_model(torch.randn(1,3,224,224).to(device)).shape}")


# â”€â”€ Distillation loss â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

ALPHA = 0.7

# Frozen copy of the teacher's classifier used to compute CE on student features
phase2_classifier = copy.deepcopy(teacher_model.classifier).to(device)
for p in phase2_classifier.parameters():
    p.requires_grad = False


def train_batch_distillation(student, teacher, classifier, data, optimizer, alpha=ALPHA):
    """Loss = Î± Â· MSE(student_pooled, teacher_pooled) + (1-Î±) Â· CE(cls(student_pooled), y)"""
    student.train(); teacher.eval(); classifier.eval()
    images, targets = data
    images, targets = images.to(device), targets.to(device)

    optimizer.zero_grad()

    with torch.no_grad():
        teacher_pooled = teacher.get_pooled(images)   # (B, 512), no grad

    student_pooled = student(images)                   # (B, 512), with grad

    mse_loss = nn.MSELoss()(student_pooled, teacher_pooled)

    # Teacher's classifier expects (B, 512, 1, 1) because of Flatten inside
    pooled_4d = student_pooled.unsqueeze(-1).unsqueeze(-1)
    preds     = classifier(pooled_4d)
    ce_loss   = nn.CrossEntropyLoss()(preds, targets.long())

    loss = alpha * mse_loss + (1 - alpha) * ce_loss
    loss.backward()
    optimizer.step()

    acc = (torch.max(preds, 1)[1] == targets.data).float().mean().item()
    return loss.item(), mse_loss.item(), ce_loss.item(), acc


@torch.no_grad()
def validate_batch_distillation(student, teacher, classifier, data, alpha=ALPHA):
    student.eval(); teacher.eval(); classifier.eval()
    images, targets = data
    images, targets = images.to(device), targets.to(device)

    teacher_pooled = teacher.get_pooled(images)
    student_pooled = student(images)

    mse_loss = nn.MSELoss()(student_pooled, teacher_pooled)

    pooled_4d = student_pooled.unsqueeze(-1).unsqueeze(-1)
    preds     = classifier(pooled_4d)
    ce_loss   = nn.CrossEntropyLoss()(preds, targets.long())

    loss = alpha * mse_loss + (1 - alpha) * ce_loss
    acc  = (torch.max(preds, 1)[1] == targets.data).float().mean().item()
    sim  = nn.CosineSimilarity(dim=1)(student_pooled, teacher_pooled).mean().item()

    return loss.item(), mse_loss.item(), ce_loss.item(), acc, sim


# â”€â”€ Phase 2 training loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

print("=" * 70)
print(f"PHASE 2: Training Student  (alpha={ALPHA})")
print(f"Loss = {ALPHA} * MSE + {1-ALPHA} * CrossEntropy")
print("=" * 70)

student_optimizer = optim.Adam(student_model.parameters(), lr=1e-4)
lr_scheduler      = optim.lr_scheduler.StepLR(student_optimizer, step_size=15, gamma=0.5)
n_epochs_phase2   = 50

for epoch in range(n_epochs_phase2):
    trn = {"loss": [], "mse": [], "ce": [], "acc": []}
    for data in trainload:
        l, m, c, a = train_batch_distillation(
            student_model, teacher_model, phase2_classifier, data, student_optimizer, ALPHA
        )
        trn["loss"].append(l); trn["mse"].append(m)
        trn["ce"].append(c);   trn["acc"].append(a)

    val = {"loss": [], "mse": [], "ce": [], "acc": [], "sim": []}
    for data in validload:
        l, m, c, a, s = validate_batch_distillation(
            student_model, teacher_model, phase2_classifier, data, ALPHA
        )
        val["loss"].append(l); val["mse"].append(m)
        val["ce"].append(c);   val["acc"].append(a); val["sim"].append(s)

    lr_scheduler.step()

    if (epoch + 1) % 10 == 0:
        def _m(d, k): return sum(d[k]) / len(d[k])
        print(
            f"  Epoch {epoch+1:>3}/{n_epochs_phase2}"
            f"  trn_loss={_m(trn,'loss'):.4f}  trn_mse={_m(trn,'mse'):.4f}"
            f"  trn_ce={_m(trn,'ce'):.4f}  trn_acc={_m(trn,'acc'):.4f}"
            f"  val_loss={_m(val,'loss'):.4f}  val_sim={_m(val,'sim'):.4f}"
        )

print("\nPhase 2 training complete!")


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# â•‘  PHASE 3 â€” Evaluate the Distilled Student
# â•‘  Student encoder + predictor combined with the teacher's frozen classifier.
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class DistilledModel(nn.Module):
    """Student encoder+predictor â†’ teacher's frozen classifier.

    The student already outputs a C-dim vector; reshape to (B,C,1,1) before
    passing through the teacher's classifier (which starts with Flatten).
    """

    def __init__(self, trained_student: StudentModel, teacher_classifier: nn.Module):
        super().__init__()
        self.features   = copy.deepcopy(trained_student.features)
        self.predictor  = copy.deepcopy(trained_student.predictor)
        self.classifier = copy.deepcopy(teacher_classifier)
        for p in self.classifier.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.predictor(x)                   # (B, 512)
        x = x.unsqueeze(-1).unsqueeze(-1)        # (B, 512, 1, 1)
        return self.classifier(x)


distilled_model = DistilledModel(student_model, teacher_model.classifier).to(device)

print("=" * 70)
print("PHASE 3: Evaluating Distilled Student Model")
print("=" * 70)

distilled_loss, distilled_acc = test_model(distilled_model, testload)

print(f"\nDistilled Student  loss={distilled_loss:.6f}  acc={distilled_acc:.4f} ({distilled_acc*100:.2f}%)")
print(f"Teacher            loss={teacher_loss:.6f}  acc={teacher_acc:.4f} ({teacher_acc*100:.2f}%)")
print(f"Difference         {(distilled_acc - teacher_acc)*100:+.2f}%")


# â”€â”€ Model Size Comparison â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


teacher_total,  teacher_trainable  = count_parameters(teacher_model)
distilled_total, distilled_trainable = count_parameters(distilled_model)

print("\n" + "=" * 70)
print("MODEL SIZE COMPARISON")
print("=" * 70)
print(f"Teacher total parameters   : {teacher_total:,}")
print(f"Distilled total parameters : {distilled_total:,}")
print(f"\nStudent is {teacher_total / distilled_total:.2f}x lighter")
print(f"Student has {(1 - distilled_total/teacher_total)*100:.1f}% fewer parameters")


# â”€â”€ Summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Teacher  : VGG16 (frozen backbone + trained classifier head)
# Distill  : Pooled-output â€” predict C-dim vector
# Loss     : Î±Â·MSE(student_pooled, teacher_pooled) + (1-Î±)Â·CE(cls(student_pooled), y)
# Alpha    : 0.7
# Eval     : student encoder+predictor + teacher classifier (no fine-tuning)