from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split

from dataset.dataset import SingleClassData, SingleDomainData, MultiDomainData
from util.util import *
from util.util import ConnectedDataIterator

def get_transform(instr, small_img=False, color_jitter=True, random_grayscale=True, domain=None):
    img_tr = []
    if small_img == False:
        img_tr.append(transforms.RandomResizedCrop((224, 224), (0.8, 1.0)))
        img_tr.append(transforms.RandomHorizontalFlip(0.5))
        # Domain-specific color jitter
        if domain == 'sketch':
            img_tr.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05))
        elif domain == 'photo':
            img_tr.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2))
        else:
            img_tr.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))
        # Texture
        img_tr.append(transforms.RandomAutocontrast(p=0.2))
        img_tr.append(transforms.RandomEqualize(p=0.2))
        img_tr.append(transforms.RandomPosterize(bits=4, p=0.2))
        # Geometric
        img_tr.append(transforms.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(0.9,1.1), shear=10, fill=0))
        img_tr.append(transforms.RandomPerspective(distortion_scale=0.2, p=0.2))
        img_tr.append(transforms.RandomVerticalFlip(0.1))
    else:
        img_tr.append(transforms.Resize((32, 32)))
    if color_jitter and domain is None:
        img_tr.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4))
    if random_grayscale:
        img_tr.append(transforms.RandomGrayscale(0.1))
    img_tr.append(transforms.ToTensor())
    if small_img == False:
        img_tr.append(transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    else:
        img_tr.append(transforms.Normalize([0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    transform = transforms.Compose(img_tr)
    # val/test不变
    if instr == 'train':
        return transform
    elif small_img == False:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


def get_dataloader(root_dir, domain, classes, batch_size, domain_class_dict=None, get_domain_label=True, get_class_label=True, instr="train", small_img=False, shuffle=True, drop_last=True, num_workers=4, split_rate=0.8, crossval=False, pin_memory=True):
    if not isinstance(domain, list): 
        domain = [domain]

    if isinstance(root_dir, list): 
        dataset_list = []
        for path in root_dir:
            sub_dataset = MultiDomainData(root_dir=path, domain=domain, classes=classes, domain_class_dict=domain_class_dict, get_domain_label=get_domain_label, get_classes_label=get_class_label, transform=get_transform(instr, small_img=small_img))
            dataset_list.append(sub_dataset)
        dataset = ConcatDataset(dataset_list)
    else:    
        # 只对单domain时传递domain[0]，多domain时可自定义
        domain_for_aug = domain[0] if len(domain) == 1 else None
        dataset = MultiDomainData(root_dir=root_dir, domain=domain, classes=classes, domain_class_dict=domain_class_dict, get_domain_label=get_domain_label, get_classes_label=get_class_label, transform=get_transform(instr, small_img=small_img, domain=domain_for_aug))

    val_loader = None
    if crossval: 
        train_size = int(len(dataset)*split_rate)
        val_size = len(dataset) - train_size
        dataset, val = random_split(dataset, [train_size, val_size])
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, pin_memory=pin_memory)
    
    return dataloader, val_loader


def get_domain_specific_dataloader(root_dir, domain, classes, group_length, batch_size, small_img, split_rate=0.8, crossval=False, val_workers=4, pin_memory=True):
    domain_specific_loader = []
    val_list = [] 

    for domain_name in domain: 
        dataloader_list = []
        if group_length == 1:
            for i, class_name in enumerate(classes):
                dataset = SingleClassData(root_dir=root_dir, domain=domain_name, classes=class_name, domain_label=-1, classes_label=i, transform=get_transform("train", small_img=small_img, domain=domain_name))

                if crossval:
                    train_size = int(len(dataset)*split_rate)
                    val_size = len(dataset) - train_size
                    scd, val = random_split(dataset, [train_size, val_size])
                    val_list.append(val)
                else:
                    scd = dataset

                loader = DataLoader(dataset=scd, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1, pin_memory=pin_memory)
                dataloader_list.append(loader)
        else:
            classes_partition = split_classes(classes_list=classes, index_list=[i for i in range(len(classes))], n=group_length)
            for class_name, class_to_idx in classes_partition:
                dataset = SingleDomainData(root_dir=root_dir, domain=domain_name, classes=class_name, domain_label=-1, get_classes_label=True, class_to_idx=class_to_idx, transform=get_transform("train", small_img=small_img, domain=domain_name))

                if crossval:
                    train_size = int(len(dataset)*split_rate)
                    val_size = len(dataset) - train_size
                    sdd, val = random_split(dataset, [train_size, val_size])
                    val_list.append(val)
                else:
                    sdd = dataset

                loader = DataLoader(dataset=sdd, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1, pin_memory=pin_memory)
                dataloader_list.append(loader)
        domain_specific_loader.append(ConnectedDataIterator(dataloader_list, batch_size=batch_size))
    val_loader = None
    if crossval and len(val_list) > 0:
        val_loader = DataLoader(ConcatDataset(val_list), batch_size=batch_size, shuffle=False, drop_last=False, num_workers=val_workers, pin_memory=pin_memory)
    return domain_specific_loader, val_loader

    








