from torch.utils.data import DataLoader, Dataset, random_split
from data import CustomData
from torchvision import transforms as T


def get_dl(root, transformations, bs, split = [0.9, 0.06, 0.04]):
    
    mean, std = [0.2250, 0.2250, 0.2250], [0.2505, 0.2505, 0.2505]
    tfs = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean = mean, std = std)])
   
    ds = CustomData(root =root,transformation=tfs)
    len_data = len(ds)
    tr_len = int(len_data * split[0])
    val_len = int(len_data * split[1])
    ts_len = len_data - (tr_len + val_len)
    tr_ds, val_ds, ts_ds = random_split(ds, lengths= [tr_len, val_len , ts_len])
    
    print(f'Trainset: {len(tr_ds)}')
    print(f'Validationset: {len(val_ds)}')
    print(f'Testset: {len(ts_ds)}')
    
    tr_dl = DataLoader(dataset=tr_ds, batch_size=bs, shuffle=True)
    val_dl = DataLoader(dataset=val_ds, batch_size=bs, shuffle=False)
    ts_dl = DataLoader(dataset=ts_ds, batch_size=1, shuffle=False)
    
    print("\n")   
    print(f'Train_dl: {len(tr_dl)}')
    print(f'Val_dl: {len(val_dl)}')
    print(f'Test_dl: {len(ts_dl)}')
    print(f'Classes: {ds.classes}')
    
    return tr_dl, val_dl, ts_dl, ds.classes 

# mean, std = [0.2250, 0.2250, 0.2250], [0.2505, 0.2505, 0.2505]
# tfs = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean = mean, std = std)])
# tr_dl, val_dl, ts_dl, classes = get_dl(root=root, transformations=tfs, bs=64)