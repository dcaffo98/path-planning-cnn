import torch


class Checkpoint(object):
    def __init__(self, model, epoch, min_loss, optimizer, scheduler=None, path=None):
        self.data = {
            'model': model.state_dict(),
            'epoch': epoch,
            'min_loss': min_loss,
            'optimizer': optimizer.state_dict()
        }
        if scheduler is not None:
            self.data['scheduler'] = scheduler.state_dict()
        if not path:
            path = "checkpoints/checkpoint_{}.pt".format(epoch)
        self.path = path

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def _save(self, path=None):
        if not path:
            path = self.path
        torch.save(self, path)
    
    @staticmethod
    def load_checkpoint(path, model, optimizer=None, scheduler=None, device=None):
        if not device:
            device = next(model.parameters()).device
        cp = torch.load(path, map_location=device)
        model.load_state_dict(cp['model'])
        if optimizer is not None:
            optimizer.load_state_dict(cp['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(cp['scheduler'])
        return cp

    @staticmethod
    def save(model, epoch, min_loss, optimizer, scheduler=None, path=None, **kwargs):
        cp = Checkpoint(model, epoch, min_loss, optimizer, scheduler, path)
        for k, v in kwargs.items():
            cp[k] = v
        cp._save()

    

    
if __name__ == '__main__':
    # Checkpoint.save(None, None, None, None, None, None, foo='qwerty', xxx=3)
    from model import OACNet
    from torch.optim import Adam
    model = OACNet().to('cuda:0')
    optimizer = Adam(model.parameters())
    Checkpoint.load_checkpoint('checkpoints/checkpoint_5.pt', model,optimizer)
        