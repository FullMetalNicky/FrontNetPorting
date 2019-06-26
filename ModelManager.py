import torch


class ModelManager:
    def Read(self, filename, model, optimizer):
        checkpoint_dict = torch.load(filename)
        epoch = checkpoint_dict['epoch']
        model.load_state_dict(checkpoint_dict['model'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint_dict['optimizer'])
        return epoch

    def Write(self, optimizer, model, epoch, filename):
        checkpoint_dict = {
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint_dict, filename)