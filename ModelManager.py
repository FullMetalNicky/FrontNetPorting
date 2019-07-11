import torch


class ModelManager:
    def Read(self, filename, model):

        checkpoint_dict = torch.load(filename)
        epoch = checkpoint_dict['epoch']

        if torch.cuda.is_available():
            model.load_state_dict(checkpoint_dict['model'])

        else:
            state_dict = torch.load(filename, map_location='cpu')
            model.load_state_dict(state_dict, strict=True)

        return epoch

    def Write(self, model, epoch, filename):
        checkpoint_dict = {
            'model': model.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint_dict, filename)