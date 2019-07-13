import torch


class ModelManager:
    @staticmethod
    def Read(filename, model):

        state_dict = torch.load(filename, map_location='cpu')

        model.load_state_dict(state_dict['model'])
        epoch = state_dict['epoch']

        return epoch

    @staticmethod
    def Write(model, epoch, filename):
        checkpoint_dict = {
            'model': model.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint_dict, filename)