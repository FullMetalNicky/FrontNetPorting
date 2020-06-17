import torch


class ModelManager:
    @staticmethod
    def Read(filename, model):
        """Reads model file

            Parameters
            ----------
            filename : str
                location of the model file
            model : NN class
                A PyTorch NN object

            Returns
            -------
            int
                the number of epochs this model was trained
            """

        state_dict = torch.load(filename, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        epoch = state_dict['epoch']


        return epoch

    @staticmethod
    def ReadQ(filename, model):
        """Reads model file

            Parameters
            ----------
            filename : str
                location of the model file
            model : NN class
                A PyTorch NN object

            Returns
            -------
            int
                the number of epochs this model was trained
            """

        state_dict = torch.load(filename, map_location='cpu')
        epoch = state_dict['epoch']
        try:
            prec_dict = state_dict['precision']
        except KeyError:
            prec_dict = None

        try:
            model.load_state_dict(state_dict['state_dict'], strict=True)
        except KeyError:
            model.load_state_dict(state_dict, strict=False)

        return epoch, prec_dict

    @staticmethod
    def Write(model, epoch, filename):
        """writes model to file

        Parameters
        ----------
        model : NN class
                A PyTorch NN object
        epoch: int
            the number of epochs this model was trained
        filename : str
            name and location of the model to be saved
        """

        checkpoint_dict = {
            'model': model.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint_dict, filename)
