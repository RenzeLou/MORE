import os
import torch
import torch.nn as nn


class BasicModel(nn.Module):
    def __init__(self, save_dir):
        super(BasicModel, self).__init__()
        # self.save_path="/media/iot538/a73dbfc5-a8a0-4021-a841-3b7d7f3fd964/mnt/lrz/RSN_torch_by_lrz/checkpoints"
        # self.save_path = "/home/wyt/1_lrz/8-21/RSN_torch_by_lrz3/checkpoints"
        # self.save_path = "/media/iot538/a73dbfc5-a8a0-4021-a841-3b7d7f3fd964/mnt/lrz/87/RSN_torch_by_lrz1/checkpoints"
        self.save_path = os.path.join(save_dir, "checkpoints")  # "/home/wyt/lrz/nyt_test/RRL_nyt2/checkpoints"

    def load_model(self, model_name):
        direc = os.path.join(self.save_path, model_name)
        try:
            self.load_state_dict(torch.load(direc))
            print('model loaded from ' + direc)
        except:
            print("loading failed!")

    def save_model(self, model_name=None, global_step=0):
        try:
            direc = os.path.join(self.save_path, model_name)
            torch.save(self.state_dict(), direc + "_step_" + str(global_step) + ".pt")
            torch.save(self.state_dict(), direc + "_best.pt")
        except:
            if model_name is None:
                print("model_name can't be None!")
            else:
                print("save failed!No such path:", self.save_path)
        else:
            print("model saved.")

    def save_last_step(self, model_name=None):
        try:
            direc = os.path.join(self.save_path, model_name)
            torch.save(self.state_dict(), direc + "_last_step.pt")
        except:
            if model_name is None:
                print("model_name can't be None!")
            else:
                print("save failed!No such path:", self.save_path)
        else:
            print("model saved.")


if __name__ == "__main__":
    BM = BasicModel()
    BM.save_model(model_name="test")
    BM.load_model("test_best.pt")
