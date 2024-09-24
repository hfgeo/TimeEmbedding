from netquery.module import *
from netquery.date2vec.Date2VecModel import Date2VecConvert
import netquery.date2vec.Date2VecModel


# time-NN forward encoder    -- xubing 2023-10-10 15:17:53
class TimeForwardEncoderNN(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function

    """

    def __init__(self, spa_embed_dim, coord_dim=6, ffn=None, device="cpu"):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(TimeForwardEncoderNN, self).__init__()

        self.spa_embed_dim = spa_embed_dim
        self.coord_dim = coord_dim
        self.ffn = ffn

        self.device = device

    def make_input_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception("Unknown coords data type for GridCellSpatialRelationEncoder")

        # (batch_size, num_context_pt, coord_dim)
        coords_mat = np.asarray(coords).astype(float)

        return coords_mat

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        spr_embeds = self.make_input_embeds(coords)

        # spr_embeds: (batch_size, num_context_pt, input_embed_dim)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:
            return spr_embeds


class TimeDirectEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function

    """

    def __init__(self, spa_embed_dim, coord_dim=6, ffn=None, device="cpu"):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(TimeDirectEncoder, self).__init__()

        self.spa_embed_dim = spa_embed_dim
        self.coord_dim = coord_dim
        self.ffn = ffn

        self.device = device

    def make_input_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception("Unknown coords data type for GridCellSpatialRelationEncoder")

        # (batch_size, num_context_pt, coord_dim)
        coords_mat = np.asarray(coords).astype(float)

        return coords_mat

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        spr_embeds = self.make_input_embeds(coords)

        # spr_embeds: (batch_size, num_context_pt, input_embed_dim)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:
            return spr_embeds


class Time2VecEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function

    """

    def __init__(self, spa_embed_dim, coord_dim=6, ffn=None, device="cpu", model_path='../date2vec/models/d2v_cos'
                                                                                      '/d2v_104675_1.6184912734574624.pth'):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(Time2VecEncoder, self).__init__()

        self.spa_embed_dim = spa_embed_dim
        self.coord_dim = coord_dim
        self.ffn = ffn

        self.device = device
        self.model_path = model_path

    def make_input_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception("Unknown coords data type for GridCellSpatialRelationEncoder")

        # (batch_size, num_context_pt, coord_dim)
        coords = np.asarray(coords).astype(float)
        # coords = np.array(coords, dtype=float)

        coords = coords.reshape(coords.shape[0], -1)
        model = Date2VecConvert(model_path=self.model_path)
        coords_mat = model(coords) / 50
        # coords_mat = model(coords)

        # model = torch.load(self.model_path, map_location=self.device).eval()
        #
        # with torch.no_grad():
        #     coords_mat = model.encode(torch.Tensor(coords)).squeeze(0)
        coords_mat = coords_mat.unsqueeze(1)
        return coords_mat

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        spr_embeds = self.make_input_embeds(coords)

        # spr_embeds: (batch_size, num_context_pt, input_embed_dim)

        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:
            return spr_embeds




class TheoryTimeRelationEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function
    """

    def __init__(self, spa_embed_dim, min_time, max_time, use_log=None, coord_dim=12, sigmoid="sin", ffn=None,
                 device="cpu", is_time_period=True, single_ffn=None):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(TheoryTimeRelationEncoder, self).__init__()

        self.spa_embed_dim = spa_embed_dim
        self.coord_dim = coord_dim
        self.ffn = ffn
        self.sigmoid = sigmoid
        self.offset = 0
        self.min_time = min_time
        self.max_time = max_time
        self.use_log = use_log
        self.is_time_period = is_time_period
        self.single_ffn = single_ffn

        # there unit vectors which is 120 degree apart from each other
        self.unit_vec1 = np.asarray(
            [1.0, 1.0 / 12, 1.0 / 12 / 30, 1.0 / 12 / 30 / 24, 1.0 / 12 / 30 / 24 / 60,
             1.0 / 12 / 30 / 24 / 60 / 60])  # 正向
        self.unit_vec2 = np.asarray(
            [-1.0, -1.0 / 12, -1.0 / 12 / 30, -1.0 / 12 / 30 / 24, -1.0 / 12 / 30 / 24 / 60,
             -1.0 / 12 / 30 / 24 / 60 / 60])  # 负向

        # # there unit vectors which is 120 degree apart from each other
        # self.unit_vec1 = np.asarray(
        #     [1.0, 1.0 / 12, 1.0 / 12 / 30, 1.0 / 12 / 30 / 24, 1.0 / 12 / 30 / 24 / 60, 1.0 / 12 / 30 / 24 / 60 / 60,
        #      1.0, 1.0 / 12, 1.0 / 12 / 30, 1.0 / 12 / 30 / 24, 1.0 / 12 / 30 / 24 / 60,
        #      1.0 / 12 / 30 / 24 / 60 / 60])  # 正向
        # self.unit_vec2 = np.asarray([-1.0, -1.0 / 12, -1.0 / 12 / 30, -1.0 / 12 / 30 / 24, -1.0 / 12 / 30 / 24 / 60,
        #                              -1.0 / 12 / 30 / 24 / 60 / 60, -1.0, -1.0 / 12, -1.0 / 12 / 30,
        #                              -1.0 / 12 / 30 / 24, -1.0 / 12 / 30 / 24 / 60,
        #                              -1.0 / 12 / 30 / 24 / 60 / 60])  # 负向

        # self.unit_vec1 = np.asarray(
        #     [1.0, 1.0,1.0,1.0, 1.0,1.0,1.0, 1.0,1.0,1.0, 1.0,1.0])  # 正向
        # self.unit_vec2 = np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])  # 负向

        self.device = device

    def make_input_embeds1(self, coords):

        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception("Unknown coords data type for GridCellSpatialRelationEncoder")

        # (batch_size, num_context_pt, coord_dim)
        # coords_mat = np.asarray(coords).astype(float)

        # 设置偏移量
        if self.min_time < 0:
            self.offset = abs(self.min_time)

        # (batch_size, num_context_pt, coord_dim)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]

        # 偏移量
        if self.offset != 0:
            coords_mat[:, 0, 0] += self.offset
            coords_mat[:, 0, 6] += self.offset

        # 使用对数方法调整输入尺度
        self.use_log = False
        if self.use_log:
            log_result = np.floor(np.log10(self.min_time))
            coords_mat[:, 0, 0] /= np.exp(log_result - 1)
            coords_mat[:, 0, 6] /= np.exp(log_result - 1)

        # compute the dot product between [deltaX, deltaY] and each unit_vec
        # (batch_size, num_context_pt, 1)
        angle_mat1 = np.expand_dims(np.multiply(coords_mat, self.unit_vec1), axis=-1).reshape(batch_size, 1, 12)
        # (batch_size, num_context_pt, 1)
        angle_mat2 = np.expand_dims(np.multiply(coords_mat, self.unit_vec2), axis=-1).reshape(batch_size, 1, 12)

        if self.sigmoid == "sin":

            angle_sig1 = np.sin(angle_mat1)[:, :, :6]
            angle_sig2 = np.sin(angle_mat2)[:, :, :6]
        elif self.sigmoid == "cos":
            angle_sig1 = np.cos(angle_mat1)[:, :, :6]
            angle_sig2 = np.cos(angle_mat2)[:, :, :6]
        else:
            angle_sig1 = 0
            angle_sig2 = 0
            pass

        mid = (angle_mat1[:, :, :6] + angle_mat1[:, :, 6:]) / 2
        period = angle_mat1[:, :, 6:] - angle_mat1[:, :, :6]
        # (batch_size, num_context_pt, 6)
        angle_mat = np.concatenate([coords_mat, angle_mat1, angle_mat2, mid, period, angle_sig1, angle_sig2], axis=-1)
        # (batch_size, num_context_pt, 1, 6)
        angle_mat = np.expand_dims(angle_mat, axis=-2)
        # (batch_size, num_context_pt, frequency_num*6)
        spr_embeds = np.reshape(angle_mat, (batch_size, num_context_pt, -1))

        return spr_embeds

    def make_input_embeds(self, coords):

        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception("Unknown coords data type for GridCellSpatialRelationEncoder")

        # (batch_size, num_context_pt, coord_dim)
        # coords_mat = np.asarray(coords).astype(float)

        # 设置偏移量
        if self.min_time < 0:
            self.offset = abs(self.min_time)

        # (batch_size, num_context_pt, coord_dim)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]

        start_time = coords_mat[:, :, :6]
        end_time = coords_mat[:, :, 6:]

        # 偏移量
        if self.offset != 0:
            coords_mat[:, 0, 0] += self.offset
            coords_mat[:, 0, 6] += self.offset

        # 使用对数方法调整输入尺度
        self.use_log = False
        if self.use_log:
            log_result = np.floor(np.log10(self.min_time))
            start_time[:, 0, 0] /= np.exp(log_result - 1)
            end_time[:, 0, 6] /= np.exp(log_result - 1)

        # compute the dot product between [deltaX, deltaY] and each unit_vec
        # (batch_size, num_context_pt, 1)
        angle_mat1 = np.expand_dims(np.multiply(start_time, self.unit_vec1), axis=-1).reshape(batch_size, 1, 6)
        # (batch_size, num_context_pt, 1)
        angle_mat2 = np.expand_dims(np.multiply(end_time, self.unit_vec2), axis=-1).reshape(batch_size, 1, 6)

        if self.sigmoid == "sin":

            angle_sig1 = np.sin(angle_mat1)
            angle_sig2 = np.sin(angle_mat2)
        elif self.sigmoid == "cos":
            angle_sig1 = np.cos(angle_mat1)
            angle_sig2 = np.cos(angle_mat2)
        else:
            angle_sig1 = 0
            angle_sig2 = 0
            pass

        start_time = torch.FloatTensor(start_time).to(self.device)
        end_time = torch.FloatTensor(end_time).to(self.device)
        angle_mat1 = torch.FloatTensor(angle_mat1).to(self.device)
        angle_mat2 = torch.FloatTensor(angle_mat2).to(self.device)
        angle_sig1 = torch.FloatTensor(angle_sig1).to(self.device)
        angle_sig2 = torch.FloatTensor(angle_sig2).to(self.device)

        mat1 = self.single_ffn(start_time)
        mat2 = self.single_ffn(end_time)

        mat3 = self.single_ffn(angle_mat1)
        mat4 = self.single_ffn(angle_mat2)

        mat5 = self.single_ffn(angle_sig1)
        mat6 = self.single_ffn(angle_sig2)

        mid = (mat1 + mat2) / 2
        period = mat2 - mat1
        # (batch_size, num_context_pt, 6)
        spr_embeds = torch.cat([mat1, mat2, mat3, mat4, mat5, mat6, period, mid], dim=-1)
        # angle_mat = np.concatenate([mat1, mat2, mat3, mat4, mat5, mat6, period, mid], axis=-1)
        # # (batch_size, num_context_pt, 1, 6)
        # angle_mat = np.expand_dims(angle_mat, axis=-2)
        # # (batch_size, num_context_pt, frequency_num*6)
        # spr_embeds = np.reshape(angle_mat, (batch_size, num_context_pt, -1))

        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        spr_embeds = self.make_input_embeds(coords)

        # spr_embeds: (batch_size, num_context_pt, input_embed_dim)

        # spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:
            return spr_embeds



if __name__ == "__main__":
    x1 = torch.Tensor([[[2019, 7, 23, 13, 23, 30]], [[2019, 7, 23, 13, 23, 30]]]).float()
    # x = x.reshape(-1, 1, 6)

    # model = Date2VecConvert('./date2vec/models/d2v_cos/d2v_0_377576.5.pth')
    # a = model(x)
    # print(a)
    x = [[[2020, 7, 23, 13, 23, 30]], [[2019, 7, 23, 13, 23, 30]]]
    spa = Time2VecEncoder(64, 6, model_path='./date2vec/models/d2v_cos/d2v_104675_1.6184912734574624.pth')
    # spa = TimeDirectEncoder(64,6)
    out = spa(x)
    print(out)

    # from utils import get_ffn,get_activation_function
    # # act = get_activation_function('leakyrelu')
    # import argparse
    #
    # # 创建一个 ArgumentParser 对象
    # parser = argparse.ArgumentParser(description="Description of your script")
    #
    # # 添加命令行参数
    # parser.add_argument('--use_layn', type=str, default="T",
    #                     choices=["T", "F"], help="Set use_layn to 'T' or 'F'")
    # parser.add_argument('--skip_connection', type=str, default="T",
    #                     choices=["T", "F"], help="Set skip_connection to 'T' or 'F'")
    # parser.add_argument('--spa_embed_dim', type=int, default=64,
    #                     help="Set spa_embed_dim (integer value)")
    # parser.add_argument('--num_hidden_layer', type=int, default=1,
    #                     help="Set num_hidden_layer (integer value)")
    # parser.add_argument('--dropout', type=float, default=0.5,
    #                     help="Set dropout (float value)")
    # parser.add_argument('--hidden_dim', type=int, default=512,
    #                     help="Set hidden_dim (integer value)")
    #
    # # 解析命令行参数
    # args = parser.parse_args()
    #
    # # coords = np.asarray(x1).astype(float)
    # # coords = np.array(coords, dtype=float)
    #
    # # coords = coords.reshape(coords.shape[0], -1)
    # ffn = get_ffn(args=args,input_dim=6,f_act='leakyrelu')
    # out1 = ffn(x1)
    # print(out1)
