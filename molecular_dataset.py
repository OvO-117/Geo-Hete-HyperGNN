from torch_geometric.data import InMemoryDataset, Data
import torch
import os.path as osp

class PretrainDataset(InMemoryDataset):
    def __init__(self, data_list, root='18w', transform=None, pre_transform=None, pre_filter=None):
        """
        Args:
            data_list: 包含PyG Data对象的列表
            root: 数据集根目录（可选）
            transform: 数据变换函数（可选）
            pre_transform: 预处理变换函数（可选）
            pre_filter: 数据过滤函数（可选）
        """
        self.data_list = data_list
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []  # 没有原始文件

    @property
    def processed_file_names(self):
        return ['data.pt']  # 处理后的文件名

    def download(self):
        pass  # 不需要下载

    def process(self):
        # 应用预处理过滤器和变换
        data_list = self.data_list
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # 保存处理后的数据
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])