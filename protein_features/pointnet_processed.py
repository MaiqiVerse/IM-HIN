"""
Author: Benny
Date: Nov 2019
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='1', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, help='Experiment root', default='pointnet2_cls_ssg')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    parser.add_argument('--path_embedding', type=str, default='./embeddings/', help='Path to save embeddings')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = './coordinates/'

    # test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
    test_dataset = CustomDataset(root=data_path, args=args)
    # testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    os.makedirs(args.path_embedding, exist_ok=True)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        # instance_acc, class_acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_class=num_class)
        # log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
        generate_embedding(classifier.eval(), test_dataset, num_class=num_class, vote_num=args.num_votes, path=args.path_embedding)



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, args):
        super().__init__()
        self.root = root
        self.num = len(os.listdir(root))
        self.files = sorted(os.listdir(root))

    def __getitem__(self, index):
        file = self.files[index]
        points = []
        temperatures = []
        with open(os.path.join(self.root, file), 'r') as f:
            data = f.readlines()
            data = [d.strip().split(",") for d in data]
            for line in data:
                points.append([float(line[0]), float(line[1]), float(line[2])])
                temperatures.append(float(line[3]))
        if len(points) > 1024:
            # based on temperature (higher temperature, higher priority)
            index_range = np.argsort(temperatures)[::-1][:1024]
            index_range = sorted(index_range)
            points = [points[i] for i in index_range]

        points = np.array(points) / 300
        points = torch.tensor(points).float().transpose(1, 0)
        points = points.unsqueeze(0)
        self.index = index
        self.file_name = file
        return points

    def get_file_name(self):
        return self.file_name
        
    def __len__(self):
        return self.num


def generate_embedding(model, loader, num_class=40, vote_num=1, path=None):
    classifier = model.eval()
    embeddings = []
    for j, points in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points = points.cuda()

        # points = points.transpose(2, 1)
        vote_pool = torch.zeros(points.size()[0], 1024).cuda()

        for _ in range(vote_num):
            embedding = classifier.embed(points)
            vote_pool += embedding
        embedding = vote_pool / vote_num
        file_name = loader.get_file_name()
        embedding = embedding.cpu().numpy()
        file_name = file_name.replace('.txt', '.npy')
        np.save(os.path.join(path, file_name), embedding)



if __name__ == '__main__':
    args = parse_args()
    main(args)
