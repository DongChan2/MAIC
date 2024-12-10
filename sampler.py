import torch
from torch.utils.data import Sampler
import random

class PositiveLabelSampler(Sampler):
    def __init__(self, data_source, num_samples, positive_label=1):
        self.data_source = data_source
        self.num_samples = num_samples
        self.positive_label = positive_label

        # 양성 레이블 샘플의 인덱스를 추출
        self.positive_indices = [i for i, data in enumerate(data_source) if data[1] == positive_label]

        # 전체 인덱스
        self.total_indices = list(range(len(data_source)))

    def __iter__(self):
        batch = []
        for _ in range(self.num_samples):
            # 무작위로 하나의 양성 레이블 샘플을 선택
            pos_sample = random.choice(self.positive_indices)
            batch.append(pos_sample)

            # 나머지 샘플을 무작위로 선택
            while len(batch) < self.num_samples:
                idx = random.choice(self.total_indices)
                if idx not in batch:
                    batch.append(idx)

            yield batch
            batch = []

    def __len__(self):
        return len(self.data_source)

# 예시 사용 방법
# dataset = YourDataset()
# sampler = PositiveLabelSampler(dataset, num_samples=your_batch_size)
# dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
