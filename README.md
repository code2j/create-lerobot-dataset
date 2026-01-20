###  데이터셋 시각화 명령어
```shell
lerobot-dataset-viz     --repo-id uon/multi-cam-joint-task     --root outputs/uon/multi-cam-joint-task     --episode-index 1
```

해당 데이터셋 안에서 사용하면 됨
```shell
cd dataset/uon/test_dataset
lerobot-dataset-viz --root . --repo-id test_dataset --episode-index 0
```


### 모델 학습 명령어
```shell
lerobot-train \
    --dataset.repo_id uon/multi-cam-joint-task \
    --dataset.root /home/jusik/TEST/test_download-dataset/outputs/uon/multi-cam-joint-task \
    --policy.type act \
    --output_dir outputs/train/act_uon \
    --batch_size 1 \
    --steps 50000 \
    --policy.push_to_hub false
```

```shell
lerobot-train \
     --dataset.repo_id test_dataset \
     --dataset.root /home/jusik/workspace/lerobot-dataset-collector/dataset/test_dataset \
     --policy.type act \
     --output_dir dataset/train/act_uon \
     --batch_size 1 \
     --steps 50000 \
     --policy.push_to_hub false
```

dev1
- main3.py를 위한 개발 브렌치