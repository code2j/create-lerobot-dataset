import torch
import shutil
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def force_generate_all_files():
    # 저장 경로 설정 (사용자 경로에 맞춤)
    repo_id = "uon/triple-cam-task"
    root_path = Path("../outputs/dataset")
    dataset_path = root_path / repo_id

    # 1. 초기화: 기존에 실패한 빈 폴더가 있다면 삭제해야 에러가 안 납니다.
    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    # 2. 데이터셋 객체 생성 (이때 info.json만 생김)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root_path,
        fps=30,
        features={
            "observation.state": {"dtype": "float32", "shape": (6,)},
            "action": {"dtype": "float32", "shape": (6,)},
        },
        use_videos=True  # 테스트를 위해 비디오는 끔
    )

    # 3. 프레임 데이터 추가 (버퍼에만 쌓임)
    # v3.0 스펙상 'task'는 필수입니다.
    for i in range(10):
        dataset.add_frame({
            "observation.state": torch.randn(6),
            "action": torch.randn(6),
            "task": "test task"
        })

    # 4. 에피소드 저장 (이때 stats.json, data 청크, episodes 청크가 생성됨)
    print("💾 에피소드 저장 중... (이 단계에서 파일들이 생성됩니다)")
    dataset.save_episode()

    # 5. 최종 확정 (파일 라이터를 닫고 데이터셋 완성)
    dataset.finalize()
    print("✅ 모든 파일 생성 완료!")

if __name__ == "__main__":
    force_generate_all_files()