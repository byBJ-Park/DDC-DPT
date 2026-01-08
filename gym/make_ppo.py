import gym  # gymnasium 대신 구버전 gym을 임포트합니다.
from stable_baselines3 import PPO
import os

# 저장 디렉토리 설정
SAVE_DIR = "Expert_policy"
os.makedirs(SAVE_DIR, exist_ok=True)

def train_full_ppo(env_id: str, save_name: str, total_steps: int):
    print(f"\n=== Training FULL PPO on {env_id} ({total_steps} steps) ===")
    
    # gym 0.21.0 방식으로 환경 생성
    try:
        env = gym.make(env_id)
    except gym.error.NameError:
        print(f"Error: {env_id} 환경을 찾을 수 없습니다. 패키지 설치를 확인하세요.")
        return

    # SB3 1.7.0 파라미터 설정
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        n_steps=2048, 
        batch_size=64, 
        gamma=0.99, 
        gae_lambda=0.95,
        device="auto" # GPU가 있다면 자동으로 사용합니다.
    )
    
    # 학습 시작
    model.learn(total_timesteps=total_steps)

    # 모델 저장 (.zip 확장자는 자동으로 붙습니다)
    save_path = os.path.join(SAVE_DIR, save_name)
    model.save(save_path)  
    print(f"Saved FULL trained PPO model: {save_path}.zip")
    
    env.close()

# 1. CartPole-v1
train_full_ppo("CartPole-v1", "CartPole-v1_PPO", 100_000)

# 2. LunarLander-v2 (box2d-py 설치 필요)
train_full_ppo("LunarLander-v2", "LunarLander-v2_PPO", 150_000)

# 3. Acrobot-v1
train_full_ppo("Acrobot-v1", "Acrobot-v1_PPO", 100_000)