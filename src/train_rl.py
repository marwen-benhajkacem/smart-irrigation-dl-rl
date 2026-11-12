"""Entraîne un agent d'irrigation (PPO / DQN / RecurrentPPO) sur gym-DSSAT.

Exemples :
  python -m src.train_rl --algo ppo --timesteps 20000 --tag smoke            # test rapide
  python -m src.train_rl --algo ppo --timesteps 500000 --seed 0 --tag ppo_v1_s0
  python -m src.train_rl --algo dqn --timesteps 500000 --seed 0 --tag dqn_s0
  python -m src.train_rl --algo ppo --forecaster models/forecaster_sim_lstm_s0.pt --tag ppo_v2_s0   # agent v2
  python -m src.train_rl --algo rppo --profile sensor --tag rppo_s0                                  # POMDP
Sortie : models/<tag>/{model.zip, vecnormalize.pkl, meta.json, train.monitor.csv}
"""
import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from . import config as C
from .env_wrapper import IrrigationEnv


def make_env_fn(seed, args, out_dir=None):
    def _f():
        fc = None
        if args.forecaster:
            from .forecaster_runtime import ForecasterRuntime
            fc = ForecasterRuntime(args.forecaster)
        env = IrrigationEnv(seed=seed, action_mode=args.action, profile=args.profile,
                            lam=args.lam, mu=args.mu, forecaster=fc)
        return Monitor(env, filename=str(out_dir / "train") if out_dir else None)
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="ppo", choices=["ppo", "dqn", "rppo"])
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--action", default="discrete", choices=["discrete", "continuous"])
    ap.add_argument("--profile", default="sensor", choices=list(C.OBS_PROFILES))
    ap.add_argument("--lam", type=float, default=C.LAMBDA_WATER)
    ap.add_argument("--mu", type=float, default=C.MU_STRESS)
    ap.add_argument("--forecaster", default="")
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--n_steps", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--eval_freq", type=int, default=20_000)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    if args.algo == "dqn" and args.action != "discrete":
        raise SystemExit("DQN exige des actions discrètes (--action discrete).")

    tag = args.tag or f"{args.algo}_{args.profile}_s{args.seed}"
    out = C.MODELS / tag
    out.mkdir(parents=True, exist_ok=True)

    venv = DummyVecEnv([make_env_fn(args.seed, args, out)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=args.gamma)
    eval_env = VecNormalize(DummyVecEnv([make_env_fn(C.SEEDS["val"][0], args)]),
                            training=False, norm_obs=True, norm_reward=False, clip_obs=10.0)

    try:
        import tensorboard  # noqa: F401
        tb = str(C.RESULTS / "tb")
    except ImportError:
        tb = None

    if args.algo == "ppo":
        model = PPO("MlpPolicy", venv, learning_rate=args.lr or 3e-4, n_steps=args.n_steps, batch_size=args.batch,
                    n_epochs=10, gamma=args.gamma, gae_lambda=0.95, clip_range=0.2, ent_coef=args.ent,
                    policy_kwargs=dict(net_arch=[64, 64]), seed=args.seed, verbose=1, tensorboard_log=tb, device="cpu")
    elif args.algo == "dqn":
        model = DQN("MlpPolicy", venv, learning_rate=args.lr or 1e-4, buffer_size=100_000, learning_starts=5_000,
                    batch_size=64, gamma=args.gamma, train_freq=4, gradient_steps=1, target_update_interval=1_000,
                    exploration_fraction=0.3, exploration_final_eps=0.05, policy_kwargs=dict(net_arch=[128, 128]),
                    seed=args.seed, verbose=1, tensorboard_log=tb, device="cpu")
    else:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO("MlpLstmPolicy", venv, learning_rate=args.lr or 3e-4, n_steps=1024, batch_size=256,
                             n_epochs=10, gamma=args.gamma, gae_lambda=0.95, ent_coef=args.ent,
                             policy_kwargs=dict(lstm_hidden_size=64, n_lstm_layers=1), seed=args.seed,
                             verbose=1, tensorboard_log=tb, device="cpu")

    callbacks = [
        EvalCallback(eval_env, n_eval_episodes=10, eval_freq=args.eval_freq, deterministic=True,
                     log_path=str(out / "eval"), verbose=1),
        CheckpointCallback(save_freq=100_000, save_path=str(out / "ckpt"), name_prefix="model", save_vecnormalize=True),
    ]
    if args.wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        wandb.init(project="irrigation-agent", name=tag, config=vars(args), sync_tensorboard=True)
        callbacks.append(WandbCallback())

    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)

    model.save(str(out / "model.zip"))
    venv.save(str(out / "vecnormalize.pkl"))
    meta = dict(vars(args), levels=C.ACTION_LEVELS_MM, tag=tag)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("Modèle sauvegardé dans", out)


if __name__ == "__main__":
    main()
