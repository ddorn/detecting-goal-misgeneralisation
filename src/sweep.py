import click
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import wandb

import __init__ as M

config = dict(
    method="bayes",
    name="3goals-blind",
    metric=dict(
        goal="maximize",
        name="rollout/ep_rew_mean",
    ),
    parameters=dict(
        lr=dict(min=1e-7, max=1e-2, distribution="log_uniform_values"),
        env_size=dict(value=4),
        n_layers=dict(min=2, max=4)
    ),
)


def do_run():
    with wandb.init(sync_tensorboard=True) as run:

        env = M.wrap(
            lambda: M.ThreeGoalsEnv(run.config.env_size),
            lambda e: M.ColorBlindWrapper(e, reduction='max', reward_indistinguishable_goals=True),
            M.AddTrueGoalToObsFlat,
            lambda e: M.AddSwitch(e, 1, lambda _: 0),  # No switch, but we still use SwitchMLP as the architecture...
        )

        n_env = 4

        policy = PPO(
            M.SwitchActorCriticPolicy,
            make_vec_env(env, n_envs=n_env, seed=42),
            policy_kwargs=dict(
                arch_kwargs=dict(
                    switched_layer=0,
                    hidden=[64, 32, 32][:run.config.n_layers - 1],
                    out_dim=32,
                    n_switches=1,
                    l1_reg=3e-5,
                ),
            ),
            learning_rate=run.config.lr,
            n_steps=2_048 // n_env,
            tensorboard_log="../run_logs",
            seed=42,
            device='cpu',
        )

        policy.learn(total_timesteps=1_000_000, callback=M.WandbWithBehaviorCallback(env()))

        stats = M.evaluate(policy, env(), add_to_wandb=True, plot=False)
        run.summary.update(stats)


@click.group()
def cli():
    pass


@cli.command()
def init():
    sweep_id = wandb.sweep(config, project="ppo_7x7")
    print(f"Created sweep with id {sweep_id}")


@cli.command()
@click.argument("sweep_id")
@click.option("--count", default=1)
def run(sweep_id: str, count: int):
    wandb.agent(sweep_id, do_run, count=count, project="ppo_7x7")


if __name__ == "__main__":
    cli()
