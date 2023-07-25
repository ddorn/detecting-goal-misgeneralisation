import click
import wandb
import train

config = dict(
    method="bayes",
    name="ppo_7env",
    metric=dict(
        goal="maximize",
        name="rollout/ep_len_mean",
    ),
    parameters=dict(
        lr=dict(min=1e-6, max=1e-2, distribution="log_uniform_values"),
        n_epochs=dict(min=1, max=100),
        total_timesteps=dict(min=10_000, max=100_000, distribution="q_log_uniform_values"),
        batch_size=dict(min=10, max=20_000),
        n_steps=dict(min=500, max=20_000),
        env_size=dict(value=7),
        first_layer_size=dict(min=10, max=100),
        second_layer_size=dict(min=10, max=100),
        weight_decay=dict(min=0.0, max=0.01),
    )
)


def do_run():
    with wandb.init() as run:
        config = run.config
        policy, perfs = train.get_agent(
            bottom_right_odds=1 / 24,
            total_timesteps=config.total_timesteps,
            env_size=config.env_size,
            use_wandb=True,

            net_arch=(config.first_layer_size, config.second_layer_size),
            learning_rate=config.lr,
            n_epochs=config.n_epochs,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            weight_decay=config.weight_decay,
        )
        run.summary["perfs"] = perfs
        run.summary["model_size"] = sum(p.numel() for p in policy.policy.parameters())
        return perfs


@click.group()
def cli():
    pass


@cli.command()
def init():
    sweep_id = wandb.sweep(config)
    print(f"Created sweep with id {sweep_id}")


@cli.command()
@click.argument("sweep_id")
@click.option("--count", default=1)
def run(sweep_id: str, count: int):
    wandb.agent(sweep_id, do_run, count=count)


if __name__ == "__main__":
    cli()
