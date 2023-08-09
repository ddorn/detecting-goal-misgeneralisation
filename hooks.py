from contextlib import contextmanager
from torch import nn, Tensor
import torch


class Cache(dict[str, Tensor]):

    def __str__(self):
        return "Cached activations:\n" + "\n".join(
            f"- {name}: {tuple(activation.shape)}" for name, activation in self.items()
        )

    def __getitem__(self, item: str) -> Tensor:
        # Find the key that matches and make sure it's unique.
        if item in self:
            return super().__getitem__(item)

        keys = [key for key in self.keys() if item in key]
        if len(keys) == 0:
            raise KeyError(item)
        elif len(keys) > 1:
            raise KeyError(f"Multiple keys match {item}: {keys}")
        return super().__getitem__(keys[0])


@contextmanager
def record_activations(module: nn.Module) -> Cache:
    """Context manager to record activations from a module and its submodules.

    Args:
        module (nn.Module): Module to record activations from.

    Yields:
        dist[str, Tensor]: Dictionary of activations, that will be populated once the
            context manager is exited.
    """

    activations = Cache()
    hooks = []

    skipped = set()
    module_to_name = {m: f"{n} {m.__class__.__name__}" for n, m in module.named_modules()}

    def hook(m: nn.Module, input: Tensor, output: Tensor):
        name = module_to_name[m]
        if not isinstance(output, Tensor):
            skipped.add(name)
        elif name not in activations:
            activations[name] = output.detach()
        else:
            activations[name] = torch.cat([activations[name], output.detach()], dim=0)

    for module in module.modules():
        hooks.append(module.register_forward_hook(hook))

    try:
        yield activations
    finally:
        for hook in hooks:
            hook.remove()

    print("Skipped:")
    for name in skipped:
        print("-", name)
