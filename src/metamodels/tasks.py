import torch
from jaxtyping import Float
from torch import Tensor


def gen_eigenvalues(
        batch_size: int,
        mat_size: int) -> tuple[Float[Tensor, "batch size size"], Float[Tensor, "batch size"]]:
    """
    Generate random matrices (inputs) and their eigenvalues (targets).
    """
    # Sample eigenvalues first
    eigenvalues = torch.rand(batch_size, mat_size) * 2 - 1
    eigenvalues = torch.sort(eigenvalues, dim=-1, descending=True).values
    # Sample a random rotation
    rotation = torch.randn(batch_size, mat_size, mat_size)
    # Orthonormalize the rotation
    rotation, _ = torch.linalg.qr(rotation)
    # Generate the matrices
    matrices = torch.matmul(
        torch.matmul(rotation, torch.diag_embed(eigenvalues)),
        rotation.transpose(-1, -2),
    )
    return matrices, eigenvalues
