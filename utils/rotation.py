import torch

def compute_rotation_matrix(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    assert X.shape == Y.shape
    """
    Compute the exact rotation matrix W such that Y = WX using SVD.
    Args:
        X (torch.Tensor): Input vector of shape (d,)
        Y (torch.Tensor): Target vector of shape (d,)
    Pre-cond:
        norm(X) == norm(Y)
    Returns:
        W (torch.Tensor): Rotation matrix of shape (d, d)
    """
    # Ensure vectors are normalized
    X = X / X.norm(p=2)
    Y = Y / Y.norm(p=2)

    # Compute the outer product matrix
    M = torch.ger(Y, X)  # Equivalent to Y @ X^T for 1D vectors

    # Perform SVD on M
    U, S, Vt = torch.linalg.svd(M)

    # Construct the rotation matrix
    W = U @ Vt

    # Ensure det(W) = 1 (rotation matrix condition, not reflection)
    if torch.det(W) < 0:
        # Adjust to ensure positive determinant
        U[:, -1] *= -1  # Flip the sign of the last column of U
        W = U @ Vt

    # print('Origin is: ', X)
    # print('Target is: ', Y)
    # print("Error:", (Y.double() - W.double() @ X.double()).norm(p=2))                 # Should be close to zero
    # print("Error:", torch.mean(torch.abs( (Y.double() - W.double() @ X.double()) )))  # Should be close to zero
    return W
