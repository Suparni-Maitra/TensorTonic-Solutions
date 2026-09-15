import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(x: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    x=np.asarray(x,dtype=float)
    y=np.asarray(y,dtype=float)
    N,D=x.shape
    w=np.zeros(D)
    b=0.0
    for _ in range (steps):
        z=x@w+b
        p=_sigmoid(z)
        dw=x.T@(p-y)/N
        db=np.mean(p-y)
        w-=lr*dw
        b-=lr*db
    return w,b
    
    
    pass