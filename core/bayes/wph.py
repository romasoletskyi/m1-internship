import pywph as pw
import time
import torch
import scipy.optimize as opt

def create_wph_operator(img, device=0):
    M, N = img.shape
    J = 6
    L = 4
    dn = 5

    wph_op = pw.WPHOp(M, N, J, L=L, dn=dn).to(device)
    return wph_op

def denoise(wph_op, d_coeffs, y0, sample_size=16, device=0, verbose=True):
    M, N = y0.shape
    eval_cnt = 0

    def objective(y):
        nonlocal eval_cnt
        start_time = time.time()
        
        if verbose:
            print(f"Evaluation: {eval_cnt}")

        y = y.reshape((M, N))
        n = torch.randn(sample_size, *y.shape, device=device)
        y_curr = torch.tensor(y, device=device) + n

        loss_tot = torch.zeros(1)
        y_curr, nb_chunks = wph_op.preconfigure(y_curr, requires_grad=True)
        for i in range(nb_chunks):
            y_coeffs_chunk, indices = wph_op.apply(y_curr, i, norm='auto', ret_indices=True)
            loss = torch.sum(torch.abs(y_coeffs_chunk.mean(dim=0) - d_coeffs[indices]) ** 2)
            loss.backward(retain_graph=True)
            loss_tot += loss.detach().cpu()
            del y_coeffs_chunk, indices, loss        
        
        if verbose:
            print(f"Loss: {loss_tot.item()} (computed in {(time.time() - start_time):.2f}s)")
            
        eval_cnt += 1
        y_grad = y_curr.grad.mean(dim=0).cpu().numpy().astype(y.dtype)
        
        return loss_tot.item(), y_grad.ravel()

    result = opt.minimize(objective, y0.ravel(), method='L-BFGS-B', jac=True, tol=None, options={"maxiter": 50, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20})
    final_loss, y_final, niter, msg = result['fun'], result['x'], result['nit'], result['message']
    y_final = y_final.reshape(y0.shape)
    
    if verbose:
        print(f"Synthesis ended in {niter} iterations with optimizer message: {msg}")
    
    return y_final, final_loss