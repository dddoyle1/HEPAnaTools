import numpy as np
import scipy.optimize
import time    

class LocalTemplateFit:
    def __init__(self, cov, templates, decorrelate=False):
        self.V = cov

        # invert using svd decomposition
        u,s,v = np.linalg.svd(self.V)

        self.Vinv = np.dot(u, np.dot(np.diag((s+1e-5)**-1),u.T))

        # condition number -- a measure of the accuracy of the inversion
        self.kV = np.linalg.cond(self.V)
        
        self.T = np.array(templates)


        # bookkeeping for fixing templates in the fit
        self._param_mask = np.identity(self.T.shape[0])
        self._fixed_params = np.empty(self.T.shape[0])
        self._fixed_params[:] = np.nan

        
    def __call__(self, params):
        u = self.U(params)
        return np.dot((self.X - u), np.dot(self.Vinv, (self.X - u).T))

    def U(self, params):
        return np.dot(self._to_user_coords(params), self.T)

    def fix_template(self, template_idx, val):
        if np.isnan(self._param_mask.sum(axis=1)[template_idx]):
            self._fixed_params[template_idx] = val
        else:        
            self._param_mask = np.delete(self._param_mask, template_idx,axis=1)
            self._param_mask[:][template_idx] = np.nan
            self._fixed_params[template_idx] = val

    def release_template(self, template_idx):
        if np.isnan(self._param_mask.sum(axis=1)[template_idx]):
            self._param_mask = np.insert(self._param_mask, template_idx, np.zeros(self.T.shape[0]),axis=1)
            self._param_mask[:][template_idx] = 0
            self._param_mask[template_idx][template_idx] = 1
            self._fixed_params[template_idx] = np.nan

    def release_templates(self):
        self._fixed_params = np.empty(self.T.shape[0])
        self._fixed_params[:] = np.nan
        self._param_mask = np.identity(self.T.shape[0])

    
    def chisq(self, params, data):
        self.X = data
        return self(params)
    
    def minimize(self, data, *args, **kwargs):
        self.X = data
        
        bounds = [(0, 10) for _ in range(self._param_mask.shape[1])]

        start_time = time.perf_counter()
        results = scipy.optimize.dual_annealing(self, bounds, *args, local_search_options={'method': 'Newton-CG', 'jac': self._jacobian}, **kwargs)        
        end_time = time.perf_counter()
        
        results.x = self._to_user_coords(results.x)
        results.update({'k(V)': self.kV,
                        'time': end_time - start_time})
        return results

    def MINOS(self, data, *args, **kwargs):
        best_results = self.minimize(data, *args, **kwargs)
        start = best_results.x
        fun0 = best_results.fun

        errors = []
        error_def = 1
        minos_results = {}
        for i, s in enumerate(start):
            def func_MINOS(x):
                self.release_templates()
                self.fix_template(i, x)
                return self.minimize(data).fun - fun0 - error_def
    
            a = s
            b = s + 0.1
            f_a = func_MINOS(a)
    
            # find b such f(b) has the opposite sign of f(a)
            while True:        
                f_b = func_MINOS(b)
                if f_a * f_b < 0: break
                b += 0.1        

            up, results_up = scipy.optimize.bisect(func_MINOS, a, b, full_output=True)
            minos_results['e%d_up' % i] = results_up
    
            a = s
            b = s - 0.1
            # find b such f(b) has the opposite sign of f(a)
            while True:
                f_b = func_MINOS(b)
                if f_a * f_b < 0: break
                b -= 0.1
                
            dw, results_dw = scipy.optimize.bisect(func_MINOS, a, b, full_output=True)
            minos_results['e%d_dw' % i] = results_dw

            errors.append((dw, up))

        self.release_templates()        
        minos_results['dx'] = errors

        best_results.update(minos_results)
        return best_results
        
    def _jacobian(self, params):
        u = self.U(params)
        return -1 * self._to_optimizer_coords(np.dot(self.T, np.dot(self.Vinv, (self.X - u).T)) + np.dot(self.X - u, np.dot(self.Vinv, self.T.T)))
    #return self._to_optimizer_coords(-2 * np.dot(self.T, np.dot(self.Vinv, (self.X - u).T)))

    def _to_user_coords(self, optimizer_params):
        return np.nan_to_num(np.dot(self._param_mask, optimizer_params)) + np.nan_to_num(self._fixed_params)
                             
    def _to_optimizer_coords(self, user_params):
        return user_params[~np.isnan(self._param_mask.sum(axis=1))]

    
class GlobalTemplateFit(LocalTemplateFit):
    def __init__(self, cov, templates, decorrelate=False):
        """ 
        arguments:
           cov       : (NxN)   np.array
           templates : (txpxqxn) np.array 
                       where p x q x n = N
                       t is number of components
                       t x p x q = number of fit parameters
        """
        self.V = cov
        
        # invert using svd decomposition
        u,s,v = np.linalg.svd(self.V)
        self.Vinv = np.dot(v.transpose(), np.dot(np.diag(s**-1),u.transpose()))
        
        # condition number -- a measure of the accuracy of the inversion
        self.kV = np.linalg.cond(self.V)
        
        self.T = np.array(templates)

        if np.prod(self.T.shape[1:]) != self.V.shape[0]:
            raise TypeError(f'Template shape ({self.T.shape}) not compatible with covariance matrix ({self.V.shape})')
        
        self._param_mask = np.identity(np.prod(self.T.shape[:-1]))
        self._fixed_params = np.empty(np.prod(self.T.shape[:-1]))
        self._fixed_params[:] = np.nan

        # save PxP identity where P=txpxq fit parameters
        # used for jacobian 
        self.I = np.identity(np.prod(self.T.shape[:-1]))

        
    def release_templates(self):
        self._fixed_params = np.empty(np.prod(self.T.shape[:-1]))
        self._fixed_params[:] = np.nan
        self._param_mask = np.identity(np.prod(self.T.shape[:-1]))
        self.I = np.identity(np.prod(self.T.shape[:-1]))
        
    def fix_template(self, template_idx, val):        
        super().fix_template(template_idx, val)
        self.I = np.identity(self._param_mask.shape[1])
        
    def U(self, params):
        # flatten phase space
        flat_T = self.T.reshape((np.prod(self.T.shape[:-1]), self.T.shape[-1]))
        return np.dot(np.diag(self._to_user_coords(params)), flat_T).reshape((self.T.shape[0], np.prod(self.T.shape[1:]))).sum(axis=0)
    
    def _jacobian(self, params):
        N_t = np.array([self.U(i) for i in self.I])
        u = self.U(params)
        return -2 * np.dot(N_t, np.dot(self.Vinv, (self.X - u).T))
